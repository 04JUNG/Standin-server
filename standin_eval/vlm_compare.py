"""VLM 프롬프트 A/B 비교 게이트.

새 프롬프트(예: p2-person-tags)를 켜기 전에, 기존 항목(route·인원수·박스·얽힘)의 판단이
흔들리지 않는지 같은 이미지에서 확인한다.

- 이미지마다 A와 B를 번갈아 `repeats`번씩 부른다(반복마다 시작 순서를 바꾼다).
- 같은 프롬프트끼리의 차이(A↔A)를 잡음 바닥으로 삼고, A↔B 차이가 그보다 얼마나 큰지로
  판정한다. VLM은 temperature 0이어도 응답이 흔들리므로, 바닥 없이 비교하면 잡음을 회귀로
  읽는다.
- 응답 캐시는 쓰지 않는다. 같은 응답을 재생하면 잡음 바닥이 0이 된다.

보고서에는 컷 ID와 집계 수치만 남는다. 이미지와 응답 원문은 저장하지 않는다.
실제 provider로 돌리려면 API 키와, 그 이미지들을 provider에 보낼 권리가 있어야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
import statistics
import time
from typing import Callable, Iterable, Optional

# 계획(P2 평가 게이트)의 통과 기준. 바꾸면 docs/POSE_GAP_LOOP.md도 함께 고친다.
GATE = {
    "parse_failures_b_max": 0,
    "route_change_excess_max": 0.02,
    "count_agreement_drop_max": 0.03,
    "box_iou_median_drop_max": 0.03,
    "entangled_change_excess_max": 1.0,
    "latency_p95_increase_max": 0.15,
    "alignment_rate_min": 0.95,
}


@dataclass
class Sample:
    """VLM 호출 한 번의 결과에서 비교에 쓰는 값만."""
    cut_id: str
    arm: str                      # "A" | "B"
    ok: bool
    latency_ms: float
    error_type: Optional[str] = None
    route: Optional[str] = None
    num_people: Optional[int] = None
    entangled: Optional[bool] = None
    # 이미지 크기로 나눈 0~1 박스. 형식이 잘못된 박스는 None 자리로 남는다.
    boxes: list = field(default_factory=list)
    # 인물별 태그 배열이 인원수와 길이가 맞았는가. 프롬프트가 묻지 않았으면 None.
    aligned: Optional[bool] = None
    tagged_people: int = 0        # action이나 view가 하나라도 있는 인물 수


def summarize(vlm, cut_id: str, arm: str, latency_ms: float, width: int, height: int) -> Sample:
    from src.person_tags import PERSON_TAG_KEYS
    from src.routing import route

    boxes = []
    for box in vlm.approx_boxes:
        boxes.append(None if box is None else (
            box.x1 / width, box.y1 / height, box.x2 / width, box.y2 / height))
    asked = any(key in vlm.raw for key in PERSON_TAG_KEYS)
    aligned = None
    if asked:
        aligned = all(isinstance(vlm.raw.get(key), list) and len(vlm.raw[key]) == vlm.num_people
                      for key in PERSON_TAG_KEYS)
    tagged = sum(
        1 for index in range(vlm.num_people)
        if (index < len(vlm.person_actions) and vlm.person_actions[index] is not None)
        or (index < len(vlm.person_views) and vlm.person_views[index] is not None))
    return Sample(cut_id=cut_id, arm=arm, ok=True, latency_ms=latency_ms, route=route(vlm),
                  num_people=vlm.num_people, entangled=vlm.relationship.is_entangled,
                  boxes=boxes, aligned=aligned, tagged_people=tagged)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def matched_box_iou(first: list, second: list) -> Optional[float]:
    """박스를 IoU가 큰 쌍부터 하나씩 짝지은 평균 IoU. 짝이 하나도 없으면 None."""
    left = [box for box in first if box is not None]
    right = [box for box in second if box is not None]
    pairs = sorted(((_iou(a, b), i, j) for (i, a), (j, b)
                    in product(enumerate(left), enumerate(right))), reverse=True)
    used_left, used_right, scores = set(), set(), []
    for score, i, j in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        scores.append(score)
    return sum(scores) / len(scores) if scores else None


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _pair_stats(pairs: Iterable[tuple[Sample, Sample]]) -> dict:
    route_diff = count_same = entangled_diff = total = 0
    ious = []
    for first, second in pairs:
        total += 1
        route_diff += first.route != second.route
        count_same += first.num_people == second.num_people
        entangled_diff += first.entangled != second.entangled
        iou = matched_box_iou(first.boxes, second.boxes)
        if iou is not None:
            ious.append(iou)
    return {
        "pairs": total,
        "route_change_rate": route_diff / total if total else 0.0,
        "count_agreement": count_same / total if total else 1.0,
        "entangled_change_rate": entangled_diff / total if total else 0.0,
        "box_iou_median": statistics.median(ious) if ious else None,
    }


def per_cut_disagreement(by_cut: dict) -> list[dict]:
    """컷별로 A↔B가 어긋난 쌍 수. 전체 비율만으로는 "몇몇 컷이 끄는지"를 알 수 없다.

    남기는 값은 컷 ID와 숫자뿐이다 — 이미지도 응답 원문도 보고서에 넣지 않는다.
    어긋남이 0인 컷은 빼서 보고서가 컷 수만큼 길어지지 않게 한다.
    """
    rows = []
    for cut_id, arms in by_cut.items():
        a = [s for s in arms["A"] if s.ok]
        b = [s for s in arms["B"] if s.ok]
        pairs = list(product(a, b))
        if not pairs:
            continue
        count_bad = sum(1 for first, second in pairs if first.num_people != second.num_people)
        route_bad = sum(1 for first, second in pairs if first.route != second.route)
        if not count_bad and not route_bad:
            continue
        rows.append({
            "cut_id": cut_id,
            "cross_pairs": len(pairs),
            "count_mismatch_pairs": count_bad,
            "route_mismatch_pairs": route_bad,
            # 어느 쪽이 몇 명으로 봤는지. 한쪽만 흔들리는지 둘 다 흔들리는지 구분된다.
            "a_counts": sorted({s.num_people for s in a if s.num_people is not None}),
            "b_counts": sorted({s.num_people for s in b if s.num_people is not None}),
        })
    rows.sort(key=lambda row: (-row["count_mismatch_pairs"], -row["route_mismatch_pairs"]))
    return rows


def evaluate(samples: list[Sample], gate: dict = GATE) -> dict:
    """샘플 목록 → 지표와 통과 여부. 순수 함수다."""
    by_cut: dict[str, dict[str, list[Sample]]] = {}
    for sample in samples:
        by_cut.setdefault(sample.cut_id, {"A": [], "B": []})[sample.arm].append(sample)
    floor_pairs, cross_pairs = [], []
    for arms in by_cut.values():
        a = [s for s in arms["A"] if s.ok]
        b = [s for s in arms["B"] if s.ok]
        floor_pairs.extend(combinations(a, 2))
        cross_pairs.extend(product(a, b))
    floor, cross = _pair_stats(floor_pairs), _pair_stats(cross_pairs)

    a_ok = [s for s in samples if s.arm == "A" and s.ok]
    b_ok = [s for s in samples if s.arm == "B" and s.ok]
    failures = {arm: sum(1 for s in samples if s.arm == arm and not s.ok) for arm in ("A", "B")}
    asked = [s for s in b_ok if s.aligned is not None]
    alignment = sum(1 for s in asked if s.aligned) / len(asked) if asked else None
    p95_a = _p95([s.latency_ms for s in a_ok])
    p95_b = _p95([s.latency_ms for s in b_ok])
    people_b = sum(s.num_people or 0 for s in b_ok)

    metrics = {
        "parse_failures": failures,
        "route_change_excess": cross["route_change_rate"] - floor["route_change_rate"],
        "count_agreement_drop": floor["count_agreement"] - cross["count_agreement"],
        "box_iou_median_drop": (
            floor["box_iou_median"] - cross["box_iou_median"]
            if floor["box_iou_median"] is not None and cross["box_iou_median"] is not None
            else None),
        # 비율 차이를 A↔B 쌍 수에 곱해 "바닥보다 몇 건 더"로 읽는다.
        "entangled_change_excess": (
            (cross["entangled_change_rate"] - floor["entangled_change_rate"]) * cross["pairs"]),
        "latency_p95_ms": {"A": p95_a, "B": p95_b},
        "latency_p95_increase": (p95_b / p95_a - 1) if p95_a and p95_b is not None else None,
        "alignment_rate": alignment,
        "person_tag_fill_rate": (
            sum(s.tagged_people for s in b_ok) / people_b if people_b else None),
    }
    checks = {
        "parse_failures_b": failures["B"] <= gate["parse_failures_b_max"],
        "route_change": metrics["route_change_excess"] <= gate["route_change_excess_max"],
        "count_agreement": metrics["count_agreement_drop"] <= gate["count_agreement_drop_max"],
        "box_iou": (metrics["box_iou_median_drop"] is None
                    or metrics["box_iou_median_drop"] <= gate["box_iou_median_drop_max"]),
        "entangled": (metrics["entangled_change_excess"]
                      <= gate["entangled_change_excess_max"]),
        "latency": (metrics["latency_p95_increase"] is None
                    or metrics["latency_p95_increase"] <= gate["latency_p95_increase_max"]),
        # B가 인물별 배열을 묻지 않는 프롬프트면 정렬률은 따지지 않는다.
        "alignment": alignment is None or alignment >= gate["alignment_rate_min"],
    }
    # A끼리 비교할 쌍이 없으면 잡음 바닥을 모른다. 통과로 치지 않는다.
    enough = floor["pairs"] > 0 and cross["pairs"] > 0
    return {
        "passed": enough and all(checks.values()),
        "enough_samples": enough,
        "checks": checks,
        "metrics": metrics,
        "noise_floor": floor,
        "cross": cross,
        "gate": dict(gate),
        "cuts": len(by_cut),
        "samples": {"A": len(a_ok) + failures["A"], "B": len(b_ok) + failures["B"]},
        # 진단용. 어긋난 컷만 들어간다(많이 어긋난 순).
        "per_cut_disagreement": per_cut_disagreement(by_cut),
    }


def run_vlm_compare(cuts: list[dict], *, prompt_a: str, prompt_b: str, repeats: int,
                    client, load_image: Callable[[dict], object],
                    clock: Callable[[], float] = time.monotonic) -> list[Sample]:
    """컷마다 A·B를 번갈아 부르고 Sample 목록을 돌려준다.

    프롬프트는 `CFG.vlm_prompt_version`으로 고르고, 끝나면 원래 값으로 되돌린다.
    """
    from src.config import CFG
    from src.vlm import prompts

    for version in (prompt_a, prompt_b):
        prompts.user_template(version)  # 모르는 버전이면 여기서 멈춘다
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to measure the noise floor")
    previous = CFG.vlm_prompt_version
    samples: list[Sample] = []
    try:
        for cut in cuts:
            image = load_image(cut)
            width, height = image.size
            for index in range(repeats):
                order = (("A", prompt_a), ("B", prompt_b))
                for arm, version in (order if index % 2 == 0 else order[::-1]):
                    CFG.vlm_prompt_version = version
                    started = clock()
                    try:
                        vlm = client.analyze(image, width, height)
                    except Exception as exc:  # noqa: BLE001 - 실패도 지표다
                        samples.append(Sample(cut_id=cut["cut_id"], arm=arm, ok=False,
                                              latency_ms=(clock() - started) * 1000,
                                              error_type=type(exc).__name__))
                        continue
                    samples.append(summarize(vlm, cut["cut_id"], arm,
                                             (clock() - started) * 1000, width, height))
    finally:
        CFG.vlm_prompt_version = previous
    return samples


def write_vlm_compare(dataset, *, provider: str, prompt_a: str, prompt_b: str, repeats: int,
                      output_root: str) -> dict:
    """데이터셋 전체로 비교를 돌려 보고서를 쓴다. 요청한 provider가 실제로 떴는지 먼저 본다."""
    from pathlib import Path

    from PIL import Image

    from src.config import CFG
    from src.vlm.client import build_vlm_client

    from .fixtures import _backend_name
    from .util import resolve_path, utc_now, write_json

    previous = CFG.vlm_provider
    CFG.vlm_provider = provider
    try:
        client = build_vlm_client()
    finally:
        CFG.vlm_provider = previous
    actual = _backend_name(client)
    if actual != provider:
        raise RuntimeError(f"requested VLM {provider}, actual {actual}")

    def load_image(cut: dict):
        with Image.open(resolve_path(cut["image_path"])) as opened:
            return opened.convert("RGB")

    samples = run_vlm_compare(dataset.cuts, prompt_a=prompt_a, prompt_b=prompt_b,
                              repeats=repeats, client=client, load_image=load_image)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset_id": dataset.dataset_id,
        "provider": provider,
        "prompts": {"A": prompt_a, "B": prompt_b},
        "repeats": repeats,
        **evaluate(samples),
    }
    stamp = report["created_at"].replace(":", "").replace("-", "")
    path = Path(output_root) / f"{dataset.dataset_id}-{prompt_a}-vs-{prompt_b}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    report["report_path"] = str(path)
    return report
