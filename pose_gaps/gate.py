"""재생 게이트 — 새 라이브러리가 기존 검색 결과를 망치지 않는지 배포 전에 확인한다.

포즈를 더하면 Top-1 거리는 줄어들 수밖에 없다. 문제는 기하적으로 더 가깝지만 의미가
엉뚱한 포즈가 기존의 좋은 매칭을 밀어내는 경우와, 기울인 구도 변형 같은 한 포즈가
수많은 쿼리의 Top-1을 독차지하는(hubness) 경우다. 그래서 공백만이 아니라 **적격 관측
전체**와 제공 러프 coverage 세트로 기준 번들과 후보 번들을 비교한다.

판정(모두 통과해야 passed):
1. 나빠진 쿼리(Δ > regress_delta) 비율 ≤ max_regressed_share
2. 기준에서 high(≤ high_edge)였던 쿼리가 high를 벗어나지 않는다. 기준 Top-1 포즈를
   의도적으로 뺀 경우만 Δ ≤ high_slack까지 허용
3. 원래 잘 맞던 쿼리(기준 거리 ≤ τ_soft) 안에서 어떤 pose family도 Top-1 점유율이
   max(floor, 기준 × multiplier)를 넘지 않는다
4. 신규 포즈마다 좌우 반전 짝이 있다(면제 목록 제외)
5. 후보 번들에 모든 포즈의 BVH·썸네일이 있다
6. 같은 계산을 두 번 하면 같은 결과가 나온다

공백 메움률은 보고만 한다(차단 조건이 아니다). 보고서에는 집계만 남긴다.
`scripts/build_pose_bundle.py record-gate`가 이 보고서의 status를 manifest에 기록한다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.features import normalize_skeleton

from .config import GapConfig
from .eligibility import quantitative_eligible
from .libraries import Library
from .replay import Replayed

VIEWS = ("front", "three_quarter", "side", "back")


@dataclass(frozen=True)
class Query:
    feature: np.ndarray
    mask: np.ndarray
    metric: str
    source: str          # snapshot | coverage
    gap: bool = False


def queries_from_snapshot(items: list[Replayed]) -> list[Query]:
    return [Query(i.feature, i.mask, (i.obs.distance_metric or "pos").lower(), "snapshot",
                  gap=i.label == "gap_open")
            for i in items if i.excluded is None and i.prod_distance is not None]


def queries_from_coverage(extraction_dir: Path, kpt_thr: float = 0.3) -> list[Query]:
    """`pose_curation/coverage` 추출 결과 중 제공 러프만 쓴다(`user_*` 파일은 건너뛴다).

    coverage.evaluate와 같은 규칙: 원래 모델 점수 0.3 이상을 마스크로, 정량 조건 통과자만.
    """
    out = []
    for path in sorted(Path(extraction_dir).glob("*.json")):
        if path.name == "model.json" or path.name.startswith("user_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for person in payload.get("people", []):
            points = np.asarray(person["keypoints"], dtype=np.float32)
            scores = np.asarray(person["scores"], dtype=np.float32)
            if not quantitative_eligible(points, scores, kpt_thr):
                continue
            feature = normalize_skeleton(points, scores).reshape(17, 2)
            out.append(Query(feature, scores >= kpt_thr, "pos", "coverage"))
    return out


def _top1_all(library: Library, queries: list[Query]):
    hits = [library.top1(q.feature, q.mask, metric=q.metric) for q in queries]
    return [(h.pose_id, float(h.distance)) if h is not None else (None, float("inf")) for h in hits]


def _family_shares(results, family_by_pose: dict[str, str]) -> Counter:
    counts = Counter(family_by_pose.get(pose, pose) for pose, _ in results if pose is not None)
    total = sum(counts.values()) or 1
    return Counter({family: n / total for family, n in counts.items()})


def _mirror_orphans(candidate: Library, new_ids: set[str], waived: set[str]) -> list[str]:
    families = candidate.family_by_pose
    members: dict[str, set] = {}
    for pose, family in families.items():
        members.setdefault(family, set()).add(pose)
    orphans = []
    for pose in sorted(new_ids):
        if pose in waived:
            continue
        twin = (pose[:-len("_mirror")] if pose.endswith("_mirror") else f"{pose}_mirror")
        if twin in families or len(members.get(families[pose], ())) > 1:
            continue
        orphans.append(pose)
    return orphans


def _missing_assets(candidate: Library) -> int:
    root = candidate.path if candidate.path.is_dir() else candidate.path.parent
    if not (root / "bvh").is_dir():
        return -1   # DB 파일만 받은 경우 자산 검사는 배포 검증기에 맡긴다
    missing = 0
    seen = set()
    for entry in candidate.entries:
        if entry.pose_id in seen:
            continue
        seen.add(entry.pose_id)
        name = Path(str(entry.bvh_path or "").replace("\\", "/")).name
        if not name or not (root / "bvh" / name).is_file():
            missing += 1
        missing += sum(not (root / "thumbs" / f"{entry.pose_id}__{view}.jpg").is_file()
                       for view in VIEWS)
    return missing


def run_gate(baseline: Library, candidate: Library, queries: list[Query], cfg: GapConfig, *,
             mirror_waivers: set[str] = frozenset()) -> dict:
    if not queries:
        raise ValueError("게이트에 쓸 쿼리가 없습니다(스냅샷·coverage 세트를 확인)")
    gate = cfg.gate
    before = _top1_all(baseline, queries)
    after = _top1_all(candidate, queries)
    again = _top1_all(candidate, queries)

    deltas = np.array([a[1] - b[1] for a, b in zip(after, before)], dtype=np.float64)
    finite = np.isfinite(deltas)
    regressed = int(np.sum(deltas[finite] > gate.regress_delta))
    improved = int(np.sum(deltas[finite] < gate.improve_delta))
    removed = baseline.pose_ids - candidate.pose_ids
    crossings, excused = 0, 0
    for (old_pose, old_d), (_, new_d) in zip(before, after):
        if old_d <= gate.high_edge < new_d:
            if old_pose in removed and new_d <= old_d + gate.high_slack:
                excused += 1
            else:
                crossings += 1

    # 쏠림은 원래 잘 맞던 쿼리(기준 거리 ≤ τ_soft)에서만 잰다. 공백 쿼리를 새 포즈가
    # 가져가는 것은 목적 그 자체라, 전체에서 재면 정당한 메움을 hub로 오판한다.
    good = [i for i, (_, d) in enumerate(before) if d <= cfg.tau_soft]
    shares_before = _family_shares([before[i] for i in good], baseline.family_by_pose)
    shares_after = _family_shares([after[i] for i in good], candidate.family_by_pose)
    hubs = sorted(family for family, share in shares_after.items()
                  if share > max(gate.max_family_share_floor,
                                 gate.max_family_share_multiplier * shares_before.get(family, 0.0)))
    new_ids = candidate.pose_ids - baseline.pose_ids
    orphans = _mirror_orphans(candidate, new_ids, set(mirror_waivers))
    missing_assets = _missing_assets(candidate)

    gap_queries = [i for i, q in enumerate(queries) if q.gap]
    filled_before = sum(before[i][1] <= cfg.tau_fill for i in gap_queries)
    filled_after = sum(after[i][1] <= cfg.tau_fill for i in gap_queries)

    checks = [
        {"name": "regressed_share", "passed": regressed / len(queries) <= gate.max_regressed_share,
         "value": round(regressed / len(queries), 4), "limit": gate.max_regressed_share},
        {"name": "high_crossings", "passed": crossings == 0, "value": crossings,
         "excused_removed_pose": excused},
        {"name": "hubness", "passed": not hubs, "families": hubs[:20]},
        {"name": "mirror_closure", "passed": not orphans, "orphans": orphans[:50],
         "orphan_count": len(orphans)},
        {"name": "assets", "passed": missing_assets <= 0, "missing": max(missing_assets, 0),
         "checked": missing_assets >= 0},
        {"name": "deterministic", "passed": after == again},
    ]
    by_source = Counter(q.source for q in queries)
    return {
        "schema_version": 1,
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "thresholds_version": cfg.version,
        "baseline": {"library_version": baseline.version, "poses": len(baseline.pose_ids)},
        "candidate": {"library_version": candidate.version, "poses": len(candidate.pose_ids),
                      "new_poses": len(new_ids), "removed_poses": len(removed)},
        "summary": {
            "queries": len(queries),
            "queries_by_source": dict(sorted(by_source.items())),
            "improved": improved,
            "regressed": regressed,
            "regressed_share": round(regressed / len(queries), 4),
            "high_crossings": crossings,
            "median_delta": round(float(np.median(deltas[finite])), 4) if finite.any() else None,
            "gap_queries": len(gap_queries),
            "gap_filled_before": filled_before,
            "gap_filled_after": filled_after,
            "max_family_share_after": round(max(shares_after.values()), 4) if shares_after else 0.0,
            "unique_top1_families_after": len(shares_after),
        },
        "checks": checks,
    }
