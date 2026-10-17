"""04J-286 라우팅 대표 사례 재현 + 기존/라우팅 Top-5 나란히 저장(04J-291 준비).

같은 러프에 대해 파이프라인을 두 번 돌린다.
  baseline  SEARCH_ROUTING=0 — 기존 기하 검색만
  routed    SEARCH_ROUTING=1 — 인물별 경로 분기
VLM 분석 결과는 이미지마다 한 번만 받아 두 실행이 **같은 분석**을 공유한다(공정 비교).

VLM 모드
  --vlm live     실제 provider(VLM_PROVIDER=gemini 등). 분석·자세 설명 모두 실호출.
  --vlm replay   운영 Job 상세(jobs.json)의 인원수·박스·shot을 재생한다(네트워크 없음).
                 자세 설명은 --describe-fixtures가 있으면 그 값을, 없으면 오류
                 (reason=describer_offline)로 처리한다 — 판단 불가로 섞지 않는다.

    # 맥(실 Gemini): 운영과 같은 env(.env)에서
    SEARCH_ROUTING=1 VLM_PROVIDER=gemini POSE_BACKEND=rtmlib \
    python scripts/eval_search_routing.py --manifest <testset>/manifest.csv \
        --images <testset> --db <bundle>/poses.db --tags <viewer>/dist/data/search-v2.json \
        --thumbs <bundle>/thumbs --vlm live --out out/routing-eval-live

출력: results.json(인물별 전체 기록), side_by_side.csv, REPORT.md, index.html(썸네일 비교)
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.pose import MockPoseModel, build_pose_model  # noqa: E402
from src.repo import load_entries  # noqa: E402
from src.schema import (Action, BBox, Relationship, Shot, View,  # noqa: E402
                        VLMAnalysis)
from src.search_routing import DescribeOutcome, PoseDescription  # noqa: E402
from src.semantic_search import build_semantic_searcher  # noqa: E402
from src.vlm.client import BaseVLMClient, MockVLMClient, build_vlm_client  # noqa: E402


# ---------------------------------------------------------------- VLM 래퍼


class CachedVLM(BaseVLMClient):
    """이미지 키 단위로 analyze/describe를 1회만 호출한다(baseline·routed 공유)."""

    def __init__(self, inner):
        self.inner = inner
        self.key = None
        self._analysis = {}
        self._describe = {}
        self.provider = type(inner).__name__

    def analyze(self, image, img_w, img_h):
        if self.key not in self._analysis:
            started = time.perf_counter()
            self._analysis[self.key] = (self.inner.analyze(image, img_w, img_h),
                                        (time.perf_counter() - started) * 1000.0)
        return _clone(self._analysis[self.key][0])

    def tag_people(self, image, boxes):
        return self.inner.tag_people(image, boxes)

    def describe_people(self, image, boxes):
        key = (self.key, tuple(None if b is None else tuple(round(v, 4) for v in b)
                               for b in boxes))
        if key not in self._describe:
            self._describe[key] = self.inner.describe_people(image, boxes)
        return self._describe[key]

    def analysis_ms(self):
        return self._analysis.get(self.key, (None, None))[1]


def _clone(vlm: VLMAnalysis) -> VLMAnalysis:
    import copy
    clone = copy.copy(vlm)
    clone.raw = dict(vlm.raw)
    clone.approx_boxes = list(vlm.approx_boxes)
    return clone


class ReplayVLM(BaseVLMClient):
    """운영 Job 상세의 인물 수·박스·태그로 VLMAnalysis를 다시 만든다."""

    def __init__(self, jobs_path: Path, fixtures: dict | None = None):
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
        self.jobs = {}
        for record in data.get("records", []):
            job = record.get("job") or {}
            self.jobs[job.get("jobId")] = record
        self.fixtures = fixtures or {}
        self.current = None      # (test_id, job_id)

    def analyze(self, image, img_w, img_h):
        test_id, job_id = self.current
        record = self.jobs.get(job_id)
        if record is None:
            raise KeyError(f"replay: job {job_id} missing")
        detail = record.get("detail") or {}
        people = sorted(detail.get("people") or [], key=lambda p: p.get("person_index", 0))
        src_w = (record.get("job") or {}).get("inputWidth") or img_w
        src_h = (record.get("job") or {}).get("inputHeight") or img_h
        sx, sy = img_w / float(src_w), img_h / float(src_h)
        boxes, tags = [], {}
        for person in people:
            box = json.loads(person["bbox_json"]) if person.get("bbox_json") else None
            boxes.append(None if box is None else BBox(
                box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy, "vlm", 0.5))
            tags = json.loads(person.get("tags_json") or "{}") or tags
        num = len(people)

        def pick(enum, value, default):
            try:
                return enum(value)
            except ValueError:
                return default

        return VLMAnalysis(
            num_people=num,
            shot=pick(Shot, tags.get("shot"), Shot.FULL_HALF),
            action=pick(Action, tags.get("action"), Action.OTHER),
            view=pick(View, tags.get("view"), View.FRONT),
            relationship=pick(Relationship, tags.get("relationship"),
                              Relationship.SOLO if num <= 1 else Relationship.TALKING),
            approx_boxes=boxes,
            raw={"replay_job": job_id},
            lower_body_visible=[False] * num,
            lower_body_visibility_known=[False] * num,
            stated_tags=dict(tags) if tags else None,
        )

    def describe_people(self, image, boxes):
        test_id, _job = self.current
        entries = self.fixtures.get(test_id)
        if isinstance(entries, dict) and "_error_status" in entries:
            # 장애 주입: 상류 오류를 Gemini 경로와 같은 분류기로 옮긴다.
            from src.vlm.client import describe_failure_reason

            class _UpstreamError(Exception):
                def __init__(self, code):
                    super().__init__(f"injected upstream {code}")
                    self.code = code

            reason, retryable, delay = describe_failure_reason(
                _UpstreamError(int(entries["_error_status"])))
            return DescribeOutcome("error", error_reason=reason, retryable=retryable,
                                   retry_after_seconds=delay, provider="replay:fault")
        if entries is None:
            return DescribeOutcome("error", error_reason="describer_offline",
                                   retryable=False, provider="replay")
        descriptions = {}
        for index, box in enumerate(boxes):
            match = _nearest_fixture(entries, box)
            descriptions[index] = None if match is None else PoseDescription(
                **{k: v for k, v in match.items() if k in PoseDescription.__dataclass_fields__})
        return DescribeOutcome("ok", descriptions=descriptions,
                               provider="fixture:" + str(entries[0].get("source", "manual"))
                               if entries else "fixture")


def _nearest_fixture(entries, box):
    """fixture의 대략 중심(cx, cy; 0~1)과 가장 가까운 항목. 박스 없으면 첫 항목."""
    if not entries:
        return None
    if box is None:
        return entries[0]
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    def distance(entry):
        ex, ey = entry.get("center", (0.5, 0.5))
        return (ex - cx) ** 2 + (ey - cy) ** 2

    return min(entries, key=distance)


# ---------------------------------------------------------------- 실행


def _top5(result, index):
    cands = result.person_candidates[index] if index < len(result.person_candidates) else []
    return [{"pose_id": c.pose_id, "view": c.view.value, "distance": round(float(c.distance), 4)}
            for c in cands[:5]]


def _routed_top5(result, index):
    record = result.person_routing[index] if index < len(result.person_routing) else None
    if record and record["route"] == "semantic":
        hits = result.person_semantic_candidates[index] or []
        return [{"pose_id": h["pose_id"], "view": h.get("pose", {}).get("view", "front"), "score": h["score"],
                 **({"preview_camera": h["pose"]["preview_camera"]} if h.get("pose", {}).get("preview_camera") else {})}
                for h in hits[:5]]
    return _top5(result, index)


def run(args) -> int:
    from PIL import Image

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if args.test_ids:
        wanted = set(args.test_ids.split(","))
        rows = [row for row in rows if (row.get("test_id") or row.get("id")) in wanted]
    if not rows:
        raise ValueError("No matching evaluation cases")
    if args.limit:
        rows = rows[:args.limit]
    entries = load_entries(str(args.db))
    fixtures = json.loads(args.describe_fixtures.read_text(encoding="utf-8")) \
        if args.describe_fixtures else None
    inner = (ReplayVLM(args.jobs, fixtures) if args.vlm == "replay" else build_vlm_client())
    if args.vlm == "live" and isinstance(inner, MockVLMClient):
        raise RuntimeError("Live evaluation requires a real VLM; mock fallback rejected")
    vlm = CachedVLM(inner)
    searcher = build_semantic_searcher(
        str(args.tags), library_pose_ids={entry.pose_id for entry in entries})
    pose = build_pose_model()
    if CFG.pose_backend == "rtmlib" and isinstance(pose, MockPoseModel):
        raise RuntimeError("RTMPose evaluation cannot use a mock fallback")
    pipeline = Pipeline(entries, vlm_client=vlm, pose_model=pose,
                        semantic_searcher=searcher)
    args.out.mkdir(parents=True, exist_ok=True)
    results, table = [], []
    for row in rows:
        test_id = row.get("test_id") or row.get("id")
        image_path = args.images / row["test_path"]
        vlm.key = test_id
        if isinstance(inner, ReplayVLM):
            inner.current = (test_id, row.get("job_id"))
        with Image.open(image_path) as image:
            width, height = image.size
        record = {"test_id": test_id, "outcome": row.get("outcome"),
                  "framing": row.get("framing"), "image": str(row["test_path"])}
        try:
            CFG.search_routing_enabled = False
            started = time.perf_counter()
            baseline = pipeline.process_cut(str(image_path), width, height)
            baseline_ms = (time.perf_counter() - started) * 1000.0
            CFG.search_routing_enabled = True
            started = time.perf_counter()
            routed = pipeline.process_cut(str(image_path), width, height)
            routed_ms = (time.perf_counter() - started) * 1000.0
        except Exception as error:  # noqa: BLE001 - 사례 단위 실패로 기록
            record["error"] = f"{type(error).__name__}: {error}"
            results.append(record)
            table.append({"test_id": test_id, "person": "", "error": record["error"]})
            _write_results(args, pipeline, searcher, results)
            print(f"{test_id}: {record['error']}", flush=True)
            continue
        finally:
            CFG.search_routing_enabled = False
        record.update({
            "vlm_count": routed.vlm_count,
            "vlm_analysis_ms": vlm.analysis_ms(),
            "baseline_ms": round(baseline_ms, 1), "routed_ms": round(routed_ms, 1),
            "routing_summary": routed.routing_summary,
            "people": [],
        })
        for index in range(len(routed.descriptors)):
            routing = routed.person_routing[index] if index < len(routed.person_routing) else None
            box = routed.descriptors[index].box
            person = {
                "index": index,
                "box": box.as_list() if box is not None else None,
                "baseline_top5": _top5(baseline, index),
                "baseline_confidence": baseline.person_confidence[index]
                if index < len(baseline.person_confidence) else None,
                "routed_top5": _routed_top5(routed, index),
                "routing": routing,
                "semantic_candidates": (routed.person_semantic_candidates[index]
                                        if index < len(routed.person_semantic_candidates)
                                        else []),
            }
            record["people"].append(person)
            joint = (routing or {}).get("joint_evidence", {})
            semantic = (routing or {}).get("semantic_evidence", {})
            table.append({
                "test_id": test_id, "outcome": row.get("outcome"), "framing": row.get("framing"),
                "person": index,
                "route": (routing or {}).get("route"),
                "reason": (routing or {}).get("reason"),
                "joint_reasons": "|".join(joint.get("reasons", [])),
                "valid_bones": joint.get("metrics", {}).get("valid_bone_count"),
                "torso_bones": joint.get("metrics", {}).get("torso_bone_count"),
                "coverage": joint.get("metrics", {}).get("coverage_class"),
                "semantic_status": semantic.get("status"),
                "semantic_error": semantic.get("error_reason"),
                "search_status": (routing or {}).get("search", {}).get("status"),
                "baseline_top5": " ".join(c["pose_id"] for c in person["baseline_top5"]),
                "routed_top5": " ".join(c["pose_id"] for c in person["routed_top5"]),
                "same_top5": (person["baseline_top5"] == person["routed_top5"]
                              if (routing or {}).get("route") == "geometric" else
                              [c["pose_id"] for c in person["baseline_top5"]]
                              == [c["pose_id"] for c in person["routed_top5"]]),
                "baseline_ms": round(baseline_ms, 1), "routed_ms": round(routed_ms, 1),
                "describe_ms": (routed.routing_summary or {}).get("describe", {})
                .get("elapsed_ms"),
            })
        if not routed.descriptors:
            table.append({"test_id": test_id, "outcome": row.get("outcome"),
                          "framing": row.get("framing"), "person": "-",
                          "route": "(no person from VLM)", "baseline_ms": round(baseline_ms, 1),
                          "routed_ms": round(routed_ms, 1)})
        results.append(record)
        _write_results(args, pipeline, searcher, results)
        print(f"{test_id}: people={len(routed.descriptors)} routes="
              f"{[p['routing']['route'] if p['routing'] else None for p in record['people']]}",
              flush=True)

    _write_results(args, pipeline, searcher, results)
    fields = sorted({key for row in table for key in row},
                    key=lambda k: list(table[0].keys()).index(k) if k in table[0] else 99)
    with (args.out / "side_by_side.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    _write_report(args, results, table)
    if args.thumbs:
        _write_html(args, results)
    return int(any(r.get("error") for r in results) or any(
        row.get("route") == "geometric" and not row.get("same_top5") for row in table))


def _write_results(args, pipeline, searcher, results):
    """Persist every case so interrupted live calls leave reviewable evidence."""
    (args.out / "results.json").write_text(
        json.dumps({"vlm_mode": args.vlm,
                    "vlm_provider": CFG.vlm_provider if args.vlm == "live" else "replay",
                    "vlm_model": (CFG.gemini_model if args.vlm == "live"
                                  and CFG.vlm_provider == "gemini" else None),
                    "vlm_prompt_version": CFG.vlm_prompt_version,
                    "pose_variant": CFG.pose_model_variant,
                    "routing_config": asdict(pipeline.routing_config),
                    "semantic_backend": getattr(searcher, "backend", None),
                    "cases": results}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")


def _write_report(args, results, table):
    from collections import Counter
    provider = getattr(args, "vlm_provider", CFG.vlm_provider)
    model = getattr(args, "vlm_model", CFG.gemini_model)

    people = [row for row in table if isinstance(row.get("person"), int)]
    routes = Counter(row["route"] for row in people)
    by_outcome = Counter((row.get("outcome"), row["route"]) for row in people)
    status = Counter((row["route"], row.get("search_status")) for row in people)
    changed = sum(1 for row in people if not row.get("same_top5"))
    geometric = [row for row in people if row.get("route") == "geometric"]
    regressions = sum(not row.get("same_top5") for row in geometric)
    descriptions = [(r.get("routing_summary") or {}).get("describe") or {} for r in results]
    requested = [d for d in descriptions if d.get("requested_people", 0)]
    semantic_people = [row for row in people if (row.get("semantic_status") or "")
                       not in ("", "not_requested")]
    valid = sum(row.get("semantic_status") == "valid" for row in semantic_people)
    lines = [
        "# 04J-286 라우팅 재현 결과",
        "",
        f"- VLM 모드: `{args.vlm}`  포즈: `{CFG.pose_model_variant}`",
        f"- VLM 설정: provider=`{provider}`, model=`{model}` "
        f"(실호출 Gemini일 때 적용)",
        f"- 사례 {len(results)}장, 인물 {len(people)}명, VLM 인물 0명 컷 "
        f"{sum(1 for r in results if r.get('vlm_count') == 0)}장",
        f"- 경로 분포: " + ", ".join(f"{k}={v}" for k, v in sorted(routes.items())),
        f"- 기존 대비 Top-5가 달라진 인물: {changed}명",
        f"- geometric Top-5 회귀: {regressions}/{len(geometric)}명",
        f"- 설명을 요청한 인물의 valid 비율: {valid}/{len(semantic_people)}명",
        f"- 설명 호출: {len(requested)}회, 상태 " + str(Counter(d["status"] for d in requested)),
        f"- 사례 처리 오류: {sum(bool(r.get('error')) for r in results)}장",
        "",
        "| 결과 유형 | 경로 | 인원 |", "|---|---|---:|",
    ]
    lines += [f"| {o} | {r} | {n} |" for (o, r), n in sorted(by_outcome.items())]
    lines += ["", "| 경로 | 검색 상태 | 인원 |", "|---|---|---:|"]
    lines += [f"| {r} | {s} | {n} |" for (r, s), n in sorted(status.items(), key=str)]
    if args.vlm == "replay":
        lines += ["", "주의: VLM 분석은 운영 기록 재생이며, 자세 설명은 fixture 또는 오류 대역이다.",
                  "이 결과는 분기·검색 연결 검증용이다. Gemini 설명 품질과 실제 호출 지연은 미측정이다."]
    if args.vlm == "cached-live":
        lines += ["", "주의: 실제 Gemini 설명을 고정하여 검색만 재생한 결과다.",
                  "아래 설명 호출 수·지연은 원본 실호출 기록이며 이번 재생에서 추가 VLM 호출은 0회다."]
    if args.vlm != "replay" and requested:
        import numpy as np
        times = [d["elapsed_ms"] for d in requested]
        p50, p95 = np.percentile(times, [50, 95])
        lines += ["", f"설명 호출 지연: p50={p50:.1f}ms, p95={p95:.1f}ms "
                  f"(호출 {len(times)}회, 실패 포함)."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _write_html(args, results):
    import shutil

    thumbs_out = args.out / "thumbs"
    rough_out = args.out / "roughs"
    thumbs_out.mkdir(exist_ok=True)
    rough_out.mkdir(exist_ok=True)

    def thumb(pose_id, view):
        for candidate in (f"{pose_id}__{view}.jpg", f"{pose_id}__{view}.png",
                          f"{pose_id}__front.jpg"):
            source = args.thumbs / candidate
            if source.exists():
                target = thumbs_out / candidate
                if not target.exists():
                    shutil.copy(source, target)
                return f"thumbs/{html.escape(candidate)}"
        return None

    def strip(cands, key):
        cells = []
        for cand in cands:
            src = thumb(cand["pose_id"], cand.get("view", "front"))
            label = f"{cand['pose_id'][:22]}<br>{key}={cand.get(key)}"
            if cand.get("preview_camera"):
                label += "<br>기존 yaw 썸네일 · pitch 미적용"
            cells.append(f"<figure><img src='{src}' loading='lazy'><figcaption>{label}"
                         f"</figcaption></figure>" if src else f"<figure>{label}</figure>")
        return "".join(cells) or "<span class=muted>후보 없음</span>"

    parts = []
    for case in results:
        rough_src = args.images / case["image"]
        target = rough_out / f"{case['test_id']}{rough_src.suffix}"
        if rough_src.exists() and not target.exists():
            shutil.copy(rough_src, target)
        people_html = []
        for person in case.get("people", []):
            routing = person.get("routing") or {}
            joint = routing.get("joint_evidence", {})
            semantic = routing.get("semantic_evidence", {})
            desc = (semantic.get("description") or {}).get("search_text") or ""
            people_html.append(
                f"<div class=person><h4>인물 {person['index']} · 경로 "
                f"<b class='r-{routing.get('route')}'>{routing.get('route')}</b> · 검색 "
                f"{(routing.get('search') or {}).get('status')}</h4>"
                f"<p class=muted>{html.escape(str(routing.get('reason')))} · 관절 사유: "
                f"{html.escape(', '.join(joint.get('reasons', [])) or '없음')} · 뼈 "
                f"{joint.get('metrics', {}).get('valid_bone_count')}/몸통 "
                f"{joint.get('metrics', {}).get('torso_bone_count')} · 의미 "
                f"{semantic.get('status')} {html.escape(desc)}</p>"
                f"<div class=row><span class=lbl>기존</span>"
                f"{strip(person['baseline_top5'], 'distance')}</div>"
                f"<div class=row><span class=lbl>라우팅</span>"
                f"{strip(person['routed_top5'], 'score' if routing.get('route') == 'semantic' else 'distance')}"
                f"</div></div>")
        parts.append(
            f"<section><div class=rough><img src='roughs/{html.escape(target.name)}'>"
            f"<p><b>{case['test_id']}</b> {html.escape(str(case.get('outcome')))} · "
            f"{html.escape(str(case.get('framing')))}</p></div>"
            f"<div class=people>{''.join(people_html) or '<p class=muted>VLM 인물 0명</p>'}"
            f"{'<p class=err>' + html.escape(case['error']) + '</p>' if case.get('error') else ''}"
            f"</div></section>")
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Routing comparison</title>
<style>
:root{{--bg:#fff;--fg:#1f1f1c;--muted:#6b6a65;--line:#e4e2da;--card:#f7f6f2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1c1a;--fg:#ecebe6;--muted:#a3a29c;--line:#3a3935;--card:#262624}}}}
body{{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px}}
section{{display:flex;gap:16px;border-bottom:1px solid var(--line);padding:16px 0;flex-wrap:wrap}}
.rough{{width:220px}} .rough img{{width:100%;background:#fff;border:1px solid var(--line)}}
.people{{flex:1;min-width:280px}} .person{{background:var(--card);border-radius:10px;padding:10px;margin-bottom:10px}}
.row{{display:flex;gap:6px;align-items:flex-start;overflow-x:auto;margin:6px 0}}
.lbl{{width:48px;flex:none;color:var(--muted)}} figure{{margin:0;width:92px;flex:none;font-size:11px}}
figure img{{width:92px;height:92px;object-fit:contain;background:#fff;border-radius:6px}}
.muted{{color:var(--muted)}} .err{{color:#c0392b}} h4{{margin:0 0 4px;font-weight:500}}
.r-semantic{{color:#7c5cd6}} .r-undetermined{{color:#b07a12}} .r-error{{color:#c0392b}}
</style>
<p class=muted>VLM {html.escape(args.vlm)} · 포즈 {html.escape(CFG.pose_model_variant)} · 기존 = SEARCH_ROUTING 0, 라우팅 = 1 (같은 VLM 분석 공유)</p>
{''.join(parts)}"""
    (args.out / "index.html").write_text(page, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="manifest test_path 기준 폴더")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True, help="search-v2.json")
    parser.add_argument("--thumbs", type=Path)
    parser.add_argument("--vlm", choices=("live", "replay"), default="live")
    parser.add_argument("--jobs", type=Path, help="replay용 jobs.json")
    parser.add_argument("--describe-fixtures", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-ids", default="", help="Comma-separated case IDs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.vlm == "replay" and not args.jobs:
        parser.error("--vlm replay requires --jobs")
    os.environ.setdefault("SEARCH_ROUTING", "0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
