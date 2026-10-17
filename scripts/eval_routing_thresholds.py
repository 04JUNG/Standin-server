"""운영 기록으로 04J-286 관절 근거 기준값을 점검한다(오프라인, 모델·API 호출 없음).

클로즈베타 운영 응답(jobs.json, Job 상세의 인물별 skeleton·quality trace)에 라우터의
관절 근거 판정을 그대로 다시 적용하고, 사람 눈 시각 분류와 대조한다.

    python scripts/eval_routing_thresholds.py \
        --jobs standin_beta_2026-09-04/jobs.json \
        --visual standin_beta_2026-09-04/visual_classification.csv \
        --out out/routing-thresholds

출력: people.csv(인물별 판정), sweep.csv(기준값 민감도), REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import BBox  # noqa: E402
from src.search_routing import RoutingConfig, assess_joint_evidence  # noqa: E402

VISUAL_SHORT = {
    "01_인식_추천불가": "no_result",
    "02_인식_포즈차이": "pose_differs",
    "03_러프와_3D포즈_유사": "similar",
}


def _load_json(text, default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def person_slots(jobs_path: Path, visual_path: Path):
    data = json.loads(jobs_path.read_text(encoding="utf-8"))
    visual = {}
    if visual_path and visual_path.exists():
        with visual_path.open(encoding="utf-8-sig") as handle:
            visual = {row["job_id"]: row for row in csv.DictReader(handle)}
    for record in data.get("records", []):
        job = record.get("job") or {}
        detail = record.get("detail") or {}
        img_h = job.get("inputHeight") or 0
        for person in detail.get("people") or []:
            context = _load_json(person.get("refine_context_json"), {}) or {}
            trace = context.get("qualityTrace") or {}
            mask = (trace.get("evidence_valid_joint_mask")
                    or trace.get("search_valid_joint_mask") or [False] * 17)
            raw = _load_json(person.get("raw_scores_json"), None) or [0.0] * 17
            bbox = _load_json(person.get("bbox_json"), None)
            skeleton = _load_json(person.get("skeleton_json"), None)
            evidence = SimpleNamespace(
                coverage_class=person.get("coverage_class") or "insufficient",
                valid_joint_mask=np.asarray(mask, dtype=bool),
                raw_scores=np.asarray(raw, dtype=float),
                valid_bone_count=int(trace.get("valid_bone_count") or 0),
                torso_bone_count=int(trace.get("torso_bone_count") or 0),
            )
            slot = SimpleNamespace(
                evidence=evidence,
                skeleton=skeleton,
                state=person.get("skeleton_state") or "missing",
                skeleton_source=person.get("skeleton_source") or "none",
                slot_origin=person.get("slot_origin") or "vlm",
                retry_count=int(trace.get("retry_count") or 0),
                lower_body_visibility_known=bool(trace.get("lower_body_visibility_known")),
                lower_body_observed=bool(person.get("lower_body_observed")),
                result_box=(BBox(*bbox) if isinstance(bbox, list) and len(bbox) == 4 else None),
            )
            row = visual.get(job.get("jobId"), {})
            yield {
                "job_id": job.get("jobId"),
                "person_index": person.get("person_index"),
                "visual": VISUAL_SHORT.get(row.get("visual_category"), "unknown"),
                "visual_reason": row.get("visual_reason", ""),
                "candidate_count": person.get("candidate_count"),
                "img_h": img_h,
                "slot": slot,
            }


def evaluate(people, rcfg: RoutingConfig):
    rows = []
    for person in people:
        desc = SimpleNamespace(coverage_class=person["slot"].evidence.coverage_class,
                               output_scope=None)
        joint = assess_joint_evidence(person["slot"], desc, person["img_h"], rcfg)
        rows.append({**{k: v for k, v in person.items() if k != "slot"},
                     "sufficient": joint.sufficient,
                     "reasons": "|".join(joint.reasons),
                     **{f"m_{k}": v for k, v in joint.metrics.items()
                        if k in ("skeleton_state", "coverage_class", "valid_bone_count",
                                 "torso_bone_count", "valid_body_joints",
                                 "mean_valid_body_score", "box_height_px",
                                 "box_height_ratio", "skeleton_source")}})
    return rows


def confusion(rows):
    table = {}
    for row in rows:
        key = (row["visual"], "geometric" if row["sufficient"] else "not_geometric")
        table[key] = table.get(key, 0) + 1
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    people = list(person_slots(args.jobs, args.visual))
    base = RoutingConfig()
    rows = evaluate(people, base)
    with (args.out / "people.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sweep = []
    for bones in (4, 5, 6, 7, 8, 9, 10):
        for torso in (3, 4):
            for box_px in (32.0, 64.0, 96.0):
                rcfg = replace(base, joint_min_valid_bones=bones, joint_min_torso_bones=torso,
                               joint_min_box_px=box_px)
                result = evaluate(people, rcfg)
                table = confusion(result)
                # 기존에 후보가 있었는데 geometric에서 빠지는 인물 = 라우팅이 만들 회귀 후보
                lost = sum(1 for row in result
                           if not row["sufficient"] and (row["candidate_count"] or 0) > 0)
                lost_similar = sum(1 for row in result
                                   if not row["sufficient"] and row["visual"] == "similar")
                kept_failed = sum(1 for row in result
                                  if row["sufficient"] and (row["candidate_count"] or 0) == 0)
                sweep.append({
                    "min_valid_bones": bones, "min_torso_bones": torso, "min_box_px": box_px,
                    "geometric": sum(1 for row in result if row["sufficient"]),
                    "not_geometric": sum(1 for row in result if not row["sufficient"]),
                    "lost_people_with_candidates": lost,
                    "lost_visually_similar": lost_similar,
                    "geometric_but_zero_candidates": kept_failed,
                    **{f"{v}->{r}": table.get((v, r), 0)
                       for v in ("no_result", "pose_differs", "similar")
                       for r in ("geometric", "not_geometric")},
                })
    with (args.out / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)

    table = confusion(rows)
    lines = [
        "# 04J-286 관절 근거 기준값 점검(운영 기록 재적용)",
        "",
        f"- 입력: `{args.jobs.name}` 인물 {len(rows)}명",
        f"- 기준값: `{json.dumps(base.joint_thresholds(), ensure_ascii=False)}`",
        "",
        "| 시각 분류 | geometric | not_geometric |",
        "|---|---:|---:|",
    ]
    for visual in ("no_result", "pose_differs", "similar", "unknown"):
        g, n = table.get((visual, "geometric"), 0), table.get((visual, "not_geometric"), 0)
        if g or n:
            lines.append(f"| {visual} | {g} | {n} |")
    lines += ["", "기준값 민감도는 sweep.csv, 인물별 판정은 people.csv."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
