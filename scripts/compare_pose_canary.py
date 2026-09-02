#!/usr/bin/env python3
"""Compare two immutable detector-inclusive pose canary snapshots."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_responses(run: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run / "responses").glob("*.json"))
    }


def candidate_ids(person: dict) -> list[str]:
    return [item["pose_id"] for item in person["candidates"]]


def local_url(path: Path, output_dir: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output_dir.resolve()), safe="/._-")


def cascade_promotion_decision(
    acceptance_rate: float | None, wrong_owner_count: int | None,
    local_checks_pass: bool = True,
) -> str:
    """Fail-closed rollout gate from POSE_CASCADE_DESIGN D10."""
    if wrong_owner_count is not None and wrong_owner_count >= 1:
        return "rollback_current_x_and_stop_promotion"
    if acceptance_rate is not None and acceptance_rate < 0.50:
        return "remove_cascade_and_return_current_x"
    if wrong_owner_count is None:
        return "blocked_pending_wrong_owner_review"
    if acceptance_rate is None:
        return "blocked_missing_acceptance_measurement"
    if not local_checks_pass:
        return "blocked_failed_local_shadow_checks"
    return "eligible_for_next_canary_stage"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--wrong-owner-count", type=int,
        help="blind ownership review count; required to clear a cascade shadow gate",
    )
    args = parser.parse_args()
    if args.wrong_owner_count is not None and args.wrong_owner_count < 0:
        raise ValueError("--wrong-owner-count must be non-negative")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite comparison: {args.out}")
    args.out.mkdir(parents=True)

    current_summary = json.loads((args.current / "summary.json").read_text())
    candidate_summary = json.loads((args.candidate / "summary.json").read_text())
    candidate_variant = str(candidate_summary.get("variant", "humanart-m"))
    current = load_responses(args.current)
    candidate = load_responses(args.candidate)
    if current.keys() != candidate.keys():
        raise ValueError("canary request sets differ")
    corpus = json.loads(args.boxes.read_text(encoding="utf-8"))

    detector_mismatches = []
    top1_changes = 0
    candidate_pairs = 0
    overlaps = []
    rescued = []
    regressions = []
    for image_stem in current:
        left = current[image_stem]
        right = candidate[image_stem]
        if left["detector_count"] != right["detector_count"]:
            detector_mismatches.append(image_stem)
        if len(left["people"]) != len(right["people"]):
            raise ValueError(f"person output mismatch for {image_stem}")
        for index, (left_person, right_person) in enumerate(zip(left["people"], right["people"])):
            left_ids = candidate_ids(left_person)
            right_ids = candidate_ids(right_person)
            if left_ids and right_ids:
                candidate_pairs += 1
                top1_changes += int(left_ids[0] != right_ids[0])
                overlaps.append(len(set(left_ids[:5]) & set(right_ids[:5])) / 5.0)
            elif not left_ids and right_ids:
                rescued.append(f"{image_stem}::p{index}")
            elif left_ids and not right_ids:
                regressions.append(f"{image_stem}::p{index}")

    current_results = current_summary["results"]
    candidate_results = candidate_summary["results"]
    current_perf = current_summary["performance"]
    candidate_perf = candidate_summary["performance"]
    checks = {
        "same_request_set": current.keys() == candidate.keys(),
        "same_detector_counts": not detector_mismatches,
        "all_person_slots_returned": (
            candidate_results["returned_people"]
            == candidate_summary["inputs"]["expected_people"]
        ),
        "top5_coverage_non_regression": (
            candidate_results["top5_people"] >= current_results["top5_people"]
        ),
        "hard_fallback_non_regression": (
            candidate_results["zero_candidate_people"]
            <= current_results["zero_candidate_people"]
        ),
        "high_confidence_non_regression": (
            candidate_results["high_confidence_people"]
            >= current_results["high_confidence_people"]
        ),
        "top1_distance_p95_guard": (
            candidate_results["top1_distance_p95"]
            <= current_results["top1_distance_p95"] + 0.05
        ),
        "latency_p95_guard": (
            candidate_perf["request_ms_p95"] <= current_perf["request_ms_p95"] * 1.2
        ),
        "rss_guard": (
            candidate_perf["max_rss_platform_units"]
            <= current_perf["max_rss_platform_units"] * 1.2
        ),
    }
    rescue_summary = candidate_summary.get("rescue", {})
    rescue_acceptance_rate = rescue_summary.get("acceptance_rate")
    cascade_gate = candidate_variant == "cascade"
    if cascade_gate:
        checks["cascade_acceptance_rate_at_least_50pct"] = bool(
            isinstance(rescue_acceptance_rate, (int, float))
            and rescue_acceptance_rate >= 0.50
        )
        checks["wrong_owner_review_complete"] = args.wrong_owner_count is not None
        checks["wrong_owner_zero"] = args.wrong_owner_count == 0
    rollback_required = bool(
        cascade_gate
        and args.wrong_owner_count is not None
        and args.wrong_owner_count >= 1
    )
    if cascade_gate:
        promotion_decision = cascade_promotion_decision(
            rescue_acceptance_rate, args.wrong_owner_count,
            local_checks_pass=all(checks.values()),
        )
    elif all(checks.values()):
        promotion_decision = "eligible_for_next_canary_stage"
    else:
        promotion_decision = "blocked"
    remaining_blockers = [
        "human review of wrist/self-occlusion and Top-5 review queue",
        "approved Human-Art/model license review",
        "concurrency/worker matrix on deployment hardware",
        "live shadow metrics and 5% canary observation",
    ]
    if rollback_required:
        remaining_blockers.insert(
            0, "wrong-owner observed: immediately deploy POSE_MODEL_VARIANT=current-x"
        )
    elif cascade_gate and args.wrong_owner_count is None:
        remaining_blockers.insert(0, "blind wrong-owner review is incomplete")
    elif (cascade_gate
          and isinstance(rescue_acceptance_rate, (int, float))
          and rescue_acceptance_rate < 0.50):
        remaining_blockers.insert(0, "cascade acceptance below 50%: remove cascade")
    elif cascade_gate and not all(checks.values()):
        failed_checks = ", ".join(
            name for name, passed in checks.items() if not passed
        )
        remaining_blockers.insert(
            0, f"local shadow checks failed: {failed_checks}"
        )
    report = {
        "checks": checks,
        "local_shadow_pass": all(checks.values()),
        "production_promotion_ready": False,
        "candidate_variant": candidate_variant,
        "rescue": rescue_summary,
        "wrong_owner_count": args.wrong_owner_count,
        "rollback_required": rollback_required,
        "rollback_variant": "current-x" if rollback_required else None,
        "promotion_decision": promotion_decision,
        "detector_count_mismatches": detector_mismatches,
        "candidate_pairs": candidate_pairs,
        "top1_changes": top1_changes,
        "top1_change_rate": top1_changes / candidate_pairs if candidate_pairs else None,
        "top5_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
        "top5_overlap_p50": float(np.percentile(overlaps, 50)) if overlaps else None,
        "rescued_current_failures": rescued,
        "new_candidate_regressions": regressions,
        "current": current_summary,
        "candidate": candidate_summary,
        "remaining_blockers": remaining_blockers,
    }
    (args.out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cards = []
    for image_name, item in corpus["images"].items():
        stem = Path(image_name).stem
        image_path = (ROOT / item["path"]).resolve()
        rows = []
        for index, (left_person, right_person) in enumerate(zip(
            current[stem]["people"], candidate[stem]["people"]
        )):
            columns = []
            for label, person in (
                ("current-X", left_person), (candidate_variant, right_person),
            ):
                candidates = []
                for rank, item_candidate in enumerate(person["candidates"], start=1):
                    thumb = ROOT / "data/thumbs" / (
                        f"{item_candidate['pose_id']}__{item_candidate['view']}.png"
                    )
                    image = (
                        f'<img src="{local_url(thumb, args.out)}">'
                        if thumb.is_file() else '<div class="noimg">no thumb</div>'
                    )
                    candidates.append(
                        f'<div class="candidate">{image}<b>{rank}</b> '
                        f"{html.escape(item_candidate['pose_id'])}"
                        f"<small>d={item_candidate['distance']:.4f}</small></div>"
                    )
                columns.append(
                    f"<section><h4>{label}</h4>"
                    f"<p>{person['skeleton_state']} · {person['coverage_class']} · "
                    f"{person['confidence']}</p><div class=\"grid\">"
                    f"{''.join(candidates)}</div></section>"
                )
            rows.append(
                f"<h3>Person {index}</h3><div class=\"variants\">{''.join(columns)}</div>"
            )
        cards.append(
            f"<article><header><h2>{html.escape(image_name)}</h2>"
            f'<img class="rough" src="{local_url(image_path, args.out)}"></header>'
            f"{''.join(rows)}</article>"
        )
    review = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>Detector-inclusive {html.escape(candidate_variant)} canary review</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f4f4f1;color:#171717}}
article{{background:white;padding:18px;margin-bottom:26px;border-radius:14px}}
header{{display:flex;gap:18px;align-items:flex-start}}.rough{{max-width:280px;max-height:340px;object-fit:contain;border:1px solid #aaa}}
.variants{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}section{{border:1px solid #ddd;padding:10px;border-radius:10px;min-width:0}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:7px}}.candidate{{font-size:11px;overflow-wrap:anywhere}}
.candidate img,.noimg{{width:100%;aspect-ratio:1;object-fit:contain;background:#fafafa;border:1px solid #eee}}small{{display:block;color:#666}}
@media(max-width:1000px){{.variants{{grid-template-columns:1fr}}}}
</style></head><body><h1>{html.escape(candidate_variant)} detector-inclusive `/analyze` review</h1>{''.join(cards)}</body></html>"""
    (args.out / "REVIEW.html").write_text(review, encoding="utf-8")

    lines = [
        f"# Detector-inclusive {candidate_variant} local shadow comparison",
        "",
        f"- Local shadow gate: **{'PASS' if report['local_shadow_pass'] else 'FAIL'}**",
        "- Production promotion: **BLOCKED**",
        f"- Promotion decision: **{promotion_decision}**",
        f"- Wrong-owner count: {args.wrong_owner_count if args.wrong_owner_count is not None else 'NOT REVIEWED'}",
        f"- Detector-count mismatches: {len(detector_mismatches)}",
        f"- Top-5 people: current-X {current_results['top5_people']}/37 → {candidate_variant} {candidate_results['top5_people']}/37",
        f"- Zero-candidate people: {current_results['zero_candidate_people']} → {candidate_results['zero_candidate_people']}",
        f"- High-confidence people: {current_results['high_confidence_people']} → {candidate_results['high_confidence_people']}",
        f"- Top-1 distance p95: {current_results['top1_distance_p95']:.4f} → {candidate_results['top1_distance_p95']:.4f}",
        f"- Request latency p50/p95 ms: {current_perf['request_ms_p50']:.1f}/{current_perf['request_ms_p95']:.1f} → {candidate_perf['request_ms_p50']:.1f}/{candidate_perf['request_ms_p95']:.1f}",
        f"- Top-1 changed among common candidate slots: {top1_changes}/{candidate_pairs} ({report['top1_change_rate']:.1%})",
        f"- Mean Top-5 overlap: {report['top5_overlap_mean']:.3f}",
        f"- Rescued current-X failures: {len(rescued)}",
        f"- New {candidate_variant} failures: {len(regressions)}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    lines.extend(["", "## Remaining blockers", ""])
    lines.extend(f"- {item}" for item in report["remaining_blockers"])
    (args.out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["rollback_required"]:
        return 3
    return 0 if report["local_shadow_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
