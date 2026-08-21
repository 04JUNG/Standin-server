#!/usr/bin/env python3
"""Frozen D0에서 Refine v2.5 final artifact의 품질·지연을 반복 측정한다."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from standin_eval.refine_evaluator import evaluate_refine_artifacts  # noqa: E402
from src.config import CFG  # noqa: E402
from src.refine import REFINE_V2_CODE_VERSION, refine_bvh  # noqa: E402


METRICS = {
    "joint_nme": None,
    "endpoint_nme": None,
    "hand_pair_error": "hand_pair",
    "lower_pair_error": "lower_pair",
    "lap_contact_error": "lap_contact",
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _materialize(result, destination: Path) -> None:
    source = Path(result.bvh_path)
    if source.resolve() == destination.resolve():
        return
    shutil.copyfile(source, destination)


def _metric(evaluation: dict, name: str) -> float | None:
    row = (evaluation.get("result_metrics") or {}).get(name) or {}
    value = row.get("value") if row.get("available") else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _percentiles(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)) if values else None,
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def _allowed_joint_suffixes(limbs: list[str]) -> list[str]:
    mapping = {
        "left_arm": ("LeftArm", "LeftForeArm"),
        "right_arm": ("RightArm", "RightForeArm"),
        "left_leg": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
        "right_leg": ("RightUpLeg", "RightLeg", "RightFoot"),
    }
    return [joint for limb in limbs for joint in mapping.get(limb, ())]


def run(source_dir: Path, output_dir: Path, repeats: int) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = source_dir / "gap_labels.locked.jsonl"
    labels = {
        row["unit_id"]: row for row in _read_jsonl(labels_path)
    } if labels_path.exists() else {}
    # 버전별 재분류는 파일명 순서대로 누적하고 최신 판단이 앞선 판단을 덮는다.
    for overrides_path in sorted(source_dir.glob("gap_labels.v*overrides.jsonl")):
        labels.update({
            row["unit_id"]: row for row in _read_jsonl(overrides_path)
        })
    all_units = [
        row for row in _read_jsonl(source_dir / "frozen_units.jsonl")
        if row.get("status") == "evaluated"
    ]
    excluded_units = [
        row["unit_id"] for row in all_units
        if labels.get(row["unit_id"], {}).get("gap_type") != "near_gap"
    ]
    units = [
        row for row in all_units if row["unit_id"] not in excluded_units
    ]

    cfg = copy.copy(CFG)
    cfg.refine_enabled = True
    cfg.refine_v2_enabled = True
    cfg.refine_v2_lower_body = True
    cfg.refine_v2_torso_enabled = False
    cfg.refine_default_mode = "aggressive"
    cfg.refine_v25_selector_enabled = True

    rows = []
    aggressive_latencies: list[float] = []
    phase_latencies: dict[str, list[float]] = {
        "prepare_ms": [], "conservative_ms": [], "aggressive_ms": [],
        "selector_ms": [], "final_postcheck_ms": [],
    }
    for number, unit in enumerate(units, 1):
        unit_id = str(unit["unit_id"])
        slug = unit_id.replace(":", "__").replace("/", "_")
        unit_dir = output_dir / "units" / slug
        unit_dir.mkdir(parents=True, exist_ok=True)
        base = Path(unit["base_bvh"])
        kp = np.asarray(unit["frozen_keypoints"], dtype=np.float64)
        scores = np.asarray(unit["frozen_scores"], dtype=np.float64)
        limbs = list(unit.get("allowed_limbs") or [])
        target_parts = set(labels.get(unit_id, {}).get("target_parts", []))
        lower_body_observed = bool(
            target_parts & {"left_leg", "right_leg", "lower_pair"}
        )
        view = str(unit["selected_view"])
        first = {}
        repeat_rows = []
        for repeat in range(repeats):
            paths = {
                "conservative": unit_dir / f"conservative-r{repeat}.bvh",
                "aggressive": unit_dir / f"aggressive-final-r{repeat}.bvh",
            }
            results = {}
            timings = {}
            for mode in ("conservative", "aggressive"):
                started = time.perf_counter()
                timeout = max(float(cfg.refine_timeout_seconds), 0.0)
                deadline = None if timeout == 0.0 else time.monotonic() + timeout
                result = refine_bvh(
                    str(base), kp, scores, view,
                    out_path=str(paths[mode]), allowed_limbs=limbs,
                    lower_body_observed=lower_body_observed,
                    refine_mode=mode, deadline=deadline, cfg=cfg,
                )
                timings[mode] = (time.perf_counter() - started) * 1000.0
                _materialize(result, paths[mode])
                results[mode] = result
            aggressive_latencies.append(timings["aggressive"])
            budget = results["aggressive"].diagnostics.get("time_budget", {})
            for key in phase_latencies:
                value = budget.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    phase_latencies[key].append(float(value))
            repeat_rows.append({
                "repeat": repeat,
                "latency_ms": timings,
                "mode_applied": results["aggressive"].diagnostics.get("mode_applied"),
                "candidate_status": results["aggressive"].diagnostics.get(
                    "candidate_status"
                ),
                "selector_fallback_reason": results[
                    "aggressive"
                ].diagnostics.get("selector", {}).get("fallback_reason"),
                "reason": results["aggressive"].reason,
                "time_budget": budget,
            })
            if repeat == 0:
                first = {"paths": paths, "results": results}

        allowed = _allowed_joint_suffixes(limbs)
        evaluations = {
            mode: evaluate_refine_artifacts(
                base, first["paths"][mode], kp, scores, view,
                score_threshold=float(unit["score_threshold"]),
                allowed_joint_suffixes=allowed,
            ) for mode in ("conservative", "aggressive")
        }
        result = first["results"]["aggressive"]
        row = {
            "unit_id": unit_id,
            "image": unit.get("image"),
            "target_parts": labels.get(unit_id, {}).get("target_parts", []),
            "mode_applied": result.diagnostics.get("mode_applied", "base"),
            "candidate_status": result.diagnostics.get("candidate_status"),
            "reason": result.reason,
            "selector": result.diagnostics.get("selector", {}),
            "diagnostics": result.diagnostics,
            "evaluations": evaluations,
            "repeats": repeat_rows,
        }
        rows.append(row)
        with (unit_dir / "result.json").open("w", encoding="utf-8") as sink:
            json.dump(_jsonable(row), sink, ensure_ascii=False, indent=2, sort_keys=True)
        print(
            f"[{number:02d}/{len(units):02d}] {unit_id} "
            f"mode={row['mode_applied']} latency={repeat_rows[0]['latency_ms']['aggressive']:.1f}ms",
            flush=True,
        )

    metric_summary = {}
    for name, target_part in METRICS.items():
        pairs = []
        for row in rows:
            if target_part is not None and target_part not in row["target_parts"]:
                continue
            before = _metric(row["evaluations"]["conservative"], name)
            after = _metric(row["evaluations"]["aggressive"], name)
            if before is not None and after is not None:
                pairs.append((before, after))
        c_mean = float(np.mean([pair[0] for pair in pairs])) if pairs else None
        a_mean = float(np.mean([pair[1] for pair in pairs])) if pairs else None
        metric_summary[name] = {
            "n": len(pairs), "conservative_mean": c_mean, "final_mean": a_mean,
            "error_reduction_pct": (
                (c_mean - a_mean) / c_mean * 100.0
                if c_mean is not None and a_mean is not None and c_mean > 1e-12
                else None
            ),
            "better": sum(after < before - 1e-9 for before, after in pairs),
            "tie": sum(abs(after - before) <= 1e-9 for before, after in pairs),
            "worse": sum(after > before + 1e-9 for before, after in pairs),
        }

    proxy_types = {
        "foot_direction_regression", "ground_contact_regression",
        "lap_contact_regression",
    }
    hard_violations = []
    proxy_alerts = []
    selector_regressions = []
    timeouts = []
    for row in rows:
        for violation in row["evaluations"]["aggressive"].get("safety", {}).get("violations", []):
            target = proxy_alerts if violation.get("type") in proxy_types else hard_violations
            target.append({"unit_id": row["unit_id"], **violation})
        regressions = row["selector"].get("metrics", {}).get("regressions", [])
        if row["mode_applied"] == "aggressive" and regressions:
            selector_regressions.append({"unit_id": row["unit_id"], "metrics": regressions})
        if (row.get("candidate_status") == "timeout"
                or "timeout" in str(row["selector"].get("fallback_reason") or "")
                or any(
                    repeat["candidate_status"] == "timeout"
                    or "timeout" in str(
                        repeat["selector_fallback_reason"] or ""
                    )
                    or repeat["reason"] == "timeout"
                    for repeat in row["repeats"]
                )):
            timeouts.append(row["unit_id"])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source_dir),
        "code_version": REFINE_V2_CODE_VERSION,
        "unit_n": len(rows),
        "excluded_non_near_gap_units": excluded_units,
        "repeats": repeats,
        "modes": dict(Counter(row["mode_applied"] for row in rows)),
        "metrics": metric_summary,
        "latency_ms": {
            "aggressive_request": _percentiles(aggressive_latencies),
            **{key: _percentiles(values) for key, values in phase_latencies.items()},
        },
        "hard_violation_count": len(hard_violations),
        "hard_violations": hard_violations,
        "proxy_alert_count": len(proxy_alerts),
        "proxy_alerts": proxy_alerts,
        "selector_regression_count": len(selector_regressions),
        "selector_regressions": selector_regressions,
        "timeout_unit_ids": sorted(set(timeouts)),
        "rows": rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as sink:
        json.dump(_jsonable(summary), sink, ensure_ascii=False, indent=2, sort_keys=True)
    lines = [
        "# Refine v2.5 frozen D0 benchmark", "",
        f"- units: {len(rows)}", f"- repeats: {repeats}",
        f"- excluded non-near-gap units: `{excluded_units}`",
        f"- modes: `{summary['modes']}`",
        f"- code version: `{REFINE_V2_CODE_VERSION}`",
        f"- hard violations: {len(hard_violations)}",
        f"- proxy alerts: {len(proxy_alerts)}",
        f"- selector regressions: {len(selector_regressions)}",
        f"- timeout units: `{summary['timeout_unit_ids']}`", "",
        "| metric | n | C mean | final mean | reduction | better/tie/worse |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in metric_summary.items():
        reduction = values["error_reduction_pct"]
        lines.append(
            f"| {name} | {values['n']} | {values['conservative_mean']} | "
            f"{values['final_mean']} | {reduction}% | "
            f"{values['better']}/{values['tie']}/{values['worse']} |"
        )
    latency = summary["latency_ms"]
    lines.extend([
        "", "| latency | mean | p50 | p95 | max |", "|---|---:|---:|---:|---:|",
    ])
    for name in (
        "aggressive_request", "prepare_ms", "conservative_ms",
        "aggressive_ms", "selector_ms", "final_postcheck_ms",
    ):
        values = latency[name]
        lines.append(
            f"| {name} | {values['mean']} | {values['p50']} | "
            f"{values['p95']} | {values['max']} |"
        )
    lines.extend([
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve(), max(int(args.repeats), 1))
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
