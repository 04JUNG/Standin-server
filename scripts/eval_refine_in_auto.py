#!/usr/bin/env python3
"""Run an automatic v1-vs-v2 refine engineering pilot on ``in/`` images.

This is intentionally *not* a GT accuracy benchmark.  RTMPose detections are
frozen as pseudo evaluation units, the same geometric Top-1 base is reused for
both arms, and the external refine evaluator measures query-fit and safety.

Default arms mirror the product/design settings:

* B1: v1, conservative, arms only, production search-distance gate
* B2: v2.4, aggressive, lower body on, torso off

The output is an auditable run directory containing the frozen queries, BVHs,
per-unit JSON records, a machine-readable summary, and a short Markdown report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.config import CFG  # noqa: E402
from src.features import normalize_skeleton  # noqa: E402
from src.pose import RTMPoseModel  # noqa: E402
from src.refine import refine_bvh  # noqa: E402
from src.repo import load_entries  # noqa: E402
from src.search import knn_geometric  # noqa: E402
from standin_eval.refine_evaluator import (  # noqa: E402
    EVALUATOR_VERSION,
    evaluate_refine_artifacts,
    query_evidence,
)


PILOT_VERSION = "in-auto-refine-v1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
METRICS = (
    "joint_nme",
    "limb_direction_error_deg",
    "endpoint_nme",
    "hand_pair_error",
    "lower_pair_error",
    "lap_contact_error",
)
LIMB_KEYPOINTS = {
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}
ALLOWED_JOINTS = {
    "left_arm": ("LeftArm", "LeftForeArm"),
    "right_arm": ("RightArm", "RightForeArm"),
    "left_leg": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
    "right_leg": ("RightUpLeg", "RightLeg", "RightFoot"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_." else "_" for char in value
    ).strip("._") or "unit"


def _person_x(skeleton) -> float:
    scores = np.asarray(skeleton.scores, dtype=np.float64)
    points = np.asarray(skeleton.keypoints, dtype=np.float64)
    visible = points[np.isfinite(points).all(axis=1) & (scores >= 0.3)]
    return float(np.median(visible[:, 0])) if len(visible) else float("inf")


def _allowed_limbs(scores: np.ndarray, threshold: float) -> list[str]:
    return [
        limb for limb, indices in LIMB_KEYPOINTS.items()
        if bool(np.all(scores[np.asarray(indices)] >= threshold))
    ]


def _allowed_joints(limbs: list[str]) -> list[str]:
    return sorted({joint for limb in limbs for joint in ALLOWED_JOINTS[limb]})


def _metric_value(evaluation: dict, metric: str) -> float | None:
    row = (evaluation.get("result_metrics") or {}).get(metric) or {}
    value = row.get("value") if row.get("available") else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _bootstrap_mean_ci(values: list[float], seed: int = 20260814,
                       repetitions: int = 10_000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    # Chunking keeps the memory bound independent of a future larger input set.
    for start in range(0, repetitions, 1000):
        count = min(1000, repetitions - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start:start + count] = array[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def _aggregate_metric(records: list[dict], metric: str) -> dict:
    pairs = []
    for record in records:
        if record.get("status") != "evaluated":
            continue
        v1 = _metric_value(record["arms"]["B1_v1"]["evaluation"], metric)
        v2 = _metric_value(record["arms"]["B2_v24_aggressive"]["evaluation"], metric)
        if v1 is not None and v2 is not None:
            pairs.append((v1, v2))
    v1_values = [pair[0] for pair in pairs]
    v2_values = [pair[1] for pair in pairs]
    improvements = [v1 - v2 for v1, v2 in pairs]
    tolerance = 1e-9
    v1_mean = statistics.fmean(v1_values) if v1_values else None
    v2_mean = statistics.fmean(v2_values) if v2_values else None
    return {
        "n_paired": len(pairs),
        "v1_mean": v1_mean,
        "v2_mean": v2_mean,
        "v2_minus_v1_mean": (
            v2_mean - v1_mean if v1_mean is not None and v2_mean is not None
            else None
        ),
        "v2_error_reduction_pct": (
            (v1_mean - v2_mean) / v1_mean * 100.0
            if v1_mean is not None and v2_mean is not None and v1_mean > 1e-12
            else None
        ),
        "mean_v1_minus_v2": (
            statistics.fmean(improvements) if improvements else None
        ),
        "mean_v1_minus_v2_bootstrap_95_ci": _bootstrap_mean_ci(improvements),
        "v2_better": sum(delta > tolerance for delta in improvements),
        "tie": sum(abs(delta) <= tolerance for delta in improvements),
        "v2_worse": sum(delta < -tolerance for delta in improvements),
    }


def _arm_summary(records: list[dict], arm: str) -> dict:
    rows = [record["arms"][arm] for record in records
            if record.get("status") == "evaluated" and arm in record.get("arms", {})]
    completed = [row for row in rows if not row.get("error")]
    latencies = [float(row["latency_ms"]) for row in completed]
    return {
        "n": len(rows),
        "completed": len(completed),
        "operational_errors": len(rows) - len(completed),
        "refined": sum(bool(row["solver"]["refined"]) for row in completed),
        "unchanged_or_reverted": sum(
            not bool(row["solver"]["refined"]) for row in completed
        ),
        "exact_fallback": sum(
            (not bool(row["solver"]["refined"]))
            and bool(row["evaluation"].get("identity", {}).get("content_equal"))
            for row in completed
        ),
        "new_hard_safety_violations": sum(
            bool(row["evaluation"].get("safety", {}).get("new_hard_violation"))
            for row in completed
        ),
        "evaluator_failures": sum(
            not bool(row["evaluation"].get("ok")) for row in completed
        ),
        "reasons": dict(Counter(
            (str(row["solver"]["reason"]) if not row.get("error")
             else "operational_error:" + str(row["error"]["type"]))
            for row in rows
        )),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
    }


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _config_snapshot(v1_cfg, v2_cfg) -> dict:
    shared = {
        "distance_metric": CFG.distance_metric,
        "skeleton_kpt_threshold": CFG.skeleton_kpt_threshold,
        "min_skeleton_score": CFG.min_skeleton_score,
        "fallback_distance": CFG.fallback_distance,
        "refine_timeout_seconds": CFG.refine_timeout_seconds,
    }
    return {
        "shared": shared,
        "B1_v1": {
            "refine_v2_enabled": v1_cfg.refine_v2_enabled,
            "refine_limbs": v1_cfg.refine_limbs,
            "refine_mode": "conservative",
            "search_distance_gate": True,
        },
        "B2_v24_aggressive": {
            "refine_v2_enabled": v2_cfg.refine_v2_enabled,
            "refine_v2_lower_body": v2_cfg.refine_v2_lower_body,
            "refine_v2_torso_enabled": v2_cfg.refine_v2_torso_enabled,
            "refine_mode": "aggressive",
            "search_distance_passed_for_diagnostics": True,
        },
    }


def _markdown(summary: dict, run_dir: Path) -> str:
    primary = summary["metrics"]["joint_nme"]
    v1 = summary["arms"]["B1_v1"]
    v2 = summary["arms"]["B2_v24_aggressive"]

    def number(value, digits=4):
        return "N/A" if value is None else f"{value:.{digits}f}"

    reduction = primary["v2_error_reduction_pct"]
    ci = primary["mean_v1_minus_v2_bootstrap_95_ci"]
    ci_text = "N/A" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    metric_rows = []
    for name, row in summary["metrics"].items():
        metric_rows.append(
            f"| {name} | {row['n_paired']} | {number(row['v1_mean'])} | "
            f"{number(row['v2_mean'])} | "
            f"{number(row['v2_error_reduction_pct'], 2)}% | "
            f"{row['v2_better']}/{row['tie']}/{row['v2_worse']} |"
        )
    metric_table = "\n".join(metric_rows)
    return f"""# `in/` 자동 refine v1-v2 엔지니어링 파일럿

## 결론

- 이 결과는 **GT 정확도 평가가 아니다.** RTMPose가 한 번 추출한 스켈레톤을 고정해 동일 Top-1 BVH에 대한 v1/v2의 2D query-fit과 안전성을 비교한 진단 결과다.
- 1차 지표 `joint_nme`: v1 `{number(primary['v1_mean'])}` → v2 `{number(primary['v2_mean'])}`, 상대 오차 감소 `{number(reduction, 2)}%` (`n={primary['n_paired']}`).
- paired 개선량 `v1-v2`의 평균은 `{number(primary['mean_v1_minus_v2'])}`, bootstrap 95% CI는 `{ci_text}`다. 양수면 v2가 낫다.
- 단위별 v2 개선/동률/악화는 `{primary['v2_better']}/{primary['tie']}/{primary['v2_worse']}`다.
- 새 hard safety violation은 v1 `{v1['new_hard_safety_violations']}`, v2 `{v2['new_hard_safety_violations']}`건이며, 운영 오류는 각각 `{v1['operational_errors']}` / `{v2['operational_errors']}`건이다.

## 입력과 포함 현황

- 이미지: {summary['input']['images_total']}장
- RTMPose 검출 인물: {summary['input']['people_detected']}명
- 외부 평가 가능: {summary['input']['units_evaluated']}명
- 제외: {summary['input']['units_excluded']}명 (주로 어깨·골반 4점 evidence 부족)
- 운영/검색/이미지 오류 합계: {summary['input']['errors']}건 (arm 운영 오류: v1 {v1['operational_errors']}, v2 {v2['operational_errors']})

## 공통 자동지표

| 지표(낮을수록 좋음) | paired n | v1 평균 | v2 평균 | v2 오차감소 | 개선/동률/악화 |
|---|---:|---:|---:|---:|---:|
{metric_table}

## 실행·안전

| arm | 완료/오류 | refined | 그대로/복구 | exact fallback | 새 hard violation | p50 / p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| v1 | {v1['completed']}/{v1['operational_errors']} | {v1['refined']} | {v1['unchanged_or_reverted']} | {v1['exact_fallback']} | {v1['new_hard_safety_violations']} | {number(v1['latency_ms']['p50'], 1)} / {number(v1['latency_ms']['p95'], 1)} ms |
| v2.4 aggressive | {v2['completed']}/{v2['operational_errors']} | {v2['refined']} | {v2['unchanged_or_reverted']} | {v2['exact_fallback']} | {v2['new_hard_safety_violations']} | {number(v2['latency_ms']['p50'], 1)} / {number(v2['latency_ms']['p95'], 1)} ms |

## 해석 제한

1. RTMPose 출력이 pseudo target이므로, 실제 그림의 관절 위치나 작가가 원하는 자세에 대한 정확도를 증명하지 않는다.
2. VLM 사람 슬롯/소유권을 쓰지 않아 다인·겹침 컷의 검출 누락, 중복, 잘못된 인물 귀속을 평가하지 못한다.
3. 같은 라이브러리와 Top-1을 쓰므로 검색 자체의 정답률·Recall@K도 측정하지 않는다.
4. 따라서 이 결과로 v2를 승격시키면 안 된다. **코드 경로의 자동 회귀/개선 신호**로 쓰고, 승격 판단은 GT 러프+관절 라벨 및 블라인드 작가 평가로 해야 한다.

원시 결과: `{run_dir / 'summary.json'}`, `{run_dir / 'records.json'}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="in")
    parser.add_argument("--db", default="data/poses.db")
    parser.add_argument("--out", help="새 결과 폴더. 생략하면 timestamp 폴더")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_dir = (_REPO / args.in_dir).resolve() if not Path(args.in_dir).is_absolute() else Path(args.in_dir)
    db_path = (_REPO / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    if args.out:
        run_dir = (_REPO / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _REPO / "out" / "eval" / f"in_refine_auto_{stamp}"
    if run_dir.exists():
        raise FileExistsError(f"output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise RuntimeError(f"no images found in {input_dir}")
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    entries = load_entries(str(db_path))
    if not entries:
        raise RuntimeError(f"pose DB has no entries: {db_path}")
    pose_model = RTMPoseModel()

    v1_cfg = copy.copy(CFG)
    v1_cfg.refine_enabled = True
    v1_cfg.refine_v2_enabled = False
    v1_cfg.refine_limbs = "arms"
    v2_cfg = copy.copy(CFG)
    v2_cfg.refine_enabled = True
    v2_cfg.refine_v2_enabled = True
    v2_cfg.refine_v2_lower_body = True
    v2_cfg.refine_v2_torso_enabled = False

    manifest = {
        "pilot_version": PILOT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "started_at_local": datetime.now().astimezone().isoformat(),
        "claim_level": "engineering_pseudo_target_not_gt_accuracy",
        "input_dir": str(input_dir),
        "db": {"path": str(db_path), "sha256": _sha256(db_path),
               "projection_entries": len(entries)},
        "images": [{"path": str(path), "sha256": _sha256(path)} for path in images],
        "git": _git_state(),
        "config": _config_snapshot(v1_cfg, v2_cfg),
        "frozen_unit_rule": (
            "one RTMPose pass per image; all detections sorted left-to-right; "
            "same target skeleton and same geometric Top-1 base reused by both arms"
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)

    records = []
    errors = 0
    detected_people = 0
    run_started = time.perf_counter()
    threshold = float(CFG.skeleton_kpt_threshold)
    for image_index, image_path in enumerate(images, 1):
        print(f"[{image_index:02d}/{len(images):02d}] {image_path.name}", flush=True)
        try:
            skeletons = sorted(
                pose_model.estimate(str(image_path), [], 0, 0), key=_person_x,
            )
        except Exception as exc:  # keep batch evidence even if one image fails
            errors += 1
            records.append({
                "unit_id": f"{image_path.stem}:image_error",
                "image": str(image_path),
                "status": "image_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        detected_people += len(skeletons)
        print(f"  detections={len(skeletons)}", flush=True)
        for person_index, skeleton in enumerate(skeletons):
            unit_id = f"{image_path.stem}:p{person_index}"
            unit_dir = run_dir / "units" / (
                f"{image_index:02d}_{_safe_name(image_path.stem)}_p{person_index}"
            )
            unit_dir.mkdir(parents=True)
            kp = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
            scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
            evidence = query_evidence(kp, scores, score_threshold=threshold)
            query = {
                "unit_id": unit_id,
                "image": str(image_path),
                "image_sha256": _sha256(image_path),
                "person_index_left_to_right": person_index,
                "keypoints": kp.tolist(),
                "scores": scores.tolist(),
                "query_evidence": evidence,
            }
            _write_json(unit_dir / "query.json", query)
            if not evidence.get("valid"):
                records.append({
                    **query,
                    "status": "excluded_invalid_query_evidence",
                    "exclusion_reason": evidence.get("error"),
                })
                print(f"    p{person_index}: EXCLUDED {evidence.get('error')}", flush=True)
                continue
            valid_mask = np.asarray(evidence["target_valid_mask"], dtype=bool)
            feature = normalize_skeleton(
                kp, scores, kpt_thr=threshold, valid_mask=valid_mask,
            )
            hits = knn_geometric(
                entries, feature, top_k=1, query_valid_mask=valid_mask,
            )
            if not hits:
                errors += 1
                records.append({**query, "status": "search_error",
                                "error": "no geometric Top-1 candidate"})
                print(f"    p{person_index}: SEARCH ERROR", flush=True)
                continue
            top1 = hits[0]
            allowed_limbs = _allowed_limbs(scores, threshold)
            base_path = Path(top1.bvh_path)
            if not base_path.is_absolute():
                base_path = (_REPO / base_path).resolve()
            if not base_path.is_file():
                errors += 1
                records.append({**query, "status": "base_error",
                                "error": f"base BVH not found: {base_path}"})
                print(f"    p{person_index}: BASE ERROR {base_path}", flush=True)
                continue

            base_eval = evaluate_refine_artifacts(
                base_path, base_path, kp, scores, top1.view.value,
                score_threshold=threshold, allowed_joint_suffixes=[],
            )
            arms = {}
            arm_specs = (
                ("B1_v1", v1_cfg, "conservative",
                 [limb for limb in allowed_limbs if limb.endswith("arm")]),
                ("B2_v24_aggressive", v2_cfg, "aggressive", allowed_limbs),
            )
            for arm, config, mode, arm_limbs in arm_specs:
                out_bvh = unit_dir / f"{arm}.bvh"
                started = time.perf_counter()
                try:
                    timeout = max(float(config.refine_timeout_seconds), 0.0)
                    deadline = None if timeout == 0.0 else time.monotonic() + timeout
                    result = refine_bvh(
                        str(base_path), kp, scores, top1.view.value,
                        out_path=str(out_bvh), search_distance=float(top1.distance),
                        allowed_limbs=allowed_limbs, refine_mode=mode, cfg=config,
                        deadline=deadline,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    evaluation = evaluate_refine_artifacts(
                        base_path, result.bvh_path, kp, scores, top1.view.value,
                        score_threshold=threshold,
                        allowed_joint_suffixes=_allowed_joints(arm_limbs),
                    )
                    arms[arm] = {
                        "solver": result.to_dict(),
                        "latency_ms": latency_ms,
                        "evaluation": evaluation,
                    }
                except Exception as exc:  # operational failures are pilot results
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    arms[arm] = {
                        "solver": None,
                        "latency_ms": latency_ms,
                        "evaluation": {},
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                    print(
                        f"    p{person_index} {arm}: ERROR "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            record = {
                **query,
                "status": "evaluated",
                "valid_joint_count": int(valid_mask.sum()),
                "allowed_limbs": allowed_limbs,
                "top1": {
                    "pose_id": top1.pose_id,
                    "pose_family_id": top1.pose_family_id,
                    "view": top1.view.value,
                    "distance": float(top1.distance),
                    "base_bvh": str(base_path),
                    "base_bvh_sha256": _sha256(base_path),
                },
                "B0_no_refine": {"evaluation": base_eval},
                "arms": arms,
            }
            records.append(record)
            v1_nme = _metric_value(arms["B1_v1"]["evaluation"], "joint_nme")
            v2_nme = _metric_value(arms["B2_v24_aggressive"]["evaluation"], "joint_nme")
            print(
                f"    p{person_index}: top1={top1.pose_id[:30]} "
                f"d={top1.distance:.3f} NME "
                f"{v1_nme if v1_nme is not None else 'NA'} -> "
                f"{v2_nme if v2_nme is not None else 'NA'}",
                flush=True,
            )
            _write_json(unit_dir / "record.json", record)
            _write_json(run_dir / "records.json", records)

    evaluated = [record for record in records if record.get("status") == "evaluated"]
    excluded = [record for record in records
                if record.get("status") == "excluded_invalid_query_evidence"]
    summary = {
        "pilot_version": PILOT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "claim_level": "engineering_pseudo_target_not_gt_accuracy",
        "decision": "diagnostic_only_not_eligible_for_promotion",
        "elapsed_sec": time.perf_counter() - run_started,
        "input": {
            "images_total": len(images),
            "people_detected": detected_people,
            "units_evaluated": len(evaluated),
            "units_excluded": len(excluded),
            "errors": errors + sum(
                bool(arm_row.get("error"))
                for record in evaluated
                for arm_row in record.get("arms", {}).values()
            ),
        },
        "arms": {
            arm: _arm_summary(evaluated, arm)
            for arm in ("B1_v1", "B2_v24_aggressive")
        },
        "metrics": {metric: _aggregate_metric(evaluated, metric) for metric in METRICS},
        "limitations": [
            "RTMPose detections are pseudo targets, not human-labelled GT",
            "person count, assignment, ownership, misses, and duplicates are not validated",
            "search Top-1 correctness and Recall@K are not measured",
            "2D query fit cannot establish 3D/depth correctness",
        ],
    }
    _write_json(run_dir / "records.json", records)
    _write_json(run_dir / "summary.json", summary)
    with (run_dir / "REPORT.md").open("w", encoding="utf-8") as stream:
        stream.write(_markdown(summary, run_dir))
    print(f"[done] {run_dir}", flush=True)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
