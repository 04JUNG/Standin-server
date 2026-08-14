#!/usr/bin/env python3
"""Refine v2 합성 보정 루프.

라이브러리 포즈 A를 2D query/3D GT로 사용하고, 같은 view에서 가장 가까운 다른
포즈 B를 base로 고른 뒤 B→A refine의 2D/3D 개선과 안전 진단을 기록한다.
사람 holdout을 대체하지 않으며 가중치·trust-region 1차 스크리닝용이다.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bvh import load_coco17  # noqa: E402
from src.config import CFG  # noqa: E402
from src.features import _BODY, pose_distance  # noqa: E402
from src.library import (pose_to_feature, project_3d_to_2d,  # noqa: E402
                         view_angle)
from src.refine import REFINE_V2_CODE_VERSION, refine_bvh  # noqa: E402


def _target_2d(path: str, view: str):
    joints, scores = load_coco17(path)
    keypoints = project_3d_to_2d(joints, view_angle(view)).copy()
    keypoints[:, 1] *= -1.0
    return joints.astype(np.float64), scores, keypoints


def _torso(kp) -> float:
    shoulder = (kp[5] + kp[6]) * 0.5
    hip = (kp[11] + kp[12]) * 0.5
    value = float(np.linalg.norm(shoulder - hip))
    return value if value > 1e-8 else 1.0


def _error_3d(candidate, target) -> float:
    scale = _torso(target)
    return float(np.linalg.norm(
        np.asarray(candidate)[_BODY] - np.asarray(target)[_BODY], axis=1
    ).mean() / scale)


def run_synthetic_loop(paths: list[str], output: str, *, view: str = "front",
                       torso: bool = False, cfg=CFG) -> dict:
    if len(paths) < 2:
        raise ValueError("합성 loop에는 서로 다른 BVH가 최소 2개 필요합니다")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    working_cfg = copy.copy(cfg)
    working_cfg.refine_enabled = True
    working_cfg.refine_v2_enabled = True
    working_cfg.refine_v2_lower_body = True
    working_cfg.refine_v2_torso_enabled = bool(torso)

    records = []
    loaded = {}
    for path in paths:
        kp, scores, keypoints = _target_2d(path, view)
        loaded[path] = {
            "kp3d": kp, "scores": scores, "keypoints": keypoints,
            "feature": pose_to_feature(kp, view, scores),
        }

    started = time.perf_counter()
    for target_path in paths:
        target = loaded[target_path]
        neighbors = sorted(
            (
                (pose_distance(target["feature"], loaded[path]["feature"]), path)
                for path in paths if path != target_path
            ),
            key=lambda item: (item[0], item[1]),
        )
        search_distance, base_path = neighbors[0]
        base = loaded[base_path]
        target_id = Path(target_path).stem
        base_id = Path(base_path).stem
        output_bvh = out / f"{target_id}__from__{base_id}.bvh"
        allowed = []
        scores = target["scores"]
        for name, indices in {
            "left_arm": (5, 7, 9), "right_arm": (6, 8, 10),
            "left_leg": (11, 13, 15), "right_leg": (12, 14, 16),
        }.items():
            if bool(np.all(scores[list(indices)] >= 0.3)):
                allowed.append(name)
        tick = time.perf_counter()
        result = refine_bvh(
            base_path, target["keypoints"], target["scores"], view,
            out_path=str(output_bvh), search_distance=search_distance,
            allowed_limbs=allowed, cfg=working_cfg,
        )
        result_path = result.bvh_path if result.refined else base_path
        adopted, _ = load_coco17(result_path)
        base_error = _error_3d(base["kp3d"], target["kp3d"])
        adopted_error = _error_3d(adopted, target["kp3d"])
        records.append({
            "target_pose_id": target_id,
            "target_bvh": str(Path(target_path).resolve()),
            "base_pose_id": base_id,
            "base_bvh": str(Path(base_path).resolve()),
            "view": view,
            "search_distance": float(search_distance),
            "allowed_limbs": allowed,
            "refined": result.refined,
            "reason": result.reason,
            "refine_outcome": result.refine_outcome,
            "result_bvh": str(Path(result_path).resolve()),
            "loss_base": result.loss_base,
            "loss_final": result.loss_final,
            "hybrid_loss_base": result.diagnostics.get("hybrid_loss_base"),
            "hybrid_loss_adopted": result.diagnostics.get("hybrid_loss_adopted"),
            "error_3d_base": base_error,
            "error_3d_adopted": adopted_error,
            "error_3d_improved": adopted_error < base_error,
            "latency_ms": (time.perf_counter() - tick) * 1000.0,
            "limbs": list(result.limbs),
            "limb_decisions": result.limb_decisions,
            "diagnostics": result.diagnostics,
        })

    manifest = {
        "schema_version": 1,
        "kind": "refine_v2_synthetic_correction",
        "refine_version": REFINE_V2_CODE_VERSION,
        "view": view,
        "torso_enabled": bool(torso),
        "source_count": len(paths),
        "elapsed_seconds": time.perf_counter() - started,
        "note": "1차 스크리닝 전용; 웹툰 blind holdout 승격 판정을 대체하지 않음",
        "summary": {
            "attempted": len(records),
            "refined": sum(row["refined"] for row in records),
            "reverted_or_unchanged": sum(not row["refined"] for row in records),
            "error_3d_improved": sum(row["error_3d_improved"] for row in records),
            "accepted_3d_regressions": sum(
                row["refined"] and row["error_3d_adopted"] > row["error_3d_base"]
                for row in records
            ),
        },
        "records": records,
    }
    temporary = out / ".manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as sink:
        json.dump(manifest, sink, ensure_ascii=False, indent=2)
    os.replace(temporary, out / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--view", default="front",
        choices=("front", "three_quarter", "side", "back"),
    )
    parser.add_argument("--torso", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.bvh_dir, "*.bvh")))
    if args.limit > 0:
        paths = paths[:args.limit]
    manifest = run_synthetic_loop(paths, args.out, view=args.view, torso=args.torso)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
