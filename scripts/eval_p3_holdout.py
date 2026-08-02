"""P3 일반화용 블라인드 평가.

Gemini/VLM 없이 RTMPose가 검출한 모든 인물을 왼쪽→오른쪽으로 정렬하고, 인물별
Top-5를 현재 P3 설정으로 refine한다. P3가 실제로 팔을 복구한 후보는 P3를 끈
full P1 결과도 같이 저장해 사람이 P1 vs P3를 비교할 수 있게 한다.

기본값은 P3 임계값 보정에 쓴 124629/171734/124702 세 러프를 제외한다.

    python scripts/eval_p3_holdout.py
    python scripts/eval_p3_holdout.py --exclude 124629,171734,124702

산출물:
    out/refine_p3_holdout/summary.json
    out/refine_p3_holdout/cutXX_<id>/pN/{manifest.json,sheet.png,Top-5 BVH}
    out/refine_p3_holdout/flagged/*.png  # P3 개입 후보의 base/P1/P3 4-view
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.bvh import load_coco17                                      # noqa: E402
from src.config import CFG                                           # noqa: E402
from src.features import angle_distance, normalize_skeleton          # noqa: E402
from src.library import pose_to_feature                              # noqa: E402
from src.pose import RTMPoseModel                                    # noqa: E402
from src.refine import refine_bvh                                    # noqa: E402
from src.repo import load_entries                                    # noqa: E402
from eval_search import draw                                         # noqa: E402
from refine_top5 import (_proj, _safe, save_sheet, save_thumb,       # noqa: E402
                         search_topk)


def _cut_token(name: str, index: int) -> str:
    digits = re.findall(r"\d{6}", name)
    suffix = digits[-1] if digits else "image"
    return f"cut{index:02d}_{suffix}"


def _person_x(skeleton) -> float:
    scores = np.asarray(skeleton.scores)
    points = np.asarray(skeleton.keypoints)
    visible = points[scores > 0]
    return float(np.median(visible[:, 0])) if len(visible) else float("inf")


def _collision_summary(decisions: dict):
    collisions = {
        limb: decision.get("collision")
        for limb, decision in (decisions or {}).items()
        if decision.get("collision") is not None
    }
    rolled = [
        limb for limb, decision in (decisions or {}).items()
        if decision.get("reason") == "self_collision"
    ]
    depths = [
        float(c["solved_depth"])
        for c in collisions.values()
        if c.get("solved_depth") is not None
    ]
    return collisions, rolled, max(depths, default=0.0)


def _save_4view_compare(path, base_bvh, p1_bvh, p3_bvh, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.library import VIRTUAL_CAMERAS

    views = [view.value for view in VIRTUAL_CAMERAS]
    rows = (("base", base_bvh, "#444"),
            ("P1 full", p1_bvh, "#d07a00"),
            ("P3 final", p3_bvh, "#1769aa"))
    fig, axes = plt.subplots(3, len(views), figsize=(3 * len(views), 8.5))
    for r, (label, bvh, color) in enumerate(rows):
        for c, view in enumerate(views):
            draw(axes[r][c], *_proj(bvh, view), f"{label} — {view}", color=color)
            axes[r][c].invert_yaxis()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="in")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--out-dir", default="out/refine_p3_holdout")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--exclude", default="124629,171734,124702")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-distance-gate", action="store_true",
                    help="프로덕션 base_mismatch 게이트 사용. 기본은 P3 진단 표본 확보를 위해 끔")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    excludes = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    extensions = {".png", ".jpg", ".jpeg"}
    files = [
        name for name in sorted(os.listdir(args.in_dir))
        if os.path.splitext(name)[1].lower() in extensions
        and not any(token in name for token in excludes)
    ]
    if args.limit:
        files = files[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    flagged_dir = os.path.join(args.out_dir, "flagged")
    os.makedirs(flagged_dir, exist_ok=True)
    entries = load_entries(args.db)
    pose_model = RTMPoseModel()
    p1_cfg = copy.copy(CFG)
    p1_cfg.refine_move_gate = False
    p1_cfg.refine_collision_gate = False

    started = time.time()
    all_candidates = []
    image_records = []
    detected_people = 0

    for image_index, filename in enumerate(files, 1):
        image_path = os.path.join(args.in_dir, filename)
        cut_name = os.path.splitext(filename)[0]
        cut_token = _cut_token(cut_name, image_index)
        try:
            skeletons = sorted(
                pose_model.estimate(image_path, [], 0, 0), key=_person_x
            )
        except Exception as exc:
            image_records.append({
                "cut": cut_name, "file": filename, "error": str(exc), "people": 0,
            })
            print(f"[{cut_token}] ERROR {exc}")
            continue

        detected_people += len(skeletons)
        image_records.append({
            "cut": cut_name, "file": filename, "people": len(skeletons),
        })
        print(f"[{cut_token}] people={len(skeletons)}")
        cut_dir = os.path.join(args.out_dir, cut_token)
        os.makedirs(cut_dir, exist_ok=True)

        for person_index, skeleton in enumerate(skeletons):
            person_dir = os.path.join(cut_dir, f"p{person_index}")
            os.makedirs(person_dir, exist_ok=True)
            kp = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
            scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
            feature = normalize_skeleton(kp, scores)
            hits = search_topk(entries, feature, args.topk)
            query = {
                "cut": cut_name,
                "file": filename,
                "person_index": person_index,
                "keypoints": kp.astype(float).tolist(),
                "scores": scores.astype(float).tolist(),
            }
            with open(os.path.join(person_dir, "query.json"), "w", encoding="utf-8") as f:
                json.dump(query, f, ensure_ascii=False, indent=2)

            rows = []
            for rank, (distance, entry) in enumerate(hits, 1):
                stem = f"{rank:02d}_{_safe(entry.pose_id)}"
                p3_bvh = os.path.join(person_dir, stem + ".bvh")
                result = refine_bvh(
                    entry.bvh_path, kp, scores, entry.view.value,
                    out_path=p3_bvh,
                    search_distance=(float(distance) if args.use_distance_gate else None),
                )
                if not result.refined:
                    shutil.copyfile(entry.bvh_path, p3_bvh)

                collisions, rolled, max_depth = _collision_summary(
                    result.limb_decisions
                )
                p1_name = None
                if rolled:
                    p1_bvh = os.path.join(person_dir, stem + "__p1_full.bvh")
                    p1_result = refine_bvh(
                        entry.bvh_path, kp, scores, entry.view.value,
                        out_path=p1_bvh, search_distance=None, cfg=p1_cfg,
                    )
                    if not p1_result.refined:
                        shutil.copyfile(entry.bvh_path, p1_bvh)
                    p1_name = os.path.basename(p1_bvh)
                    if not args.no_render:
                        flagged_name = f"{cut_token}_p{person_index}_r{rank}.png"
                        _save_4view_compare(
                            os.path.join(flagged_dir, flagged_name),
                            entry.bvh_path, p1_bvh, p3_bvh,
                            f"{cut_token} p{person_index} #{rank} "
                            f"rollback={','.join(rolled)} depth={max_depth:.3f}",
                        )

                base_kp3, base_sc = load_coco17(entry.bvh_path)
                final_kp3, final_sc = load_coco17(p3_bvh)
                fit_base = angle_distance(
                    pose_to_feature(base_kp3, entry.view.value, base_sc), feature
                )
                fit_final = angle_distance(
                    pose_to_feature(final_kp3, entry.view.value, final_sc), feature
                )
                row = {
                    "rank": rank,
                    "pose_id": entry.pose_id,
                    "view": entry.view.value,
                    "distance": round(float(distance), 6),
                    "refined": bool(result.refined),
                    "reason": result.reason,
                    "limbs": list(result.limbs),
                    "limb_decisions": result.limb_decisions,
                    "collision_rollback_limbs": rolled,
                    "max_solved_collision_depth": round(max_depth, 6),
                    "fit_base": round(float(fit_base), 6),
                    "fit_final": round(float(fit_final), 6),
                    "bvh": os.path.basename(p3_bvh),
                    "p1_full_bvh": p1_name,
                    "thumb": stem + ".png",
                    "_base_draw": _proj(entry.bvh_path, entry.view.value),
                    "_ref_draw": _proj(p3_bvh, entry.view.value),
                }
                rows.append(row)
                all_candidates.append({
                    "cut": cut_name,
                    "cut_token": cut_token,
                    "person_index": person_index,
                    **{key: value for key, value in row.items()
                       if not key.startswith("_")},
                })
                if not args.no_render:
                    save_thumb(
                        os.path.join(person_dir, row["thumb"]),
                        *row["_ref_draw"],
                        f"#{rank} {entry.pose_id[:22]}\n{result.reason}",
                        "#1a6fb5" if result.refined else "#999",
                    )

            if rows and not args.no_render:
                save_sheet(
                    os.path.join(person_dir, "sheet.png"),
                    f"{cut_token} p{person_index}", kp, scores, rows,
                )
            for row in rows:
                row.pop("_base_draw", None)
                row.pop("_ref_draw", None)
            manifest = {
                "cut": cut_name,
                "file": filename,
                "cut_token": cut_token,
                "person_index": person_index,
                "query": query,
                "candidates": rows,
            }
            with open(os.path.join(person_dir, "manifest.json"), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            rollbacks = sum(bool(row["collision_rollback_limbs"]) for row in rows)
            reasons = ",".join(row["reason"] for row in rows)
            print(f"  p{person_index}: top={len(rows)} p3_rollbacks={rollbacks} "
                  f"reasons={reasons}")

    rollback_candidates = [
        row for row in all_candidates if row["collision_rollback_limbs"]
    ]
    measured_depths = []
    for row in all_candidates:
        for decision in (row.get("limb_decisions") or {}).values():
            collision = decision.get("collision")
            if collision and collision.get("solved_depth") is not None:
                measured_depths.append(float(collision["solved_depth"]))
    positive_depths = sorted(depth for depth in measured_depths if depth > 0)
    reason_hist = {}
    for row in all_candidates:
        reason_hist[row["reason"]] = reason_hist.get(row["reason"], 0) + 1

    summary = {
        "excluded": list(excludes),
        "images": len(files),
        "detected_people": detected_people,
        "candidates": len(all_candidates),
        "p3_rollback_candidates": len(rollback_candidates),
        "p3_rollback_limbs": sum(
            len(row["collision_rollback_limbs"]) for row in rollback_candidates
        ),
        "reason_histogram": reason_hist,
        "collision_depth": {
            "measured_arms": len(measured_depths),
            "positive_arms": len(positive_depths),
            "max": max(positive_depths, default=0.0),
            "p50_positive": (float(np.median(positive_depths))
                             if positive_depths else 0.0),
            "values_positive": positive_depths,
        },
        "image_records": image_records,
        "rollback_candidates": [
            {
                "cut_token": row["cut_token"],
                "person_index": row["person_index"],
                "rank": row["rank"],
                "pose_id": row["pose_id"],
                "limbs": row["collision_rollback_limbs"],
                "depth": row["max_solved_collision_depth"],
                "reason": row["reason"],
                "fit_base": row["fit_base"],
                "fit_final": row["fit_final"],
            }
            for row in rollback_candidates
        ],
        "elapsed_sec": round(time.time() - started, 2),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== P3 holdout summary ===")
    print(f"images={summary['images']} people={detected_people} "
          f"candidates={len(all_candidates)}")
    print(f"rollback candidates={len(rollback_candidates)} "
          f"limbs={summary['p3_rollback_limbs']}")
    print(f"reason={reason_hist}")
    print(f"saved: {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
