"""Compare the current XY retrieval with an oracle XY+Z upper bound.

Example:
    python scripts/experiment_z_axis_search.py \
      --library-root /path/to/pose-library/source_bundle \
      --samples 200 --out out/z-axis-search-20261012
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bvh import load_coco17
from standin_eval.z_axis_search import Pose3D, benchmark, build_search_index


def _family_id(pose_id: str, metadata: dict) -> str:
    explicit = metadata.get("pose_family_id")
    if explicit:
        return str(explicit)
    suffix = "_mirror"
    return pose_id[:-len(suffix)] if pose_id.lower().endswith(suffix) else pose_id


def load_library(library_root: Path, max_library: int | None = None) -> list[Pose3D]:
    db_path = library_root / "poses.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"poses.db not found: {db_path}")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT pose_id, bvh_path, meta_json FROM poses ORDER BY pose_id"
        ).fetchall()
    finally:
        connection.close()
    if max_library is not None:
        rows = rows[:max_library]

    poses: list[Pose3D] = []
    skipped: list[str] = []
    for row in rows:
        metadata = json.loads(row["meta_json"] or "{}")
        stored = Path(row["bvh_path"])
        candidates = [stored, library_root / stored, library_root / "bvh" / stored.name]
        bvh_path = next((path for path in candidates if path.is_file()), None)
        if bvh_path is None:
            skipped.append(row["pose_id"])
            continue
        try:
            joints, scores = load_coco17(str(bvh_path))
        except Exception as error:
            skipped.append(f"{row['pose_id']} ({type(error).__name__})")
            continue
        poses.append(
            Pose3D(
                pose_id=row["pose_id"],
                family_id=_family_id(row["pose_id"], metadata),
                joints=np.asarray(joints, dtype=np.float32),
                scores=np.asarray(scores, dtype=np.float32),
            )
        )
    if not poses:
        raise RuntimeError("no usable BVH poses were loaded")
    if skipped:
        print(f"[load] skipped {len(skipped)} / {len(rows)} poses")
        for item in skipped[:10]:
            print(f"  - {item}")
    print(f"[load] {len(poses)} poses from {library_root}")
    return poses


def _markdown_report(result: dict) -> str:
    protocol = result["protocol"]
    metrics = result["metrics"]
    baseline = metrics.get("0")
    best_weight, best = max(
        metrics.items(),
        key=lambda item: (item[1]["top5"], item[1]["top1"], item[1]["mrr"]),
    )
    top5_delta = best["top5"] - baseline["top5"]
    top1_delta = best["top1"] - baseline["top1"]
    if best_weight == "0" or (top5_delta < 0.03 and top1_delta < 0.03):
        decision = "KILL: oracle Z improvement is below the 3 percentage-point gate."
    else:
        decision = (
            f"CONTINUE: z_weight={best_weight} clears the gate "
            f"(Top-1 {top1_delta:+.1%}, Top-5 {top5_delta:+.1%})."
        )

    lines = [
        "# Z-axis search experiment",
        "",
        "> This is an oracle-Z upper-bound test, not evidence that an image model can infer Z reliably.",
        "",
        "## Protocol",
        "",
        f"- Library poses: {protocol['library_poses']:,}",
        f"- Queries: {protocol['samples']:,}",
        f"- Query yaw: {', '.join(map(str, protocol['query_angles_degrees']))} degrees",
        f"- 3D Gaussian noise: {protocol['noise_torso_ratio']:.1%} of torso length",
        f"- Missing limb joints per query: {protocol['dropout_joints']}",
        "- Target: recover the source pose family after folding views/mirrors",
        "",
        "## Results",
        "",
        "| Z weight | Top-1 | Top-5 | Top-10 | MRR | Median rank |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for weight, values in metrics.items():
        lines.append(
            f"| {weight} | {values['top1']:.1%} | {values['top5']:.1%} | "
            f"{values['top10']:.1%} | {values['mrr']:.3f} | "
            f"{values['median_rank']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Automatic gate",
            "",
            decision,
            "",
            "The gate is intentionally simple: continue only when ideal depth adds at least "
            "3 percentage points to Top-1 or Top-5. A continuation still requires a second "
            "experiment on real rough images with predicted Z and human relevance labels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--dropout-joints", type=int, default=1)
    parser.add_argument("--z-weights", default="0,0.25,0.5,1")
    parser.add_argument("--max-library", type=int)
    parser.add_argument("--out", type=Path, default=Path("out/z-axis-search"))
    args = parser.parse_args()

    weights = tuple(float(value) for value in args.z_weights.split(",") if value.strip())
    poses = load_library(args.library_root.resolve(), args.max_library)
    print("[index] building 0/45/90/180-degree XY+Z rows")
    index = build_search_index(poses)
    result = benchmark(
        poses,
        index,
        samples=args.samples,
        seed=args.seed,
        noise=args.noise,
        dropout_joints=args.dropout_joints,
        z_weights=weights,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "results.json"
    report_path = args.out / "REPORT.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    print(_markdown_report(result))
    print(f"[saved] {json_path}")
    print(f"[saved] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
