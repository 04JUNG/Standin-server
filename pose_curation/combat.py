"""Select combat frames from verified CC0 motion pools, then reuse review stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np

from src.bvh import load_coco17
from .candidates import build_candidates
from .quality import ReferenceIndex
from .selection import Selection, select_frames
from .storage import read_json, sha256, utc_now, write_json


def choose_samples(points, times, count):
    """A static action is already one pose; moving actions need diversity sampling."""
    if count < 1 or len(points) < 1:
        raise ValueError("at least one source sample and a positive count required")
    if len(points) == 1:
        return [Selection(0, "static_source_pose", 0.0, 0.0)]
    return select_frames(
        points, times, count=count, minimum_distance=0.22, minimum_seconds=0.08
    )


def run(
    pools: list[Path], batch: Path, data_dir: Path, count: int = 3, labels=None
) -> dict:
    if (batch / "manifest.json").exists():
        raise ValueError("choose a new batch; never overwrite reviewed revisions")
    reference = ReferenceIndex.load(data_dir)
    poses = []
    for pool_path in pools:
        pool = read_json(pool_path)
        for action in sorted({row["action"] for row in pool["poses"]}):
            rows = [row for row in pool["poses"] if row["action"] == action]
            points = np.stack([load_coco17(row["bvh"])[0] for row in rows])
            times = np.array([row["frame"] / pool["fps"] for row in rows])
            picks = choose_samples(points, times, count)
            for pick in picks:
                row = rows[pick.sample_index]
                source = Path(row["bvh"])
                if sha256(source) != row["bvh_sha256"]:
                    raise ValueError("source frame changed")
                destination = batch / "bvh" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                nearest = reference.nearest(points[pick.sample_index])
                poses.append(
                    {
                        "pose_id": row["pose_id"],
                        "clip": action,
                        "style": action,
                        "movement": "combat",
                        "source": "quaternius",
                        "author": "Quaternius / Gonzalo Furnier",
                        "license": "CC0-1.0",
                        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                        "source_url": "https://opengameart.org/content/universal-animation-library"
                        + ("-2" if pool["version"] == 2 else ""),
                        "source_sha256": pool["source_sha256"],
                        "source_frame_0based": row["frame"],
                        "source_time_seconds": row["frame"] / pool["fps"],
                        "frame_time": 1 / pool["fps"],
                        "selection": asdict(pick),
                        "bvh": destination.relative_to(batch).as_posix(),
                        "bvh_sha256": row["bvh_sha256"],
                        "source_hand_joints": 30,
                        "thumbnails": {},
                        "preview_kind": "pending",
                        "rig_profile": "mixamo_noprefix",
                        "retarget_status": "not_validated",
                        "nearest_existing": nearest,
                        "near_duplicate": nearest["distance"] < 0.15,
                        "checks": row["checks"],
                        "transform": "Sample original animation; preserve 52 joints including 30 fingers; remove horizontal root travel; verified FK export in centimetres.",
                    }
                )
                # Presentation labels cannot override source hashes, checks or provenance.
                label = (labels or {}).get(action, {})
                poses[-1].update(
                    {
                        key: label[key]
                        for key in (
                            "style",
                            "movement",
                            "category",
                            "category_label",
                            "prop_guides",
                        )
                        if key in label
                    }
                )
    poses.sort(key=lambda row: row["pose_id"])
    count = build_candidates(poses, batch, batch.name)
    manifest = {
        "schema_version": 1,
        "batch_id": batch.name,
        "created_at": utc_now(),
        "status": "complete",
        "poses": poses,
        "failures": [],
        "summary": {"poses": len(poses), "projections": count},
        "source_pools": [{"path": str(p), "sha256": sha256(p)} for p in pools],
    }
    write_json(batch / "manifest.json", manifest)
    (batch / "ATTRIBUTION.txt").write_text(
        "Quaternius / Gonzalo Furnier, Universal Animation Libraries 1 and 2, CC0.\n"
        "Official author distribution: https://opengameart.org/users/quaternius\n"
        "Changes: selected static frames; renamed bones; Y-up centimetre BVH; removed horizontal root travel. Original finger animation preserved.\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    print(
        run(
            args.pool,
            args.batch,
            Path("data"),
            args.count,
            read_json(args.labels) if args.labels else None,
        )["summary"]
    )
