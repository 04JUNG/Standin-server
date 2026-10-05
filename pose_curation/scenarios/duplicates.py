"""Geometric similarity is a review warning, never a keyword-based approval."""

import argparse
from pathlib import Path
import numpy as np
from src.bvh import load_coco17
from ..motion import normalized_body
from ..quality import ReferenceIndex
from ..storage import contained_path, read_json, write_json


def annotate(records, batch, data_dir=Path("data")):
    if not records:
        raise ValueError("no scenarios to compare")
    reference = ReferenceIndex.load(data_dir)
    points = [load_coco17(str(contained_path(batch, p["bvh"])))[0] for p in records]
    bodies = np.stack([normalized_body(p) for p in points])
    for i, (pose, points3d) in enumerate(zip(records, points)):
        nearest = reference.nearest(points3d)
        distances = np.sqrt(
            np.mean(np.sum((bodies - bodies[i]) ** 2, axis=-1), axis=-1)
        )
        distances[i] = np.inf
        j = int(np.argmin(distances))
        pose["nearest_existing"] = nearest
        pose["nearest_scenario"] = (
            {
                "pose_id": records[j]["pose_id"],
                "distance": round(float(distances[j]), 6),
            }
            if len(records) > 1
            else None
        )
        pose["near_duplicate"] = bool(nearest["distance"] < 0.15 or distances[j] < 0.15)


def run(batch, data_dir=Path("data")):
    """Refresh the whole batch after revisions, without changing review decisions."""
    batch = Path(batch)
    manifest = read_json(batch / "manifest.json")
    if manifest.get("character_render", {}).get("status") == "rendering":
        raise ValueError("cannot annotate a rendering batch")
    annotate(manifest["poses"], batch, data_dir)
    from ..candidates import build_candidates

    build_candidates(manifest["poses"], batch, manifest["batch_id"])
    write_json(batch / "manifest.json", manifest)
    return {
        "compared": len(manifest["poses"]),
        "similarity_warnings": sum(p["near_duplicate"] for p in manifest["poses"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(run(args.batch, args.data_dir))
