"""Apply reviewed presentation labels without changing BVH or source provenance.

This is deliberately separate from candidate generation: a label revision cannot
silently replace source hashes, geometry evidence, or a prior review decision.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .candidates import build_candidates
from .storage import read_json, write_json


FIELDS = frozenset({"style", "movement", "category", "category_label"})


def apply(batch: Path, labels_path: Path) -> dict:
    manifest_path = batch / "manifest.json"
    manifest = read_json(manifest_path)
    labels = read_json(labels_path)
    if not isinstance(labels, dict) or not labels:
        raise ValueError("labels must be a nonempty object")
    if any(not isinstance(value, dict) or set(value) != FIELDS for value in labels.values()):
        raise ValueError("each label requires only style, movement, category, category_label")
    if any(not all(isinstance(value, str) and value for value in row.values()) for row in labels.values()):
        raise ValueError("label values must be nonempty strings")
    poses = manifest["poses"]
    sources = {pose["clip"].split("_", 1)[0] for pose in poses}
    if sources != set(labels):
        raise ValueError(f"source styles and label keys differ: {sources ^ set(labels)}")
    if any(pose["source"] != "100style" for pose in poses):
        raise ValueError("this label map is for 100STYLE batches only")
    for pose in poses:
        pose.update(labels[pose["clip"].split("_", 1)[0]])
    manifest["presentation_labels"] = {
        "source": labels_path.as_posix(),
        "note": "Style labels describe the visible movement; source clip and original motion are unchanged.",
    }
    write_json(manifest_path, manifest)
    build_candidates(poses, batch, batch.name)
    return {"poses": len(poses), "styles": len(labels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    print(apply(args.batch, args.labels))
