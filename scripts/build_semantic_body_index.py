"""Build semantic-only body metadata from existing BVHs, without modifying poses.db.

Example: python scripts/build_semantic_body_index.py --db ... --tags ...
         --bvh-root ... --out tmp/semantic-body-v1.json
Rebuild whenever BVH contents, tagging records, or algorithm/thresholds change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.library import load_bvh_pose
from src.semantic_body_index import BODY_INDEX_VERSION, THRESHOLDS, describe_library_body, records_digest


def build_index(db, tags, bvh_root):
    started = perf_counter()
    raw = json.loads(Path(tags).read_text(encoding="utf-8"))
    records = raw["poses"] if isinstance(raw, dict) else raw
    tagged = {r["pose_id"] for r in records}
    root = Path(bvh_root).resolve()
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT pose_id, bvh_path FROM poses ORDER BY pose_id").fetchall()
    poses, errors = {}, []
    for pose_id, stored_path in rows:
        if pose_id not in tagged:
            continue
        try:
            # A relocated bundle uses the registry's basename under an explicit root.
            # Never search unrelated directories or silently substitute a different pose.
            name = Path(stored_path).name
            if name != pose_id + ".bvh":
                raise ValueError("registry filename differs from pose_id")
            path = (root / name).resolve()
            if path.parent != root:
                raise ValueError("BVH outside explicit root")
            joints, scores = load_bvh_pose(str(path), frame=0)
            value = describe_library_body(joints, scores)
            value["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            poses[pose_id] = value
        except Exception as error:
            errors.append({"pose_id": pose_id, "error": type(error).__name__})
    return {"version": BODY_INDEX_VERSION, "thresholds": THRESHOLDS,
            "tags_sha256": records_digest(records), "frame": 0, "poses": poses,
            "errors": errors, "summary": {
                "requested": sum(p in tagged for p, _ in rows), "built": len(poses),
                "failed": len(errors), "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                "observations": {key: dict(Counter(v["observations"][key] for v in poses.values()))
                                 for key in ("knees", "legs", "arms", "torso")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "tags", "bvh-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build_index(args.db, args.tags, args.bvh_root)
    if not result["poses"]:
        raise RuntimeError("no BVH metadata built")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
