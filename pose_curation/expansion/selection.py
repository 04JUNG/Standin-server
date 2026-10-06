"""Select missing body shapes against a frozen, published library.

This is a candidate filter, not an anatomy or artistic-quality gate. Existing
reviewed source frames are not revived, and every survivor still needs rendering
and the normal visual review. Selection never changes the production database.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import re
import shutil
import sqlite3

import numpy as np

from src.bvh import load_coco17
from ..candidates import build_candidates
from ..motion import normalized_body
from ..quality import existing_bvh_path
from ..storage import read_json, sha256, utc_now, write_json


def published_reference(database: Path, data: Path) -> tuple[list[str], np.ndarray]:
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute("SELECT pose_id,bvh_path FROM poses ORDER BY pose_id").fetchall()
    ids, bodies = [], []
    for identity, raw in rows:
        points, scores = load_coco17(str(existing_bvh_path(data, raw)))
        if not np.all(scores[5:] > 0):
            raise ValueError(f"reference pose has missing body joints: {identity}")
        ids.append(identity)
        bodies.append(normalized_body(points))
    if not ids:
        raise ValueError("reference library must not be empty")
    return ids, np.stack(bodies)


def select(manifests, destination, database, *, data=Path("data"), curation=Path("data/curation"),
           minimum_distance=0.15, per_clip=5, limit=250):
    destination, database = Path(destination), Path(database)
    if (destination / "manifest.json").exists():
        raise ValueError("choose a fresh batch; reviewed batches are immutable")
    if not 0 < minimum_distance < 1 or per_clip < 1 or limit < 1:
        raise ValueError("invalid novelty selection limits")
    ids, reference = published_reference(database, Path(data))
    reference_sha256 = sha256(database)
    known_ids, known_frames = set(ids), set()
    for path in Path(curation).glob("batches/*/manifest.json"):
        for row in read_json(path).get("poses", []):
            known_ids.add(row["pose_id"])
            if row.get("source") != "authored_scenario":
                known_frames.add((row.get("source_sha256"), row.get("clip"), row.get("source_frame_0based")))
    pool, skipped = [], []
    seen = set()
    for path in map(Path, manifests):
        for record in read_json(path)["poses"]:
            identity = record["pose_id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity):
                raise ValueError(f"unsafe candidate ID: {identity!r}")
            source_key = (record.get("source_sha256"), record.get("clip"), record.get("source_frame_0based"))
            if identity in known_ids or identity in seen or source_key in known_frames:
                skipped.append({"pose_id": identity, "reason": "previously_reviewed_or_repeated_source"})
                continue
            source = (path.parent / record["bvh"]).resolve()
            if sha256(source) != record["bvh_sha256"]:
                raise ValueError(f"source BVH changed: {identity}")
            points, scores = load_coco17(str(source))
            if not np.all(scores[5:] > 0):
                raise ValueError(f"candidate has missing body joints: {identity}")
            body = normalized_body(points)
            distances = np.sqrt(np.mean(np.sum((reference - body) ** 2, axis=-1), axis=-1))
            index = int(distances.argmin())
            pool.append({"record": deepcopy(record), "source": source, "body": body,
                         "nearest": {"pose_id": ids[index], "distance": float(distances[index])}})
            seen.add(identity)
    # Prefer genuinely missing shapes. Stable ID ordering makes ties reproducible.
    pool.sort(key=lambda p: (-p["nearest"]["distance"], p["record"]["pose_id"]))
    selected, bodies, counts = [], [], Counter()
    for item in pool:
        record = item["record"]
        clip = (record["source"], record["clip"])
        distance = item["nearest"]["distance"]
        if bodies:
            distance = min(distance, float(np.sqrt(np.mean(np.sum((np.stack(bodies) - item["body"]) ** 2, axis=-1), axis=-1)).min()))
        reason = None
        if distance < minimum_distance:
            reason = "body_shape_already_covered"
        elif counts[clip] >= per_clip or len(selected) >= limit:
            reason = "selection_budget"
        if reason:
            skipped.append({"pose_id": record["pose_id"], "reason": reason, "distance": distance})
            continue
        target = destination / "bvh" / (record["pose_id"] + ".bvh")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], target)
        record.update(bvh=target.relative_to(destination).as_posix(), thumbnails={}, preview_kind="pending",
                      retarget_status="not_validated", near_duplicate=False, nearest_existing=item["nearest"],
                      expansion_selection={"body_novelty": distance, "reference_db_sha256": reference_sha256})
        for key in ("anatomy_check", "preview", "quality_review"):
            record.pop(key, None)
        selected.append(record)
        bodies.append(item["body"])
        counts[clip] += 1
    if not selected:
        raise ValueError("no novel candidates")
    projections = build_candidates(selected, destination, destination.name)
    manifest = {"schema_version": 1, "batch_id": destination.name, "created_at": utc_now(),
                "status": "complete", "poses": selected, "failures": [],
                "summary": {"poses": len(selected), "projections": projections},
                "expansion": {"reference_count": len(ids), "reference_sha256": reference_sha256,
                              "minimum_distance": minimum_distance, "per_clip": per_clip,
                              "automatic_approval": False, "skipped": skipped}}
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--database", type=Path, default=Path("data/curation/library/poses.db"))
    p.add_argument("--minimum-distance", type=float, default=0.15)
    p.add_argument("--per-clip", type=int, default=5)
    p.add_argument("--limit", type=int, default=250)
    a = p.parse_args()
    print(select(a.manifest, a.batch, a.database, minimum_distance=a.minimum_distance,
                 per_clip=a.per_clip, limit=a.limit)["summary"])
