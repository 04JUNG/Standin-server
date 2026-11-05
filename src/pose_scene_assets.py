"""Physical scene availability is separate from semantic object tags.

Only content-bound, reviewed geometry is advertised. This module does not
create objects, infer grip sides, or authorize a different/refined BVH.
"""
from functools import lru_cache
import json
from pathlib import Path
import re

CATALOG = Path(__file__).resolve().parents[1] / "config/pose_scene_assets.v1.json"
HASH = re.compile(r"[a-f0-9]{64}")


@lru_cache(maxsize=1)
def load_assets():
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported scene asset catalog")
    rows = {}
    for row in payload["poses"]:
        if (not isinstance(row["pose_id"], str) or row["pose_id"] in rows
                or any(not HASH.fullmatch(row[k]) for k in
                       ("source_bvh_sha256", "reviewed_fbx_sha256", "recipe_token"))
                or not isinstance(row["objects"], list)
                or not all(isinstance(o, str) and o for o in row["objects"])):
            raise ValueError("invalid scene asset binding")
        rows[row["pose_id"]] = row
    return rows


def evidence(pose, catalog=None):
    """Unknown/mismatching sources never acquire a prop from a shared label."""
    row = (load_assets() if catalog is None else catalog).get(pose.get("pose_id"))
    source = (pose.get("meta") or {}).get("bvh_sha256")
    if not row or source != row["source_bvh_sha256"]:
        return {"status": "not_registered", "objects": []}
    return {"status": "reviewed_placement", "objects": list(row["objects"]),
            "quality": "primitive-placement-guide",
            "source_bvh_sha256": source, "recipe_token": row["recipe_token"],
            "reviewed_fbx_sha256": row["reviewed_fbx_sha256"],
            "scope": "registered_source_default_character",
            "complete_collision_certificate": False}


def prioritize(candidates, requested_objects):
    """Stable reorder inside an already validated group; no score/limit changes."""
    requested = set(requested_objects)
    for candidate in candidates:
        candidate.matched["scene_props"] = evidence(candidate.pose)
    if requested:
        candidates.sort(key=lambda c: not requested.issubset(
            set(c.matched["scene_props"]["objects"])))
    for rank, candidate in enumerate(candidates, 1):
        candidate.rank = rank
