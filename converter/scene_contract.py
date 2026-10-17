"""One-cut composition contract; safe in both HTTP Python and Blender."""

from __future__ import annotations

import math
import re

SCENE_VERSION = "cut-scene-v1"
COORDINATE_SYSTEM = "threejs-y-up-m"
MAX_PERSONS = 8
MAX_FBX_BYTES = 32 * 1024 * 1024
MAX_SCENE_BYTES = 128 * 1024 * 1024
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


def _number(value, name, limit):
    if type(value) not in (int, float) or abs(value) > limit or not math.isfinite(value):
        raise ValueError(f"invalid {name}")
    return float(value)


def validate_manifest(value):
    """Reject implicit coordinates, duplicate people, paths and non-finite values."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "cut_id", "coordinate_system", "persons"
    }:
        raise ValueError("invalid scene fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported scene schema")
    cut_id = value["cut_id"]
    if not isinstance(cut_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", cut_id):
        raise ValueError("invalid cut_id")
    if value["coordinate_system"] != COORDINATE_SYSTEM:
        raise ValueError("coordinate_system must be threejs-y-up-m")
    rows = value["persons"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PERSONS:
        raise ValueError(f"scene must contain 1..{MAX_PERSONS} persons")
    people, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - {
            "person_index", "fbx_sha256", "position", "rotation_y_degrees", "scale"
        } or not {"person_index", "fbx_sha256", "position"} <= set(row):
            raise ValueError("invalid person fields")
        index = row["person_index"]
        if type(index) is not int or not 0 <= index <= 9999 or index in seen:
            raise ValueError("person_index must be unique and in 0..9999")
        seen.add(index)
        digest = row["fbx_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid fbx_sha256")
        position = row["position"]
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("position must contain x, y, z")
        position = [_number(n, "position", 1000) for n in position]
        yaw = _number(row.get("rotation_y_degrees", 0), "rotation_y_degrees", 360)
        scale = _number(row.get("scale", 1), "scale", 10)
        if scale < 0.01:
            raise ValueError("scale must be in 0.01..10")
        people.append(dict(person_index=index, fbx_sha256=digest, position=position,
                           rotation_y_degrees=yaw, scale=scale))
    return dict(schema_version=1, cut_id=cut_id, coordinate_system=COORDINATE_SYSTEM,
                persons=people)
