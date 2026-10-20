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
    if not isinstance(value, dict) or set(value) - {
        "rough_layout",
        "schema_version", "cut_id", "coordinate_system", "persons"
    } or not {"schema_version", "cut_id", "coordinate_system", "persons"} <= set(value):
        raise ValueError("invalid scene fields")
    rough = value.get("rough_layout")
    if rough is not None:
        if not isinstance(rough, dict) or set(rough) != {"version", "width", "height"}:
            raise ValueError("invalid rough layout")
        if rough["version"] != "rough-layout-v1" or any(
            type(rough[k]) is not int or not 16 <= rough[k] <= 50000 for k in ("width", "height")
        ):
            raise ValueError("invalid rough image dimensions")
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
            "person_index", "fbx_sha256", "position", "rotation_y_degrees", "scale", "observation"
        } or not {"person_index", "fbx_sha256"} <= set(row):
            raise ValueError("invalid person fields")
        if (rough is None and ("position" not in row or "observation" in row)) or (
            rough is not None and ("observation" not in row or any(k in row for k in ("position", "scale", "rotation_y_degrees")))
        ):
            raise ValueError("use either automatic observations or explicit placement")
        index = row["person_index"]
        if type(index) is not int or not 0 <= index <= 9999 or index in seen:
            raise ValueError("person_index must be unique and in 0..9999")
        seen.add(index)
        digest = row["fbx_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid fbx_sha256")
        position = row.get("position", [0, 0, 0])
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("position must contain x, y, z")
        position = [_number(n, "position", 1000) for n in position]
        yaw = _number(row.get("rotation_y_degrees", 0), "rotation_y_degrees", 360)
        scale = _number(row.get("scale", 1), "scale", 10)
        if scale < 0.01:
            raise ValueError("scale must be in 0.01..10")
        people.append(dict(person_index=index, fbx_sha256=digest, position=position,
                           rotation_y_degrees=yaw, scale=scale))
        if rough is not None:
            o = row["observation"]
            if not isinstance(o, dict) or set(o) != {"box", "keypoints", "scores"}:
                raise ValueError("invalid observation")
            box = o["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("invalid observed box")
            box = [_number(v, "box", 100000) for v in box]
            if not (0 <= box[0] < box[2] <= rough["width"] and 0 <= box[1] < box[3] <= rough["height"]):
                raise ValueError("box must be clipped to the source image")
            points, scores = o["keypoints"], o["scores"]
            if not isinstance(points, list) or len(points) != 17 or not isinstance(scores, list) or len(scores) != 17:
                raise ValueError("expected COCO17 observations")
            clean = []
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("invalid observed joint")
                clean.append([_number(v, "joint", 100000) for v in point])
            scores = [_number(v, "joint score", 1) for v in scores]
            if min(scores) < 0:
                raise ValueError("invalid joint score")
            people[-1] = dict(person_index=index, fbx_sha256=digest,
                              observation=dict(box=box, keypoints=clean, scores=scores))
    result = dict(schema_version=1, cut_id=cut_id, coordinate_system=COORDINATE_SYSTEM, persons=people)
    if rough is not None:
        result["rough_layout"] = dict(rough)
    return result
