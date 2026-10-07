"""Strict normalized-coordinate contract, independent of GT and experiment files."""

import math

from ..schema import COCO17


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


coordinate = {"type": ["number", "null"], "minimum": 0, "maximum": 1000}
point = obj(
    {
        "name": {"type": "string", "enum": COCO17},
        "x": coordinate,
        "y": coordinate,
        "state": {
            "type": "string",
            "enum": ["visible", "occluded_estimated", "not_present"],
        },
    }
)
box = obj(
    {
        k: {"type": "number", "minimum": 0, "maximum": 1000}
        for k in ("x1", "y1", "x2", "y2")
    }
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1000


def valid_box(value, allow_degenerate=False):
    if not isinstance(value, dict) or set(value) != set(box["properties"]):
        raise ValueError("Invalid bbox fields")
    if not all(finite(x) for x in value.values()):
        raise ValueError("Invalid bbox coordinates")
    if value["x1"] > value["x2"] or value["y1"] > value["y2"]:
        raise ValueError("Inverted bbox")
    if not allow_degenerate and (
        value["x1"] == value["x2"] or value["y1"] == value["y2"]
    ):
        raise ValueError("New-person bbox must have positive area")


def valid_point(value):
    if not isinstance(value, dict) or value.get("name") not in COCO17:
        raise ValueError("Invalid joint name")
    if value.get("state") not in point["properties"]["state"]["enum"]:
        raise ValueError("Invalid joint state")
    if "x" not in value or "y" not in value:
        raise ValueError("Missing coordinates")
    if value["state"] == "not_present":
        if value["x"] is not None or value["y"] is not None:
            raise ValueError("Absent joint coordinates must be null")
    elif not finite(value["x"]) or not finite(value["y"]):
        raise ValueError("Invalid joint coordinates")


def validate_prediction(data):
    """Validate without reordering IDs, dropping scores, or changing coordinates."""
    if not isinstance(data, dict) or set(data) != {"people"}:
        raise ValueError("Invalid prediction fields")
    people = data["people"]
    if not isinstance(people, list) or len(people) > 20:
        raise ValueError("Expected at most 20 people")
    ids = set()
    for person in people:
        if not isinstance(person, dict) or set(person) != {
            "person_index",
            "bbox",
            "keypoints",
        }:
            raise ValueError("Invalid person fields")
        pid = person["person_index"]
        if type(pid) is not int or pid < 0 or pid in ids:
            raise ValueError("Invalid or duplicate person ID")
        ids.add(pid)
        # An RTM detection with no localized points can have a zero-area bbox.
        valid_box(person["bbox"], allow_degenerate=True)
        points = person["keypoints"]
        if not isinstance(points, list) or len(points) != 17:
            raise ValueError("Expected 17 COCO entries per person")
        for expected, value in zip(COCO17, points):
            valid_point(value)
            if value["name"] != expected:
                raise ValueError("Noncanonical or duplicate COCO entry")
            if set(value) - {"name", "x", "y", "state", "score"}:
                raise ValueError("Unexpected joint fields")
            score = value.get("score")
            if score is not None and (
                type(score) not in (int, float) or not math.isfinite(score)
            ):
                raise ValueError("Invalid confidence score")
    return data


def keypoint_bounds(prediction):
    """Derived extents, NOT detection boxes; null when no joint is localized.

    Keep these separate from the original proposal bbox so repaired coordinates
    are not accidentally matched using a stale/zero-area RTM bbox.
    """
    result = []
    for person in prediction["people"]:
        points = [p for p in person["keypoints"] if p["state"] != "not_present"]
        bounds = None
        if points:
            bounds = {
                "x1": min(p["x"] for p in points),
                "y1": min(p["y"] for p in points),
                "x2": max(p["x"] for p in points),
                "y2": max(p["y"] for p in points),
            }
        result.append({"person_index": person["person_index"], "bbox": bounds})
    return result
