"""Versioned shape vocabulary. Body shape and fictional character presentation are independent signals."""
from __future__ import annotations

import math

SCHEMA_VERSION = "body-match.v1"
OBSERVATION_VERSION = "body-observation.v3.presentation"
REGIONS = ("head", "torso", "arms", "legs")
# Head classes are visual proportion buckets, not measured head counts or age.
AXES = {
    "head_ratio_class": ("h3", "h4", "h5", "h6", "h7", "h8"),
    "body_build": ("small_frame", "slim", "regular", "athletic", "muscular", "chubby", "large_frame"),
    "frame_width": ("narrow", "regular", "broad"),
    "limb_proportion": ("short", "regular", "long"),
    "muscularity_visual": (0, 1, 2, 3, 4),
    "soft_volume": (0, 1, 2, 3, 4),
    "volume_distribution": ("even", "abdomen", "lower_body"),
}
NOMINAL_AXES = {"body_build", "volume_distribution"}
GEOMETRY_VERSION = "body-projected-ratios.v1"
GEOMETRY_AXES = ("shoulder_torso", "hip_torso", "left_arm_torso", "right_arm_torso",
                 "left_leg_torso", "right_leg_torso")
VISIBILITY = {"visible": 1.0, "uncertain": 0.25, "unknown": 0.0}


def finite_number(value, low=0.0, high=20.0):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def valid_axis_value(axis, value):
    return type(value) is type(AXES[axis][0]) and value in AXES[axis]


def unknown_attributes():
    return {axis: {"value": None, "visibility": "unknown", "evidence": "not_observed"}
            for axis in AXES}


def parse_person(raw, expected_id):
    if not isinstance(raw, dict) or raw.get("person_id") != expected_id:
        raise ValueError("body_person_id_mismatch")
    allowed = {"person_id", "attributes", "clothing_occlusion", "foreshortening", "ownership_ambiguous"}
    if "presentation" in raw:
        allowed.add("presentation")
    extra = {"coverage", "full_body_visible"}
    extended = extra.issubset(raw)
    if set(raw) != allowed | (extra if extended else set()) or not isinstance(raw["attributes"], dict):
        raise ValueError("invalid_body_observation_fields")
    if set(raw["attributes"]) != set(AXES):
        raise ValueError("invalid_body_attribute_fields")
    result = {"person_id": expected_id, "attributes": unknown_attributes()}
    for flag in allowed - {"person_id", "attributes", "presentation"}:
        if type(raw[flag]) is not bool:
            raise ValueError("invalid_body_observation_flag")
        result[flag] = raw[flag]
    if extended:
        coverage = raw["coverage"]
        if (not isinstance(coverage, dict) or set(coverage) != set(REGIONS)
                or any(v not in VISIBILITY for v in coverage.values())
                or type(raw["full_body_visible"]) is not bool):
            raise ValueError("invalid_body_coverage")
        result.update(coverage=dict(coverage), full_body_visible=raw["full_body_visible"])
    for axis, item in raw["attributes"].items():
        if not isinstance(item, dict) or set(item) != {"value", "visibility", "evidence"}:
            raise ValueError("invalid_body_attribute")
        value, visibility, evidence = item["value"], item["visibility"], item["evidence"]
        if visibility not in VISIBILITY or not isinstance(evidence, str) or len(evidence) > 240:
            raise ValueError("invalid_body_evidence")
        if value is not None and not valid_axis_value(axis, value):
            raise ValueError("invalid_body_axis_value")
        # Providers can emit null+uncertain or a value+unknown. Both carry no
        # usable evidence; mask this axis without discarding other people/axes.
        result["attributes"][axis] = (dict(value=None, visibility="unknown", evidence=evidence)
                                      if visibility == "unknown" or value is None else dict(item))
    if "presentation" in raw:
        from .presentation import parse_presentation
        result["presentation"] = parse_presentation(raw["presentation"], ambiguous=result["ownership_ambiguous"])
    # Never interpret covered volume or an ambiguous person's outline as body evidence.
    if result["ownership_ambiguous"]:
        result["attributes"] = unknown_attributes()
    elif extended:
        supports = {
            "body_build": ("torso", "arms", "legs"),
            "frame_width": ("torso",), "muscularity_visual": ("torso", "arms", "legs"),
            "soft_volume": ("torso",), "volume_distribution": ("torso",),
            "limb_proportion": ("arms", "legs"), "head_ratio_class": REGIONS,
        }
        for axis, regions in supports.items():
            item = result["attributes"][axis]
            strength = max(VISIBILITY[result["coverage"][region]] for region in regions)
            if axis == "head_ratio_class" and (not result["full_body_visible"] or result["foreshortening"]
                    or any(result["coverage"][r] != "visible" for r in ("head", "legs"))):
                strength = 0
            if strength == 0:
                result["attributes"][axis] = dict(value=None, visibility="unknown", evidence="coverage_excluded")
            elif strength < 1 and item["visibility"] == "visible":
                item["visibility"] = "uncertain"
    elif result["clothing_occlusion"]:
        for axis in ("body_build", "frame_width", "muscularity_visual", "soft_volume", "volume_distribution"):
            result["attributes"][axis] = {"value": None, "visibility": "unknown", "evidence": "clothing_occlusion"}
    return result


def response_schema():
    attrs = {}
    for axis, values in AXES.items():
        attrs[axis] = {"type": "object", "additionalProperties": False,
                      "properties": {
                          "value": {"anyOf": [{"type": "integer" if type(values[0]) is int else "string", "enum": list(values)}, {"type": "null"}]},
                          "visibility": {"type": "string", "enum": list(VISIBILITY)},
                          "evidence": {"type": "string", "maxLength": 240}},
                      "required": ["value", "visibility", "evidence"]}
    properties = {"person_id": {"type": "string"},
                  "attributes": {"type": "object", "properties": attrs, "required": list(attrs), "additionalProperties": False},
                  **{key: {"type": "boolean"} for key in ("clothing_occlusion", "foreshortening", "ownership_ambiguous")}}
    properties.update(full_body_visible={"type": "boolean"}, coverage={
        "type": "object", "properties": {r: {"type": "string", "enum": list(VISIBILITY)} for r in REGIONS},
        "required": list(REGIONS), "additionalProperties": False})
    from .presentation import presentation_schema
    properties["presentation"] = presentation_schema()
    return {"type": "object", "additionalProperties": False, "required": ["people"],
            "properties": {"people": {"type": "array", "items": {
                "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}}
