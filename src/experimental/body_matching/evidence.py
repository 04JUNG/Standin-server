"""Mask symbolic lines and missing shape evidence without inventing body dimensions."""
from __future__ import annotations

REPRESENTATIONS = ("outline", "shaded", "skeletal", "unknown")
SURFACE_DETAIL = ("drawn", "omitted", "unknown")
SHAPE_REGIONS = {
    "head_ratio_class": ("head", "legs"),
    "body_build": ("torso", "arms", "legs"),
    "frame_width": ("torso",),
    "limb_proportion": ("arms", "legs"),
    "muscularity_visual": ("torso", "arms", "legs"),
    "soft_volume": ("torso",),
    "volume_distribution": ("torso",),
}
# Correlated labels from one outline must not become independent evidence votes.
ATTRIBUTE_GROUPS = {
    "head_ratio_class": "head_proportion", "limb_proportion": "limb_proportion",
    **{k: "body_contour" for k in ("body_build", "frame_width", "muscularity_visual",
                                   "soft_volume", "volume_distribution")},
}


def mask_shape_attributes(observation):
    """Called after schema validation. Legacy observations remain readable, not applicable."""
    representation = observation.get("representation")
    if representation is None:
        return observation
    for axis, regions in SHAPE_REGIONS.items():
        supported = supported_regions(observation, axis)
        unclear = supported_regions(observation, axis, visibility="uncertain")
        if axis == "head_ratio_class":
            valid = (set(regions).issubset(supported) and observation.get("full_body_visible")
                     and not observation.get("foreshortening")
                     and representation.get("torso") != "skeletal")
        else:
            valid = bool(supported or unclear)
        if not valid:
            observation["attributes"][axis] = dict(value=None, visibility="unknown",
                                                   evidence="shape_representation_excluded")
        elif not supported and observation["attributes"][axis]["visibility"] == "visible":
            observation["attributes"][axis]["visibility"] = "uncertain"
    muscle = observation["attributes"]["muscularity_visual"]
    if muscle["value"] == 0 and observation.get("surface_detail") != "drawn":
        observation["attributes"]["muscularity_visual"] = dict(
            value=None, visibility="unknown", evidence="missing_muscle_lines_not_zero_volume")
    return observation


def supported_regions(observation, axis, *, visibility="visible"):
    """Visibility and a shape contour must describe the same evidenced region."""
    regions = observation.get("evidence_regions", {}).get(axis, SHAPE_REGIONS[axis])
    coverage, representation = observation.get("coverage", {}), observation.get("representation", {})
    return {r for r in regions if r in SHAPE_REGIONS[axis]
            and coverage.get(r) == visibility and representation.get(r) in {"outline", "shaded"}}


def independent_groups(observation, axes, geometry_axes):
    groups = evidence_groups(axes, geometry_axes)
    # Count a cropped torso and arms separately only with explicit, disjoint
    # frame/volume evidence. Multiple labels on one outline remain one group.
    explicit = observation.get("evidence_regions", {})
    frame = supported_regions(observation, "frame_width") if "frame_width" in axes and "frame_width" in explicit else set()
    if frame:
        for axis in ("muscularity_visual", "soft_volume", "volume_distribution"):
            value = observation["attributes"][axis]["value"]
            salient = value != "even" if axis == "volume_distribution" else value is not None and value > 0
            if axis not in axes or axis not in explicit or not salient:
                continue
            volume = supported_regions(observation, axis)
            if volume and frame.isdisjoint(volume):
                return sorted((set(groups) - {"body_contour"}) | {"frame_structure", "regional_volume"})
    return groups


def mask_geometry(observation):
    """COCO joints in a stick figure are pose evidence, not measured body proportions."""
    geometry = observation.get("geometry", {})
    representation = observation.get("representation", {})
    coverage = observation.get("coverage", {})
    if observation.get("ownership_ambiguous"):
        return {}
    def visible(region):
        return (coverage.get(region) == "visible"
                and representation.get(region) in {"outline", "shaded"})
    return {axis: value for axis, value in geometry.items()
            if visible("torso") and (axis in {"shoulder_torso", "hip_torso"}
                or ("arm" in axis and visible("arms"))
                or ("leg" in axis and visible("legs")))}


def evidence_groups(axes, geometry_axes):
    groups = {ATTRIBUTE_GROUPS[a] for a in axes}
    if set(geometry_axes) & {"shoulder_torso", "hip_torso"}:
        groups.add("body_contour")
    if any("arm" in a or "leg" in a for a in geometry_axes):
        groups.add("limb_proportion")
    return sorted(groups)
