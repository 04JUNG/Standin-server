"""Portable framing data for a rough-aligned scene, independent of Blender.

A camera embedded in FBX is not a promise that CSP selects it on import.
The consumer must apply the complete recipe, including its focal point.
"""
from __future__ import annotations

import math

VERSION = "scene-camera-v1"
CSP_FORMULA_SOURCE = "https://www.clip-studio.com/clip_site/support/help/detail/svc/53/tid/88435"


def framing_recipe(cfg):
    """Convert our centered, front-facing Blender camera to CSP cm/y-up.

    Reject unsupported cameras instead of returning a plausible but wrong view.
    The source is the final layout camera, never per-cut hand-tuned values.
    """
    width, height = cfg["width"], cfg["height"]
    if any(type(v) is not int or not 16 <= v <= 50000 for v in (width, height)):
        raise ValueError("invalid camera image dimensions")
    fov = cfg["horizontal_fov_degrees"]
    if type(fov) not in (int, float) or not 0 < fov < 160 or not math.isfinite(fov):
        raise ValueError("invalid horizontal field of view")
    position = cfg["position_blender"]
    if not isinstance(position, (list, tuple)) or len(position) != 3 or any(
        type(v) not in (int, float) or abs(v) > 1000 or not math.isfinite(v) for v in position
    ):
        raise ValueError("invalid camera position")
    if cfg.get("look_direction_blender", [0, 1, 0]) != [0, 1, 0]:
        raise ValueError("only front-facing layout cameras are supported")
    if cfg.get("projection", "perspective") != "perspective":
        raise ValueError("only perspective layout cameras are supported")
    principal = [width / 2, height / 2]
    if cfg.get("principal_pixels", principal) != principal:
        raise ValueError("off-center projection requires a separate viewport adapter")
    tangent = math.tan(math.radians(fov / 2)) * height / width
    vertical_fov = math.degrees(2 * math.atan(tangent))
    perspective = 100 * tangent / math.tan(math.radians(80))
    x, y, z = position
    camera_cm = [100 * x, 100 * z, -100 * y]
    return {
        "version": VERSION,
        "source_dimensions": [width, height],
        "projection": "perspective",
        "horizontal_fov_degrees": fov,
        "vertical_fov_degrees": vertical_fov,
        "principal_pixels": principal,
        "csp": {
            "coordinate_system": "y-up-centimeters",
            "camera_position": camera_cm,
            "focal_point_position": [camera_cm[0], camera_cm[1], camera_cm[2] - 100],
            "roll_degrees": 0.0,
            "perspective": perspective,
            "perspective_ui_value": round(perspective, 2),
            "link_camera_advancing_retreating": False,
            "adapt_view_to_object": False,
        },
        "requires_camera_application": True,
        "automatic_import_framing_verified": False,
        "csp_recipe_application_verified": False,
        "preconditions": ["same canvas aspect ratio", "unchanged imported scene transform"],
        "formula_source": CSP_FORMULA_SOURCE,
    }


def configure_camera(camera, cfg):
    """Keep Blender/FBX camera optics and the machine-readable recipe together."""
    import json
    recipe = framing_recipe(cfg)
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.sensor_height = 36.0 * cfg["height"] / cfg["width"]
    camera.data.lens = 36.0 / (2 * math.tan(math.radians(cfg["horizontal_fov_degrees"] / 2)))
    camera["source_image_width"] = cfg["width"]
    camera["source_image_height"] = cfg["height"]
    camera["standin_framing_recipe"] = json.dumps(recipe, allow_nan=False, sort_keys=True)
    return recipe


def verify_camera_recipe(camera, expected):
    """Check serialized optics and world pose against the reported projection."""
    import json
    if json.loads(camera.get("standin_framing_recipe", "null")) != expected:
        raise ValueError("FBX camera framing metadata did not survive export")
    if [camera.get("source_image_width"), camera.get("source_image_height")] != expected["source_dimensions"]:
        raise ValueError("FBX camera image dimensions did not survive export")
    data = camera.data
    if (data.type != "PERSP"
            or any(not math.isfinite(v) or v <= 0 for v in (data.sensor_width, data.lens))
            or any(not math.isfinite(v) or abs(v) > 1e-7 for v in (data.shift_x, data.shift_y))):
        raise ValueError("FBX camera projection does not match framing metadata")
    angle = math.degrees(2 * math.atan(data.sensor_width / (2 * data.lens)))
    if not math.isfinite(angle) or abs(angle - expected["horizontal_fov_degrees"]) > 1e-4:
        raise ValueError("FBX camera optics do not match framing metadata")
    # This recipe supports +Y forward / +Z up with zero roll only. Check the
    # world matrix, so a changed parent transform cannot leave a stale receipt
    # looking valid. Camera local -Z is forward and local +Y is up in Blender.
    x, y, z = expected["csp"]["camera_position"]
    target = (
        (1.0, 0.0, 0.0, x / 100),
        (0.0, 0.0, -1.0, -z / 100),
        (0.0, 1.0, 0.0, y / 100),
        (0.0, 0.0, 0.0, 1.0),
    )
    for i, row in enumerate(camera.matrix_world):
        for j, value in enumerate(row):
            tolerance = 1e-4 if j == 3 and i < 3 else 1e-5
            if not math.isfinite(value) or abs(value - target[i][j]) > tolerance:
                raise ValueError("FBX camera world pose does not match framing metadata")
