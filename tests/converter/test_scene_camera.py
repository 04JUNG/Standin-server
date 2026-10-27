"""Camera handoff must reproduce source pixels, not merely store an FOV label."""
import json
import math

import numpy as np
import pytest

from converter.rough_layout import camera_for, project
from converter.scene_camera import framing_recipe


@pytest.mark.parametrize("width,height", [(690, 1168), (690, 1926), (283, 163), (512, 512)])
def test_csp_projection_recovers_rough_pixels(width, height):
    cfg = camera_for(width, height)
    recipe = framing_recipe(cfg)
    csp = recipe["csp"]
    # Independently project in CSP's Y-up centimetre coordinates using its
    # documented vertical perspective parameter. Include multiple depths.
    points = np.array([[-1, 0, 1], [1.4, 2, 2], [.3, -3, .4]])
    camera = np.array(csp["camera_position"])
    target = np.array(csp["focal_point_position"])
    np.testing.assert_allclose(target - camera, [0, 0, -100])
    csp_points = points[:, [0, 2, 1]] * [100, 100, -100]
    rel = csp_points - camera
    vertical_tangent = csp["perspective"] / 100 * math.tan(math.radians(80))
    focal = height / (2 * vertical_tangent)
    actual = np.column_stack((width / 2 + focal * rel[:, 0] / -rel[:, 2],
                              height / 2 - focal * rel[:, 1] / -rel[:, 2]))
    np.testing.assert_allclose(actual, project(points, [0, 0, 0], cfg), atol=1e-9)
    assert json.loads(json.dumps(recipe, allow_nan=False)) == recipe
    assert recipe["requires_camera_application"] is True
    assert recipe["automatic_import_framing_verified"] is False
    assert recipe["csp_recipe_application_verified"] is False


@pytest.mark.parametrize("patch", [
    {"width": True}, {"height": 0}, {"width": 50001},
    {"horizontal_fov_degrees": float("nan")},
    {"horizontal_fov_degrees": 180},
    {"horizontal_fov_degrees": 10 ** 1000},
    {"position_blender": [0, float("inf"), 0]},
    {"position_blender": [0, 0]},
    {"position_blender": [10 ** 1000, 0, 0]},
    {"projection": "orthographic"},
    {"look_direction_blender": [1, 0, 0]},
    {"principal_pixels": [0, 0]},
])
def test_reject_unsupported_camera_instead_of_guessing(patch):
    with pytest.raises(ValueError):
        framing_recipe({**camera_for(690, 1168), **patch})
