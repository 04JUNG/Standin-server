import copy
import hashlib

import numpy as np
import pytest

from converter.rough_layout import camera_for, fit_translation, project
from converter.scene_contract import validate_manifest


def fixture():
    camera = camera_for(765, 714)
    vertices = np.array([[-.2, -.05, 1.0], [.2, -.05, 1.0], [-.2, .08, 1.7], [.2, .08, 1.7]])
    return camera, vertices


def test_box_fit_matches_position_and_size_without_changing_geometry():
    camera, vertices = fixture()
    original = vertices.copy()
    target = project(vertices, [.4, -6, .2], camera)
    box = np.r_[target.min(0), target.max(0)]
    observation = dict(box=box.tolist(), keypoints=[[0, 0]] * 17, scores=[0] * 17)
    translation, report = fit_translation(vertices, {}, observation, camera)
    xy = project(vertices, translation, camera)
    np.testing.assert_allclose((xy.min(0) + xy.max(0))/2, (box[:2]+box[2:])/2, atol=.05)
    assert abs(np.ptp(xy[:, 1]) - (box[3]-box[1])) < .05
    np.testing.assert_array_equal(vertices, original)
    assert report["physical_scale"] == 1 and report["additional_separation_pixels"] == 0


def test_partial_scope_uses_visible_skin_height_not_full_skeleton():
    camera, vertices = fixture()
    head = np.array([0, 0, 1.58])
    desired = [.25, -5, -.15]
    target = project(vertices, desired, camera)
    head_xy = project([head], desired, camera)[0]
    points = np.tile(head_xy, (17, 1))
    observation = dict(box=np.r_[target.min(0), target.max(0)].tolist(), keypoints=points.tolist(), scores=[1]*5+[0]*12)
    translation, report = fit_translation(vertices, {-1: head, 11: [0, 0, -10]}, observation, camera)
    np.testing.assert_allclose(project([head], translation, camera)[0], head_xy, atol=.05)
    assert report["visible_height_error_pixels"] < .05
    assert report["method"] == "head-anchor-visible-height"


def manifest():
    return dict(schema_version=1, cut_id="cut", coordinate_system="threejs-y-up-m",
      rough_layout=dict(version="rough-layout-v1", width=765, height=714),
      persons=[dict(person_index=0, fbx_sha256=hashlib.sha256(b"fbx").hexdigest(),
        observation=dict(box=[100, 100, 600, 650], keypoints=[[100,100]]*17, scores=[.8]*17))])


def test_automatic_manifest_is_idempotent():
    m = manifest()
    assert validate_manifest(validate_manifest(m)) == m


@pytest.mark.parametrize("mutate", [
    lambda m: m["persons"][0].update(position=[1,2,3]),
    lambda m: m["persons"][0].update(scale=2),
    lambda m: m["persons"][0]["observation"].update(box=[0,0,800,700]),
    lambda m: m["persons"][0]["observation"].update(scores=[float('nan')]*17),
    lambda m: m["persons"][0]["observation"].update(keypoints=[[1,2]]),
    lambda m: m["rough_layout"].update(width=True),
])
def test_rejects_implicit_or_manually_overridden_observations(mutate):
    m = copy.deepcopy(manifest()); mutate(m)
    with pytest.raises(ValueError): validate_manifest(m)
