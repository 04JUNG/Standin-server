from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from src import pose_camera as camera
from converter.camera import validate_rotation
from api.candidate_presentation import refine_in_camera
from api.models import CandidateCamera
from src.bvh import fk, parse_bvh
from tests.test_pose_orientation import bvh_file


def body():
    points = np.zeros((17, 3))
    points[5:] = [(1, 3, 0), (-1, 3, 0), (1.7, 2, .8), (-1.8, 2.6, -.2),
                  (1.5, 3, 1.6), (-2.3, 3.4, .6), (.6, 0, 0), (-.6, 0, 0),
                  (.8, -1.8, .8), (-.6, -1.9, -.3), (1.2, -3.5, .6), (-.9, -3.7, 0)]
    return points


def fit(monkeypatch, tmp_path, points, rough, facing):
    path = tmp_path / 'dummy.bvh'
    path.write_text('unused')
    monkeypatch.setattr(camera, '_source', lambda *args: (points, np.ones(17), camera.canonical_yaw(points), 'a'*64))
    return camera.fit_camera(path, rough, np.ones(17), facing)


@pytest.mark.parametrize('source_yaw', [0, 57, 170, -110])
def test_raw_library_heading_does_not_change_fitted_front(monkeypatch, tmp_path, source_yaw):
    original = body()
    desired = camera.camera_matrix(12, 18, -8)
    rough = (original @ desired.T)[:, :2] * [1, -1] * 120 + [340, 210]
    source = original @ camera.camera_matrix(source_yaw).T
    result = fit(monkeypatch, tmp_path, source, rough, 'front')
    assert result['status'] == 'fitted'
    assert result['display_view'] == 'front'
    actual = source @ np.asarray(result['rotation']).T
    assert np.allclose(actual, original @ desired.T, atol=1e-5)
    assert np.linalg.det(result['rotation']) == pytest.approx(1)


def test_upside_down_pose_is_not_stood_upright():
    inverted = body() @ camera.camera_matrix(72, 0, 180).T
    canonical = inverted @ camera.camera_matrix(camera.canonical_yaw(inverted)).T
    assert canonical[5, 1] < canonical[11, 1]
    assert np.allclose(canonical, body() @ camera.camera_matrix(0, 0, 180).T)


def test_no_pseudo_face_keypoints_and_no_mirrored_fit(monkeypatch, tmp_path):
    points = body()
    rough = points[:, :2] * [1, -1]
    rough[:5] = 1e9
    result = fit(monkeypatch, tmp_path, points, rough, 'front')
    assert result['fit_error'] < 1e-5
    back = fit(monkeypatch, tmp_path, points, rough, 'back')
    assert back['display_view'] == 'back'
    # A wrong facing prior cannot be concealed by reflection or negative scale.
    assert back['status'] == 'canonical_fallback'


@pytest.mark.parametrize('matrix', [[[1,0,0],[0,1,0],[0,0,-1]], [[2,0,0],[0,1,0],[0,0,1]], [[float('nan')]*3]*3, [1,2,3]])
def test_reject_reflection_scale_and_malformed_rotation(matrix):
    with pytest.raises(ValueError):
        validate_rotation(matrix)


@pytest.mark.parametrize('order', list(permutations('XYZ')))
def test_refine_frame_roundtrip_preserves_children_and_root_position(tmp_path, order):
    path = tmp_path / 'pose.bvh'
    bvh_file(path, order)
    raw = path.read_text()
    matrix = camera.camera_matrix(123, -25, 11)
    rotated = camera.rotate_bvh_text(raw, matrix.tolist())
    restored = camera.rotate_bvh_text(rotated, matrix.T.tolist())
    target = tmp_path / 'restored.bvh'
    target.write_text(restored)
    j, f = parse_bvh(str(path)); jj, ff = parse_bvh(str(target))
    assert np.array_equal(f[:, 6:], ff[:, 6:])
    assert np.array_equal(f[:, :3], ff[:, :3])
    for i, value in fk(j, f[0]).items():
        assert np.allclose(value, fk(jj, ff[0])[i])


def test_refine_rejects_stale_source_before_calling_solver(tmp_path):
    path = tmp_path / 'pose.bvh'; bvh_file(path, 'XYZ')
    with pytest.raises(ValueError, match='source changed'):
        refine_in_camera(lambda *a, **kw: pytest.fail('called solver'), str(path), [], [], 'back',
                         camera=SimpleNamespace(source_bvh_sha256='a'*64))


def test_sparse_observation_does_not_invent_camera(monkeypatch, tmp_path):
    points = body(); path=tmp_path/'x'; path.write_text('x')
    monkeypatch.setattr(camera, '_source', lambda *args: (points, np.ones(17), 0, 'a'*64))
    scores = np.zeros(17); scores[:7] = 1
    assert camera.fit_camera(path, points[:, :2], scores, 'front') is None
