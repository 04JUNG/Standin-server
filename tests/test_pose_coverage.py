"""Geometry and private-file boundaries for the offline coverage workflow."""
from pathlib import Path
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from pose_curation.authored import rotation_between, solve_two_bone
from pose_curation.kinematics import foot_rotation, transported_rotation
from pose_curation.coverage.inputs import import_zip
from pose_curation.review.coverage import coverage_router
from pose_curation.storage import write_json
from pose_curation.coverage.orientation import (
    composition_pose_id, orient_bvh, require_supplied_query,
)
from pose_curation.motion import Motion
from src.bvh import parse_bvh, load_coco17
from scipy.spatial.transform import Rotation
from tests.test_pose_curation import make_motion


@pytest.mark.parametrize('target', [[.2, .8, .3], [4., 0., 0.], [.001, .001, 0.]])
def test_ik_preserves_bone_lengths_at_reach_limits(target):
    origin = np.zeros(3)
    middle, end, adjustment = solve_two_bone(origin, target, [0, 0, 1], .4, .43)
    assert np.linalg.norm(middle - origin) == pytest.approx(.4)
    assert np.linalg.norm(end - middle) == pytest.approx(.43)
    assert np.isfinite(middle).all() and np.isfinite(end).all()
    assert adjustment >= 0


def test_ik_rejects_ambiguous_pole_and_zero_target():
    with pytest.raises(ValueError):
        solve_two_bone(np.zeros(3), [1, 0, 0], [2, 0, 0], .4, .4)
    with pytest.raises(ValueError):
        solve_two_bone(np.zeros(3), [0, 0, 0], [0, 1, 0], .4, .4)


def test_antiparallel_rotation_is_proper_rotation():
    rotation = rotation_between([0, -1, 0], [0, 1, 0])
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.allclose(rotation @ [0, -1, 0], [0, 1, 0])


def test_airborne_foot_follows_turned_shin_instead_of_world_up():
    shin = Rotation.from_euler('ZYX', [90, 20, -15], degrees=True).as_matrix()
    foot, report = foot_rotation(shin, {'mode': 'follow_shin', 'pitch_degrees': 12})
    relative = Rotation.from_matrix(shin.T @ foot)
    assert relative.magnitude() == pytest.approx(np.deg2rad(12))
    assert np.allclose(foot, shin @ Rotation.from_euler('X', 12, degrees=True).as_matrix())
    assert report['applied_degrees'] == 12


def test_ground_target_cannot_force_large_ankle_rotation():
    shin = Rotation.from_euler('Z', 100, degrees=True).as_matrix()
    foot, report = foot_rotation(shin, {'mode': 'planted'})
    assert Rotation.from_matrix(shin.T @ foot).magnitude() == pytest.approx(np.deg2rad(45))
    assert report['contact_limited']
    with pytest.raises(ValueError):
        foot_rotation(shin, {'mode': 'follow_shin', 'pitch_degrees': 90})


def test_limb_transport_preserves_parent_roll_when_direction_already_matches():
    parent = Rotation.from_euler('XYZ', [35, 50, 70], degrees=True).as_matrix()
    edge = np.array([0., -1., 0.])
    result = transported_rotation(parent, edge, parent @ edge, rotation_between)
    assert np.allclose(result, parent)


def test_composition_only_changes_root_and_preserves_rotated_geometry(tmp_path):
    motion = tmp_path / 'motion.bvh'; source = tmp_path / 'source.bvh'; target = tmp_path / 'target.bvh'
    make_motion(motion); Motion.load(motion).export(90, source)
    matrix = Rotation.from_euler('YXZ', [-30, 40, 20], degrees=True).as_matrix()
    orient_bvh(source, target, matrix)
    before_joints, before = parse_bvh(str(source)); after_joints, after = parse_bvh(str(target))
    assert np.allclose(before[:, 6:], after[:, 6:])
    assert [j[0] for j in before_joints] == [j[0] for j in after_joints]
    assert np.allclose(load_coco17(str(source))[0] @ matrix.T, load_coco17(str(target))[0], atol=.0002)


def test_composition_ids_never_carry_the_query_image():
    pid = composition_pose_id('combat_ual1_Punch_Jab', (20, -45, 0))
    assert pid.startswith('composition_combat_ual1_punch_jab_') and len(pid.rsplit('_', 1)[1]) == 8
    assert pid == composition_pose_id('combat_ual1_Punch_Jab', np.array([20, -45, 0]))
    assert pid != composition_pose_id('combat_ual1_Punch_Jab', (0, -45, 0))
    for identity in ('rough_287', 'supplied-cut-03'):
        require_supplied_query(identity)
    for identity in ('user_0123456789ab', 'inst_1b9d6bcd-bbfd-4b2d', 'job_7c9e6679-7425-40de'):
        with pytest.raises(ValueError, match='supplied roughs only'):
            require_supplied_query(identity)


def test_image_import_flattens_paths_and_never_copies_user_uploads(tmp_path):
    image = tmp_path / 'input.png'; Image.new('RGB', (30, 20)).save(image)
    archive = tmp_path / 'images.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('../../rough.png', image.read_bytes())
        out.writestr('__MACOSX/._rough.png', b'ignored')
        out.writestr('instructions.py', b'ignored')
    rows = import_zip(archive, tmp_path / 'safe')
    assert len(rows) == 1 and Path(rows[0]['path']).parent == tmp_path / 'safe'
    assert {row['origin'] for row in rows} == {'supplied'}
    # 사용자 업로드 원본(BetaData)을 로컬로 복사하는 진입점은 없어야 한다.
    import pose_curation.coverage.inputs as inputs
    assert not [name for name in dir(inputs) if 'private' in name.lower()]


def test_coverage_routes_hide_paths_and_confine_downloads(tmp_path):
    root = tmp_path / 'coverage' / '20261001'; root.mkdir(parents=True)
    image = root / 'rough.png'; Image.new('RGB', (30, 20)).save(image)
    write_json(root / 'report.json', {'summary': {'poses': 5},
        'files': {'safe': 'rough.png', 'escape': '../../outside.png'}})
    app = FastAPI(); app.include_router(coverage_router(tmp_path, Path('unused')))
    client = TestClient(app)
    response = client.get('/api/coverage')
    assert response.status_code == 200 and 'files' not in response.json()
    assert response.headers['cache-control'] == 'no-store'
    assert client.get('/api/coverage/image/safe').status_code == 200
    assert client.get('/api/coverage/image/escape').status_code == 404
    assert client.get('/api/coverage/image/not-in-report').status_code == 404


def test_report_refuses_changed_queries_and_missing_related_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pose_curation.coverage import report
    image = tmp_path / 'rough.png'
    Image.new('RGB', (30, 20)).save(image)
    monkeypatch.setattr(report, 'Catalog', lambda *_: SimpleNamespace(all=lambda: []))
    write_json(tmp_path / 'inputs.json', [{'id': 'rough', 'path': str(image), 'sha256': 'image-hash', 'origin': 'supplied'}])
    person = {'person': 0, 'keypoints': [[1, 2]], 'scores': [.8], 'body_visible': 0,
              'quantitative_eligible': False, 'hits': []}
    measured = {'model': {'name': 'real-model'}, 'poses': 1, 'database_sha256': 'db-hash',
                'method': 'same-query', 'images': [{'id': 'rough', 'image_sha256': 'image-hash', 'people': [person]}]}
    note = {'image_id': 'rough', 'related_ids': []}
    write_json(tmp_path / 'annotations.json', {'cases': [note], 'method': 'diagnostic', 'limitations': []})
    write_json(tmp_path / 'before.json', measured)
    write_json(tmp_path / 'after.json', measured)
    assert report.build_report(tmp_path, tmp_path, tmp_path)['summary']['case_groups'] == 1
    person['keypoints'] = [[8, 9]]
    write_json(tmp_path / 'after.json', measured)
    with pytest.raises(ValueError, match='query joints changed'):
        report.build_report(tmp_path, tmp_path, tmp_path)
    person['keypoints'] = [[1, 2]]
    write_json(tmp_path / 'after.json', measured)
    note['related_ids'] = ['nonexistent']
    write_json(tmp_path / 'annotations.json', {'cases': [note], 'method': 'diagnostic', 'limitations': []})
    with pytest.raises(ValueError, match='missing input'):
        report.build_report(tmp_path, tmp_path, tmp_path)
