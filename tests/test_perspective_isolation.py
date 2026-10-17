import copy
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.confirm_perspective_isolation import select, strong_camera, projection_strength, success_gate, GROUPS
from standin_eval.perspective_search import Camera, project, normalize_batch
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose


def test_family_sampling_excludes_used_and_replays():
    poses = [Pose3D(str(i), str(i//2), np.zeros((17, 3)), np.ones(17)) for i in range(20)]
    a = select(poses, {'0', '2'}, seed=981, count=6)
    b = select(poses, {'0', '2'}, seed=981, count=6)
    assert [p.pose_id for p in a] == [p.pose_id for p in b]
    assert len({p.family_id for p in a}) == 6
    assert not {p.family_id for p in a} & {'0', '2'}


def test_strong_projection_changes_relative_shape_and_matches_renderer():
    from scripts.experiment_z_rendered import render_pose
    joints = _synthetic_pose('running')
    camera = strong_camera(joints, 75, 20)
    assert 1.5 <= projection_strength(joints, camera) <= 2.00001
    xy, valid = project(joints, camera)
    ortho, _ = project(joints, Camera(75, 20))
    assert valid and not np.allclose(normalize_batch(xy), normalize_batch(ortho), atol=.01)
    _, rendered = render_pose(joints, 75, size=128, style='rough', pitch=20, distance=camera.distance)
    assert np.allclose(normalize_batch(xy), normalize_batch(rendered), atol=1e-5)
    assert projection_strength(joints, Camera(75, 20)) == 1.


def test_gate_requires_primary_and_control_and_stress_conditions():
    good = {g: {'counts': {'pooled_39': {'top1': 10, 'top5': 20},
                          'pooled_117': {'top1': 11, 'top5': 30}},
                'projection_vs_orthographic': {'delta': .1, 'paired_bootstrap_95ci': [.02, .18], 'exact_mcnemar_p': .01}}
            for g in GROUPS}
    assert success_gate(good)
    for group, metric in [('perspective', 'top1'), ('orthographic_control', 'top1'),
                          ('orthographic_control', 'top5'), ('strong_perspective', 'top5')]:
        bad = copy.deepcopy(good)
        bad[group]['counts']['pooled_117'][metric] = 0
        assert not success_gate(bad)
    bad = copy.deepcopy(good)
    bad['perspective']['projection_vs_orthographic']['paired_bootstrap_95ci'][0] = 0
    assert not success_gate(bad)


if __name__ == '__main__':
    for name, function in sorted(list(globals().items())):
        if name.startswith('test_'):
            function()
            print('PASS', name)
