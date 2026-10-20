import copy
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_near_camera import near_cameras, select_policy, GROUPS, ARMS
from standin_eval.perspective_search import camera_grid, build_camera_index, score_camera_index, normalize_batch
from standin_eval.dual_label_search import merged_order
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose


def test_near_grid_only_adds_distances():
    old = camera_grid()['yaw_pitch_perspective_117']
    near = near_cameras()
    assert len(near) == 78 and not set(old) & set(near)
    assert len(set(old + near)) == 195
    assert {(c.yaw, c.pitch) for c in old} == {(c.yaw, c.pitch) for c in near}
    assert {c.distance for c in near} == {2., 3.}


def test_pooled_partitions_equal_full_index():
    rng = np.random.default_rng(809)
    poses = [Pose3D(str(i), str(i), _synthetic_pose('running') + rng.normal(0, .03, (17, 3)), np.ones(17)) for i in range(3)]
    old, near = camera_grid()['yaw_pitch_perspective_117'], near_cameras()
    query = normalize_batch(rng.normal(size=(17, 2)))
    mask = np.ones(17, bool)
    a = score_camera_index(build_camera_index(poses, old), query, mask)
    b = score_camera_index(build_camera_index(poses, near), query, mask)
    full = score_camera_index(build_camera_index(poses, old+near), query, mask)
    assert np.allclose(np.minimum(a, b), full)
    assert merged_order(a, b).tolist() == np.argsort(full, kind='stable').tolist()
    kept = merged_order(a, b, preserve_first=True)
    assert kept[0] == np.argmin(a) and len(set(kept)) == len(kept)


def test_selection_requires_gain_and_all_protections():
    counts = {arm: {g: {'top1': 20, 'top5': 40} for g in GROUPS} for arm in ARMS}
    assert select_policy(counts) == []
    counts['expanded_195']['strong_perspective']['top5'] = 46
    assert select_policy(counts) == ['expanded_195']
    counts['keepfirst_195']['strong_perspective']['top5'] = 46
    assert select_policy(counts)[0] == 'keepfirst_195'
    for group, metric in [(g, 'top1') for g in GROUPS] + [('perspective', 'top5'), ('orthographic_control', 'top5')]:
        bad = copy.deepcopy(counts)
        for arm in ARMS[1:]:
            bad[arm][group][metric] = 0
        assert select_policy(bad) == []


if __name__ == '__main__':
    for name, function in sorted(list(globals().items())):
        if name.startswith('test_'):
            function()
            print('PASS', name)
