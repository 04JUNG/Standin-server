"""Tests for the diagnostic, using toy values rather than research outcomes."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_dual_camera_coverage import ARMS, summarize, valid_query
from standin_eval.perspective_search import camera_grid


def test_camera_arms_are_nested_and_last_step_only_adds_distance():
    grids = list(camera_grid().values())
    assert [len(g) for g in grids] == [4, 13, 39, 117]
    for first, second in zip(grids, grids[1:]):
        assert set(first) <= set(second)
    assert {(c.yaw, c.pitch) for c in grids[2]} == {(c.yaw, c.pitch) for c in grids[3]}
    assert all(c.distance is None for c in grids[2])
    assert {c.distance for c in grids[3]} == {None, 4., 8.}


def test_query_mask_and_raw_observations_are_preserved():
    xy = np.arange(34).reshape(17, 2).tolist()
    scores = [1.] * 17
    scores[9] = .2
    prediction = {'keypoints2d': xy, 'scores': scores}
    raw, mask = valid_query(prediction)
    assert raw.tolist() == xy and not mask[9]
    assert prediction['keypoints2d'] == xy and prediction['scores'] == scores
    scores[5] = .2
    assert valid_query(prediction) is None
    assert valid_query({'error': 'failed'}) is None


def test_failure_denominator_and_exact_id_gains_losses():
    rows = []
    truth = {}
    for i in range(3):
        row = {'dataset': 'toy', 'image': str(i)}
        for arm in ARMS:
            row[arm] = ['target' if i == 0 else 'target_mirror']
        truth[('toy', str(i))] = {'group': 'perspective', 'family_id': str(i), 'pose_id': 'target'}
        rows.append(row)
    rows[0]['pooled_117'] = ['target_mirror']
    rows[1]['pooled_117'] = ['target']
    rows[2]['error'] = 'failure'
    result = summarize(rows, truth)['groups']['perspective']
    assert result['n'] == 3 and result['failures'] == 1
    assert result['counts']['pooled_39'] == {'top1': 1, 'top5': 1}
    assert result['counts']['pooled_117'] == {'top1': 1, 'top5': 1}
    contrast = result['contrasts']['pooled_117_vs_pooled_39']['top5']
    assert (contrast['gains'], contrast['losses'], contrast['net']) == (1, 1, 0)


def test_reject_duplicate_or_missing_queries():
    row = {'dataset': 'toy', 'image': '0', 'error': 'failed'}
    truth = {('toy', '0'): {'group': 'perspective', 'family_id': 'x', 'pose_id': 'x'}}
    for rows in ([], [row, row]):
        try:
            summarize(rows, truth)
        except ValueError:
            pass
        else:
            raise AssertionError('mismatched rows were accepted')


if __name__ == '__main__':
    for name, function in sorted(list(globals().items())):
        if name.startswith('test_'):
            function()
            print('PASS', name)
