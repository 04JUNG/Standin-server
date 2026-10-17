import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_depth_style import fixed_infer, pair_counts, features
from standin_eval.z_diagnostics import undo_quarter_turn


def test_fixed_inference_uses_supplied_crop_and_rotation():
    rgb = np.arange(9*12*3, dtype=np.uint8).reshape(9, 12, 3)
    xy = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
    k3 = np.zeros((1, 17, 3)); k3[0, :, 2] = np.arange(17)
    bbox = [1, 2, 7, 8]
    for turn in range(4):
        def model(pixels, bboxes):
            assert np.array_equal(pixels, np.rot90(rgb, turn))
            assert pixels.flags.c_contiguous and bboxes == [bbox]
            return k3, np.ones((1, 17)), None, xy
        result = fixed_infer(rgb, model, turn, bbox)
        assert np.array_equal(result['keypoints2d'], undo_quarter_turn(xy[0], turn, 12, 9))
        assert result['z_meters'] == list(range(17))


def test_pair_counts_obey_mask_and_front_back_controls():
    result = pair_counts(np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1]),
                         np.array([True, True, False, False]))
    assert result == {'n': 2, 'correct': 1, 'always_front': 2, 'always_back': 0}
    assert pair_counts(np.ones(4), np.ones(4), np.zeros(4, bool))['n'] == 0


def test_features_preserve_human_mask_and_do_not_use_confidence_as_z_gate():
    xy = np.column_stack((np.arange(17), np.arange(17)**2)).tolist()
    row = {'human': {'keypoints2d': xy}, 'mask': [True]*17}
    row['mask'][9] = False
    pred = {'keypoints2d': xy, 'z_meters': list(np.arange(17)*.2), 'scores': [0]*17}
    result = features(row, pred)['hypotheses']['xy_aligned']
    assert result['all'] == [True, False, True, True, True, True, True, True]
    assert result['all'] == result['xy_gate']
    assert not row['mask'][9] and pred['z_meters'][9] == 1.8


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn(); print('PASS', name)
