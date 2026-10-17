import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standin_eval.dual_label_search import LEFT_RIGHT, alternate_labels, merged_order
from standin_eval.perspective_search import normalize_batch, score_camera_index


def test_alternate_is_label_permutation_not_image_mirror():
    xy = np.arange(34).reshape(17, 2); original = xy.copy()
    mask = np.ones(17, dtype=bool); mask[9] = False; before = mask.copy()
    out, valid = alternate_labels(xy, mask)
    assert np.array_equal(out[10], xy[9]) and not valid[10] and valid[9]
    assert np.array_equal(xy, original) and np.array_equal(mask, before)
    again, again_mask = alternate_labels(out, valid)
    assert np.array_equal(again, xy) and np.array_equal(again_mask, mask)


def test_normalization_anchors_survive_global_side_swap():
    xy = np.random.default_rng(54).normal(size=(17, 2))
    assert np.allclose(normalize_batch(xy)[LEFT_RIGHT], normalize_batch(xy[LEFT_RIGHT]))


def test_preserves_first_without_duplicate_pose_or_oracle():
    d = np.array([.4, .1, .5, .6, .7, .8])
    a = np.array([.02, .2, .01, .03, .04, .05])
    assert merged_order(d, a).tolist() == [2, 0, 3, 4, 5, 1]
    assert merged_order(d, a, preserve_first=True).tolist() == [1, 2, 0, 3, 4, 5]
    assert merged_order(d, d, preserve_first=True).tolist() == np.argsort(d).tolist()


def test_cost_ties_and_invalid_values_are_deterministic():
    assert merged_order([.2, .2], [.2, .2]).tolist() == [0, 1]
    assert merged_order([np.nan, np.inf, -1], [np.inf, .2, -3], preserve_first=True).tolist() == [1]
    assert len(merged_order([np.inf], [np.nan])) == 0


def test_alternate_scoring_recovers_same_pose_with_observation_mask():
    rng = np.random.default_rng(908)
    target = normalize_batch(rng.normal(size=(17, 2)))
    different = normalize_batch(rng.normal(size=(17, 2)))
    index = (np.stack([target, different])[None, ...], np.ones((1, 2), bool))
    mask = np.ones(17, bool); mask[9] = False
    query, qm = alternate_labels(target, mask)
    corrected, cm = alternate_labels(query, qm)
    direct = score_camera_index(index, query, qm)
    alternate = score_camera_index(index, corrected, cm)
    assert alternate[0] == 0 and direct[0] > 0
    assert merged_order(direct, alternate)[0] == 0
    assert np.array_equal(cm, mask)


if __name__ == '__main__':
    for n, f in sorted(list(globals().items())):
        if n.startswith('test_'): f(); print('PASS', n)
