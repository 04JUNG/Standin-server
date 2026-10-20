import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standin_eval.pose_geometry_diagnostic import normalize_body, yaw_aligned_rms, body_label_errors, BODY_LEFT_RIGHT


def fixture():
    return np.random.default_rng(23).normal(size=(17, 3))


def test_translation_scale_and_yaw_invariance():
    p = fixture()
    theta = .731
    r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    q = normalize_body(p)
    c = normalize_body((p @ r.T) * 3.1 + [8, -2, 4])
    assert yaw_aligned_rms(q, c) < 1e-12
    assert np.allclose(yaw_aligned_rms(q, np.stack([q, c])), [0, 0], atol=1e-12)


def test_no_reflection_or_label_swap():
    q = normalize_body(fixture())
    assert yaw_aligned_rms(q, q * [-1, 1, 1]) > .1
    c = q.copy(); c[[0, 1]] = c[[1, 0]]
    assert yaw_aligned_rms(q, c) > .1


def test_closed_form_matches_dense_angle_search_and_is_symmetric():
    q = normalize_body(fixture())
    c = normalize_body(np.random.default_rng(9).normal(size=(17, 3)))
    theta = np.linspace(-np.pi, np.pi, 30001)
    rotated = np.broadcast_to(c, (len(theta), 12, 3)).copy()
    rotated[:, :, 0] = np.cos(theta)[:, None]*c[:, 0] + np.sin(theta)[:, None]*c[:, 2]
    rotated[:, :, 2] = -np.sin(theta)[:, None]*c[:, 0] + np.cos(theta)[:, None]*c[:, 2]
    brute = np.sqrt(((rotated-q)**2).sum(axis=-1).mean(axis=-1)).min()
    exact = yaw_aligned_rms(q, c)
    assert abs(exact-brute) < 1e-7
    assert abs(exact-yaw_aligned_rms(c, q)) < 1e-12


def test_degenerate_and_nonfinite_rejected_without_mutation():
    p = fixture(); old = p.copy(); normalize_body(p)
    assert np.array_equal(p, old)
    for bad in [np.zeros((17, 3)), np.full((17, 3), np.nan)]:
        try: normalize_body(bad)
        except ValueError: pass
        else: raise AssertionError('bad skeleton accepted')


def test_xy_label_diagnostic_keeps_query_mask_and_labels():
    gt = fixture()[:, :2]; q = gt.copy(); mask = np.ones(17, dtype=bool)
    direct, swapped = body_label_errors(q, gt, mask)
    assert direct == 0 and swapped > 0
    q[5:17] = gt[BODY_LEFT_RIGHT]; original = q.copy()
    direct, swapped = body_label_errors(q, gt, mask)
    assert direct > 0 and swapped == 0 and np.array_equal(q, original)
    q[9] += 100; mask[9] = False
    assert body_label_errors(q, gt, mask)[1] == 0


if __name__ == '__main__':
    for name, f in sorted(list(globals().items())):
        if name.startswith('test_'): f(); print('PASS', name)
