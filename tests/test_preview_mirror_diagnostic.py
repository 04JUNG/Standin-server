import numpy as np
from standin_eval.preview_mirror_diagnostic import SWAP, pair_metrics, torso_front


def fixture():
    rng = np.random.RandomState(20261014)
    p = rng.uniform(-.7, .7, (17, 3))
    p[11] = [.2, 0, 0]; p[12] = [-.2, 0, 0]
    p[5] = [.4, 1, 0]; p[6] = [-.4, 1, 0]
    return p


def test_reflection_with_relabel_keeps_front_as_polar_vector():
    p = fixture(); sign = np.array([-1, 1, 1])
    q = p[SWAP] * sign
    assert pair_metrics(p, q)['classification'] == 'reflection_with_lr_swap'
    np.testing.assert_allclose(torso_front(q), torso_front(p) * sign)


def test_reflection_without_relabel_flips_front():
    p = fixture(); sign = np.array([-1, 1, 1])
    q = p * sign
    assert pair_metrics(p, q)['classification'] == 'reflection_without_lr_swap'
    np.testing.assert_allclose(torso_front(q), -torso_front(p) * sign)


def test_translation_does_not_change_pair_match():
    p = fixture(); q = p[SWAP] * [-1, 1, 1]
    a, b = pair_metrics(p, q), pair_metrics(p + [3, 0, 0], q + [0, 4, 0])
    assert a['classification'] == b['classification']
    for x, y in zip(a['hypotheses'], b['hypotheses']):
        assert x['matches'] == y['matches']
        np.testing.assert_allclose(x['max_joint_error_torso_units'], y['max_joint_error_torso_units'], atol=1e-14)
        np.testing.assert_allclose(x['reflected_front_dot_preview_front'], y['reflected_front_dot_preview_front'], atol=1e-14)


def test_unrelated_pose_is_not_classified_as_mirror():
    p = fixture(); q = p.copy(); q[9] += [.2, .3, .4]
    assert pair_metrics(p, q)['classification'] == 'not_axis_reflection'
