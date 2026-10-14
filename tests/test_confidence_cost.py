import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standin_eval.confidence_cost import joint_weights, frozen_costs
from standin_eval.dual_label_search import LEFT_RIGHT
from scripts.experiment_confidence_cost import BASE, STRENGTHS, GROUPS, select_policy


def test_positive_floor_and_equal_scores_recover_uniform():
    assert np.allclose(joint_weights(np.array([.1,.3,.5,.7,1.]+[.5]*12))[:5], [.25,.25,.625,1,1])
    rng = np.random.default_rng(10); f = rng.normal(size=(3,17,2)); q = rng.normal(size=(17,2))
    mask = np.ones(17, bool); mask[10] = False
    selected = np.flatnonzero(mask[5:])+5
    reference = np.linalg.norm(f[:,selected]-q[selected],axis=-1).mean(axis=-1)
    uniform, weighted = frozen_costs(f, q, mask, np.full(17,.5), reference)
    assert np.allclose(uniform, weighted)


def test_label_permutation_carries_scores_and_mask():
    rng = np.random.default_rng(12); f = rng.normal(size=(5,17,2)); q = rng.normal(size=(17,2))
    mask = np.ones(17, bool); mask[[7,10]] = False; scores = rng.random(17)
    selected = np.flatnonzero(mask[5:])+5
    reference = np.linalg.norm(f[:,selected]-q[selected],axis=-1).mean(axis=-1)
    a = frozen_costs(f,q,mask,scores,reference)
    b = frozen_costs(f,q[LEFT_RIGHT],mask[LEFT_RIGHT],scores[LEFT_RIGHT],reference)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    for bad in (np.full(17,np.nan), np.ones(16)):
        try: joint_weights(bad)
        except ValueError: pass
        else: raise AssertionError('invalid scores accepted')


def test_freezes_original_label_even_when_other_label_is_better():
    q = np.arange(34,dtype=float).reshape(17,2)
    f = q[LEFT_RIGHT][None]; mask = np.ones(17,bool)
    reference = np.linalg.norm(f[:,5:]-q[5:],axis=-1).mean(axis=-1)
    uniform, weighted = frozen_costs(f,q,mask,np.ones(17),reference)
    assert reference[0] > 0 and np.array_equal(uniform,reference) and np.array_equal(weighted,reference)


def test_selection_requires_three_gains_and_no_group_regression():
    c = {a:{g:{'top1':40,'top5':80} for g in GROUPS} for a in (BASE,*STRENGTHS)}
    c['confidence025']['strong_perspective']['top5'] = 82
    assert select_policy(c) == []
    c['confidence025']['strong_perspective']['top5'] = 83
    assert select_policy(c) == ['confidence025']
    c['confidence025']['orthographic_control']['top5'] = 79
    assert select_policy(c) == []


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'): fn(); print('PASS', name)
