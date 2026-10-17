import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.consistency_cost import consistency_weights,fixed_weighted_cost
from standin_eval.dual_label_search import LEFT_RIGHT
from scripts.experiment_consistency_cost import ARMS,BASE,CONFIDENCE,CONSISTENCY,COMBINED,GROUPS,select_policy


def test_half_weight_boundary_and_unknown_fallback():
    signal={'joints':[{'index':j,'support':0,'disagreement':None} for j in range(5,17)]}
    signal['joints'][0].update(support=2,disagreement=.25)
    signal['joints'][1].update(support=3,disagreement=.249)
    w=consistency_weights(signal)
    assert w[5]==.5 and np.all(w[6:]==1)
    signal['joints'][2].update(support=1,disagreement=0.)
    try:consistency_weights(signal)
    except ValueError:pass
    else:raise AssertionError('unknown falsely accepted as stable')


def test_fixed_label_weight_permutation_and_positive_weights():
    rng=np.random.default_rng(1);f=rng.normal(size=(3,17,2));q=rng.normal(size=(17,2))
    mask=np.ones(17,bool);mask[9]=False;w=rng.uniform(.125,1,17)
    ix=np.flatnonzero(mask[5:])+5;reference=np.linalg.norm(f[:,ix]-q[ix],axis=-1).mean(axis=-1)
    a,_=fixed_weighted_cost(f,q,mask,w,reference)
    b,_=fixed_weighted_cost(f,q[LEFT_RIGHT],mask[LEFT_RIGHT],w[LEFT_RIGHT],reference)
    assert np.array_equal(a,b)
    try:fixed_weighted_cost(f,q,mask,np.zeros(17),reference)
    except ValueError:pass
    else:raise AssertionError('zero weights accepted')


def test_gate_requires_beyond_confidence_and_original():
    c={a:{g:{'top1':40,'top5':80} for g in GROUPS} for a in ARMS}
    c[CONSISTENCY]['strong_perspective']['top5']=83;assert select_policy(c)==[CONSISTENCY]
    c[CONFIDENCE]['strong_perspective']['top5']=83;assert select_policy(c)==[]
    c[COMBINED]['strong_perspective']['top5']=84;assert select_policy(c)==[COMBINED]
    c[CONFIDENCE]['orthographic_control']['top5']=81;assert select_policy(c)==[]


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
