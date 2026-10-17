import sys
import copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experiment_refinement_blend import blend_cost,select_policy,POLICIES,BASELINES,GROUPS


def test_blending_limits_and_monotonic_interpolation():
    a=np.array([2.,4.,6.]);b=np.array([1.,0.,3.])
    assert np.array_equal(blend_cost(a,b,0),a)
    assert np.array_equal(blend_cost(a,b,1),b)
    assert np.allclose(blend_cost(a,b,.25),[1.75,3.,5.25])
    try:blend_cost(a,b,1.1)
    except ValueError:pass
    else:raise AssertionError('invalid alpha accepted')


def test_selection_protects_baselines_and_prefers_lower_alpha_on_tie():
    c={a:{g:{'top1':20,'top5':60} for g in GROUPS} for a in (*BASELINES,*POLICIES)}
    for a in POLICIES:c[a]['strong_perspective']['top5']=66
    assert select_policy(c)[0]=='keepfirst_blend25'
    for a in POLICIES:c[a]['orthographic_control']['top5']=59
    assert select_policy(c)==[]


def test_canonical_arithmetic_preserves_exact_tie_order():
    base=np.array([.11933832809423049,.11933832809423049])
    fitted=np.array([.11807805924171468,.11807805924171466])
    canonical=blend_cost(base,fitted,.25)
    assert canonical[0]==canonical[1]
    assert np.argsort(canonical,kind='stable').tolist()==[0,1]
    alternative=base+.25*(fitted-base)
    assert np.allclose(canonical,alternative,rtol=0,atol=1e-15)


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn();print('PASS',name)
