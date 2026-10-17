import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.ordinal_depth import PAIRS,query_relations,ordinal_penalty,rerank


def test_front_back_and_unknown_are_distinct():
    z=np.zeros(17);z[9]=-.4
    signs,use=query_relations(z,np.ones(17,bool))
    assert signs[1]==-1 and use.sum()==1
    gallery=np.stack((z,-z,np.zeros(17)))
    assert np.allclose(ordinal_penalty(gallery,signs,use),[0,1,.5])


def test_root_translation_does_not_change_relations():
    z=np.random.default_rng(32).normal(size=17)
    a,m=query_relations(z,np.ones(17,bool));b,n=query_relations(z+5,np.ones(17,bool))
    assert np.array_equal(a,b) and np.array_equal(m,n)


def test_gating_checks_both_corresponding_joints_without_swapping():
    xy=np.zeros((17,2));xy[5:7,1]=1
    other=xy.copy();other[9,0]=1.
    z=np.zeros(17);z[9]=-.4;z[7]=-.3
    _,mask=query_relations(z,np.ones(17,bool),xy,other,.5)
    assert mask[0] and not mask[1]


def test_zero_weight_unknown_and_tail_preserve_baseline():
    costs=np.array([.2,.1,.3,.4]);penalty=np.array([0,1,0,0])
    baseline=np.argsort(costs,kind='stable')
    assert np.array_equal(rerank(costs,penalty,0),baseline)
    assert np.array_equal(rerank(costs,np.zeros(4),.25),baseline)
    assert np.array_equal(rerank(costs,penalty,.25,top_n=2),[0,1,2,3])


if __name__=='__main__':
    for name,f in sorted(list(globals().items())):
        if name.startswith('test_'):f();print('PASS',name)
