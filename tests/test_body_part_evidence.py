import numpy as np
from standin_eval.body_part_evidence import body_part_errors
from standin_eval.dual_label_search import LEFT_RIGHT


def pose():
    return np.random.default_rng(734).normal(size=(17,2))


def test_identity_and_single_global_swap():
    q=pose();m=np.ones(17,bool)
    for c in (q,q[LEFT_RIGHT]):
        got=body_part_errors(q,m,c)
        assert all(got[k]==0 for k in ('body','arms','legs','torso'))
    assert body_part_errors(q,m,q[LEFT_RIGHT])['global_swap']


def test_masked_errors_do_not_affect_region_cost():
    q=pose();c=q.copy();c[9]+=100;m=np.ones(17,bool);m[9]=False
    assert body_part_errors(q,m,c)['arms']==0
    assert body_part_errors(q,m,c)['valid_counts']['arms']==5


def test_region_tradeoff_is_not_hidden_by_mean():
    q=pose();c=q.copy();c[13]+=[.1,0];c[15]+=[.1,0]
    r=body_part_errors(q,np.ones(17,bool),c)
    assert r['legs']>r['body']>r['arms'] and r['arms']==0


def test_too_few_observations_are_not_scored():
    m=np.zeros(17,bool);m[5:10]=True
    try:body_part_errors(pose(),m,pose())
    except ValueError:pass
    else:raise AssertionError('invalid query scored')
