import numpy as np
from standin_eval.joint_part_guard import guarded_first


def parts(a,l,t):return {'arms':a,'legs':l,'torso':t}


def test_a_better_feature_cannot_sacrifice_legs():
    order=np.arange(20);cost=np.zeros(20);shape=np.zeros(20);shape[0]=.3
    regions=[parts(.2,.2,.2)]+[parts(.1,.3,.1) for _ in range(4)]
    new,meta=guarded_first(order,cost,shape,regions)
    assert new[0]==0 and meta['eligible_top5_positions']==[0]


def test_dominating_candidate_can_move_and_top5_tail_stay_fixed():
    order=np.arange(20);cost=np.arange(20)*.01;shape=np.zeros(20);shape[0]=.3
    regions=[parts(.2,.2,.2),parts(.1,.1,.1)]+[parts(.3,.3,.3) for _ in range(3)]
    new,meta=guarded_first(order,cost,shape,regions)
    assert new[:5].tolist()==[1,0,2,3,4] and np.array_equal(new[5:],order[5:])
    assert meta['winner_previous_position']==1


def test_ties_keep_original_first():
    order=np.arange(20);regions=[parts(.2,.2,.2)]*5
    new,_=guarded_first(order,np.zeros(20),np.zeros(20),regions)
    assert np.array_equal(new,order)


def test_incomplete_region_is_rejected():
    regions=[parts(.2,.2,.2)]*4+[parts(None,.1,.1)]
    try:guarded_first(np.arange(20),np.zeros(20),np.zeros(20),regions)
    except ValueError:pass
    else:raise AssertionError('missing evidence scored')
