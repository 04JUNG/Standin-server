import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.joint_shape import elbow_angles,leg_crossing,shape_penalties,reorder_shortlist,segment_relation
from standin_eval.joint_ties import reorder_near_ties
from standin_eval.dual_label_search import alternate_labels


def pose(crossed=False):
    q=np.zeros((17,2));q[5]=[-.5,-1];q[7]=[-.5,0];q[9]=[.5,0]
    q[6]=[.5,-1];q[8]=[.5,0];q[10]=[1.5,0]
    q[11]=[-.3,0];q[12]=[.3,0];q[13]=[-.4,1];q[14]=[.4,1]
    q[15]=[.4 if crossed else -.5,2];q[16]=[-.4 if crossed else .5,2]
    return q


def test_right_angle_and_masked_wrist():
    q=pose();mask=np.ones(17,bool)
    assert np.allclose(elbow_angles(q,mask),[90,90])
    mask[9]=False
    assert elbow_angles(q,mask)[0] is None
    q[9]=[1e4,-1e4];assert elbow_angles(q,mask)[0] is None


def test_crossing_rotation_invariant_and_contact_uncertain():
    mask=np.ones(17,bool);q=pose(True)
    assert leg_crossing(q,mask) is True and leg_crossing(pose(),mask) is False
    rotation=np.array([[0,-1],[1,0]])
    assert leg_crossing(q@rotation.T+[13,-8],mask) is True
    assert segment_relation(np.array([0.,0]),np.array([1.,0]),np.array([1.,0]),np.array([1.,1])) is None
    mask[15]=False;assert leg_crossing(q,mask) is None


def test_same_pose_zero_and_global_side_swap():
    q=pose(True);mask=np.ones(17,bool);swapped,_=alternate_labels(q,mask)
    a,c,meta=shape_penalties(q,mask,np.stack([q,swapped,pose()]))
    assert np.allclose(a[:2],0) and np.allclose(c,[0,0,1])
    assert meta['swapped_candidates'][1]


def test_missing_query_features_are_neutral():
    q=pose();mask=np.ones(17,bool);mask[[9,10,15]]=False
    a,c,meta=shape_penalties(q,mask,np.stack([q,pose(True)]))
    assert np.allclose(a,0) and np.allclose(c,0) and meta['used_arms']==0 and not meta['used_crossing']


def test_first_can_move_tail_and_ties_preserved():
    order=np.arange(25);cost=np.arange(25)*.01;penalty=np.zeros(20);penalty[0]=1
    new=reorder_shortlist(order,cost,penalty,.15)
    assert new[0]==1 and np.array_equal(new[20:],order[20:]) and set(new[:20])==set(order[:20])
    assert np.array_equal(reorder_shortlist(order,np.zeros(25),np.zeros(20),1),order)


def test_bad_shape_and_nonfinite_are_rejected():
    q=pose();q[7,0]=np.nan
    try:shape_penalties(q,np.ones(17,bool),np.stack([pose()]))
    except ValueError:pass
    else:raise AssertionError('nonfinite observations accepted')


def test_near_ties_protect_top5_and_exclude_distant_feature_match():
    order=np.arange(25);costs=np.arange(25)*.01;shape=np.zeros(25);shape[0]=.5;shape[1]=.4
    updated=reorder_near_ties(order,costs,shape)
    assert updated[:5].tolist()==[2,1,0,3,4]
    assert set(updated[:5])==set(order[:5]) and np.array_equal(updated[5:],order[5:])


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
