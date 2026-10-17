import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.contour_rerank import anchor_affine,prepare_query,contour_cost,distance_field,rerank,candidate_evidence
from scripts.experiment_z_rendered import render_pose
from src.library import _synthetic_pose


def test_anchor_alignment_preserves_hip_and_torso_without_rotating():
    xy=np.column_stack((np.arange(17),np.arange(17)**2)).astype(float)
    target=xy*2.3+[8,-7];m=anchor_affine(xy,target)
    assert np.allclose(m,[[2.3,0,8],[0,2.3,-7]])
    assert np.allclose(xy@m[:,:2].T+m[:,2],target)


def test_symmetric_cost_matches_bruteforce_and_caps_outliers():
    a=np.zeros((20,20),bool);a[2,2:14]=True
    b=np.zeros_like(a);b[6,3:15]=True
    query={'mask':a,'distance':distance_field(a),'torso':5}
    ab=np.argwhere(a);bb=np.argwhere(b)
    distances=np.linalg.norm(ab[:,None]-bb[None],axis=2)/5
    expected=(np.minimum(distances.min(0),.5).mean()+np.minimum(distances.min(1),.5).mean())/2
    assert abs(contour_cost(query,b)-expected)<1e-7
    assert contour_cost(query,a)==0 and contour_cost(query,b)<=.5


def test_identical_render_scores_zero_at_fixed_camera():
    joints=_synthetic_pose('running');image,xy=render_pose(joints,37,size=256,style='rough',seed=0,pitch=-22,distance=5)
    query=prepare_query(np.asarray(image),xy)
    cost,affine,mask=candidate_evidence(query,joints,[37,-22,.2])
    assert cost==0 and np.array_equal(mask,query['mask'])
    assert np.allclose(affine,[[1,0,0],[0,1,0]])


def test_zero_weight_first_tail_and_stable_ties():
    order=np.arange(25);scores=np.arange(25)*.01;p=np.zeros(20);p[1:5]=.5
    assert np.array_equal(rerank(order,scores,p,0),order)
    ranked=rerank(order,scores,p,1)
    assert ranked[0]==0 and np.array_equal(ranked[20:],order[20:]) and ranked[1]==5
    assert set(ranked[:20])==set(order[:20])
    assert np.array_equal(rerank(order,np.zeros(25),np.zeros(20),1),order)


def test_empty_raster_and_distortion_rejected():
    try:distance_field(np.zeros((20,20),bool))
    except ValueError:pass
    else:raise AssertionError('empty boundary accepted')
    try:prepare_query(np.zeros((20,30,3),np.uint8),np.zeros((17,2)))
    except ValueError:pass
    else:raise AssertionError('aspect distortion accepted')


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
