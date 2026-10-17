import sys
import copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.camera_regularization import optimize_one,regularized_costs,penalty
from standin_eval.camera_refinement import normalized_joints,refine_costs,parameter_cost
from standin_eval.perspective_search import Camera,build_camera_index,project,normalize_batch
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose
from scripts.experiment_camera_regularization import select_policy,BASE,WEIGHTS,GROUPS,order


def fixture():
    poses=[Pose3D(n,n,_synthetic_pose(n),np.ones(17)) for n in ('running','standing')]
    cams=[Camera(y,p,d) for y in (0,60,90) for p in (0,30) for d in (None,2.,4.)]
    points=normalized_joints(np.stack([p.joints for p in poses]));index=build_camera_index(poses,cams)
    q=normalize_batch(project(poses[0].joints,Camera(75,20,3.2))[0])
    return points,index,cams,q,np.ones(17,bool)


def test_zero_weight_reproduces_original_costs():
    args=fixture();a,_=refine_costs(*args);b,_=regularized_costs(*args,0)
    assert np.allclose(a,b,atol=1e-6)


def test_cost_and_returned_camera_consistent_and_objective_decreases():
    points,index,cams,q,mask=fixture()
    for w in WEIGHTS.values():
        before,_=optimize_one(points,index,cams,q,mask,w,steps=())
        cost,params=optimize_one(points,index,cams,q,mask,w)
        assert np.all(cost<=before+1e-10)
        assert np.allclose(cost,parameter_cost(points,params,q,mask)+penalty(params,w))


def test_masked_xy_does_not_affect_either_label():
    points,index,cams,q,mask=fixture();mask[9]=False
    a,_=regularized_costs(points,index,cams,q,mask,.02)
    q[9]=1e6;b,_=regularized_costs(points,index,cams,q,mask,.02)
    assert np.array_equal(a,b)


def test_protection_and_energy_gate_and_first_result():
    counts={a:{g:{'top1':40,'top5':80} for g in GROUPS} for a in (BASE,*WEIGHTS)}
    energy={a:80. for a in (BASE,*WEIGHTS)};energy[BASE]=100
    assert select_policy(counts,energy)==list(WEIGHTS)
    bad=copy.deepcopy(counts)
    for a in WEIGHTS:bad[a]['strong_perspective']['top5']-=1
    assert not select_policy(bad,energy)
    assert not select_policy(counts,{a:100. for a in energy})
    assert order(np.array([.1,.2,.3]),np.array([.9,.2,.1]))[0]==0


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
