import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.confidence_camera_fit import feature_cost, objective, fit_confidence
from standin_eval.confidence_cost import joint_weights
from standin_eval.camera_refinement import normalized_joints, refine_costs
from standin_eval.dual_label_search import LEFT_RIGHT, alternate_labels
from standin_eval.perspective_search import Camera,project,normalize_batch,build_camera_index
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose
from scripts.experiment_confidence_camera import gate,ARMS,BASE,CONTROL,JOINT,RERANK,GROUPS


def test_camera_objective_warm_start_and_label_consistency():
    poses=[Pose3D(n,n,_synthetic_pose(n),np.ones(17)) for n in ('running','standing')]
    cams=[Camera(y,p,d) for y in (0,60,90) for p in (0,30) for d in (None,2.,4.)]
    points=normalized_joints(np.stack([p.joints for p in poses])); index=build_camera_index(poses,cams)
    q=normalize_batch(project(poses[0].joints,Camera(75,20,3.2))[0]);q[9]+=.5
    mask=np.ones(17,bool);mask[10]=False;scores=np.linspace(.3,.9,17)
    body,warm=refine_costs(points,index,cams,q,mask)
    alt,am=alternate_labels(q,mask)
    for weighted in (False,True):
        w=joint_weights(scores) if weighted else np.ones(17)
        old=np.minimum(objective(points,warm,q,mask,w),objective(points,warm,alt,am,w[LEFT_RIGHT]))
        cost,params,labels=fit_confidence(points,index,cams,q,mask,warm,scores,weighted=weighted)
        assert np.all(cost<=old+1e-10)
        dc=objective(points,params,q,mask,w);ac=objective(points,params,alt,am,w[LEFT_RIGHT])
        assert np.array_equal(cost,np.where(labels,ac,dc))
        c2,p2,l2=fit_confidence(points,index,cams,alt,am,warm,scores[LEFT_RIGHT],weighted=weighted)
        assert np.array_equal(cost,c2) and np.array_equal(params,p2)


def test_invalid_projection_and_weights_rejected():
    f=np.zeros((2,17,2));q=np.ones((17,2));mask=np.ones(17,bool)
    costs=feature_cost(f,np.array([True,False]),q,mask,np.ones(17))
    assert np.isclose(costs[0],2**.5) and np.isinf(costs[1])
    for w in (np.zeros(17),np.full(17,np.nan),np.ones(16)):
        try:feature_cost(f,np.ones(2,bool),q,mask,w)
        except ValueError:pass
        else:raise AssertionError('invalid weights accepted')


def test_gate_protects_matched_control_and_frozen_rerank():
    c={a:{g:{'top1':40,'top5':80} for g in GROUPS} for a in ARMS}
    c[JOINT]['strong_perspective']['top5']=83;assert gate(c)
    c[CONTROL]['strong_perspective']['top5']=81;assert not gate(c)
    c[JOINT]['strong_perspective']['top5']=84;assert gate(c)
    c[RERANK]['orthographic_control']['top5']=81;assert not gate(c)


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
