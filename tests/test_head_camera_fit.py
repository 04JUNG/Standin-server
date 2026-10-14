import sys
import copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.head_camera_fit import objective,fit_joint
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.head_hint import head_hint
from standin_eval.perspective_search import Camera,project,normalize_batch,build_camera_index
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose
from scripts.experiment_head_camera_fit import gate,ARMS,BASE,CONTROL,JOINT,GROUPS


def test_equivalent_weight_and_warm_start_protection():
    poses=[Pose3D(n,n,_synthetic_pose(n),np.ones(17)) for n in ('running','standing')]
    cams=[Camera(y,p,d) for y in (0,60,90) for p in (0,30) for d in (None,2.,4.)]
    points=normalized_joints(np.stack([p.joints for p in poses]));index=build_camera_index(poses,cams)
    q=normalize_batch(project(poses[0].joints,Camera(75,20,3.2))[0]);q[0]+=.5;mask=np.ones(17,bool)
    body,warm=refine_costs(points,index,cams,q,mask);_,head=head_hint(points,warm,q,.9)
    for w in (0.,.1):
        cost,params=fit_joint(points,index,cams,q,mask,warm,w)
        assert np.all(cost<=body+(w/.75)*head+1e-6)
        d=objective(points,params,q,mask,w)
        assert np.all(cost<=d+1e-8)
    direct_body=objective(points,warm,q,mask,0)
    direct_joint=objective(points,warm,q,mask,.1)
    assert np.allclose(.75*direct_joint,.75*direct_body+.1*head)


def test_gate_requires_improvement_over_matched_budget_control():
    c={a:{g:{'top1':40,'top5':80} for g in GROUPS} for a in ARMS}
    c[JOINT]['strong_perspective']['top5']=83
    assert gate(c)
    c[CONTROL]['strong_perspective']['top5']=81
    assert not gate(c)
    c[JOINT]['strong_perspective']['top5']=84
    assert gate(c)
    c[JOINT]['orthographic_control']['top5']=79
    assert not gate(c)


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
