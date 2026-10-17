import sys
import copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.head_hint import penalty_from_errors,head_hint
from standin_eval.camera_refinement import normalized_joints
from standin_eval.perspective_search import Camera,project,normalize_batch
from src.library import _synthetic_pose
from scripts.experiment_head_hint import select_policy,BASE,WEIGHTS,GROUPS,order


def test_confidence_deadzone_and_cap():
    e=np.array([0,.25,.5,2.])
    assert np.array_equal(penalty_from_errors(e,.7),[0,0,.25,1])
    assert np.array_equal(penalty_from_errors(e,.699),np.zeros(4))
    assert np.array_equal(penalty_from_errors(e,float('nan')),np.zeros(4))


def test_matching_projection_has_no_penalty_or_query_mutation():
    joints=_synthetic_pose('running');cam=Camera(75,20,3.2)
    q=normalize_batch(project(joints,cam)[0]);before=q.copy()
    e,p=head_hint(normalized_joints(joints[None]),np.array([[75,20,1/3.2]]),q,.9)
    assert e[0]<1e-5 and p[0]==0 and np.array_equal(q,before)


def test_selection_requires_gain_and_control_protection():
    c={a:{g:{'top1':40,'top5':80} for g in GROUPS} for a in (BASE,*WEIGHTS)}
    assert select_policy(c)==[]
    for a in WEIGHTS:c[a]['strong_perspective']['top5']=83
    assert select_policy(c)==list(WEIGHTS)
    bad=copy.deepcopy(c)
    for a in WEIGHTS:bad[a]['orthographic_control']['top5']-=1
    assert select_policy(bad)==[]
    assert order(np.array([.1,.2,.3]),np.array([.9,.3,.1]))[0]==0


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):fn();print('PASS',name)
