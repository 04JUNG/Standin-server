import numpy as np
from standin_eval.partial_similarity import similarity_cost,refine_partial
from standin_eval.perspective_search import Camera,project,build_camera_index
from standin_eval.camera_refinement import normalized_joints
from standin_eval.z_axis_search import Pose3D
from src.library import _synthetic_pose


def mask4():
    m=np.zeros(17,bool);m[[7,8,9,10]]=True;return m


def test_positive_scale_translation_invariant():
    q=np.random.default_rng(744).normal(size=(17,2));g=(q-[3,7])/4
    assert similarity_cost(g[None],q,mask4())[0]<1e-12


def test_query_hip_shoulders_and_legs_do_not_affect_selected_fit():
    q=np.random.default_rng(744).normal(size=(17,2));g=q[None].copy();m=mask4()
    expected=similarity_cost(g,q,m);q[~m]=1e6
    assert np.allclose(similarity_cost(g,q,m),expected)


def test_negative_scale_and_degenerate_queries_rejected():
    q=np.random.default_rng(744).normal(size=(17,2))
    assert np.isinf(similarity_cost((-q)[None],q,mask4())[0])
    try:similarity_cost(q[None],np.zeros((17,2)),mask4())
    except ValueError:pass
    else:raise AssertionError('degenerate input accepted')


def test_camera_fit_replay_and_missing_joint_invariance():
    p=_synthetic_pose('running');points=normalized_joints(p[None]);poses=[Pose3D('unit','unit',p,np.ones(17))]
    cams=[Camera(0),Camera(45,20,4),Camera(90,0,3)]
    index=build_camera_index(poses,cams);q=project(p,Camera(75,15,4.5))[0];m=mask4()
    costs,params,coarse,_=refine_partial(points,index,cams,q,m)
    assert costs[0]<=coarse[0]+1e-5 and np.isfinite(costs[0])
    q[11:]=1e4;changed,_,_,_=refine_partial(points,index,cams,q,m)
    assert np.allclose(costs,changed)
