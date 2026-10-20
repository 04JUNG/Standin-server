import numpy as np
from standin_eval.manual_pose_cues import angle,describe,satisfies
from standin_eval.perspective_search import Camera,project
from standin_eval.folded_arm_contact import folded_arm_contact


def scaffold():
    p=np.zeros((17,3));p[5]=[.3,1,0];p[6]=[-.3,1,0];p[11]=[.2,0,0];p[12]=[-.2,0,0]
    p[7]=[.4,.6,-.15];p[9]=[-.2,.6,-.15];p[8]=[-.4,.8,-.3];p[10]=[.2,.8,-.3]
    p[13]=[.2,-.7,0];p[14]=[-.2,-.7,0];p[15]=[.2,-1.4,0];p[16]=[-.2,-1.4,0]
    return p


def test_real_3d_angle_not_projected_angle():
    assert np.isclose(angle(np.array([1,0,0]),np.zeros(3),np.array([0,1,0])),90)
    assert np.isclose(angle(np.array([1,0,1]),np.zeros(3),np.array([0,1,1])),60)
    assert angle(np.zeros(3),np.zeros(3),np.ones(3)) is None


def test_folded_arms_and_side_specific_cues():
    p=scaffold();xy,_=project(p,Camera(0));f=describe(p,Camera(0),xy)
    assert f['folded_arms']
    assert satisfies(f,{'folded_arms':True})[0]
    p[9]=[.4,.2,-.15];f=describe(p,Camera(0),xy)
    assert not f['folded_arms']
    assert satisfies(f,{'elbow_degrees':{'left':[155,180]}})[0]
    assert not satisfies(f,{'elbow_degrees':{'right':[155,180]}})[0]


def test_depth_order_changes_with_view_not_anatomical_labels():
    p=scaffold();xy,_=project(p,Camera(0))
    front=describe(p,Camera(0),xy);back=describe(p,Camera(180),xy)
    assert np.isclose(front['left_arm_nearer_than_right'],-back['left_arm_nearer_than_right'])
    assert np.allclose(front['elbows_3d_degrees'],back['elbows_3d_degrees'])


def test_empty_cues_are_neutral():
    # No candidate ID or answer enters the cue predicate.
    f=describe(scaffold(),Camera(0),project(scaffold(),Camera(0))[0])
    assert satisfies(f,{})==(True,[])


def test_crossed_hands_are_not_enough_without_arm_proximity():
    p=scaffold();assert folded_arm_contact(p)[0]
    p[[9,10],2]-=1
    assert not folded_arm_contact(p)[0]


def test_bilateral_arm_proximity_is_required():
    p=scaffold();p[10]=[1.5,.5,-.1]
    assert not folded_arm_contact(p)[0]
