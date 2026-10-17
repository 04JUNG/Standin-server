import numpy as np
from standin_eval.pose_requirement_coverage import requirement_metrics


def folded_crouch():
    p=np.zeros((17,3));p[5]=[.3,1,0];p[6]=[-.3,1,0];p[11]=[.2,0,0];p[12]=[-.2,0,0]
    p[7]=[.4,.6,-.15];p[9]=[-.2,.6,-.15];p[8]=[-.4,.8,-.3];p[10]=[.2,.8,-.3]
    p[13]=[.7,0,0];p[15]=[.7,-.5,0];p[14]=[-.7,0,0];p[16]=[-.7,-.5,0]
    return p


def test_joint_requirements_need_both_features():
    p=folded_crouch();r=requirement_metrics(p)
    assert r['joint_requirements_geometry'] and np.allclose(r['knee_angles_degrees'],[90,90])
    p[9]=[.8,.6,0]
    assert requirement_metrics(p)['deep_crouch_geometry'] and not requirement_metrics(p)['joint_requirements_geometry']


def test_straight_legs_fail_crouch_even_when_arms_match():
    p=folded_crouch();p[13]=[.2,-.5,0];p[15]=[.2,-1,0];p[14]=[-.2,-.5,0];p[16]=[-.2,-1,0]
    r=requirement_metrics(p)
    assert r['bilateral_arm_proximity_geometry'] and not r['deep_crouch_geometry']


def test_translation_scale_and_world_yaw_invariance():
    p=folded_crouch();yaw=.7;r=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    original=requirement_metrics(p);changed=requirement_metrics((p@r.T)*3+[2,5,-8])
    assert original['joint_requirements_geometry']==changed['joint_requirements_geometry']
    assert np.allclose(original['knee_angles_degrees'],changed['knee_angles_degrees'])


def test_nonfinite_or_degenerate_geometry_rejected():
    for p in (np.zeros((17,3)),np.full((17,3),np.nan)):
        try:requirement_metrics(p)
        except ValueError:pass
        else:raise AssertionError('invalid geometry accepted')
