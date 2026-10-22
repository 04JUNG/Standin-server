"""Whole-rig guards must reject attractive 2D fits with unsafe 3D changes."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from converter.pose_safety import audit_pose, descendants, rotate_subtree


def rig():
    points={'Hips':[0,0,1.], 'Spine':[0,0,1.2], 'Neck':[0,0,1.6], 'Head':[0,0,1.7]}
    parents={'Hips':None,'Spine':'Hips','Neck':'Spine','Head':'Neck'}
    for side,sign in [('Left',1),('Right',-1)]:
        for name,xyz,parent in [('Arm',[sign*.2,0,1.5],'Spine'),('ForeArm',[sign*.35,.12,1.25],side+'Arm'),
            ('Hand',[sign*.4,.23,1.04],side+'ForeArm'),('UpLeg',[sign*.1,0,1.],'Hips'),
            ('Leg',[sign*.12,.08,.55],side+'UpLeg'),('Foot',[sign*.12,-.02,.1],side+'Leg'),
            ('HandIndex1',[sign*.4,.25,1.],side+'Hand')]:
            points[side+name]=xyz;parents[side+name]=parent
    names=list(points);bones=np.repeat(np.eye(4)[None],len(names),axis=0)
    bones[:,:3,3]=np.array(list(points.values()))
    camera={'rotation':np.eye(3).tolist(),'distance_torso':8.}
    return bones,names,parents,camera


def audit(base,new,names,parents,camera):
    return audit_pose(base,new,names,parents,camera)


def test_identity_checks_arms_legs_and_protected_body():
    b,n,p,c=rig();r=audit(b,b,n,p,c)
    assert r['status']=='pose_validated'
    assert len(r['chains'])==4 and len(r['joints'])==len(n)
    assert r['observed_depth_ground_truth'] is False
    assert r['anatomical_certificate'] is False


@pytest.mark.parametrize('joint',['Spine','Neck','Head','LeftHand','RightHandIndex1'])
def test_protected_parts_cannot_rotate_in_body_correction(joint):
    b,n,p,c=rig();i=n.index(joint)
    out=rotate_subtree(b,descendants(n,p,joint),i,Rotation.from_euler('x',2,degrees=True).as_matrix())
    assert 'angular_bound:'+joint in audit(b,out,n,p,c)['violations']


@pytest.mark.parametrize('joint,degree',[('LeftArm',13),('RightForeArm',13),('LeftUpLeg',9),('RightLeg',9),('LeftFoot',13)])
def test_cumulative_angular_limits(joint,degree):
    b,n,p,c=rig();i=n.index(joint)
    out=rotate_subtree(b,descendants(n,p,joint),i,Rotation.from_euler('x',degree,degrees=True).as_matrix())
    assert 'angular_bound:'+joint in audit(b,out,n,p,c)['violations']


def test_axial_twist_rejected_even_inside_swing_bound():
    b,n,p,c=rig();i=n.index('LeftArm');axis=b[n.index('LeftForeArm'),:3,3]-b[i,:3,3];axis/=np.linalg.norm(axis)
    out=rotate_subtree(b,descendants(n,p,'LeftArm'),i,Rotation.from_rotvec(axis*np.deg2rad(6)).as_matrix())
    assert 'axial_twist:LeftArm' in audit(b,out,n,p,c)['violations']


def test_camera_depth_flip_rejected_despite_identical_front_projection():
    b,n,p,c=rig();out=b.copy()
    out[n.index('LeftHand'),1,3]=-.23
    r=audit(b,out,n,p,c)
    assert any(v.startswith('depth_order:') for v in r['violations'])
    assert 'camera_depth_trust_bound' in r['violations']


def test_bone_stretch_and_nonfinite_are_not_angle_corrections():
    b,n,p,c=rig();out=b.copy();out[n.index('RightLeg'),0,3]+=.02
    assert 'bone_length_or_stretch:RightLeg' in audit(b,out,n,p,c)['violations']
    out=b.copy();out[0,0,0]=np.nan
    with pytest.raises(ValueError,match='invalid_pose_frames'):audit(b,out,n,p,c)


def test_small_swing_preserves_lengths_and_depth_prior():
    b,n,p,c=rig();i=n.index('LeftArm')
    out=rotate_subtree(b,descendants(n,p,'LeftArm'),i,Rotation.from_euler('x',1,degrees=True).as_matrix())
    r=audit(b,out,n,p,c)
    assert r['status']=='pose_validated'
    assert all(d['preserved'] for d in r['depth_pairs'])


def test_nonfinite_depth_reference_cannot_disable_depth_guards():
    b,n,p,c=rig();reference=b.copy();reference[0,0,0]=np.nan
    with pytest.raises(ValueError,match='invalid_camera_pose_frames'):
        audit_pose(b,b,n,p,c,depth_reference=reference)


def test_cyclic_rig_is_not_a_valid_relative_angle_reference():
    b,n,p,c=rig();p['Spine']='Neck'
    with pytest.raises(ValueError,match='invalid_pose_hierarchy'):audit(b,b,n,p,c)
