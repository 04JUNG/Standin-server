import numpy as np
from src.bvh import fk, coco17_from_fk
from standin_eval.bvh_orientation_diagnostic import world_transforms, relative_rotation_angle


def fixture():
    return [['Hips',-1,np.zeros(3),['Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation'],False],
            ['Head',0,np.array([0.,1.,0.]),['Yrotation','Zrotation','Xrotation'],False],
            ['Head_End',1,np.array([0.,.3,0.]),[],True]]


def test_positions_match_existing_fk_for_mixed_rotation_orders():
    joints=fixture(); frame=np.array([1.,2.,3.,24.,-12.,43.,17.,27.,8.])
    frames=world_transforms(joints,frame); pos=fk(joints,frame)
    for i in range(len(joints)): np.testing.assert_allclose(frames[i,:3,3],pos[i],atol=1e-12)
    for f in frames: np.testing.assert_allclose(f[:3,:3].T@f[:3,:3],np.eye(3),atol=1e-12)


def test_head_rotation_changes_orientation_but_not_coco_head_positions():
    joints=fixture(); a=np.zeros(9); b=a.copy(); b[6]=30
    wa=world_transforms(joints,a); wb=world_transforms(joints,b)
    np.testing.assert_array_equal(coco17_from_fk(joints,fk(joints,a))[0],coco17_from_fk(joints,fk(joints,b))[0])
    np.testing.assert_allclose(relative_rotation_angle(wa[1],wb[1]),30,atol=1e-10)


def test_parent_rotation_does_not_change_local_relative_angle():
    joints=fixture(); a=np.zeros(9); a[6]=40; b=a.copy(); b[3:6]=[43,22,-14]
    wa=world_transforms(joints,a); wb=world_transforms(joints,b)
    np.testing.assert_allclose(relative_rotation_angle(wa[0],wa[1]),relative_rotation_angle(wb[0],wb[1]),atol=1e-10)


def test_incomplete_or_nonfinite_channels_are_rejected():
    for frame in (np.zeros(8),np.full(9,np.nan)):
        try: world_transforms(fixture(),frame)
        except ValueError: pass
        else: raise AssertionError('invalid frame accepted')
