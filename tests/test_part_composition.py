"""Offline composition tests; no runtime routing change or actual mesh claim."""
from itertools import permutations

import numpy as np
import pytest

pytest.importorskip("scipy")  # optional dependency of the offline probe only
from src.bvh import _rot, channel_starts, coco17_from_fk, fk, parse_bvh, single_frame_bvh_text
from src.experimental.part_composition import (
    canonical, compose_arms, shortest_arc, world_transforms, _set_rotation, torso_frame,
)


def rig(scale=1., prefix="", order="ZXY"):
    points={"Hips":(0,0,0),"Spine":(0,.5,0),"LeftArm":(-.4,1,0),"LeftForeArm":(-.45,.6,.4),
        "LeftHand":(.2,.7,.4),"RightArm":(.4,1,0),"RightForeArm":(.45,.55,.3),"RightHand":(-.2,.65,.3),
        "Head":(0,1.3,0),"LeftUpLeg":(-.3,0,0),"LeftLeg":(-.3,0,.7),"LeftFoot":(-.3,-.7,.7),
        "RightUpLeg":(.3,0,0),"RightLeg":(.3,0,.7),"RightFoot":(.3,-.7,.7)}
    parent={"Hips":None,"Spine":"Hips","Head":"Spine", "LeftArm":"Spine","RightArm":"Spine",
        "LeftForeArm":"LeftArm","LeftHand":"LeftForeArm","RightForeArm":"RightArm","RightHand":"RightForeArm",
        "LeftUpLeg":"Hips","LeftLeg":"LeftUpLeg","LeftFoot":"LeftLeg",
        "RightUpLeg":"Hips","RightLeg":"RightUpLeg","RightFoot":"RightLeg"}
    joints=[];by_name={}
    for name,point in points.items():
        p=parent[name];pos=np.asarray(point,float)*scale
        offset=pos if p is None else pos-np.asarray(points[p])*scale
        channels=([a+'position' for a in 'XYZ'] if p is None else [])+[a+'rotation' for a in order]
        by_name[name]=len(joints);joints.append([prefix+name,-1 if p is None else by_name[p],offset,channels,False])
    return joints,np.zeros(sum(len(j[3]) for j in joints))


@pytest.mark.parametrize("order",[''.join(p) for p in permutations('XYZ')])
def test_all_euler_orders_round_trip(order):
    j,f=rig(order=order);mat=_rot('Z',33)@_rot('X',-21)@_rot('Y',52)
    _set_rotation(j,f,0,mat)
    np.testing.assert_allclose(world_transforms(j,f)[0][:3,:3],mat,atol=1e-12)


@pytest.mark.parametrize("a,b",[((1,0,0),(0,1,0)),((1,0,0),(-1,0,0)),((0,1,0),(0,1,0))])
def test_shortest_arc_is_proper_rotation(a,b):
    mat=shortest_arc(a,b)
    np.testing.assert_allclose(mat@a,b,atol=1e-12)
    np.testing.assert_allclose(mat.T@mat,np.eye(3),atol=1e-12)
    assert np.linalg.det(mat)==pytest.approx(1.)


def test_aliases_are_explicit_not_left_right_swapped():
    assert canonical('index_01_l')=='LeftHandIndex1'
    assert canonical('thumb_04_leaf_r')=='RightHandThumb3_End'
    assert canonical('mixamorig:LeftArm')=='LeftArm'


def test_different_units_and_euler_order_preserve_base_and_fold():
    base,bf=rig(scale=100, prefix='mixamorig:',order='ZYX')
    donor,df=rig(scale=.7,order='XYZ')
    start=channel_starts(base)
    # Different arms; retain the same seated lower pose.
    bf[start[2]+2]=70
    saved_bf=bf.copy();saved_df=df.copy();saved_offsets=[j[2].copy() for j in base]
    result=compose_arms(base,bf,donor,df)
    assert result.diagnostics['kinematic_probe_pass']
    assert result.diagnostics['release_allowed'] is False
    assert result.diagnostics['lower_position_delta_torso']==0
    assert result.diagnostics['protected_position_delta_torso']==0
    assert result.diagnostics['fold_geometry']['fold_geometry'] is True
    np.testing.assert_array_equal(bf,saved_bf);np.testing.assert_array_equal(df,saved_df)
    for joint,offset in zip(base,saved_offsets):np.testing.assert_array_equal(joint[2],offset)


def test_world_transform_positions_match_existing_fk_with_translation():
    j,f=rig(order='XYZ');f[:6]=[8,4,2,30,15,-5]
    for i,pos in fk(j,f).items():np.testing.assert_allclose(world_transforms(j,f)[i][:3,3],pos,atol=1e-12)
    axes=torso_frame(j,f);np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=1e-7)


def test_disproportionate_rig_loses_fold_and_is_rejected_without_lengthening():
    j,f=rig();donor,df=rig()
    for index in (3,4,6,7):j[index][2]*=.2
    before=[x[2].copy() for x in j]
    result=compose_arms(j,f,donor,df)
    assert not result.diagnostics['kinematic_probe_pass']
    assert 'chest_fold_lost' in result.diagnostics['reasons']
    assert result.diagnostics['bone_length_delta_torso']<1e-7
    for joint,offset in zip(j,before):np.testing.assert_array_equal(joint[2],offset)


def test_gimbal_euler_roundtrip_remains_valid_matrix():
    j,f=rig(order='XYZ');mat=_rot('X',12)@_rot('Y',90)@_rot('Z',27)
    assert _set_rotation(j,f,0,mat)>=1
    np.testing.assert_allclose(world_transforms(j,f)[0][:3,:3],mat,atol=1e-12)


@pytest.mark.parametrize("bad",['missing','hierarchy','nan','frame','channels'])
def test_bad_rigs_fail_explicitly(bad):
    j,f=rig();donor,df=rig()
    if bad=='missing':j[2][0]='UnknownArm'
    elif bad=='hierarchy':j[3][1]=1
    elif bad=='nan':f[0]=float('nan')
    elif bad=='frame':f=f[:-1]
    else:j[0][3][0]='Wposition'
    with pytest.raises(ValueError):compose_arms(j,f,donor,df)


def test_nonfold_donor_is_not_silently_accepted():
    j,f=rig(); donor,df=rig()
    donor[4][2]=np.array([-.4,-.5,0.])
    donor[7][2]=np.array([.4,-.5,0.])
    with pytest.raises(ValueError,match='not chest-fold'):compose_arms(j,f,donor,df)


def test_export_preserves_hierarchy_and_reloads(tmp_path):
    # Minimal file helper reuses the real single-frame writer, not generated coordinates.
    src=tmp_path/'source.bvh'
    src.write_text('HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n End Site\n { OFFSET 0 1 0 }\n}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n0 0 0 0 0 0\n')
    target=tmp_path/'result.bvh';target.write_text(single_frame_bvh_text(str(src),[1,2,3,20,5,9]))
    a,af=parse_bvh(str(src));b,bf=parse_bvh(str(target))
    assert [j[0:2]+[j[3],j[4]] for j in a]==[j[0:2]+[j[3],j[4]] for j in b]
    for x,y in zip(a,b):np.testing.assert_array_equal(x[2],y[2])
    assert np.isfinite(np.array(list(fk(b,bf[0]).values()))).all()
