"""Offline arms-only composition probe. Not connected to search/API/refine.

Keep the seated base hierarchy, offsets, translation, spine, pelvis and legs.
Transport donor arm/finger directions in a shared anatomical torso frame; solve
destination local rotations instead of copying cross-rig Euler channels.
This kinematic probe does not certify skin/contact/rig-limit correctness.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from src.bvh import _rot, channel_starts, coco17_from_fk, fk
from src.collision import arm_torso_penetration, arm_leg_penetration
from src.semantic_shapes import relational_geometry

VERSION = "offline-arms-direction-transport-v1"
ARM_ROOTS = ("LeftArm", "RightArm")
MAJOR = ("LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand")


def canonical(name):
    name = name.split(":")[-1]
    match = re.fullmatch(r"(thumb|index|middle|ring|pinky)_0([1-4])(?:_leaf)?_([lr])", name)
    if match:
        finger, segment, side = match.groups()
        prefix = ("Left" if side=="l" else "Right") + "Hand" + finger.title()
        return prefix + ("3_End" if segment=="4" else segment)
    return name


def _mapping(joints):
    names = {}
    for i,joint in enumerate(joints):
        key = canonical(joint[0])
        if key in names: raise ValueError("ambiguous canonical joint:" + key)
        names[key] = i
    return names


def _validate(joints, frame):
    frame = np.asarray(frame, dtype=float)
    if frame.shape != (sum(len(j[3]) for j in joints),) or not np.isfinite(frame).all():
        raise ValueError("invalid BVH frame")
    if not joints: raise ValueError("empty rig")
    for i,j in enumerate(joints):
        if not -1 <= j[1] < i or not np.isfinite(j[2]).all():
            raise ValueError("invalid BVH hierarchy/offset")
        if len(j[3]) != len(set(j[3])) or any(ch not in {a+k for a in 'XYZ' for k in ('position','rotation')} for ch in j[3]):
            raise ValueError("invalid BVH channels")
    return frame


def local_translation(joints, frame, idx):
    t = np.asarray(joints[idx][2], dtype=float).copy()
    start = channel_starts(joints)[idx]
    for k,ch in enumerate(joints[idx][3]):
        if ch.endswith("position"): t["XYZ".index(ch[0])] += frame[start+k]
    return t


def world_transforms(joints, frame):
    frame = _validate(joints, frame)
    result = []
    starts = channel_starts(joints)
    for i,j in enumerate(joints):
        r = np.eye(3)
        for k,ch in enumerate(j[3]):
            if ch.endswith("rotation"): r = r @ _rot(ch[0], frame[starts[i]+k])
        transform = np.eye(4)
        transform[:3,:3] = r
        transform[:3,3] = local_translation(joints,frame,i)
        result.append(transform if j[1]==-1 else result[j[1]] @ transform)
    return result


def torso_frame(joints, frame):
    p,s = coco17_from_fk(joints,fk(joints,frame))
    if not (s[[5,6,11,12]]>.0).all(): raise ValueError("missing torso mapping")
    up=(p[5]+p[6]-p[11]-p[12])*.5
    right=p[6]-p[5]
    if np.linalg.norm(up)<1e-8: raise ValueError("degenerate torso")
    up=up/np.linalg.norm(up)
    right=right-up*np.dot(right,up)
    if np.linalg.norm(right)<1e-8: raise ValueError("degenerate shoulders")
    right=right/np.linalg.norm(right)
    forward=np.cross(right,up)
    return np.column_stack((right,up,forward))


def shortest_arc(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if min(np.linalg.norm(a),np.linalg.norm(b))<1e-8: raise ValueError("zero bone vector")
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    v=np.cross(a,b);c=float(np.clip(np.dot(a,b),-1,1));s=float(np.linalg.norm(v))
    if s<1e-8:
        if c>0: return np.eye(3)
        axis=np.eye(3)[int(np.argmin(np.abs(a)))];axis=np.cross(a,axis);axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3)
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+k@k*((1-c)/(s*s))


def align_vectors(source,target):
    """Single ray: minimal rotation. Multiple finger rays: rigid least squares."""
    a=np.asarray(source,float);b=np.asarray(target,float)
    a=a/np.linalg.norm(a,axis=1)[:,None];b=b/np.linalg.norm(b,axis=1)[:,None]
    if len(a)==1 or np.linalg.matrix_rank(a,tol=1e-6)<2: return shortest_arc(a[0],b[0])
    u,_,vt=np.linalg.svd(a.T@b)
    d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
    return vt.T@d@u.T


def _set_rotation(joints,frame,idx,matrix):
    # Optional SciPy is imported only for this offline experiment.
    from scipy.spatial.transform import Rotation
    channels=joints[idx][3]; axes=[ch[0] for ch in channels if ch.endswith("rotation")]
    if len(axes)!=3 or set(axes)!=set("XYZ"): raise ValueError("unsupported rotation channels")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        angles=Rotation.from_matrix(matrix).as_euler("".join(axes),degrees=True)
    start=channel_starts(joints)[idx];n=0
    for k,ch in enumerate(channels):
        if ch.endswith("rotation"):frame[start+k]=angles[n];n+=1
    return len(recorded)


def arm_mask(joints):
    names=_mapping(joints)
    if any(k not in names for k in MAJOR): raise ValueError("missing arm chain")
    roots={names[name] for name in ARM_ROOTS}
    selected=set()
    for i,j in enumerate(joints):
        if i in roots or j[1] in selected:selected.add(i)
    for side in ("Left","Right"):
        if joints[names[side+"ForeArm"]][1]!=names[side+"Arm"] or joints[names[side+"Hand"]][1]!=names[side+"ForeArm"]:
            raise ValueError("unsupported arm hierarchy")
    return selected


@dataclass
class Composition:
    frame: np.ndarray
    diagnostics: dict


def compose_arms(base_joints,base_frame,donor_joints,donor_frame):
    base=_validate(base_joints,base_frame);donor=_validate(donor_joints,donor_frame)
    src_names=_mapping(donor_joints);dst_names=_mapping(base_joints)
    selected=arm_mask(base_joints);arm_mask(donor_joints)
    dp,ds=coco17_from_fk(donor_joints,fk(donor_joints,donor))
    if relational_geometry(dp,ds)["fold_geometry"] is not True: raise ValueError("donor is not chest-fold geometry")
    transport=torso_frame(base_joints,base)@torso_frame(donor_joints,donor).T
    src_world=world_transforms(donor_joints,donor)
    base_world=world_transforms(base_joints,base)
    for names,world in ((src_names,src_world),(dst_names,base_world)):
        for side in ("Left","Right"):
            for a,b in (("Arm","ForeArm"),("ForeArm","Hand")):
                if np.linalg.norm(world[names[side+b]][:3,3]-world[names[side+a]][:3,3])<1e-8:
                    raise ValueError("zero arm bone")
    children={i:[] for i in range(len(base_joints))}
    for i,j in enumerate(base_joints):
        if j[1]!=-1:children[j[1]].append(i)
    out=base.copy();changed=[];euler_warnings=0;skipped=[]
    for i,j in enumerate(base_joints):
        if i not in selected or j[4] or not j[3]:continue
        name=canonical(j[0]);si=src_names.get(name)
        if si is None:skipped.append(name);continue
        a=[];b=[]
        world=world_transforms(base_joints,out)
        parent_rotation=np.eye(3) if j[1]==-1 else world[j[1]][:3,:3]
        for child in children[i]:
            sc=src_names.get(canonical(base_joints[child][0]))
            if sc is None or donor_joints[sc][1]!=si:continue
            offset=local_translation(base_joints,out,child)
            desired=transport@(src_world[sc][:3,3]-src_world[si][:3,3])
            if min(np.linalg.norm(offset),np.linalg.norm(desired))<1e-8:continue
            a.append(offset);b.append(parent_rotation.T@desired)
        if not a:skipped.append(name);continue
        euler_warnings+=_set_rotation(base_joints,out,i,align_vectors(a,b));changed.append(name)
    pos_before=fk(base_joints,base);pos_after=fk(base_joints,out)
    bp,bs=coco17_from_fk(base_joints,pos_before);cp,cs=coco17_from_fk(base_joints,pos_after)
    scale=float(np.linalg.norm((bp[5]+bp[6]-bp[11]-bp[12])*.5))
    protected=max(float(np.linalg.norm(pos_after[i]-pos_before[i])) for i in range(len(base_joints)) if i not in selected)
    lower=float(np.max(np.linalg.norm(cp[11:17]-bp[11:17],axis=1)))
    lengths=max(abs(np.linalg.norm(pos_after[i]-pos_after[j[1]])-np.linalg.norm(pos_before[i]-pos_before[j[1]])) for i,j in enumerate(base_joints) if j[1]!=-1)
    errors=[]
    for parent,child in (("LeftArm","LeftForeArm"),("LeftForeArm","LeftHand"),("RightArm","RightForeArm"),("RightForeArm","RightHand")):
        actual=pos_after[dst_names[child]]-pos_after[dst_names[parent]]
        desired=transport@(src_world[src_names[child]][:3,3]-src_world[src_names[parent]][:3,3])
        angle=np.degrees(np.arccos(np.clip(np.dot(actual,desired)/(np.linalg.norm(actual)*np.linalg.norm(desired)),-1,1)))
        errors.append(float(angle))
    collision={}
    for side in ("left_arm","right_arm"):
        collision[side]={"donor":arm_torso_penetration(dp,side,ds).depth,
                         "base":arm_torso_penetration(bp,side,bs).depth,
                         "composed":arm_torso_penetration(cp,side,cs).depth}
    relation=relational_geometry(cp,cs)
    arm_leg={}
    for arm in ("left_arm","right_arm"):
        for leg in ("left_leg","right_leg"):
            arm_leg[arm+":"+leg]={"base":arm_leg_penetration(bp,arm,leg,bs).depth,
                                    "composed":arm_leg_penetration(cp,arm,leg,cs).depth}
    reasons=[]
    if protected>1e-7*scale or lower>1e-7*scale:reasons.append("protected_base_changed")
    if lengths>1e-7*scale:reasons.append("bone_length_changed")
    if max(errors)>1e-3:reasons.append("arm_direction_mismatch")
    if relation["fold_geometry"] is not True:reasons.append("chest_fold_lost")
    if any(v["composed"]>max(v["donor"],v["base"])+.02 for v in collision.values()):reasons.append("new_torso_proxy_penetration")
    if any(v["composed"]>v["base"]+.02 for v in arm_leg.values()):reasons.append("new_arm_leg_proxy_penetration")
    return Composition(out,{"version":VERSION,"scope":"arms_and_mapped_fingers_only",
        "kinematic_probe_pass":not reasons,"reasons":reasons,"release_allowed":False,
        "protected_position_delta_torso":protected/scale,"lower_position_delta_torso":lower/scale,
        "bone_length_delta_torso":float(lengths/scale),"arm_direction_error_deg":errors,
        "fold_geometry":relation,"collision_proxy":collision,"arm_leg_proxy":arm_leg,"changed_joints":changed,
        "skipped_joints":skipped,"euler_warnings":euler_warnings,
        "limitations":["no_actual_mesh_contact_check","no_joint_limit_or_hand_twist_certification",
                        "not_whole_upper_body_transplant","no_rough_3D_ground_truth"]})
