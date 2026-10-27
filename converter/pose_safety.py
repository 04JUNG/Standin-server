"""Offline whole-rig angular and camera-depth trust policy.

These conservative relative bounds are NOT a medical/anatomical certificate.
Depth comes from the selected 3D pose prior, never an invented observed Z value.
Unchanged full-mesh and final-byte verification are required in addition.
"""
from __future__ import annotations
import numpy as np

VERSION = "target-pose-depth-safety-v1"
COCO_NAMES = ["Head"]*5+["LeftArm","RightArm","LeftForeArm","RightForeArm",
    "LeftHand","RightHand","LeftUpLeg","RightUpLeg","LeftLeg","RightLeg","LeftFoot","RightFoot"]
CHAINS = {side+kind: (side+root, side+middle, side+end)
          for side in ("Left", "Right")
          for kind,root,middle,end in (("Arm","Arm","ForeArm","Hand"),("Leg","UpLeg","Leg","Foot"))}
LIMITS = {side+part: limit for side in ("Left","Right")
          for part,limit in (("Arm",12.),("ForeArm",12.),("UpLeg",8.),("Leg",8.),("Foot",12.))}


def rotation(matrix):
    u, _, v = np.linalg.svd(matrix[:3,:3])
    r = u@v
    if np.linalg.det(r) <= 0:
        raise ValueError("reflected_bone_frame")
    return r


def angle(a,b):
    den = np.linalg.norm(a)*np.linalg.norm(b)
    if den < 1e-12: raise ValueError("degenerate_joint_chain")
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/den,-1,1))))


def index_map(names):
    lookup = {n.split(":")[-1]: i for i,n in enumerate(names)}
    if len(lookup) != len(names) or not set(COCO_NAMES).issubset(lookup):
        raise ValueError("unsupported_pose_rig")
    return lookup


def camera_points(bones,names,camera):
    lookup=index_map(names)
    bones=np.asarray(bones,float)
    if bones.shape!=(len(names),4,4) or not np.isfinite(bones).all():
        raise ValueError("invalid_camera_pose_frames")
    world=bones[[lookup[n] for n in COCO_NAMES],:3,3]
    r=np.asarray(camera["rotation"],float)
    if (r.shape!=(3,3) or not np.isfinite(r).all() or
            not np.allclose(r.T@r,np.eye(3),atol=1e-7) or np.linalg.det(r)<0):
        raise ValueError("invalid_fixed_camera")
    # Native FBX world is right-handed Z-up; search camera is Y-up.
    return (world[:,[0,2,1]]*np.array([1.,1.,-1.]))@r.T


def audit_pose(base,candidate,names,parents,camera,*,depth_reference=None):
    base,candidate=np.asarray(base,float),np.asarray(candidate,float)
    failures=[]; joints=[]; chains=[]
    if (base.shape!=(len(names),4,4) or candidate.shape!=base.shape or
            not np.isfinite(base).all() or not np.isfinite(candidate).all() or
            not np.allclose(base[:,3,:],[0,0,0,1],atol=1e-8) or
            not np.allclose(candidate[:,3,:],[0,0,0,1],atol=1e-8)):
        raise ValueError("invalid_pose_frames")
    lookup=index_map(names); ids={n:i for i,n in enumerate(names)}
    if set(parents)!=set(names): raise ValueError("invalid_pose_hierarchy")
    if sum(p is None for p in parents.values())!=1:raise ValueError("invalid_pose_hierarchy")
    for name in names:
        visited=set();current=name
        while current is not None:
            if current in visited or current not in parents:raise ValueError("invalid_pose_hierarchy")
            visited.add(current);current=parents[current]
    for i,name in enumerate(names):
        parent=parents[name]
        if parent is not None and parent not in ids: raise ValueError("invalid_pose_hierarchy")
        a=base[i] if parent is None else np.linalg.inv(base[ids[parent]])@base[i]
        b=candidate[i] if parent is None else np.linalg.inv(candidate[ids[parent]])@candidate[i]
        ra,rb=rotation(a),rotation(b)
        degrees=float(np.degrees(np.arccos(np.clip((np.trace(rb@ra.T)-1)/2,-1,1))))
        shift=float(np.linalg.norm(a[:3,3]-b[:3,3]))
        # Scale/shear is not an angular correction. Compare polar stretch.
        stretch=float(np.max(abs(ra.T@a[:3,:3]-rb.T@b[:3,:3])))
        suffix=name.split(":")[-1];limit=LIMITS.get(suffix,0.)
        twist=0.
        child=next((chain[k+1] for chain in CHAINS.values() for k in (0,1)
                    if chain[k]==suffix),None)
        if child is not None and degrees>1e-6:
            axis=base[lookup[child],:3,3]-base[i,:3,3]
            if parent is not None:
                axis=np.linalg.inv(base[ids[parent],:3,:3])@axis
            axis/=np.linalg.norm(axis)
            delta=rb@ra.T;theta=np.deg2rad(degrees)
            spin=np.array([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])
            if abs(np.sin(theta))>1e-8:
                spin/=2*np.sin(theta)
                twist=abs(float(np.degrees(2*np.arctan2(np.sin(theta/2)*np.dot(spin,axis),np.cos(theta/2)))))
            else:
                twist=degrees
            if twist>4.+.002: failures.append("axial_twist:"+suffix)
        if degrees>limit+.002: failures.append("angular_bound:"+suffix)
        if shift>1e-5 or stretch>1e-4: failures.append("bone_length_or_stretch:"+suffix)
        joints.append(dict(joint=suffix,delta_degrees=degrees,limit_degrees=limit,offset_error_m=shift,
                           axial_twist_degrees=twist,axial_twist_limit_degrees=4. if child else 0.))
    for label, chain in CHAINS.items():
        idx=[lookup[n] for n in chain]
        p,q=base[idx,:3,3],candidate[idx,:3,3]
        u,v=p[1]-p[0],p[2]-p[1];un,vn=q[1]-q[0],q[2]-q[1]
        before,after=angle(-u,v),angle(-un,vn)
        oldpole,newpole=np.cross(u,v),np.cross(un,vn)
        turn=None
        if before<178. and after<178.:
            turn=angle(oldpole,newpole)
            if turn>60.: failures.append("bend_branch:"+label)
        if after<20.: failures.append("overfolded_joint:"+label)
        chains.append(dict(chain=label,base_interior_degrees=before,interior_degrees=after,pole_turn_degrees=turn))
    p,q=camera_points(base if depth_reference is None else depth_reference,names,camera),camera_points(candidate,names,camera)
    torso=float(np.linalg.norm(p[[5,6]].mean(0)-p[[11,12]].mean(0)))
    if torso<1e-6: raise ValueError("degenerate_pose_torso")
    drift=abs(q[:,2]-p[:,2])/torso
    if np.any(drift[5:]>.15+1e-6): failures.append("camera_depth_trust_bound")
    depth=[]
    for a,b in [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(7,8),(9,10),(13,14),(15,16)]:
        old,new=(p[b,2]-p[a,2])/torso,(q[b,2]-q[a,2])/torso
        active=abs(old)>.03
        ok=not active or np.sign(old)*new>=.25*abs(old)-1e-6
        if not ok: failures.append("depth_order:"+str(a)+":"+str(b))
        depth.append(dict(joints=[a,b],base_depth_torso=float(old),depth_torso=float(new),prior_constrained=bool(active),preserved=bool(ok)))
    return dict(version=VERSION,status="pose_validated" if not failures else "pose_blocked",
        violations=failures,joints=joints,chains=chains,depth_pairs=depth,
        max_depth_drift_torso=float(drift[5:].max()),depth_authority="selected-3d-pose-prior",
        observed_depth_ground_truth=False,anatomical_certificate=False)


def descendants(names,parents,root):
    active={root}
    for _ in names:
        larger=active|{n for n in names if parents[n] in active}
        if larger==active: return [i for i,n in enumerate(names) if n in active]
        active=larger
    raise ValueError("cyclic_pose_hierarchy")


def rotate_subtree(bones,indices,root,rot):
    out=bones.copy();pivot=bones[root,:3,3];delta=np.eye(4)
    delta[:3,:3]=rot;delta[:3,3]=pivot-rot@pivot
    out[indices]=delta@bones[indices]
    return out
