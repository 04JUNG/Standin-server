"""Apply a reviewed arm pose part to an imported Mixamo FBX, with atomic rollback.

Reuses converter rig validation, output modes, and minimal-rotation arm reach.
Works on the actual imported rest/pose rather than assuming an original T-pose.
"""
from __future__ import annotations

import numpy as np
import bpy
from mathutils import Matrix, Vector
from src.arm_refine.parts import validate_part, selected_sides, require_pair_quality
from src.experimental.part_composition import shortest_arc
from src.collision import arm_torso_penetration, arm_leg_penetration
from src.scene_refinement.rig import validate_rig

COCO_NAMES = {0:'Head',5:'LeftArm',6:'RightArm',7:'LeftForeArm',8:'RightForeArm',
    9:'LeftHand',10:'RightHand',11:'LeftUpLeg',12:'RightUpLeg',13:'LeftLeg',14:'RightLeg',
    15:'LeftFoot',16:'RightFoot'}


def snapshot(arm):
    return {p.name:np.array(arm.matrix_world @ p.matrix) for p in arm.pose.bones}


def points17(arm):
    kp=np.zeros((17,3));scores=np.zeros(17)
    for i,name in COCO_NAMES.items():
        pb=arm.pose.bones.get('mixamorig:'+name)
        if pb:
            kp[i]=np.array(arm.matrix_world @ pb.head);scores[i]=1
    return kp,scores


def body_frame(arm):
    p,_=points17(arm)
    up=(p[5]+p[6]-p[11]-p[12])*.5
    up/=np.linalg.norm(up)
    right=p[6]-p[5];right-=up*np.dot(right,up)
    right/=np.linalg.norm(right)
    basis=np.column_stack((right,up,np.cross(right,up)))
    if not np.isfinite(basis).all():raise ValueError('degenerate destination torso')
    return basis


def apply_part(arm, meshes, part, *, selection=None, image_goals=None, camera=None,
               connection_deg=(0.,0.,0.)):
    available=validate_part(part)
    require_pair_quality(part)
    sides=available if selection is None else selected_sides(selection)
    if not set(sides)<=set(available):raise ValueError('selection absent from part')
    if part['relation']=='paired' and tuple(sides)!=('left','right'):
        raise ValueError('paired part must be applied atomically')
    validate_rig(arm,meshes)
    # Fail before mutation on unsupported scale/shear or constraints/animation.
    # Nonuniform object scaling alters fixed-length rotation assumptions.
    scales=arm.matrix_world.to_scale()
    if max(scales)-min(scales)>1e-5 or min(scales)<=0:
        raise ValueError('unsupported nonuniform or reflected armature scale')
    if arm.animation_data or any(p.constraints for p in arm.pose.bones):
        raise ValueError('animated or constrained rig requires baking first')
    angles=np.asarray(connection_deg,float)
    if angles.shape!=(3,) or not np.isfinite(angles).all() or max(abs(angles))>45:
        raise ValueError('connection adjustment must be three finite torso angles within 45 degrees')
    if image_goals and camera is None:raise ValueError('image goals require calibrated camera')
    basis=body_frame(arm)
    from mathutils import Euler
    adjustment=np.array(Euler(tuple(np.radians(angles)), 'XYZ').to_matrix())
    before=snapshot(arm);bp,scores=points17(arm)
    bases={p.name:p.matrix_basis.copy() for p in arm.pose.bones}
    lengths={b.name:b.length for b in arm.data.bones}
    mutable={'mixamorig:'+s.title()+n for s in sides for n in ('Arm','ForeArm','Hand')}
    moving=set(mutable)
    for pb in arm.pose.bones:
        if any(parent.name in mutable for parent in pb.parent_recursive):moving.add(pb.name)
    goals=[];reasons=[];errors={};flexions={}
    try:
        for side in sides:
            prefix='mixamorig:'+side.title()
            upper=arm.pose.bones[prefix+'Arm'];fore=arm.pose.bones[prefix+'ForeArm'];hand=arm.pose.bones[prefix+'Hand']
            if fore.parent!=upper or hand.parent!=fore or upper.parent.name!=prefix+'Shoulder':
                raise ValueError('unsupported destination arm chain')
            hand_world=(arm.matrix_world @ hand.matrix).copy()
            desired=[basis @ adjustment @ np.asarray(v) for v in part['arms'][side]['directions_torso']]
            for k,(pb,child) in enumerate(((upper,fore),(fore,hand))):
                a=arm.matrix_world @ pb.head;b=arm.matrix_world @ child.head
                delta=Matrix(shortest_arc(np.array(b-a),desired[k]).tolist()).to_4x4()
                world=arm.matrix_world @ pb.matrix
                pb.matrix=arm.matrix_world.inverted() @ Matrix.Translation(a) @ delta @ Matrix.Translation(-a) @ world
                bpy.context.view_layer.update()
            hand_world.translation=arm.matrix_world @ hand.head
            hand.matrix=arm.matrix_world.inverted() @ hand_world
            bpy.context.view_layer.update()
            if image_goals and side in image_goals:
                # Existing calibrated fixed-length solver, no invented 2D->3D lift.
                from src.scene_refinement.arms import solve
                goal=image_goals[side]
                goals.append(solve(arm,side.title(),goal['wrist'],goal.get('elbow'),camera,goal['headheight']))
                if not goals[-1]['accepted']:reasons.append('image_reach_not_improved:'+side)
            else:
                rays=[np.array(arm.matrix_world @ c.head-arm.matrix_world @ p.head)
                      for p,c in ((upper,fore),(fore,hand))]
                rays=[v/np.linalg.norm(v) for v in rays]
                errors[side]=[float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1)))) for a,b in zip(rays,desired)]
                if max(errors[side])>.05:reasons.append('direction_transfer_error:'+side)
            a=np.array(arm.matrix_world @ fore.head-arm.matrix_world @ upper.head)
            b=np.array(arm.matrix_world @ hand.head-arm.matrix_world @ fore.head)
            flexions[side]=float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
            if flexions[side]>165:reasons.append('elbow_limit:'+side)
        after=snapshot(arm);ap,_=points17(arm)
        protected=max(float(np.max(np.abs(after[n]-v))) for n,v in before.items() if n not in moving)
        local_exact=all(np.max(np.abs(np.array(p.matrix_basis)-np.array(bases[p.name])))<1e-6 for p in arm.pose.bones if p.name not in mutable)
        length_delta=max(abs(b.length-lengths[b.name]) for b in arm.data.bones)
        wrist_errors={s:float(np.max(np.abs(after['mixamorig:'+s.title()+'Hand'][:3,:3]-before['mixamorig:'+s.title()+'Hand'][:3,:3]))) for s in sides}
        if protected>2e-5 or not local_exact:reasons.append('protected_pose_changed')
        if length_delta>1e-7:reasons.append('bone_length_changed')
        if max(wrist_errors.values())>2e-5:reasons.append('wrist_orientation_changed')
        collisions={}
        for side in sides:
            limb=side+'_arm'
            prior=arm_torso_penetration(bp,limb,scores);new=arm_torso_penetration(ap,limb,scores)
            collisions[side]={'before':prior.depth,'after':new.depth}
            if not (prior.available and new.available):reasons.append('collision_proxy_unavailable')
            elif new.depth>prior.depth+.02:reasons.append('new_torso_proxy_penetration:'+side)
            for leg in ('left_leg','right_leg'):
                old=arm_leg_penetration(bp,limb,leg,scores);cur=arm_leg_penetration(ap,limb,leg,scores)
                if old.available and cur.available and cur.depth>old.depth+.02:reasons.append('new_arm_leg_proxy_penetration:'+side)
        report={'accepted':not reasons,'reasons':reasons,'part_id':part['id'],'selection':list(sides),
            'protected_world_matrix_max_delta':protected,'nonselected_local_pose_preserved':bool(local_exact),
            'bone_length_max_delta':float(length_delta),'wrist_orientation_max_delta':wrist_errors,
            'elbow_flexion_deg':flexions,'direction_error_deg':errors,'collision_proxy':collisions,
            'connection_deg':angles.tolist(),'image_goals':goals,'release_allowed':False,
            'limitations':['collision_proxy_not_skin_check','shoulder_twist_limits_not_certified','manual_part_selection_not_detection']}
        if reasons:
            for name,mat in bases.items():arm.pose.bones[name].matrix_basis=mat
            bpy.context.view_layer.update()
        return report
    except Exception:
        for name,mat in bases.items():arm.pose.bones[name].matrix_basis=mat
        bpy.context.view_layer.update()
        raise
