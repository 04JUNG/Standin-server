"""Fixed-length native arm fitting for explicit asymmetric object/support roles.

No evaluation IDs or image coordinates. Each role defines body-relative goal
candidates; knee support uses actual surface samples with a reachable wrist.
Both arms are transactional and checked with the unchanged full skin checker.
"""
import math
from types import SimpleNamespace
import numpy as np
import bpy
from mathutils import Vector, Matrix, Quaternion
from .blender_adapter import snapshot,body_frame
from src.scene_refinement.rig import validate_rig
from .contact_adapter import (bones,unit,frame,set_hand_shape,hand_samples,
    elbow_for,apply_native,wrist_guard,surface)


def restore(arm,state):
    for p in arm.pose.bones:p.matrix_basis=state[p.name]
    bpy.context.view_layer.update()

def joint(arm,name):return arm.matrix_world@bones(arm)[name].head

def shape(role):
    return SimpleNamespace(hand_preset=role['hand_preset'],planar_thumb=False,patch='ulnar_edge')

def axes_for(arm,side):
    basis=body_frame(arm);up=Vector(basis[:,1]);forward=Vector(-basis[:,2]);out=Vector(-basis[:,0])*(1 if side=='Left' else -1)
    return out,up,forward

def hand_geometry(arm,meshes,side,role):
    return hand_samples(arm,meshes,side,shape(role))

def candidates(arm,meshes,side,role):
    out,up,fwd=axes_for(arm,side);s=joint(arm,side+'Arm');e=joint(arm,side+'ForeArm');w=joint(arm,side+'Hand')
    reach=(e-s).length+(w-e).length;pid=role['id']
    if pid=='umbrella_shaft_grip':
        gravity=Vector((0,0,1));u=unit(fwd-gravity*fwd.dot(gravity))
        # The hand-frame width axis points toward the pinky on Left and toward
        # the thumb on Right. A shared positive axis inverted the left grip.
        thumb_up_axis=gravity*(-1 if side=='Left' else 1)
        n=unit(thumb_up_axis.cross(u));h=frame(u,n)
        for front in [.42,.52,.62]:
            for low in [.10,.20,.30]:
                for yaw in [0,-.52,.52,-1.05,1.05]:
                    rotated=Quaternion(gravity,yaw).to_matrix()@h
                    yield s+fwd*reach*front-up*reach*low-out*reach*.10,rotated,unit(fwd+out*.5-up),None
    elif pid=='palm_up_offer':
        chest_center=(s+joint(arm,('Right' if side=='Left' else 'Left')+'Arm'))*.5
        for front in [.50,.62,.72]:
            for low in [.30,.42,.54]:
                for lateral in [-.10,0.,.10]:
                    h=frame(unit(fwd+up*.20),up)
                    yield chest_center+fwd*reach*front-up*reach*low+out*reach*lateral,h,unit(out-up*.8),None
    elif pid=='phone_hold':
        for front in [.48,.60]:
            for low in [.42,.55]:
                for tilt in [.55,.85]:
                    h=frame(unit(up*tilt+fwd*(1-tilt)),unit(fwd-up))
                    yield s+fwd*reach*front-up*reach*low-out*reach*.12,h,unit(out-up),None
    elif pid=='behind_back':
        for low in [.42,.50,.58]:
            for back in [.25,.35,.45]:
                for inward in [.15,.25,.35]:
                    # A behind-back hand crosses the lower back. A downward
                    # length axis produced a dangling arm rather than this pose.
                    h=frame(unit(-out+up*.10),fwd)
                    yield s-up*reach*low-fwd*reach*back-out*reach*inward,h,unit(out-fwd*.5-up),None
    elif pid=='lowered_hand':
        hip=joint(arm,side+'UpLeg')
        for front in [.12,.20,.28]:
            for lateral in [.10,.18,.25]:
                for low in [.93,.96,.98]:
                    goal=s-up*reach*low+out*reach*lateral+fwd*reach*front
                    h=frame(unit(fwd*.85-up*.55),-up)
                    yield goal,h,unit(out+fwd*.15-up),None
    elif pid=='raised_knee_rest':
        # Select the raised leg from the destination skeleton, independently of
        # which anatomical hand was named. Never assume the same-side knee.
        kside=max(['Left','Right'],key=lambda k:joint(arm,k+'Leg').dot(up))
        tree,pts,_=surface(meshes,kside+'Knee');center=joint(arm,kside+'Leg')
        # Support only the actual knee cap region, not any point on the thigh.
        offsets=pts-np.array(center)
        near=np.linalg.norm(offsets,axis=1)<.13
        top=near & ((offsets@np.array(up))>.015)
        support=pts[top]
        # Prefer the top/back (shoulder-near) portion of the knee surface.
        order=np.argsort(np.linalg.norm(support-np.array(s),axis=1))
        _,_,patch=hand_geometry(arm,meshes,side,role);offset=patch.mean(axis=0)
        for idx in order[::max(1,len(order)//12)][:12]:
            anchor=Vector(support[idx]);normal=tree.find_nearest(anchor)[1]
            if normal.dot(up)<.25:continue
            normal=unit(normal)
            for ray in [unit(fwd-up*.45),unit(fwd-up*.9),unit(fwd-up*1.5),-out]:
                # ulnar edge faces into the support; length axis may slope down,
                # putting the wrist above/behind the knee with real arm lengths.
                u=ray+normal*max(0,-ray.dot(normal))*.25
                z=-normal;u=unit(u-z*u.dot(z));n=unit(z.cross(u));h=frame(u,n)
                for clearance in [.002,.005]:
                    goal=anchor+normal*clearance-Vector(np.array(h)@offset)
                    yield goal,h,unit(out-up*.8),{'anchor':list(anchor),'normal':list(normal),'region':kside+'Knee','patch':'ulnar_edge'}
    else:raise ValueError('unsupported role')


def measure_arm(arm,meshes,side,role,profile,lengths,goal,contact):
    s,e,w=[joint(arm,side+t) for t in ['Arm','ForeArm','Hand']]
    swing,twist=wrist_guard(arm,side,profile);hf,_,patch=hand_geometry(arm,meshes,side,role)
    m={'wrist_swing_deg':swing,'wrist_twist_deg':twist,
       'elbow_flexion_deg':math.degrees(unit(e-s).angle(unit(w-e))),
       'bone_length_error_m':max(abs((e-s).length-lengths[0]),abs((w-e).length-lengths[1])),
       'wrist_goal_error_m':(w-Vector(goal)).length}
    if role['id']=='umbrella_shaft_grip':
        thumb=joint(arm,side+'HandThumb1');pinky=joint(arm,side+'HandPinky1')
        m.update(thumb_side_up_dot=unit(thumb-pinky).z,thumb_above_pinky_m=(thumb-pinky).z)
    if role['id']=='lowered_hand':m['hand_back_up_dot']=(-hf.col[1]).z
    if role['id']=='palm_up_offer':m['palm_up_dot']=hf.col[1].z
    if role['id']=='behind_back':
        _,_,forward=axes_for(arm,side)
        m['hand_behind_spine_m']=(joint(arm,'Spine')-w).dot(forward)
    if contact:
        tree,_,_=surface(meshes,contact['region']);pts=patch@np.array(hf).T+np.array(w)
        ds=np.array([tree.find_nearest(Vector(p))[3] for p in pts])
        m.update(patch_median_m=float(np.median(ds)),patch_p90_m=float(np.quantile(ds,.9)),patch_samples=len(ds),
                 normal_error_deg=math.degrees(hf.col[2].angle(-Vector(contact['normal']))))
    return m

def reasons_for(m):
    limits={'wrist_swing_deg':65,'wrist_twist_deg':35,'elbow_flexion_deg':165,'bone_length_error_m':2e-5,'wrist_goal_error_m':2e-5,
            'patch_median_m':.008,'patch_p90_m':.015,'normal_error_deg':25}
    reasons=[k for k,v in limits.items() if k in m and (not math.isfinite(m[k]) or m[k]>v)]
    for key,minimum in [('thumb_side_up_dot',.6),('thumb_above_pinky_m',.03),('hand_back_up_dot',.6),('palm_up_dot',.6),('hand_behind_spine_m',.06)]:
        if key in m and (not math.isfinite(m[key]) or m[key]<minimum):reasons.append(key)
    return reasons

def new_skin(skin,prior):
    old={tuple(c['regions']):c['max_depth_mm'] for c in prior['collisions']}
    return [c for c in skin['collisions'] if c['max_depth_mm']>old.get(tuple(c['regions']),0)+2]


def apply_relational(arm,meshes,plan,skin_check,profile):
    validate_rig(arm,meshes)
    if arm.animation_data or any(p.constraints for p in arm.pose.bones):raise ValueError('baked unconstrained rig required')
    scale=arm.matrix_world.to_scale()
    if min(scale)<=0 or max(scale)-min(scale)>1e-5:raise ValueError('uniform positive armature scale required')
    originals={p.name:p.matrix_basis.copy() for p in arm.pose.bones};before=snapshot(arm);baseline=skin_check(arm)
    result={'method':'skeleton_relative_relational_fit_v1','plan':plan,'accepted':False,'members':[],'initial_skin':baseline,'release_allowed':False}
    try:
        for member in plan['members']:
            side=member['side'].title();role=member['part'];state={p.name:p.matrix_basis.copy() for p in arm.pose.bones}
            s,e,w=[joint(arm,side+t) for t in ['Arm','ForeArm','Hand']];lengths=((e-s).length,(w-e).length)
            set_hand_shape(arm,side,shape(role));shaped={p.name:p.matrix_basis.copy() for p in arm.pose.bones}
            hf,_,_=hand_geometry(arm,meshes,side,role)
            options=[]
            for goal,h,pole,contact in candidates(arm,meshes,side,role):
                for swivel in [0,-.35,.35,-.7,.7]:
                    try:elbow=elbow_for(s,goal,lengths,pole,swivel)
                    except ValueError:continue
                    # A cheap orientation screen limits expensive skin checks.
                    angle=math.degrees(unit(goal-elbow).angle(h.col[0]))
                    if angle>70:continue
                    if role['id']=='umbrella_shaft_grip' and (goal-elbow).z<lengths[1]*.40:continue
                    score=angle*.001+abs(swivel)*.008+(goal-w).length*.015
                    options.append((score,goal,elbow,h,contact))
            trials=[];chosen=None
            for _,goal,elbow,h,contact in sorted(options,key=lambda o:o[0]):
                restore(arm,shaped);roll=apply_native(arm,side,elbow,goal,h,hf,profile)
                metrics=measure_arm(arm,meshes,side,role,profile,lengths,goal,contact);reasons=reasons_for(metrics)
                if not reasons:
                    skin=skin_check(arm);new=new_skin(skin,baseline)
                    if new:reasons.append('new_skin')
                trials.append({'goal':list(goal),'metrics':metrics,'reasons':reasons})
                if not reasons:
                    chosen={'side':member['side'],'part':role,'accepted':True,'metrics':metrics,'wrist_goal_world':list(goal),'contact':contact,
                       'lengths':lengths,'forearm_roll_deg':roll,'new_skin':new,'trials':trials};break
                if len(trials)>=65:break
            if chosen is None:
                result['members'].append({'side':member['side'],'part':role,'accepted':False,'trials':trials,'feasible_candidates':len(options)})
                result['reasons']=[member['side']+': no admitted relational fit'];break
            result['members'].append(chosen)
        else:
            skin=skin_check(arm);new=new_skin(skin,baseline);after=snapshot(arm)
            allowed={p.name for p in arm.pose.bones if any(p.name.split(':')[-1] in [s+t for t in ['Arm','ForeArm','Hand']] or p.name.split(':')[-1].startswith(s+'Hand') for s in [m['side'].title() for m in plan['members']])}
            protected=max(float(np.max(abs(after[n]-v))) for n,v in before.items() if n not in allowed)
            result.update(accepted=not new and protected<2e-5,reasons=(['new_skin'] if new else [])+(['protected_pose'] if protected>=2e-5 else []),
                          protected_matrix_delta=protected,final_skin=skin,new_skin=new)
    except Exception:
        restore(arm,originals);raise
    if not result['accepted']:
        result['rejected_pose_basis']={p.name:np.array(p.matrix_basis).tolist() for p in arm.pose.bones}
        restore(arm,originals)
        result['rollback_matrix_delta']=max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in before.items())
    return result
