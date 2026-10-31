"""Read actual native bones and target-specific skin distances, without posing."""
import math
import numpy as np
from mathutils import Vector
from converter.hand_geometry import hand_frame
from .blender_adapter import body_frame, snapshot
from .contact_adapter import bones, hand_samples, target_frame
from .contact_parts import select_contact_part

def _unit(v):
    v=np.asarray(v,float);length=float(np.linalg.norm(v))
    if length<1e-8:raise ValueError('degenerate native vector')
    return v/length

def _flex(a,b,c):
    return float(np.degrees(np.arccos(np.clip(_unit(b-a)@_unit(c-b),-1,1))))

def read_native_state(arm,meshes,target=None):
    before=snapshot(arm);b=bones(arm)
    p={name:np.array(arm.matrix_world@bone.head) for name,bone in b.items()}
    basis=body_frame(arm);up=basis[:,1];forward=-basis[:,2]
    shoulders=(p['LeftArm']+p['RightArm'])/2
    hips=(p['LeftUpLeg']+p['RightUpLeg'])/2
    scale=float(np.linalg.norm(shoulders-hips))
    if scale<.05:raise ValueError('invalid native torso scale')
    out={'schema':'standin.native-limb-state.v1','torso_scale_m':scale,'limbs':{},
         'side_convention':'subject_relative','source':'actual_selected_fbx_bones_and_skin',
         'object_identity_inferred':False}
    for side in ('left','right'):
        prefix=side.title()
        try:
            s,e,w=[p[prefix+name] for name in ('Arm','ForeArm','Hand')]
            bases={finger:p[prefix+'Hand'+finger.title()+'1'] for finger in ('thumb','index','middle','ring','pinky')}
            hf=hand_frame(side,w,bases)
            thumb=_unit(bases['thumb']-bases['pinky'])
            ratios=[float(np.linalg.norm(np.array(arm.matrix_world@b[prefix+'Hand'+f+'3'].tail)-w)/
                          np.linalg.norm(p[prefix+'Hand'+f+'1']-w)) for f in ('Index','Middle','Ring','Pinky')]
            row={'valid':True,'flexion_deg':_flex(s,e,w),
                 'wrist_axis_deg':float(np.degrees(np.arccos(np.clip(_unit(w-e)@hf.length_axis,-1,1)))),
                 'palm_up_dot':float(hf.palm_normal@np.array([0,0,1.])),
                 'hand_back_forward_dot':float(-hf.palm_normal@forward),
                 'thumb_up_dot':float(thumb@np.array([0,0,1.])),
                 'finger_fold_ratio':max(ratios),
                 'wrist_height_shoulder_torso':float((w-shoulders)@up/scale),
                 'wrist_height_hip_torso':float((w-hips)@up/scale),
                 'wrist_forward_torso':float((w-shoulders)@forward/scale),
                 'wrist_head_distance_torso':float(np.linalg.norm(w-p['Head'])/scale)}
            knee_heights={s:float(p[s.title()+'Leg']@up) for s in ('left','right')}
            row['raised_knee_side']=max(knee_heights,key=knee_heights.get) if abs(knee_heights['left']-knee_heights['right'])>.05 else 'unknown'
            wanted=(target or {}).get('limbs',{}).get(side+'_arm',{})
            action=wanted.get('action');contact=None
            if action in ('head_side','forehead_touch','chin_support','waist_side','knee_rest'):
                contact={'head_side':'head_side','forehead_touch':'forehead_straight',
                         'chin_support':'chin_support','waist_side':'waist_side','knee_rest':'knee_rest'}[action]
                if action=='chin_support' and wanted.get('hand_shape') in ('open','relaxed'):contact='chin_relaxed'
            if contact:
                part=select_contact_part(contact,side)
                tree,anchor,normal,frame,pole,axes=target_frame(arm,meshes,prefix,part)
                hand_basis,_,local_patch=hand_samples(arm,meshes,prefix,part)
                samples=np.asarray(local_patch)@np.array(hand_basis).T+w
                ds=[tree.find_nearest(Vector(point))[3] for point in samples]
                row['target_patch_median_m']=float(np.median(ds))
                row['target_anchor_distance_torso']=float(np.linalg.norm(np.mean(samples,axis=0)-np.array(anchor))/scale)
                row['contact_recipe_measurement']=contact
                row['target_surface_normal_error_deg']=float(np.degrees(np.arccos(np.clip(
                    hf.palm_normal@(-np.array(normal)),-1,1))))
            out['limbs'][side+'_arm']=row
        except (KeyError,ValueError,TypeError,ZeroDivisionError) as error:
            out['limbs'][side+'_arm']={'valid':False,'error_type':type(error).__name__}
        try:
            hip,knee,ankle,toe=[p[prefix+name] for name in ('UpLeg','Leg','Foot','ToeBase')]
            out['limbs'][side+'_leg']={'valid':True,'flexion_deg':_flex(hip,knee,ankle),
                'knee_height_hip_torso':float((knee-hips)@up/scale),
                'ankle_height_hip_torso':float((ankle-hips)@up/scale),
                'foot_forward_dot':float(_unit(toe-ankle)@forward)}
        except (KeyError,ValueError,ZeroDivisionError) as error:
            out['limbs'][side+'_leg']={'valid':False,'error_type':type(error).__name__}
    after=snapshot(arm)
    out['read_only_matrix_delta']=max(float(np.max(np.abs(after[name]-value))) for name,value in before.items())
    if out['read_only_matrix_delta']!=0:raise ValueError('native measurement mutated pose')
    return out
