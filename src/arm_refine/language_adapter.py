"""Apply a reviewed per-arm language plan, including inferred hand directions."""
import math
from dataclasses import replace
import numpy as np
import bpy
from .blender_adapter import snapshot,body_frame,capture_pose_channels,restore_pose_channels
from .contact_parts import select_contact_part,contact_axis
from .contact_adapter import apply_contact_part,hand_samples,target_frame
from .relational_adapter import apply_relational,restore,new_skin


def contact_directions(arm,meshes,member,part):
    side=member['side'].title();hf,_,_=hand_samples(arm,meshes,side,part)
    _,_,normal,_,_,axes=target_frame(arm,meshes,side,part);checks=[]
    for rule in member['directions']:
        axis=rule['axis'];value=rule['value']
        if axis=='fingers' and value=='down':actual=hf.col[0];expected=-axes[1];minimum=math.cos(math.radians(35))
        elif axis in ['palm','thumb','support_patch'] and value.startswith('toward_'):
            actual=hf.col[contact_axis(part)];expected=-normal;minimum=math.cos(math.radians(25))
        elif axis=='hand_back' and value=='forward' and part.target=='forehead_front':
            actual=-hf.col[1];expected=normal;minimum=math.cos(math.radians(25))
        else:raise ValueError('unsupported native contact direction check')
        dot=float(actual.dot(expected));checks.append({**rule,'alignment_dot':dot,'minimum_dot':minimum,'passed':dot>=minimum})
    return checks


def apply_language_plan(arm,meshes,plan,skin_check,profile,*,baseline_skin=None):
    original_channels=capture_pose_channels(arm)
    state={p.name:p.matrix_basis.copy() for p in arm.pose.bones};before=snapshot(arm);baseline=skin_check(arm) if baseline_skin is None else baseline_skin
    result={'plan':plan,'method':'per_arm_language_direction_fit_v2','accepted':False,'members':[],'trials':[],
            'initial_skin':baseline,'reasons':[],'release_allowed':False}
    if not plan['ready']:
        result.update(status='held',reasons=plan['reasons'],rollback_matrix_delta=0.);return result
    active=[m for m in plan['members'] if m['active']]
    engines={m['engine'] for m in active}
    try:
        if engines=={'relational'}:
            relational_plan={**plan,'members':[{'side':m['side'],'part':m['part']} for m in active]}
            native=apply_relational(arm,meshes,relational_plan,skin_check,profile,baseline_skin=baseline)
            result.update(native);result['plan']=plan;result['engine']='relational'
        elif engines=={'contact'}:
            paired=len(active)==2
            if paired and not all(m['action']=='chin_support' for m in active):
                result['reasons']=['unsupported_atomic_contact_relation']
            else:
                # Same candidate set for every selected case. No per-case tuning.
                fractions=[.26,.34] if paired else [None]
                for fraction in fractions:
                    restore(arm,state);members=[]
                    for member in active:
                        part=select_contact_part(member['action'],member['side'])
                        if paired:part=replace(part,id='language_chin_pair_member',lateral_fraction=fraction)
                        accepted=None;member_state={p.name:p.matrix_basis.copy() for p in arm.pose.bones}
                        for bias in [0.,.75,-.75]:
                            restore(arm,member_state)
                            native=apply_contact_part(arm,meshes,part,member['side'],skin_check,profile,pole_bias=bias,reuse_hand_shape=member.get('part',{}).get('reuse_existing_hand_shape',False))
                            result['trials'].append({'side':member['side'],'pole_bias':bias,'lateral_fraction':fraction,'accepted':native['accepted'],'reasons':native['reasons'],'metrics':native.get('metrics')})
                            if native['accepted']:
                                checks=contact_directions(arm,meshes,member,part)
                                native['direction_checks']=checks
                                if all(c['passed'] for c in checks):accepted=native;break
                                # Fitter acceptance cannot override semantic direction.
                                result['trials'][-1].update(accepted=False,reasons=['inferred_direction_failed'])
                                restore(arm,member_state)
                        if accepted is None:result['reasons']=[member['side']+': contact_or_direction_rejected'];break
                        members.append(accepted)
                    else:
                        skin=skin_check(arm);new=new_skin(skin,baseline)
                        if not new:
                            result.update(accepted=True,members=members,engine='contact',reasons=[],new_skin=[],final_skin=skin);break
                        result['reasons']=['new_combined_skin']
                if not result['accepted']:result['members']=members
        else:result['reasons']=['unsupported_mixed_engine_relation']
        if result['accepted']:
            after=snapshot(arm);sides=[m['side'].title() for m in active]
            allowed={n for n in before if any(n.split(':')[-1] in [s+t for t in ['Arm','ForeArm','Hand']] or n.split(':')[-1].startswith(s+'Hand') for s in sides)}
            delta=max(float(np.max(abs(after[n]-v))) for n,v in before.items() if n not in allowed)
            if delta>=2e-5:result.update(accepted=False,reasons=['protected_pose_changed'])
            result['protected_matrix_delta']=delta
    except Exception:
        restore_pose_channels(arm,original_channels);raise
    if not result['accepted']:
        result['rejected_pose_basis']={p.name:np.array(p.matrix_basis).tolist() for p in arm.pose.bones}
        restore_pose_channels(arm,original_channels);result['rollback_matrix_delta']=max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in before.items())
    result['status']='applied' if result['accepted'] else 'not_applied'
    return result
