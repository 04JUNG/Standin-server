"""Opt-in native arm stage for a leg-first v4 transaction. No API default changes.

Call after leg changes have been fixed. No target coordinates are cached across
calls; every application queries the current armature and current knee skin.
"""
import copy
import numpy as np
from .relational_parts import plan_arms


def apply_arm_stage(arm,meshes,*,clauses,skin_check,rig_profile,skeleton_scores=None):
    from .blender_adapter import snapshot
    from .relational_adapter import apply_relational
    before=snapshot(arm)
    plan=plan_arms(clauses,skeleton_scores=skeleton_scores)
    receipt=apply_relational(arm,meshes,plan,skin_check,rig_profile)
    after=snapshot(arm)
    receipt.update(selected_parts=[{'side':m['side'],**m['part']} for m in plan['members']],
      changed_joints=[n for n,v in before.items() if float(np.max(abs(after[n]-v)))>1e-6],
      joint_matrices_before={n:m.tolist() for n,m in before.items()},
      joint_matrices_after={n:m.tolist() for n,m in after.items()},
      status='applied' if receipt['accepted'] else 'not_applied')
    return receipt


def check_arm_stage(arm,meshes,receipt,*,skin_check,rig_profile):
    """Recheck the final combined native state, including current knee surface."""
    from .blender_adapter import snapshot
    from .relational_adapter import measure_arm,reasons_for,new_skin
    if not receipt['accepted']:return {'accepted':False,'reasons':['arm_stage_not_applied']}
    current=snapshot(arm);reasons=[];members=[]
    protected=[]
    for name,m in receipt['joint_matrices_before'].items():
        short=name.split(':')[-1]
        movable=any(short in [s+t for t in ['Arm','ForeArm','Hand']] or short.startswith(s+'Hand') for s in ['Left','Right'])
        if not movable:protected.append(float(np.max(abs(current[name]-np.array(m)))))
    delta=max(protected,default=0.)
    if delta>=2e-5:reasons.append('non_arm_pose_changed_after_arm_stage')
    for member in receipt['members']:
        # The actual knee tree is rebuilt inside measure_arm on this call.
        m=measure_arm(arm,meshes,member['side'].title(),member['part'],rig_profile,member['lengths'],member['wrist_goal_world'],copy.deepcopy(member['contact']))
        mr=reasons_for(m);members.append({'side':member['side'],'metrics':m,'reasons':mr});reasons += [member['side']+': '+r for r in mr]
    new=new_skin(skin_check(arm),receipt['initial_skin'])
    if new:reasons.append('new_skin')
    return {'accepted':not reasons,'reasons':reasons,'protected_matrix_delta':delta,'members':members,'new_skin':new}
