"""Atomic paired chin support; a failed member restores both arms and fingers.

This does not authorize folded-arm application or certify interlocking hands.
"""
from dataclasses import replace
import numpy as np
import bpy
from .blender_adapter import snapshot
from .contact_parts import select_contact_part
from .contact_adapter import apply_contact_part


def apply_chin_pair(arm,meshes,skin_check,rig_profile):
    before=snapshot(arm);basis={p.name:p.matrix_basis.copy() for p in arm.pose.bones}
    baseline=skin_check(arm);depth={tuple(c['regions']):c['max_depth_mm'] for c in baseline['collisions']}
    part=replace(select_contact_part('chin_support','left'),id='chin_pair_member',
                 label='양손 턱 받침의 한쪽 지지면',lateral_fraction=.26)
    result={'method':'atomic_chin_pair_v1','paired':True,'accepted':False,'members':[],
            'initial_skin':baseline,'release_allowed':False,'reasons':[]}
    try:
        for side in ['left','right']:
            member=apply_contact_part(arm,meshes,part,side,skin_check,rig_profile)
            result['members'].append(member)
            if not member['accepted']:
                result['reasons']=[side+': '+r for r in member['reasons']];break
        else:
            skin=skin_check(arm)
            new=[c for c in skin['collisions'] if c['max_depth_mm']>depth.get(tuple(c['regions']),0)+2]
            result.update(accepted=not new,final_skin=skin,reasons=['new_pair_skin'] if new else [])
    except Exception:
        for p in arm.pose.bones:p.matrix_basis=basis[p.name]
        bpy.context.view_layer.update()
        raise
    if not result['accepted']:
        for p in arm.pose.bones:p.matrix_basis=basis[p.name]
        bpy.context.view_layer.update();after=snapshot(arm)
        result['rollback_matrix_delta']=max(float(np.max(abs(after[n]-v))) for n,v in before.items())
    return result
