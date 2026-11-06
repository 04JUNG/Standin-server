"""Read-only native leg state and local straight-knee candidate, Blender only.

No ground/table is inferred from the lowest mesh vertex. This candidate path
does not grant automatic application or call any external model/service.
"""
import hashlib
import json
import math
import numpy as np

from src.arm_refine.blender_adapter import body_frame
from src.refine_v4.pipeline import Stage, snapshot, run_refine_v4, skin_nonregression
from src.refine_v4.leg_adapter import extract_native_leg_part, apply_native_leg_part
from src.refine_v4.skin_collision import inspect_self


def unit(value):
    vector = np.asarray(value, float)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length < 1e-8:
        raise ValueError('degenerate native direction')
    return vector / length


def native_state(arm, meshes):
    initial = snapshot(arm)
    bones = {b.name.split(':')[-1]: b for b in arm.pose.bones}
    points = {name: np.array(arm.matrix_world @ bone.head) for name, bone in bones.items()}
    basis = body_frame(arm)
    up = basis[:, 1]
    shoulders = (points['LeftArm'] + points['RightArm']) / 2
    hips = (points['LeftUpLeg'] + points['RightUpLeg']) / 2
    scale = float(np.linalg.norm(shoulders - hips))
    if scale < .05:
        raise ValueError('native torso scale unavailable')
    out = {'schema': 'standin.native-leg-state.v1', 'legs': {}, 'torso_scale_m': scale,
           'body_up': up.tolist(), 'ground_plane': None, 'table_plane': None,
           'support_geometry_available': False, 'support_certified': False}
    for side in ('left', 'right'):
        prefix = side.title()
        hip, knee, ankle, toe = [points[prefix + role] for role in ('UpLeg', 'Leg', 'Foot', 'ToeBase')]
        other = points[('Right' if side == 'left' else 'Left') + 'Leg']
        out['legs'][side] = {'valid': True,
            'flexion_deg': float(np.degrees(np.arccos(np.clip(unit(knee - hip) @ unit(ankle - knee), -1, 1)))),
            'knee_above_other_torso': float((knee - other) @ up / scale),
            'knee_height_hip_torso': float((knee - hips) @ up / scale),
            'ankle_height_hip_torso': float((ankle - hips) @ up / scale),
            'hip_world': hip.tolist(), 'knee_world': knee.tolist(), 'ankle_world': ankle.tolist(),
            'toe_world': toe.tolist(), 'world_up_knee_z': float(knee[2]),
            'world_up_ankle_z': float(ankle[2]), 'world_up_toe_z': float(toe[2])}
    error = max(float(np.max(abs(snapshot(arm)[name] - value))) for name, value in initial.items())
    if error != 0:
        raise ValueError('read-only measurement changed native pose')
    out['read_only_matrix_delta'] = error
    return out


def straight_candidate(arm, meshes, *, sides, source_sha256, evidence, target_flexion_deg=15.):
    """Reuse native direction transport and atomic guards, without donor answers.

    The candidate is constructed only from current native directions + the
    generic straight-leg tolerance. There are no case-specific coordinates,
    evaluated fixes, target-library answers or surface plane guesses.
    """
    if not sides or not set(sides) <= {'left', 'right'} or len(set(sides)) != len(sides):
        raise ValueError('explicit unique anatomical sides required')
    if not 0 < target_flexion_deg < 25:
        raise ValueError('generic straight-knee candidate angle required')
    plan = {'schema': 'standin.local-leg-shape-candidate.v1', 'sides': sides,
            'selection': 'both' if len(sides) == 2 else sides[0],
            'intent': {'crossing': None, 'top_side': None, 'cross_region': 'unspecified',
                       'posture': 'unspecified', 'cross_style': 'unspecified'},
            'evidence': {'source': 'cached_vlm_explicit_text', 'reference': evidence},
            'automatic_image_extraction': False, 'auto_apply_allowed': False}
    part = extract_native_leg_part(arm, plan, source_sha256=source_sha256)
    for side in sides:
        rays = part['directions_pelvis'][side]
        thigh, calf = unit(rays[0]), unit(rays[1])
        angle = float(np.arccos(np.clip(thigh @ calf, -1, 1)))
        if angle <= math.radians(25.):
            raise ValueError('already-straight leg cannot enter corrective candidate')
        target = math.radians(target_flexion_deg)
        pole = unit(calf - thigh * float(thigh @ calf))
        rays[1] = (thigh * math.cos(target) + pole * math.sin(target)).tolist()
    part['id'] = hashlib.sha256(json.dumps({k: v for k, v in part.items() if k != 'id'},
                                          sort_keys=True, allow_nan=False).encode()).hexdigest()
    before_skin = inspect_self(arm)
    stage = Stage('legs', tuple(sides), lambda a, m: apply_native_leg_part(a, m, part, plan, skin_check=inspect_self))
    result = run_refine_v4(arm, meshes, leg_stage=stage,
        final_check=lambda a, m, receipts: skin_nonregression(before_skin, inspect_self(a)))
    result.update(candidate_only=True, auto_apply_allowed=False, support_verified=False,
                  support_hold_reason='ground_and_table_geometry_not_provided',
                  target_flexion_deg=target_flexion_deg, donor_asset_used=False,
                  source='current_native_directions_and_generic_straight_leg_goal')
    return result
