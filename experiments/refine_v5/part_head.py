"""Independent head candidate with an explicit observable orientation mask."""

import numpy as np

from experiments.refine_v5.ab_common import project


HEAD_CAP_DEG = 15.
NECK_CAP_DEG = 8.


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('degenerate head axis')
    return vector / length


def head_axes(world, lookup):
    head, neck = world[lookup['Head']], world[lookup['Neck']]
    return dict(up=unit(head[:3, 1]), forward=unit(head[:3, 2]),
                neck=unit(head[:3, 3] - neck[:3, 3]))


def targets(case, world, lookup):
    goal = case['part_goal']['head']
    if goal is None:
        return {}
    axes = head_axes(world, lookup)
    if 'up_screen' in goal:
        point = world[lookup['Head'], :3, 3]
        tip = point + axes['up'] * (.20 * case['torso_world'])
        up = unit(np.diff(project(np.array([point, tip]), case['camera']), axis=0)[0])
        return {'screen_up': (up, np.asarray(goal['up_screen']))}
    baseline = {bone['name']: np.asarray(bone['matrix'])[:3, 3] for bone in case['bones']}
    forward = unit(np.asarray(case['body_forward']))
    up = unit(baseline['Spine2'] - baseline['Hips'])
    body = np.column_stack([unit(np.cross(up, forward)), up, forward])
    return {name: (axes[name], body @ direction) for name, direction in goal['axes_body'].items()}


def orientation_errors(case, world, names):
    lookup = {name: i for i, name in enumerate(names)}
    return {name: float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1., 1.))))
            for name, (a, b) in targets(case, world, lookup).items()}


def propose(case):
    from scipy.optimize import least_squares

    from experiments.refine_v5.depth_fit import candidate_row
    from experiments.refine_v5.goal_probe_v2 import fk, prepare
    from experiments.refine_v5.part_observations import validate_review

    goal = case['part_goal']['head']
    if goal is None or case['ownership_authority'] != 'offline_single_target_review':
        return None
    validate_review({'head': goal})
    names, lookup, parents, local, base = prepare(case)
    active = ['Head']
    caps = [HEAD_CAP_DEG]
    # A single projected tilt can constrain only rotation around the view axis.
    if 'up_screen' in goal:
        camera = case['camera']
        if camera['inverse_distance_torso'] != 0:
            raise ValueError('reviewed head tilt requires the fixed orthographic camera')
        view_axis = np.asarray(camera['rotation'])[2]
        directions = unit(np.linalg.solve(base[lookup['Head'], :3, :3], view_axis))[None, :, None]
    else:
        if 'neck' in goal['axes_body']:
            active.insert(0, 'Neck')
            caps.insert(0, NECK_CAP_DEG)
        directions = np.repeat(np.eye(3)[None], len(active), axis=0)
    ids = [lookup[name] for name in active]
    components = directions.shape[-1]
    box = np.repeat(np.deg2rad(caps) / np.sqrt(components), components)

    def world_pose(x):
        delta = np.einsum('nij,nj->ni', directions, x.reshape(len(active), components))
        return delta, fk(local, parents, ids, delta)

    def residual(x):
        _, world = world_pose(x)
        errors = np.concatenate([a - unit(b) for a, b in targets(case, world, lookup).values()])
        return np.r_[errors, .04 * x]

    zero = np.zeros(len(box))
    solution = least_squares(residual, zero, bounds=(-box, box), max_nfev=45)
    delta, world = world_pose(solution.x)
    before, after = (orientation_errors(case, pose, names) for pose in (base, world))
    if any(after[key] > before[key] + .05 for key in before) or sum(before.values()) - sum(after.values()) < .25:
        return None
    edits = dict(zip(active, np.rad2deg(delta).tolist(), strict=True))
    return candidate_row(case, 'head', names, world, edits,
                         profile_limits_deg=dict(zip(active, caps, strict=True)), before_deg=before, after_deg=after)
