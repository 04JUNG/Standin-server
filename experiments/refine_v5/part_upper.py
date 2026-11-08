"""Optional spine and clavicle ablation proposals from independent observations."""

import numpy as np
from scipy.optimize import least_squares

from experiments.refine_v5.ab_common import project
from experiments.refine_v5.depth_fit import candidate_row
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.part_head import orientation_errors, unit
from experiments.refine_v5.part_observations import validate_review


SPINE = ('Spine', 'Spine1', 'Spine2')
SPINE_CAP_DEG = 2.
CLAVICLE_CAP_DEG = 4.


def propose(case, *, spine, clavicle):
    upper = case['part_goal'].get('upper')
    if (not upper or case['camera']['status'] != 'fitted'
            or case['ownership_authority'] != 'offline_single_target_review'):
        return None
    validate_review({'upper': upper})
    camera = case['camera']
    names, lookup, parents, local, base = prepare(case)

    def project_heads(world, names):
        return project(world[[lookup[name] for name in names], :3, 3], camera)

    def chest_axis(world):
        hip, chest = project_heads(world, ['Hips', 'Spine2'])
        return unit(chest - hip)

    wanted_chest = upper.get('chest_up_screen')
    shoulders = upper.get('shoulders_px', {})
    active, limits = [], []
    if spine and wanted_chest is not None and np.linalg.norm(chest_axis(base) - wanted_chest) > .025:
        active.extend(SPINE)
        limits.extend([SPINE_CAP_DEG] * len(SPINE))
    if clavicle:
        for side, desired in shoulders.items():
            point = project_heads(base, [side + 'Arm'])[0]
            if np.linalg.norm(point - desired) >= 3.:
                active.append(side + 'Shoulder')
                limits.append(CLAVICLE_CAP_DEG)
    if not active:
        return None
    ids = [lookup[name] for name in active]
    box = np.repeat(np.deg2rad(limits) / np.sqrt(3), 3)
    screen_unit = camera['rough_unit_px']
    head_base = orientation_errors(case, base, names)
    distal = [lookup[name] for name in ('LeftHand', 'RightHand')]
    # Shoulder and chest goals are scored in all ablations. A limb position
    # tether limits unintended descendant movement of upper-body parents.
    def error(world):
        errors = []
        if wanted_chest is not None:
            errors.extend(chest_axis(world) - wanted_chest)
        for side, point in shoulders.items():
            errors.extend((project_heads(world, [side + 'Arm'])[0] - point) / screen_unit)
        return np.array(errors)

    def pose(x):
        return fk(local, parents, ids, x)

    def residual(x):
        world = pose(x)
        drift = (world[distal, :3, 3] - base[distal, :3, 3]) / case['torso_world']
        return np.r_[error(world), .12 * drift.ravel(), .18 * x]

    solution = least_squares(residual, np.zeros(len(box)), bounds=(-box, box), max_nfev=50)
    world = pose(solution.x)
    before, after = (float(np.linalg.norm(error(p))) for p in (base, world))
    if before - after < .002:
        return None
    if any(error > head_base[key] + .5 for key, error in orientation_errors(case, world, names).items()):
        return None
    edits = dict(zip(active, np.rad2deg(solution.x.reshape(-1, 3)).tolist(), strict=True))
    return candidate_row(case, 'upper', names, world, edits,
                         profile_limits_deg=dict(zip(active, limits, strict=True)),
                         before_residual=before, after_residual=after, requires_native_audit=True)
