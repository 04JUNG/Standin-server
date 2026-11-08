"""Fit the leading single lifted-knee profile from VLM regions and rig geometry."""

import argparse
from pathlib import Path
import re
import shutil

import numpy as np

from experiments.refine_v5.ab_common import load_json, project, save_json
from experiments.refine_v5.goal_probe_v2 import prepare
from experiments.refine_v5.learning_data import digest, frame
from experiments.refine_v5.part_grounded_observe import COLUMNS, ROWS
from experiments.refine_v5.part_single_fit import fit


def region(cell, size):
    if not isinstance(cell, str) or not re.fullmatch(r'[A-L](0[1-9]|1[0-9]|2[0-4])', cell):
        raise ValueError('no usable visual region')
    lower = np.array([ord(cell[0]) - ord('A'), int(cell[1:]) - 1])
    step = np.asarray(size) / [COLUMNS, ROWS]
    return (lower + .5) * step, step


def bend_limits(cue, maximum):
    low, high = cue['bend_min'], cue['bend_max']
    if low is None or high is None or not 0 <= low < high <= maximum:
        raise ValueError('no usable bend estimate')
    return [low, high]


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('no reliable segment direction')
    return vector / length


def direction(screen, length_px, sign, camera, body, pixel_uncertainty):
    # Recover the two orthographic depth signs from the projected length and
    # pick the one consistent with the observed direction in the body frame.
    xy = screen / length_px
    ratio = np.linalg.norm(xy)
    if ratio >= .96:
        # A region pair longer than the rig bone cannot resolve depth at its
        # centers. A one-cell error budget can support a shorter projection.
        ratio_limit = float(np.clip(1. - pixel_uncertainty / length_px, .65, .96))
        xy *= ratio_limit / ratio
    depth = np.sqrt(1. - xy @ xy)
    alternatives = np.array([[*xy, depth], [*xy, -depth]]) @ camera @ body
    result = max(alternatives, key=lambda value: sign * value[2])
    if sign * result[2] <= .10:
        raise ValueError('projected segment conflicts with depth cue')
    return result


def compile_goals(case, cue, limb_cues, seed):
    if not cue or cue['torso'] != 'front' or case['seed_view'] not in ('front', 'three_quarter'):
        raise ValueError('front-facing anatomical assignment is unverified')
    if case['camera']['status'] != 'fitted' or case['camera']['inverse_distance_torso'] != 0:
        raise ValueError('a fitted orthographic camera is required')
    raised = cue['raised']
    if raised['hip_side'] != 'image_left':
        raise ValueError('right-knee specialist profile is unsupported by this observation')
    if raised['thigh_depth'] != 'forward' or raised['shin_depth'] != 'backward':
        raise ValueError('forward lift and backward fold are unverified')
    names, lookup, _, _, world = prepare(case)
    _, _, torso, body = frame(case['bones'])
    camera = case['camera']
    pixels_per_m = camera['pixel_scale'] / camera['torso_world']
    camera_axes = np.asarray(camera['rotation'])
    size = case['dimensions']
    knee, size_px = region(raised['knee'], size)
    ankle, _ = region(raised['ankle'], size)
    toe, _ = region(raised['toe'], size)
    other, _ = region(raised['other_ankle'], size)
    leg_nodes = world[[lookup[name] for name in ('RightUpLeg', 'RightLeg', 'RightFoot')], :3, 3]
    hip = project(leg_nodes[:1], camera)[0]
    if not hip[1] < knee[1] < ankle[1] or other[1] - ankle[1] < 1.5 * size_px[1]:
        raise ValueError('image regions do not verify the raised leg trace')
    thigh_len, shin_len = np.linalg.norm(np.diff(leg_nodes, axis=0), axis=1)
    thigh = direction(knee - hip, thigh_len * pixels_per_m, 1, camera_axes, body, size_px[1])
    shin = direction(ankle - knee, shin_len * pixels_per_m, -1, camera_axes, body, size_px[1])
    # When the model's visual bend conflicts with the independently lifted
    # directions, keep the bounded geometric angle and record the conflict.
    visual_bend = bend_limits(raised, 150.)
    geometric_bend = float(np.degrees(np.arccos(np.clip(thigh @ shin, -1., 1.))))
    geometric_range = [max(0., geometric_bend - 20.), min(145., geometric_bend + 20.)]
    if visual_bend[1] >= geometric_range[0] and visual_bend[0] <= geometric_range[1]:
        flex = [max(visual_bend[0], geometric_range[0]), min(visual_bend[1], geometric_range[1])]
    else:
        flex = geometric_range
    if toe[1] < ankle[1]:
        # No usable visible toe direction. Use the source foot as a prior.
        foot = project(world[[lookup[n] for n in ('RightFoot', 'RightToeBase')], :3, 3], camera)
        toe_direction = np.diff(foot, axis=0)[0]
    else:
        toe_direction = toe - ankle
    up = thigh[1] * thigh_len / torso
    bounds = lambda center: np.array([center - .75 * size_px, center + .75 * size_px]).tolist()
    goals = dict(
        right_leg=dict(pixels={'RightLeg': bounds(knee), 'RightFoot': bounds(ankle)}, bend_deg=flex,
                       directions=[thigh.tolist(), shin.tolist()], foot_screen=unit(toe_direction).tolist(),
                       height_torso=[up - .45, up + .25],
                       shin_forward_max=min(-.15, .80 * shin[2]),
                       forward_min_torso=max(0., thigh[2] * thigh_len / torso - .25),
                       initial_bend_deg=float(np.mean(flex))),
    )
    # The arm observation is a separate inference. Bind it to the detector's
    # projected left chain before letting a moderate separation cue move skin.
    arm = limb_cues.get('left_arm', {}) if limb_cues else {}
    if (arm.get('contact') == 'clear_of_waist' and arm.get('confidence') in ('strong', 'moderate')
            and arm['joint'] == arm['endpoint'] == 'visible'):
        pixels = np.asarray(seed['keypoints'], float)[[5, 7, 9]]
        scores = np.asarray(seed['scores'], float)[[5, 7, 9]]
        shoulder, elbow, wrist = pixels
        if (np.isfinite(pixels).all() and scores.min() >= .50
                and shoulder[0] > hip[0] and elbow[1] < wrist[1] and wrist[0] > hip[0]):
            arm_bend = bend_limits(dict(bend_min=arm['bend_low_deg'], bend_max=arm['bend_high_deg']), 45.)
            upper, lower = np.diff(pixels, axis=0)
            projected = float(np.degrees(np.arccos(np.clip(unit(upper) @ unit(lower), -1., 1.))))
            if arm_bend[0] < projected + 14.:
                arm_bend[1] = min(arm_bend[1], projected + 14.)
                goals['left_arm'] = dict(pixels={'LeftForeArm': bounds(elbow), 'LeftHand': bounds(wrist)},
                                         bend_deg=arm_bend, clearance_m=.09 * torso, preserve_camera_depth=True)
    return goals, dict(visual_bend=visual_bend, geometric_bend=geometric_bend,
                       selected_bend=flex, ground_arm_side=cue['separated_arm']['shoulder_side'],
                       arm_basis='VLM separation corroborated by detected chain' if 'left_arm' in goals else None)


def run(root, seed_path, limb_path):
    case = load_json(root / 'input.json')
    response_path = root / 'grounded-response.json'
    response = load_json(response_path)
    if (case['case_id'] != response['case_id'] or digest(case['source']) != case['source_sha256']
            or digest(case['image']) != case['image_sha256'] or response['rough_sha256'] != case['image_sha256']):
        raise ValueError('frozen observation inputs changed')
    limb_response = load_json(limb_path)
    seed = next(row for row in load_json(seed_path) if row['case_id'] == case['case_id'])
    if (limb_response['rough_sha256'] != case['image_sha256'] or limb_response['case_id'] != case['case_id']
            or seed['image_sha256'] != case['image_sha256']):
        raise ValueError('additional observations refer to another rough')
    goals, plan, reason, diagnostics = None, None, None, None
    try:
        goals, diagnostics = compile_goals(case, response['goals'], limb_response['goals'], seed)
    except ValueError as exc:
        reason = str(exc)
    if goals:
        with np.load(root / 'base.npz') as mesh:
            plan = fit(case, mesh, goals)
    destination = root / 'grounded'
    destination.mkdir(exist_ok=False)
    frozen_limb = root / 'limb-response.json'
    if limb_path.resolve() != frozen_limb.resolve():
        shutil.copyfile(limb_path, frozen_limb)
    save_json(destination / 'plan.json', dict(input_sha256=digest(root / 'input.json'),
              response_sha256=digest(response_path), limb_response_sha256=digest(frozen_limb),
              result=plan, reason=reason, diagnostics=diagnostics))
    print('GROUNDED', case['case_id'], plan['after'] if plan else reason, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--limb-response', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.seeds, args.limb_response)
