"""Fit visible VLM limb goals together on one fixed B0 rig."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import load_json, project, save_json
from experiments.refine_v5.ab_geometry import contact_faces
from experiments.refine_v5.depth_fit import SkinSample, Surface, limit_vectors, subsample
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import digest, frame
from experiments.refine_v5.part_quality import segment_cos
from experiments.refine_v5.part_vlm_observe import CHAINS


DEPTH_BOUNDS = {'forward': (.22, 1.), 'backward': (-1., -.22), 'aligned': (-.22, .22)}
HEIGHT_BOUNDS = {'down': (-1., -.78), 'diagonal_down': (-.84, -.37),
                 'level': (-.30, .22), 'above': (.20, 1.)}
JOINT_CAPS = {name: cap for key, (chain, _) in CHAINS.items()
              for name, cap in zip(chain, (105., 115., 78.) if key.endswith('leg') else (90., 105.))}


def outside(value, bounds):
    return value - np.clip(value, *bounds)


def unit(vector):
    return vector / max(1e-8, np.linalg.norm(vector))


def compile_goals(case, observation, seed, visibility):
    goals, statuses = {}, {}
    if observation is None or observation['sides'] != 'resolved':
        return goals, dict.fromkeys(CHAINS, 'uncertain_anatomical_side')
    for limb, (chain, indices) in CHAINS.items():
        cue = observation[limb]
        if cue['confidence'] == 'unknown':
            statuses[limb] = 'uncertain_limb'
            continue
        side, kind = limb.split('_')
        leg = kind == 'leg'
        parts = ('thigh', 'shin') if leg else ('upper_arm', 'forearm')
        visible = [visibility[f'{side}_{part}'] for part in parts]
        if 'cropped' in visible or all(state == 'occluded' for state in visible):
            statuses[limb] = 'cropped_or_fully_occluded'
            continue
        limit = 150. if leg else 145.
        low, high = cue['bend_low_deg'], cue['bend_high_deg']
        bend = [low, min(high, limit)] if low is not None and high is not None and 0 <= low < min(high, limit) else None
        strong = cue['confidence'] == 'strong'
        depth = [DEPTH_BOUNDS.get(cue[field]) for field in ('proximal_depth', 'distal_depth')]
        height = HEIGHT_BOUNDS.get(cue['thigh_height']) if leg else None
        support = cue['support'] if leg else 'unknown'
        contact = cue['contact'] if not leg and strong else 'unknown'
        screen = {}
        camera = case['camera']
        registered = camera['status'] == 'fitted' and camera['anchor_rmse_px'] <= .27 * camera['rough_unit_px']
        if registered:
            for name, index, label in zip(chain[1:], indices[1:], ('joint', 'endpoint'), strict=True):
                point, score = np.asarray(seed['keypoints'][index], float), seed['scores'][index]
                if cue[label] == 'visible' and visible[0 if label == 'joint' else 1] == 'visible' and np.isfinite(point).all() and score >= .46:
                    tolerance = max(5., camera['rough_unit_px'] * (.07 + .20 * (1. - min(1., score))))
                    screen[name] = dict(target=point.tolist(), tolerance=tolerance)
        if bend is None and not any(depth) and height is None and not screen and contact == 'unknown':
            statuses[limb] = 'no_actionable_cue'
            continue
        goals[limb] = dict(bend=bend, depth=depth, height=height, support=support,
                           contact=contact, screen=screen, weight=1. if strong else .65)
        statuses[limb] = 'goal'
    return goals, statuses


@dataclass
class Limb:
    chain: tuple
    ids: list
    goal: dict
    is_leg: bool
    hinge_local: np.ndarray
    base_bend: float
    cap: np.ndarray


def make_limb(key, goal, lookup, base, basis, caps):
    chain = CHAINS[key][0]
    leg = key.endswith('leg')
    ids = [lookup[name] for name in chain]
    points = base[ids, :3, 3]
    first, second = np.diff(points, axis=0)
    bend = float(np.arccos(segment_cos(points)))
    axis = unit(np.cross(first, second))
    if bend < np.deg2rad(7.) or np.linalg.norm(axis) < 1e-6:
        # Near extension, use the natural knee/elbow plane in the B0 body frame.
        preferred = (-1. if leg else 1.) * basis[:, 2]
        axis = unit(np.cross(first, preferred))
        if np.linalg.norm(axis) < 1e-6:
            axis = unit(np.cross(first, basis[:, 1]))
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError('indeterminate hinge')
    hinge = unit(np.linalg.solve(base[ids[1], :3, :3], axis))
    return Limb(chain, ids, goal, leg, hinge, bend, np.deg2rad([caps[n] for n in chain[:2]]))


def fit(case, mesh, goals, *, caps=None, prior=None):
    if not goals:
        return None
    names, lookup, parents, local, base = prepare(case)
    _, _, torso, basis = frame(case['bones'])
    camera = case['camera']
    scale = max(1., camera['rough_unit_px'])
    caps = JOINT_CAPS if caps is None else caps
    prior = prior or {}
    limbs = [make_limb(key, goal, lookup, base, basis, caps) for key, goal in goals.items()]
    active = [index for limb in limbs for index in limb.ids[:2]]
    all_pts = base[:, :3, 3]
    body_up = basis[:, 1]
    vertices, faces, weights = (mesh[key] for key in ('vertices', 'faces', 'weights'))
    dominant = weights.argmax(axis=1)
    torso_indices = [lookup[name] for name in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck')]
    torso_mask = (weights[:, torso_indices].sum(axis=1)[faces] > .75).all(axis=1)
    body_surface = Surface(vertices[faces[torso_mask]]) if torso_mask.any() else None
    patches, probes = {}, {}
    for limb in limbs:
        endpoint = limb.chain[2]
        arm_or_leg = [limb.ids[1]] + [i for i, name in enumerate(names) if name.startswith(endpoint)]
        indices = np.flatnonzero(np.isin(dominant, arm_or_leg))
        if len(indices) >= 12 and body_surface is not None:
            indices = subsample(indices, 70)
            probes[limb.chain[0]] = SkinSample(vertices[indices], dominant[indices], base)
        contact = limb.goal['contact']
        if contact != 'unknown':
            side = 'Left' if endpoint.startswith('Left') else 'Right'
            region = 'waist' if contact == 'clear_of_waist' else contact
            _, palm, target = contact_faces(vertices, faces, weights, names, base, side, region)
            if len(palm) and len(target):
                sample = subsample(vertices[faces[palm]].mean(axis=1), 30)
                patches[limb.chain[0]] = Surface(vertices[faces[target]]), SkinSample(
                    sample, np.full(len(sample), limb.ids[-1]), base)

    lower, upper, initial = [], [], []
    for limb in limbs:
        proximal, hinge_cap = limb.cap
        max_bend = np.deg2rad(150. if limb.is_leg else 145.)
        lo = max(-limb.base_bend, -hinge_cap)
        hi = max(lo + 1e-5, min(max_bend - limb.base_bend, hinge_cap))
        lower.extend([-proximal] * 3 + [lo])
        upper.extend([proximal] * 3 + [hi])
        bend = limb.goal['bend']
        initial.extend([0., 0., 0., float(outside(limb.base_bend, np.radians(bend)) * -1) if bend else 0.])

    def pose(parameters):
        edits = []
        for limb, values in zip(limbs, parameters.reshape(-1, 4), strict=True):
            edits.extend([limit_vectors(values[:3], [limb.cap[0]])[0], values[3] * limb.hinge_local])
        return np.asarray(edits), fk(local, parents, active, edits)

    def observations(world, limb):
        points = world[limb.ids, :3, 3]
        axes = np.array([unit(vector) for vector in np.diff(points, axis=0)]) @ basis
        bend = float(np.degrees(np.arccos(segment_cos(points))))
        screen = dict(zip(limb.chain, project(points, camera).tolist(), strict=True))
        gap = None
        if limb.chain[0] in patches:
            surface, sample = patches[limb.chain[0]]
            distance, signed = surface.sample(sample(world))
            gap = np.quantile(distance, .25)
        return dict(bend_deg=bend, segment_forward=axes[:, 2].tolist(), proximal_height=float(axes[0, 1]),
                    endpoint_height_change_torso=float((points[-1] - all_pts[limb.ids[-1]]) @ body_up / torso),
                    screen=screen, hand_gap_m=float(gap) if gap is not None else None)

    def terms(world, limb, *, include_safety, learned=None):
        goal = limb.goal
        values = observations(world, limb)
        out = []
        for name, screen in goal['screen'].items():
            difference = np.asarray(values['screen'][name]) - screen['target']
            margin = screen['tolerance']
            out.extend(1.65 * (outside(difference, (-margin, margin)) + .14 * difference) / scale)
        if goal['bend']:
            out.append(1.45 * np.deg2rad(outside(values['bend_deg'], goal['bend'])))
        for value, interval in zip(values['segment_forward'], goal['depth'], strict=True):
            if interval:
                out.append(1.12 * outside(value, interval))
        if goal['height']:
            out.append(1.0 * outside(values['proximal_height'], goal['height']))
        if goal['support'] == 'planted':
            out.append(3. * outside(values['endpoint_height_change_torso'], (-.045, .045)))
        gap = values['hand_gap_m']
        if gap is not None:
            interval = (.040, 10.) if goal['contact'] == 'clear_of_waist' else (0., .021)
            out.append(1.8 * outside(gap, interval) / torso)
        if learned:
            directions = np.asarray([unit(v) for v in np.diff(world[limb.ids, :3, 3], axis=0)]) @ basis
            for direction, wanted in zip(directions, learned, strict=True):
                if wanted is not None:
                    out.extend(.38 * (direction - wanted))
        if include_safety and limb.chain[0] in probes:
            distance, signed = body_surface.sample(probes[limb.chain[0]](world))
            intrusion = np.where(distance < .07, np.maximum(0., .007 - signed), 0.)
            out.extend(5. * np.sort(intrusion)[-12:] / torso / np.sqrt(12.))
        return np.asarray(out) * goal['weight']

    def residual(parameters):
        edits, world = pose(parameters)
        return np.r_[np.concatenate([terms(world, limb, include_safety=True, learned=prior.get(key))
                                     for key, limb in zip(goals, limbs, strict=True)]),
                     .085 * edits.ravel()]

    zero = np.clip(np.zeros(len(lower)), lower, upper)
    seed = np.clip(initial, lower, upper)
    # Initialize one candidate, then run the optimizer once.
    start = min((zero, seed), key=lambda x: np.linalg.norm(residual(x)))
    sol = least_squares(residual, start, bounds=(lower, upper), max_nfev=105, diff_step=1e-4)
    edits, world = pose(sol.x)
    outcomes = {}
    for key, limb in zip(goals, limbs, strict=True):
        before = observations(base, limb)
        after = observations(world, limb)
        outcomes[key] = dict(before=before, after=after,
                             loss_before=float(np.linalg.norm(terms(base, limb, include_safety=False))),
                             loss_after=float(np.linalg.norm(terms(world, limb, include_safety=False))))
    before_loss = np.linalg.norm([row['loss_before'] for row in outcomes.values()])
    after_loss = np.linalg.norm([row['loss_after'] for row in outcomes.values()])
    if before_loss - after_loss < .035 or np.max(np.linalg.norm(edits, axis=1)) < np.deg2rad(2.):
        return dict(status='insufficient_goal_gain', outcomes=outcomes)
    rotations = dict(zip((names[i] for i in active), edits, strict=True))
    for limb in limbs:
        if limb.goal['support'] == 'planted':
            foot = limb.ids[-1]
            delta = Rotation.from_matrix(np.linalg.solve(world[foot, :3, :3], base[foot, :3, :3])).as_rotvec()
            rotations[names[foot]] = limit_vectors(delta, [np.deg2rad(caps[names[foot]])])[0]
    world = fk(local, parents, [lookup[name] for name in rotations], list(rotations.values()))
    for key, limb in zip(goals, limbs, strict=True):
        outcomes[key]['after'] = observations(world, limb)
    return dict(status='proposal', source_sha256=case['source_sha256'],
                edits_deg={name: np.rad2deg(value).tolist() for name, value in rotations.items()},
                expected_heads={name: matrix[:3, 3].tolist() for name, matrix in zip(names, world, strict=True)},
                evaluations=int(sol.nfev), outcomes=outcomes)


def run(root, seeds_path):
    seeds = {row['case_id']: row for row in load_json(seeds_path)}
    repo = Path(__file__).resolve().parents[2]
    reviews = {row['case_id']: row['part_visibility'] for row in
               load_json(repo / 'evaluation/refine_v5/visual_review_v2.json')['records']}
    for entry in load_json(root / 'manifest.json')['cases']:
        cid = entry['case_id']
        folder = root / cid
        case = load_json(folder / 'input.json')
        response = load_json(folder / 'vlm-response.json')
        seed = seeds[cid]
        if (case != entry or response['case_id'] != cid or
                response['rough_sha256'] != case['image_sha256'] or seed['image_sha256'] != case['image_sha256'] or
                digest(case['image']) != case['image_sha256'] or digest(case['source']) != case['source_sha256']):
            raise ValueError(f'{cid}: proposal inputs differ from frozen manifest')
        goals, statuses = compile_goals(case, response['goals'], seed, reviews[cid])
        with np.load(root / f'{cid}-B0.npz') as data:
            mesh = {key: data[key] for key in ('vertices', 'faces', 'weights')}
        plan = fit(case, mesh, goals)
        save_json(folder / 'vlm-plan.json', dict(
            input_sha256=digest(folder / 'input.json'), response_sha256=digest(folder / 'vlm-response.json'),
            goals=goals, statuses=statuses, result=plan))
        print('LIMB_FIT', cid, len(goals), plan['status'] if plan else 'no_goals', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.seeds)
