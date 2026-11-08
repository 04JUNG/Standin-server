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
from experiments.refine_v5.hand_targets import HEAD_REGIONS, behind_head, build_hand_target, rear_arm_seed
from experiments.refine_v5.learning_data import digest, frame
from experiments.refine_v5.part_head_direction import HEAD_CAP
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
        if cue.get('anatomical_side') == 'uncertain' or cue['confidence'] == 'unknown':
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
        depth_fields = ('proximal_depth', 'distal_depth')
        depth_weight = [{'clear': 1., 'indirect': .45, 'unknown': 0.}[cue.get(field + '_evidence', 'clear')]
                        for field in depth_fields]
        depth = [DEPTH_BOUNDS.get(cue[field]) if weight else None
                 for field, weight in zip(depth_fields, depth_weight, strict=True)]
        height = HEIGHT_BOUNDS.get(cue['thigh_height']) if leg else None
        support = cue['support'] if leg else 'unknown'
        contact = cue['contact'] if not leg and strong else 'unknown'
        waist_surface = cue.get('waist_surface') if contact == 'waist' else None
        screen = {}
        camera = case['camera']
        registered = camera['status'] == 'fitted' and camera['anchor_rmse_px'] <= .27 * camera['rough_unit_px']
        if registered:
            for name, index, label in zip(chain[1:], indices[1:], ('joint', 'endpoint'), strict=True):
                point, score = np.asarray(seed['keypoints'][index], float), seed['scores'][index]
                state = visible[0 if label == 'joint' else 1]
                if not np.isfinite(point).all():
                    continue
                tolerance = max(5., camera['rough_unit_px'] * (.07 + .20 * (1. - min(1., score))))
                if cue[label] == 'visible' and state == 'visible' and score >= .46:
                    screen[name] = dict(target=point.tolist(), tolerance=tolerance)
                elif cue[label] == 'inferable' and state == 'partial' and score >= .60:
                    screen[name] = dict(target=point.tolist(), tolerance=1.6 * tolerance, weight=.35)
        relations = [r for r in cue.get('hand_relations', ()) if
                     (r['position'] != 'unknown' or r['contact'] != 'unknown')
                     and r.get('target_limb') != 'unknown'
                     and not (r.get('target_limb') == limb and r['target_region'] in ('wrist', 'hand', 'fingers'))]
        if bend is None and not any(depth) and height is None and not screen and contact == 'unknown' and not relations:
            statuses[limb] = 'no_actionable_cue'
            continue
        goals[limb] = dict(bend=bend, depth=depth, depth_weight=depth_weight, height=height, support=support,
                           contact=contact, waist_surface=waist_surface, hand_relations=relations,
                           anatomical_axes='attachment_screen' in cue,
                           screen=screen, weight=1. if strong else .65)
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

    @property
    def fits_wrist(self):
        return self.goal.get('waist_surface') in ('front', 'side') or any(
            r['contact'] == 'touching' and r['contact_surface'] != 'unknown'
            for r in self.goal.get('hand_relations', ()))


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
    limits = [caps[n] for n in chain[:2]]
    if any(map(behind_head, goal.get('hand_relations', ()))):
        limits = [max(limits[0], 110.), max(limits[1], 85.)]
    return Limb(chain, ids, goal, leg, hinge, bend, np.deg2rad(limits))


def fit(case, mesh, goals, *, caps=None, prior=None, head_plan=None):
    if not goals:
        return None
    names, lookup, parents, local, base = prepare(case)
    _, _, torso, basis = frame(case['bones'])
    camera = case['camera']
    scale = max(1., camera['rough_unit_px'])
    caps = JOINT_CAPS if caps is None else caps
    prior = prior or {}
    limbs = [make_limb(key, goal, lookup, base, basis, caps) for key, goal in goals.items()]
    active = [index for limb in limbs for index in (limb.ids if limb.fits_wrist else limb.ids[:2])]
    all_pts = base[:, :3, 3]
    body_up = basis[:, 1]
    world_up = np.array([0., 0., 1.])  # Native FBX world coordinates.
    chest_up = all_pts[lookup['Neck']] - all_pts[lookup['Spine1']]
    chest_forward = unit(np.cross(basis[:, 0], chest_up))
    hip_lateral = all_pts[lookup['LeftUpLeg']] - all_pts[lookup['RightUpLeg']]
    hips = base[lookup['Hips'], :3, :3]
    # The rig's hip-to-spine rest direction follows pelvic rotation.
    pelvis_up = hips @ local[lookup['Spine'], :3, 3]
    pelvis_forward = unit(np.cross(hip_lateral, pelvis_up))
    vertices, faces, weights = (mesh[key] for key in ('vertices', 'faces', 'weights'))
    dominant = weights.argmax(axis=1)
    torso_indices = [lookup[name] for name in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck')]
    torso_mask = (weights[:, torso_indices].sum(axis=1)[faces] > .75).all(axis=1)
    body_surface = Surface(vertices[faces[torso_mask]]) if torso_mask.any() else None
    patches, probes, relations = {}, {}, {}
    triangles = vertices[faces]
    face_weights = weights[faces]
    for limb in limbs:
        endpoint = limb.chain[2]
        arm_or_leg = [limb.ids[1]] + [i for i, name in enumerate(names) if name.startswith(endpoint)]
        indices = np.flatnonzero(np.isin(dominant, arm_or_leg))
        if len(indices) >= 12 and body_surface is not None:
            if limb.fits_wrist:
                hand = np.flatnonzero(np.isin(dominant, arm_or_leg[1:]))
                forearm = np.flatnonzero(dominant == limb.ids[1])
                indices = np.r_[subsample(hand, 220), subsample(forearm, 55)]
            else:
                indices = subsample(indices, 70)
            probes[limb.chain[0]] = SkinSample(vertices[indices], dominant[indices], base)
        contact = limb.goal['contact']
        observed = limb.goal.get('hand_relations', ())
        targets = []
        for relation in observed:
            target = build_hand_target(relation, lookup, base, triangles, face_weights, torso,
                                       basis[:, 0], chest_forward, pelvis_forward)
            if target is not None:
                targets.append(target)
        if contact != 'unknown' or targets:
            side = 'Left' if endpoint.startswith('Left') else 'Right'
            region = 'waist' if contact == 'clear_of_waist' else contact
            if region == 'unknown':
                region = 'head'  # Only the source hand samples are needed.
            waist_surface = limb.goal.get('waist_surface')
            if contact == 'waist' and waist_surface is not None:
                region = {'front': 'waist_front', 'side': 'waist_side',
                          'unknown': 'waist_unspecified'}[waist_surface]
            hand, palm, target = contact_faces(vertices, faces, weights, names, base, side, region)
            hand_probe = None
            if len(hand) and any(t.uses_whole_hand for t in targets):
                hand_vertices = subsample(np.unique(faces[hand]), 84)
                hand_probe = SkinSample(vertices[hand_vertices], dominant[hand_vertices], base)
            targets = [t for t in targets if hand_probe is not None or not t.uses_whole_hand]
            if len(palm):
                sample = subsample(vertices[faces[palm]].mean(axis=1), 30)
                palm_probe = SkinSample(sample, np.full(len(sample), limb.ids[-1]), base)
                if len(target) and contact != 'unknown' and not (
                    contact == 'head' and any(t.relation['target_region'] in HEAD_REGIONS for t in targets)
                ):
                    patches[limb.chain[0]] = Surface(vertices[faces[target]]), palm_probe
            if targets and (len(palm) or hand_probe is not None):
                if not len(palm):
                    targets = [t for t in targets if t.uses_whole_hand]
                relations[limb.chain[0]] = (targets, palm_probe if len(palm) else hand_probe, hand_probe)

    head_index = lookup['Head']
    head_seed_deg = (head_plan or {}).get('edits_deg', {}).get('Head')
    coupled = head_seed_deg is not None and any(
        t.relation['target_region'] in HEAD_REGIONS for targets, _, _ in relations.values() for t in targets)
    head_seed = np.radians(head_seed_deg) if coupled else None
    lower, upper, initial, slices = [], [], [], []
    wrist_cap = np.deg2rad(40.)
    for limb in limbs:
        start_index = len(initial)
        proximal, hinge_cap = limb.cap
        max_bend = np.deg2rad(150. if limb.is_leg else 145.)
        lo = max(-limb.base_bend, -hinge_cap)
        hi = max(lo + 1e-5, min(max_bend - limb.base_bend, hinge_cap))
        lower.extend([-proximal] * 3 + [lo])
        upper.extend([proximal] * 3 + [hi])
        bend = limb.goal['bend']
        initial.extend([0., 0., 0., float(outside(limb.base_bend, np.radians(bend)) * -1) if bend else 0.])

        if limb.fits_wrist:
            lower.extend([-wrist_cap] * 3)
            upper.extend([wrist_cap] * 3)
            initial.extend([0.] * 3)
        slices.append(slice(start_index, len(initial)))

    if coupled:
        cap = np.radians(head_plan.get('head_cap_deg', HEAD_CAP))
        lower.extend([-cap] * 3)
        upper.extend([cap] * 3)
        initial.extend(head_seed)
        active.append(head_index)

    def pose(parameters):
        edits = []
        for limb, section in zip(limbs, slices, strict=True):
            values = parameters[section]
            edits.extend([limit_vectors(values[:3], [limb.cap[0]])[0], values[3] * limb.hinge_local])
            if limb.fits_wrist:
                edits.append(limit_vectors(values[4:], [wrist_cap])[0])
        if coupled:
            edits.append(limit_vectors(parameters[-3:], [cap])[0])
        return np.asarray(edits), fk(local, parents, active, edits)

    def observations(world, limb):
        points = world[limb.ids, :3, 3]
        vectors = np.array([unit(vector) for vector in np.diff(points, axis=0)])
        forward, up = basis[:, 2], body_up
        if limb.goal.get('anatomical_axes'):
            forward = pelvis_forward if limb.is_leg else chest_forward
            up = world_up
        bend = float(np.degrees(np.arccos(segment_cos(points))))
        screen = dict(zip(limb.chain, project(points, camera).tolist(), strict=True))
        gap = None
        if limb.chain[0] in patches:
            surface, sample = patches[limb.chain[0]]
            distance, signed = surface.sample(sample(world))
            gap = np.quantile(distance, .25)
        measured = []
        if limb.chain[0] in relations:
            targets, palm, hand = relations[limb.chain[0]]
            palm_points = palm(world)
            hand_points = hand(world) if hand is not None else None
            for target in targets:
                depth, lateral, distance = target.measure(world, palm_points, hand_points)
                measured.append(dict(target_region=target.relation['target_region'],
                                     depth_m=depth, lateral_m=lateral, gap_m=distance))
        return dict(hand_targets=measured, bend_deg=bend, segment_forward=(vectors @ forward).tolist(),
                    proximal_height=float(vectors[0] @ up),
                    endpoint_height_change_torso=float((points[-1] - all_pts[limb.ids[-1]]) @ body_up / torso),
                    screen=screen, hand_gap_m=float(gap) if gap is not None else None)

    def terms(world, limb, *, include_safety, learned=None):
        goal = limb.goal
        values = observations(world, limb)
        out = []
        for name, screen in goal['screen'].items():
            difference = np.asarray(values['screen'][name]) - screen['target']
            margin = screen['tolerance']
            out.extend(1.65 * screen.get('weight', 1.) * (outside(difference, (-margin, margin)) + .14 * difference) / scale)
        if goal['bend']:
            out.append(1.45 * np.deg2rad(outside(values['bend_deg'], goal['bend'])))
        for value, interval, weight in zip(values['segment_forward'], goal['depth'],
                                           goal.get('depth_weight', (1., 1.)), strict=True):
            if interval:
                out.append(1.12 * weight * outside(value, interval))
        if goal['height']:
            out.append(1.0 * outside(values['proximal_height'], goal['height']))
        if goal['support'] == 'planted':
            out.append(3. * outside(values['endpoint_height_change_torso'], (-.045, .045)))
        gap = values['hand_gap_m']
        if gap is not None:
            interval = (.040, 10.) if goal['contact'] == 'clear_of_waist' else (0., .021)
            surface_known = goal.get('waist_surface') in ('front', 'side')
            if surface_known:
                interval = (.004, .013)
            out.append((4. if surface_known else 1.8) * outside(gap, interval) / torso)
        if limb.chain[0] in relations:
            targets, _, _ = relations[limb.chain[0]]
            for target, measured in zip(targets, values['hand_targets'], strict=True):
                relation = target.relation
                head_target = relation['target_region'] in HEAD_REGIONS
                depth, lateral, distance = (measured[k] for k in ('depth_m', 'lateral_m', 'gap_m'))
                position = relation['position']
                relation_weight = 8. if head_target else 1.2
                if position in ('front', 'back'):
                    sign = 1. if position == 'front' else -1.
                    out.append(relation_weight * max(0., .055 * torso - sign * depth) / torso)
                elif position == 'side':
                    out.extend(relation_weight * np.array([
                        max(0., .04 * torso - abs(lateral)), max(0., abs(depth) - .15 * torso)]) / torso)
                if relation['contact'] == 'touching' and not (
                    relation['target_region'] == 'waist' and goal.get('waist_surface') in ('front', 'side')
                ):
                    out.append((8. if head_target else 3.) * outside(distance, (.008, .018)) / torso)
                if relation.get('spacing') == 'near':
                    out.append(2. * max(0., distance / torso - .09))
                elif relation['contact'] == 'separated':
                    minimum = .09 * torso if target.uses_whole_hand else .014
                    out.append(3. * max(0., minimum - distance) / torso)
                if include_safety and head_target and limb.chain[0] in probes:
                    intrusion = target.intrusion(world, probes[limb.chain[0]](world))
                    out.extend(55. * np.sort(intrusion)[-35:] / torso / np.sqrt(35.))
        if learned:
            directions = np.asarray([unit(v) for v in np.diff(world[limb.ids, :3, 3], axis=0)]) @ basis
            for direction, wanted in zip(directions, learned, strict=True):
                if wanted is not None:
                    out.extend(.38 * (direction - wanted))
        if include_safety and limb.chain[0] in probes:
            distance, signed = body_surface.sample(probes[limb.chain[0]](world))
            intrusion = np.where(distance < .07, np.maximum(0., .007 - signed), 0.)
            count = 30 if limb.fits_wrist else 12
            weight = 12. if limb.fits_wrist else 5.
            out.extend(weight * np.sort(intrusion)[-count:] / torso / np.sqrt(count))
        return np.asarray(out) * goal['weight']

    def residual(parameters):
        edits, world = pose(parameters)
        head_error = .9 * (edits[-1] - head_seed) if coupled else []
        return np.r_[np.concatenate([terms(world, limb, include_safety=True, learned=prior.get(key))
                                     for key, limb in zip(goals, limbs, strict=True)]),
                     .085 * edits.ravel(), head_error]

    zero = np.clip(np.zeros(len(lower)), lower, upper)
    seed = np.clip(initial, lower, upper)
    start = min((zero, seed), key=lambda x: np.linalg.norm(residual(x)))
    rear_seed = seed.copy()
    seeded = False
    _, seed_world = pose(seed)
    for limb, section in zip(limbs, slices, strict=True):
        targets, _, _ = relations.get(limb.chain[0], ((), None, None))
        rear = next((target for target in targets if behind_head(target.relation)), None)
        if rear is None:
            continue
        outward = basis[:, 0] * (1. if limb.chain[0].startswith('Left') else -1.)
        wrist = rear.seed_point(seed_world, outward)
        rotation, bend = rear_arm_seed(base, limb.ids, wrist, outward, body_up)
        rear_seed[section.start:section.start + 3] = rotation
        rear_seed[section.start + 3] = bend - limb.base_bend
        seeded = True
    starts = (start, np.clip(rear_seed, lower, upper)) if seeded else (start,)
    sol = min((least_squares(residual, x, bounds=(lower, upper), max_nfev=135 if seeded else 105,
                             diff_step=1e-4) for x in starts), key=lambda result: result.cost)
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
                evaluations=int(sol.nfev), head_coupled=coupled, outcomes=outcomes)


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
