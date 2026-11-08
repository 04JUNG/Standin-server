"""Produce paired automatic and automatic-plus-learned plans without validation GT."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import digest, frame, source_camera
from experiments.refine_v5.original_seed_oracle import CAPS
from experiments.refine_v5.original_seed_prior import checked_seed, predict
from experiments.refine_v5.part_head_direction import fit as fit_head
from experiments.refine_v5.part_single_batch import registration
from experiments.refine_v5.part_vlm_fit import compile_goals, fit


LIMB_CAPS = {side + name: value for side in ('Left', 'Right') for name, value in CAPS.items()}
HEAD_RANGES = {
    'yaw_deg': ('yaw', {'left_profile': [-108., -63.], 'left_three_quarter': [-63., -19.],
                        'frontal': [-20., 20.], 'right_three_quarter': [19., 63.],
                        'right_profile': [63., 108.]}),
    'pitch_deg': ('pitch', {'up': [-62., -24.], 'slightly_up': [-30., -8.], 'level': [-12., 12.],
                           'slightly_down': [8., 30.], 'down': [24., 62.]}),
    'roll_deg': ('lean', {'left': [-45., -9.], 'upright': [-12., 12.], 'right': [9., 45.]}),
}


def bind_leg_roles(cues, seed, size):
    """Check role-to-hip assignment using screen side, independently of shoe position."""
    if cues['sides'] != 'resolved':
        return 'unresolved_anatomy', None
    role = {side: cues[side + '_leg']['support'] for side in ('left', 'right')}
    raised, support = cues['raised_hip_screen'], cues['support_hip_screen']
    if raised == 'unknown':
        return ('raise_unverified' if 'lifted' in role.values() else 'no_unique_raise'), None
    if (support == 'unknown' or raised == support
            or cues['torso_view'] not in ('facing_camera', 'facing_away')):
        return 'screen_roles_ambiguous', None
    points = np.asarray(seed['keypoints'], float)
    scores = np.asarray(seed['scores'], float)
    left_is_screen_right = points[11, 0] > points[12, 0]
    expected = cues['torso_view'] == 'facing_camera'
    if min(scores[11:13]) < .40 or abs(points[11, 0] - points[12, 0]) < .025 * size[0] or left_is_screen_right != expected:
        return 'detector_hip_assignment_ambiguous', None
    raised_side = 'left' if (raised == 'image_right') == expected else 'right'
    support_side = 'right' if raised_side == 'left' else 'left'
    if role[raised_side] != 'lifted' or role[support_side] != 'planted':
        return 'vlm_roles_disagree', None
    return 'screen_roles_agree', support_side


def visibility(seed, cues, size):
    # Runtime evidence only. The P0 human/analyst masks are reserved for evaluation.
    observed = {}
    kp, score = np.asarray(seed['keypoints'], float), np.asarray(seed['scores'], float)
    for side, limb, parts, indices in (
        ('left', 'arm', ('upper_arm', 'forearm'), (5, 7, 9)),
        ('right', 'arm', ('upper_arm', 'forearm'), (6, 8, 10)),
        ('left', 'leg', ('thigh', 'shin'), (11, 13, 15)),
        ('right', 'leg', ('thigh', 'shin'), (12, 14, 16)),
    ):
        cue = cues[f'{side}_{limb}']
        for part, end, tag in zip(parts, indices[1:], ('joint', 'endpoint'), strict=True):
            inside = np.all(kp[end] >= 0) and np.all(kp[end] < np.asarray(size))
            if not inside:
                label = 'cropped'
            elif cue[tag] == 'visible' and score[end] >= .40:
                label = 'visible'
            else:
                label = 'partial'
            observed[f'{side}_{part}'] = label
    return observed


def compile_observation(case, cue, seed):
    cue = deepcopy(cue)
    role_status, support_side = bind_leg_roles(cue, seed, case['dimensions'])
    automatic_visibility = visibility(seed, cue, case['dimensions'])
    goals, statuses = compile_goals(case, cue, seed, automatic_visibility)
    for side in ('left', 'right'):
        name = side + '_leg'
        if role_status not in ('no_unique_raise', 'screen_roles_agree') or side == support_side:
            goals.pop(name, None)
            statuses[name] = 'locked_support' if side == support_side else role_status
        if name in goals and not goals[name]['screen']:
            goals.pop(name)
            statuses[name] = 'no_corroborating_leg_point'
        arm = side + '_arm'
        goal = goals.get(arm)
        if (goal and cue[arm]['contact'] == 'clear_of_waist' and goal['contact'] == 'unknown'
                and len(goal['screen']) == 2):
            # An independently detected wrist/forearm can corroborate a moderate
            # separation cue. Contact-seeking remains strong-only.
            goal['contact'] = 'clear_of_waist'
    return goals, statuses, role_status


def head_goal(cue, case, seed):
    if (case['camera']['status'] != 'fitted' or max(seed['scores'][:5]) < .50
            or cue['torso_view'] == 'facing_away' or not cue.get('head')):
        return {}
    return {axis: intervals[cue['head'][field]]
            for axis, (field, intervals) in HEAD_RANGES.items() if cue['head'][field] in intervals}


def compatible_prior(goals, predictions):
    cleaned = {}
    for name, goal in goals.items():
        vectors = []
        for value, interval in zip(predictions[name], goal['depth'], strict=True):
            if value is not None and interval:
                if (interval[0] > 0 and value[2] < -.10) or (interval[1] < 0 and value[2] > .10):
                    value = None
            vectors.append(value)
        cleaned[name] = vectors
    return cleaned


def compose(source, body_plan, head_plan):
    if body_plan and body_plan['status'] == 'proposal':
        edits = dict(body_plan['edits_deg'])
    else:
        edits = {}
    if head_plan:
        if edits.keys() & head_plan['edits_deg'].keys():
            raise ValueError('head and limbs share edited bones')
        edits.update(head_plan['edits_deg'])
    if not edits:
        return dict(edits_deg={}, status='no_supported_edit')
    names, lookup, parents, local, _ = prepare(source)
    world = fk(local, parents, [lookup[n] for n in edits], np.radians(list(edits.values())))
    return dict(edits_deg=edits, status='proposal',
                expected_heads={n: value[:3, 3].tolist() for n, value in zip(names, world, strict=True)},
                body_outcomes=body_plan.get('outcomes', {}) if body_plan else {},
                head_before=head_plan['before_deg'] if head_plan else None,
                head_after=head_plan['after_deg'] if head_plan else None)


def solve_case(args):
    p1, out, row, seed, model = args
    cid = row['case_id']
    source = load_json(p1 / 'p1-native' / f'{cid}-source.json')
    if digest(p1 / source['source']) != source['source_sha256']:
        raise ValueError('source pose changed')
    if digest(p1 / row['rough']) != row['rough_sha256']:
        raise ValueError('rough changed')
    observation_path = out / 'observations' / f'{cid}.json'
    response = load_json(observation_path)
    if response['case_id'] != cid or response['rough_sha256'] != row['rough_sha256']:
        raise ValueError('VLM observation changed')
    seed = checked_seed(row, {cid: seed} if seed else {})
    result = dict(source_sha256=source['source_sha256'], observation_sha256=digest(observation_path))
    if seed is None or response['goals'] is None:
        return cid, dict(result, status='missing_detector_or_observation', automatic=None, learned=None)
    with Image.open(p1 / row['rough']) as image:
        dimensions = list(image.size)
    points, scores = np.asarray(seed['keypoints']), np.asarray(seed['scores'])
    _, lookup, _, _, world = prepare(source)
    _, _, _, basis = frame(source['bones'])
    selected_view = row['selection']['selected_view']
    case = dict(source, dimensions=dimensions, base_camera=(source_camera(source['bones'], selected_view) @ basis.T).tolist(),
                torso_anchors=points[[5, 6, 11, 12]].tolist(), torso_scores=scores[[5, 6, 11, 12]].tolist())
    with np.load(out / 'base-mesh' / f'{cid}.npz') as mesh_file:
        mesh = {key: mesh_file[key] for key in ('vertices', 'faces', 'weights')}
    case['camera'] = registration(case, lookup, world, mesh['vertices'])
    cue = response['goals']
    goals, statuses, roles = compile_observation(case, cue, seed)
    head = head_goal(cue, case, seed)
    head_candidates = fit_head(dict(case, direction_goal=head)) if head else []
    head_plan = head_candidates[0] if head_candidates else None
    prior = compatible_prior(goals, predict(row, source, seed, model))
    auto_fit = fit(case, mesh, goals, caps=LIMB_CAPS)
    learned_fit = fit(case, mesh, goals, caps=LIMB_CAPS, prior=prior) if any(
        value is not None for part in prior.values() for value in part) else auto_fit
    return cid, dict(result, status='computed', camera=case['camera']['status'], leg_roles=roles,
                     goal_status=statuses, goals=goals, head_goal=head, learned_prior=prior,
                     automatic=compose(source, auto_fit, head_plan), learned=compose(source, learned_fit, head_plan))


def run(p1, out, seeds_path, workers):
    cohort_path = out / 'cohort.json'
    cohort = load_json(cohort_path)
    model = load_json(out / 'model.json')
    if digest(p1 / 'manifest.json') != cohort['p1_manifest_sha256'] or model['cohort_sha256'] != digest(cohort_path):
        raise ValueError('frozen inference binding changed')
    rows = {row['case_id']: row for row in load_json(p1 / 'manifest.json')['rows']}
    seeds = {row['case_id']: row for row in load_json(seeds_path)}
    tasks = [(p1, out, rows[cid], seeds.get(cid), model) for cid in cohort['validation']]
    cases = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cid, case in pool.map(solve_case, tasks):
            cases[cid] = case
            print('P3', cid, case['status'], case.get('leg_roles'), flush=True)
    save_json(out / 'plans.json', dict(cohort_sha256=digest(cohort_path), model_sha256=digest(out / 'model.json'),
                                      ground_truth_used=False, cases=cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('p1', 'out', 'seeds'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    run(args.p1, args.out, args.seeds, args.workers)
