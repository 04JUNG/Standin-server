"""P1 diagnostic: pass observed GT 3D goals to bounded FK from the original BVH."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.depth_fit import limit_vectors
from experiments.refine_v5.goal_probe_v2 import fk, prepare
from experiments.refine_v5.learning_data import digest, frame


CAPS = {'Arm': 54., 'ForeArm': 53., 'UpLeg': 110., 'Leg': 65., 'Foot': 72.}
PROFILES = ('local30', 'limbs_head', 'torso_assisted')
CHAINS = {
    'arm': (('Arm', 'ForeArm', 'Hand'), ('upper_arm', 'forearm')),
    'leg': (('UpLeg', 'Leg', 'Foot', 'ToeBase'), ('thigh', 'shin', 'foot')),
}


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def angular(first, second):
    return np.degrees(np.arccos(np.clip(np.sum(unit(first) * unit(second), axis=-1), -1., 1.)))


def specification(case, names, truth):
    visibility = case['part_visibility']
    available = set(names) & truth.keys()
    points, edges, bends, editable = set(), [], [], {}
    for side in ('Left', 'Right'):
        for kind, (joints, labels) in CHAINS.items():
            chain = tuple(side + joint for joint in joints)
            observed = []
            for start, end, label in zip(chain[:-1], chain[1:], labels, strict=True):
                if (visibility[f'{side.lower()}_{label}'] != 'visible'
                        or not {start, end} <= available):
                    continue
                edges.append((kind, start, end))
                observed.append((start, end))
                points.add(end)
                editable[start] = CAPS[start[len(side):]]
            if (chain[0], chain[1]) in observed and (chain[1], chain[2]) in observed:
                bends.append((kind, *chain[:3]))
    head = visibility['head'] == 'visible' and {'Head', 'Neck'} <= available
    if head:
        editable.update(Neck=12., Head=72.)
        points.add('Head')
    return sorted(points), edges, bends, head, editable


def angles_for_world(world, target, spec, lookup, target_head):
    points, edges, bends, head, _ = spec
    def chain_values(start, end, values):
        return values[lookup[end]] - values[lookup[start]]

    segments = {kind: [] for kind in CHAINS}
    for kind, start, end in edges:
        a = chain_values(start, end, world[:, :3, 3])
        b = chain_values(start, end, target)
        segments[kind].append(float(angular(a, b)))
    bend_errors = {kind: [] for kind in CHAINS}
    for kind, a, b, c in bends:
        q = world[:, :3, 3]
        bend = angular(chain_values(a, b, q), chain_values(b, c, q))
        target_bend = angular(chain_values(a, b, target), chain_values(b, c, target))
        bend_errors[kind].append(float(abs(bend - target_bend)))
    head_error = None
    if head:
        axes = world[lookup['Head'], :3, 1:3].T
        head_error = angular(axes, target_head).tolist()
    return dict(arm_direction_deg=segments['arm'], leg_direction_deg=segments['leg'],
                elbow_bend_error_deg=bend_errors['arm'], knee_bend_error_deg=bend_errors['leg'],
                head_up_forward_deg=head_error)


def problem(case, source, gt):
    names, lookup, parents, local, base = prepare(source)
    truth, gt_center, gt_scale, gt_basis = frame(gt['bones'])
    _, center, torso, basis = frame(source['bones'])
    rotation = basis @ gt_basis.T
    target = base[:, :3, 3].copy()
    for name, bone in truth.items():
        if name in lookup:
            target[lookup[name]] = center + rotation @ (np.asarray(bone['head']) - gt_center) * (torso / gt_scale)
    target_head = (rotation @ np.asarray(truth['Head']['matrix'])[:3, 1:3]).T if 'Head' in truth else None
    spec = specification(case, names, truth)
    return names, lookup, parents, local, base, target, target_head, torso, spec


def metrics(world, target, spec, lookup, target_head, torso):
    scores = angles_for_world(world, target, spec, lookup, target_head)
    indices = [lookup[name] for name in spec[0]]
    scores['endpoint_error_torso'] = (np.linalg.norm(world[indices, :3, 3] - target[indices], axis=1) / torso).tolist()
    return scores


def fit(name, case, problem_data):
    names, lookup, parents, local, base, target, target_head, torso, spec = problem_data
    points, edges, _, head, editable = spec
    caps = {k: min(30., v) if name == 'local30' and k not in ('Head', 'Neck') else v
            for k, v in editable.items()}
    extra = []
    if name == 'torso_assisted' and case['part_visibility']['torso'] == 'visible':
        extra = ['Spine', 'Spine1', 'Spine2']
        caps.update({joint: 8. / 3. for joint in extra})
        for side in ('Left', 'Right'):
            if case['part_visibility'][f'{side.lower()}_upper_arm'] == 'visible':
                caps[side + 'Shoulder'] = 5.
                extra.append(side + 'Arm')
    active = [n for n in names if n in caps]
    indices = [lookup[n] for n in active]
    cap = np.radians([caps[n] for n in active])
    tracked = [lookup[n] for n in points]
    extra_ids = [lookup[n] for n in extra]
    origin = np.asarray([lookup[start] for _, start, _ in edges], int)
    distal = np.asarray([lookup[end] for _, _, end in edges], int)
    directions = unit(target[distal] - target[origin])

    def pose(x):
        edits = limit_vectors(x, cap)
        return edits, fk(local, parents, indices, edits)

    def residual(x):
        edits, world = pose(x)
        xyz = world[:, :3, 3]
        point_error = 1.3 * (xyz[tracked] - target[tracked]).ravel() / torso
        direction_error = (unit(xyz[distal] - xyz[origin]) - directions).ravel()
        head_error = .8 * (unit(world[lookup['Head'], :3, 1:3].T) - unit(target_head)).ravel() if head else []
        torso_error = .8 * (xyz[extra_ids] - target[extra_ids]).ravel() / torso
        return np.r_[point_error, direction_error, head_error, torso_error, .045 * edits.ravel()]

    box = np.repeat(cap, 3)
    solved = least_squares(residual, np.zeros(len(box)), bounds=(-box, box), max_nfev=42, ftol=3e-5, xtol=3e-5)
    edits, actual = pose(solved.x)
    return dict(edits_deg={n: edit.tolist() for n, edit in zip(active, np.degrees(edits), strict=True)
                           if np.linalg.norm(edit) > 1e-4},
                expected_heads={n: matrix[:3, 3].tolist() for n, matrix in zip(names, actual, strict=True)},
                metrics=metrics(actual, target, spec, lookup, target_head, torso),
                evaluations=int(solved.nfev))


def solve_case(args):
    root, case, gt = args
    cid = case['case_id']
    source = load_json(root / 'p1-native' / f'{cid}-source.json')
    if source['status'] != 'ready':
        return source
    if gt['sha256'] != case['ground_truth_sha256'] or digest(root / source['source']) != source['source_sha256']:
        raise ValueError('source or target geometry differs from pinned metadata')
    data = problem(case, source, gt)
    _, lookup, _, _, base, target, head, torso, spec = data
    if not spec[-1]:
        return dict(case_id=cid, status='unobservable')
    baseline = metrics(base, target, spec, lookup, head, torso)
    profiles = {name: fit(name, case, data) for name in PROFILES}
    metadata = {k: v for k, v in source.items() if k != 'bones'}
    return dict(metadata, status='computed', split=case['split'], B0=baseline, profiles=profiles)


def run(root, gt_audit, workers):
    cohort = load_json(root / 'p1-cohort.json')
    if digest(root / 'manifest.json') != cohort['manifest_sha256']:
        raise ValueError('manifest changed')
    targets = {row['case_id']: row for row in load_json(gt_audit)}
    cases = {row['case_id']: row for row in load_json(root / 'manifest.json')['rows']}
    tasks = [(root, cases[cid], targets[cid]) for cid in cohort['case_ids']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = []
        for row in pool.map(solve_case, tasks):
            rows.append(row)
            print('ORIGINAL_ORACLE', row['case_id'], row['status'], flush=True)
    save_json(root / 'p1-plans.json', dict(manifest_sha256=cohort['manifest_sha256'], uses_ground_truth_goals=True,
                                          export_profile='limbs_head', rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gt-audit', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    run(args.root, args.gt_audit, args.workers)
