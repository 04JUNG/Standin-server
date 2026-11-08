"""Apply reviewed face direction ranges to the saved Part goals artifact."""

import argparse
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.part_head import head_axes


REVIEW = Path(__file__).resolve().parents[2] / 'evaluation/refine_v5/head_direction_review_20261108.json'
HEAD_CAP = 72.
NECK_CAP = 12.
LIMITS = {'yaw_deg': (-175., 175.), 'pitch_deg': (-80., 80.), 'neck_tilt_deg': (-175., 175.)}


def measure(case, world, names):
    lookup = {name: i for i, name in enumerate(names)}
    axes = head_axes(world, lookup)
    camera = np.asarray(case['camera']['rotation'])
    x, down, z = camera @ axes['forward']
    neck = camera @ axes['neck']
    return dict(yaw_deg=float(np.degrees(np.arctan2(x, -z))),
                pitch_deg=float(np.degrees(np.arctan2(down, np.hypot(x, z)))),
                neck_tilt_deg=float(np.degrees(np.arctan2(neck[0], -neck[1]))))


def range_error(values, goal):
    errors = {}
    for key in (name for name in LIMITS if name in goal):
        low, high = goal[key]
        value = values[key]
        delta = [(value - edge + 180.) % 360. - 180. for edge in (low, high)]
        errors[key] = 0. if low <= value <= high else min(map(abs, delta))
    return errors


def validate_goal(goal):
    if set(goal) - {*LIMITS, 'rough_crop', 'evidence'} or not {'yaw_deg', 'pitch_deg'} <= goal.keys():
        raise ValueError('invalid head review')
    for name in (key for key in LIMITS if key in goal):
        lower, upper = LIMITS[name]
        bounds = np.asarray(goal[name], float)
        if bounds.shape != (2,) or not lower <= bounds[0] < bounds[1] <= upper or bounds[1] - bounds[0] < 2.:
            raise ValueError('invalid head angle range')


def fit(case):
    from scipy.optimize import least_squares

    from experiments.refine_v5.depth_fit import candidate_row, limit_vectors
    from experiments.refine_v5.goal_probe_v2 import fk, prepare

    goal = case['direction_goal']
    validate_goal(goal)
    names, lookup, parents, local, base = prepare(case)
    initial = measure(case, base, names)
    active = ['Neck', 'Head'] if 'neck_tilt_deg' in goal else ['Head']
    indices = [lookup[n] for n in active]
    caps_deg = [NECK_CAP, HEAD_CAP] if len(active) == 2 else [HEAD_CAP]
    caps = np.deg2rad(caps_deg)
    box = np.repeat(caps, 3)
    wanted = {key: float(np.clip(initial[key], low + 1., high - 1.)) if not low <= initial[key] <= high else initial[key]
              for key, (low, high) in ((k, goal[k]) for k in LIMITS if k in goal)}

    def pose(x):
        delta = limit_vectors(x, caps)
        return delta, fk(local, parents, indices, delta)

    def residual(x):
        delta, world = pose(x)
        actual = measure(case, world, names)
        angular = np.deg2rad([(actual[k] - wanted[k] + 180.) % 360. - 180. for k in wanted])
        return np.r_[angular, .09 * delta.ravel()]

    solve = least_squares(residual, np.zeros(len(box)), bounds=(-box, box), max_nfev=60)
    delta, _ = pose(solve.x)
    baseline_error = range_error(initial, goal)
    candidates = []
    for scale in (1., .8, .6, .4):
        scaled = delta * scale
        world = fk(local, parents, indices, scaled)
        after = measure(case, world, names)
        errors = range_error(after, goal)
        if any(errors[k] > baseline_error[k] + .15 for k in errors) or sum(baseline_error.values()) - sum(errors.values()) < .5:
            continue
        edits = dict(zip(active, np.rad2deg(scaled).tolist(), strict=True))
        candidates.append(candidate_row(case, 'head_direction', names, world, edits,
                                        before_deg=initial, after_deg=after, scale=scale))
    return candidates


def prepare(prior, out, review):
    from experiments.refine_v5.learning_data import digest

    manifest = load_json(prior / 'manifest.json')
    if digest(prior / 'manifest.json') != load_json(prior / 'frozen.json')['manifest_sha256']:
        raise ValueError('Part goals manifest changed')
    original = {c['case_id']: c for c in manifest['cases']}
    source = {r['case_id']: r for r in load_json(prior / 'native/audit.json') if r['method'] == 'fused'}
    fixture = load_json(review)
    if fixture['schema'] != 'v5-rough-head-direction-v1':
        raise ValueError('unsupported head review schema')
    cases, rows = [], []
    for cid, goal in fixture['cases'].items():
        case, selected = original[cid], source[cid]
        if selected['status'] in ('fallback_exact_B0', 'reference') or selected['skin_nonregression_failures']:
            raise ValueError('Part goals did not provide an applicable body candidate')
        if selected['source_sha256'] != case['source_sha256'] or digest(selected['path']) != selected['sha256']:
            raise ValueError('Part goals file changed')
        if digest(case['image']) != case['image_sha256']:
            raise ValueError('rough changed')
        actual = {bone['name']: bone['matrix'] for bone in selected['bones']}
        bones = [dict(bone, matrix=actual[bone['name']]) for bone in case['bones']]
        composed = dict(case, bones=bones, source=selected['path'], source_sha256=selected['sha256'],
                        direction_goal=goal)
        cases.append(composed)
        rows.append(dict(case_id=cid, candidates=fit(composed)))
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'manifest.json'
    save_json(path, dict(cases=cases, prior_manifest_sha256=digest(prior / 'manifest.json'), review_sha256=digest(review)))
    save_json(out / 'head-direction-plans.json', dict(manifest_sha256=digest(path), results=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=REVIEW)
    args = parser.parse_args()
    prepare(args.prior, args.out, args.review)


if __name__ == '__main__':
    main()
