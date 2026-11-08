"""Ten-case fixed-B0 replay for the detector-driven single-solve path."""

import argparse
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, project, save_json
from experiments.refine_v5.ab_fit import cameras
from experiments.refine_v5.goal_probe_v2 import prepare
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.part_observations import projected_evidence
from experiments.refine_v5.part_single_fit import fit
from experiments.refine_v5.part_single_observe import MIN_SCORE, bend_envelope


ARM_INDICES = {'Left': (5, 7, 9), 'Right': (6, 8, 10)}


def registration(case, lookup, world, vertices):
    try:
        return cameras(case, lookup, world, vertices)['ortho_calibrated']
    except ValueError:
        # Used only to render abstentions. No rough geometry is fitted here.
        origin = world[lookup['Hips'], :3, 3]
        torso = np.linalg.norm(world[lookup['Spine2'], :3, 3] - origin)
        return dict(status='fallback_unreliable_registration', rotation=case['base_camera'],
                    origin=origin.tolist(), torso_world=torso, pixel_scale=min(case['dimensions']) / 3.,
                    pixel_offset=(np.asarray(case['dimensions']) * [.5, .56]).tolist(),
                    rough_unit_px=min(case['dimensions']) / 3., inverse_distance_torso=0.)


def arm_goals(case, seed, visibility, lookup, world):
    camera = case['camera']
    evidence = {}
    eligible = []
    for side, ids in ARM_INDICES.items():
        row = evidence[side] = {}
        points = np.asarray(seed['keypoints'], float)[list(ids)]
        scores = np.asarray(seed['scores'], float)[list(ids)]
        names = [side + bone for bone in ('Arm', 'ForeArm', 'Hand')]
        row.update(scores=scores.tolist(), points=points.tolist())
        parts = [visibility[f'{side.lower()}_{part}'] for part in ('upper_arm', 'forearm')]
        if not np.isfinite(scores).all() or not np.isfinite(points).all() or min(scores) < MIN_SCORE:
            row['status'] = 'weak_detection'
        elif parts != ['visible', 'visible']:
            row['status'] = 'arm_visibility_unverified'
        elif camera['status'] != 'fitted':
            row['status'] = 'camera_unverified'
        else:
            lengths = np.linalg.norm(np.diff(world[[lookup[n] for n in names], :3, 3], axis=0), axis=1)
            row.update(projected_evidence(points, camera['pixel_scale'], lengths, camera['torso_world']))
            bend = bend_envelope(row)
            row['bend_deg'] = bend
            if bend is None:
                row['status'] = 'projected_lengths_unreliable'
                continue
            margin = np.clip((1 - scores[1:]) * 32., 8., 18.)
            bounds = {name: [(point - half).tolist(), (point + half).tolist()]
                      for name, point, half in zip(names[1:], points[1:], margin, strict=True)}
            eligible.append((float(min(scores)), side, dict(pixels=bounds, bend_deg=bend, preserve_camera_depth=True)))
            row['status'] = 'eligible'
    if not eligible:
        return {}, evidence
    _, side, goal = max(eligible, key=lambda entry: entry[0])
    evidence[side]['status'] = 'selected'
    return {side.lower() + '_arm': goal}, evidence


def propose(root, previous):
    manifest = load_json(root / 'manifest.json')
    repo = Path(__file__).resolve().parents[2]
    reviews = {r['case_id']: r for r in load_json(repo / 'evaluation/refine_v5/visual_review_v2.json')['records']}
    seeds = {r['case_id']: r for r in load_json(previous / 'all-seeds/seeds.json')}
    for case in manifest['cases']:
        cid = case['case_id']
        seed = seeds[cid]
        if digest(case['source']) != case['source_sha256'] or digest(case['image']) != seed['image_sha256']:
            raise ValueError('source or rough changed')
        names, lookup, _, _, world = prepare(case)
        folder = root / cid
        folder.mkdir(exist_ok=False)
        with np.load(root / f'{cid}-B0.npz') as mesh:
            skin = {key: mesh[key] for key in ('vertices', 'faces', 'weights')}
        case['camera'] = registration(case, lookup, world, skin['vertices'])
        save_json(folder / 'input.json', case)
        goals, evidence = arm_goals(case, seed, reviews[cid]['part_visibility'], lookup, world)
        plan = fit(case, dict(skin, names=names), goals)
        screen = None
        if plan is not None:
            side = next(iter(goals)).split('_')[0].title()
            chain = ARM_INDICES[side][1:]
            targets = np.asarray(seed['keypoints'])[list(chain)]
            bone_names = [side + 'ForeArm', side + 'Hand']
            original = project(world[[lookup[n] for n in bone_names], :3, 3], case['camera'])
            revised = np.asarray([plan['predicted_pixels'][n] for n in bone_names])
            error = lambda xy: float(np.linalg.norm(xy - targets, axis=1).mean())
            screen = dict(before_px=error(original), proposed_px=error(revised))
            if screen['before_px'] - screen['proposed_px'] < 1.:
                plan = None
                evidence[side]['status'] = 'insufficient_screen_gain'
        output = folder / 'detector'
        output.mkdir()
        save_json(output / 'plan.json', dict(input_sha256=digest(folder / 'input.json'), result=plan))
        save_json(folder / 'evidence.json', dict(arms=evidence, screen=screen, legs='no_reviewed_3d_lift_or_support'))
        print('SINGLE_BATCH', cid, {side: item['status'] for side, item in evidence.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    propose(args.root, args.previous)
