"""Prepare and fit the same ten roughs under three frozen camera models."""

import argparse
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, prepare_manifest, save_json
from experiments.refine_v5.ab_fit import cameras
from experiments.refine_v5.depth_fit import fit
from experiments.refine_v5.goal_probe_v2 import prepare
from experiments.refine_v5.learning_data import digest, frame

METHODS = ('ortho_fixed', 'ortho_calibrated', 'perspective')
REPO = Path(__file__).resolve().parents[2]


def prepare_inputs(prior, previous, body, out):
    settings = load_json(REPO / 'evaluation/refine_v5/depth_ten_cases.json')
    old_observations = load_json(REPO / 'evaluation/refine_v5/camera_vlm_ab.json')['cases']
    old_depth = load_json(REPO / 'evaluation/refine_v5/depth_review_20261108.json')
    added = settings['added']
    case_ids = settings['case_order']
    annotations = {cid: ({k: v for k, v in added[cid].items() if k != 'depth'}
                         if cid in added else old_observations[cid]) for cid in case_ids}
    prepare_manifest(prior, previous, body, out, fixture={'cases': annotations})
    manifest = load_json(out / 'manifest.json')
    seeds = {row['case_id']: row for row in load_json(previous / 'all-seeds/seeds.json')}
    for case in manifest['cases']:
        cid = case['case_id']
        case['depth_evidence'] = added[cid]['depth'] if cid in added else old_depth['cases'][cid]
        case['depth_authority'] = 'analyst_rough_only' if cid in added else old_depth['authority']
        _, _, length, basis = frame(case['bones'])
        case.update(body_forward=basis[:, 2].tolist(), torso_world=length)
        if cid == '006':
            case['lifted_leg_targets_px'] = np.asarray(seeds[cid]['keypoints'])[[11, 13, 15]].tolist()
        if 'lifted_leg_points' in case:
            case['lifted_leg_targets_px'] = (np.asarray(case['lifted_leg_points']) * case['dimensions']).tolist()
        if 'shoulder_anchors' in case:
            case['torso_anchors'][:2] = (np.asarray(case['shoulder_anchors']) * case['dimensions']).tolist()
            case['torso_scores'][:2] = [1., 1.]
            case['torso_anchor_authority'] = 'analyst_visible_shoulders'
    manifest['schema'] = 'v5-depth-camera-comparison-ten-v1'
    manifest['scope'] = settings['scope']
    save_json(out / 'manifest.json', manifest)


def camera_and_fit(root, only=None):
    path = root / 'manifest.json'
    manifest = load_json(path)
    if (root / 'frozen.json').exists():
        raise FileExistsError('frozen results')
    output = root / 'depth-ten-proposals.json'
    results = load_json(output) if only and output.exists() else {}
    for case in manifest['cases']:
        cid = case['case_id']
        if only and cid not in only:
            continue
        if digest(case['source']) != case['source_sha256'] or digest(case['image']) != case['image_sha256']:
            raise ValueError('stale source or image')
        _, lookup, _, _, base = prepare(case)
        with np.load(root / f'{cid}-B0.npz') as mesh:
            views = cameras(case, lookup, base, mesh['vertices'])
            case['cameras'] = views
            case['camera'] = views['perspective']
            case['depth_camera_farther'] = views['perspective']['rotation'][2]
            grouped = {}
            memo = {}
            for method in METHODS:
                camera = views[method]
                key = repr([camera[k] for k in ('rotation', 'inverse_distance_torso', 'pixel_scale', 'pixel_offset')])
                if key not in memo:
                    memo[key] = fit(dict(case, camera=camera), mesh, {})
                grouped[method] = memo[key]
            results[cid] = grouped
        save_json(output, results)
        save_json(path, manifest)
        print('DEPTH_TEN_FIT', cid, views['perspective']['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    build = sub.add_parser('prepare')
    for key in ('prior', 'previous', 'body', 'out'):
        build.add_argument(f'--{key}', required=True, type=Path)
    fit_parser = sub.add_parser('fit')
    fit_parser.add_argument('--root', required=True, type=Path)
    fit_parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare_inputs(args.prior, args.previous, args.body, args.out)
    else:
        camera_and_fit(args.root, args.case)


if __name__ == '__main__':
    main()
