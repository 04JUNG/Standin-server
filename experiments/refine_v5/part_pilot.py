"""Fit the paired ordinal and enriched observation pilots without GT."""

import argparse
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.depth_fit import fit
from experiments.refine_v5.learning_data import digest

METHODS = ('ordinal', 'fused')


def model_starts(manifest, checkpoint, seeds_path):
    from experiments.refine_v5.part_observations import DEFAULT_REVIEW
    from experiments.refine_v5.predict_v3 import predict

    reviews = {r['case_id']: r for r in load_json(DEFAULT_REVIEW.with_name('visual_review_v2.json'))['records']}
    seeds = {r['case_id']: r for r in load_json(seeds_path)}
    starts = {}
    for case in manifest['cases']:
        cid = case['case_id']
        if case['part_goal']['reason'] is not None:
            continue
        seed = seeds[cid]
        if seed['image_sha256'] != case['image_sha256']:
            raise ValueError('learned prior rough differs from the frozen input')
        result = predict(checkpoint, dict(
            case_id=cid, image=case['image'], image_sha256=case['image_sha256'],
            source_fbx=case['source'], source_sha256=case['source_sha256'], source_bones=case['bones'],
            keypoints=seed['keypoints'], scores=seed['scores'], review=reviews[cid],
            seed_kind=seed['seed_kind'], fixed_camera=case['camera']['rotation'],
            ownership_verified=False,
        ))
        radians = result['proposals']['local_SO3_residual_proposals']
        starts[cid] = {name: np.rad2deg(radians[name]).tolist() for name in case['depth_evidence']['movable']}
    return starts


def propose(root, checkpoint=None, seeds=None):
    if (root / 'frozen.json').exists():
        raise FileExistsError('already frozen')
    manifest_path = root / 'manifest.json'
    manifest = load_json(manifest_path)
    if manifest['schema'] != 'v5-part-target-pilot-v1':
        raise ValueError('unexpected part pilot manifest')
    if (checkpoint is None) != (seeds is None):
        raise ValueError('checkpoint and seed observations must be supplied together')
    starts = model_starts(manifest, checkpoint, seeds) if checkpoint else {}
    methods = (*METHODS, 'learned') if checkpoint else METHODS
    previous_checkpoint = manifest.pop('learned_prior', None)
    if checkpoint:
        manifest['learned_prior'] = dict(checkpoint_sha256=digest(checkpoint), seeds_sha256=digest(seeds))
    results = {}
    for case in manifest['cases']:
        cid = case['case_id']
        if digest(case['source']) != case['source_sha256'] or digest(case['image']) != case['image_sha256']:
            raise ValueError('stale source or rough')
        with np.load(root / f'{cid}-B0.npz') as skin:
            results[cid] = {mode: fit(case, skin, starts.get(cid, {}) if mode == 'learned' else {}, part_mode=mode)
                            for mode in methods}
        print('PART_FIT', cid, {mode: sum(map(len, result.values())) for mode, result in results[cid].items()},
              case['part_goal']['reason'], flush=True)
    save_json(root / 'depth-ten-proposals.json', results)
    if manifest.get('learned_prior') != previous_checkpoint:
        save_json(manifest_path, manifest)


def propose_heads(root):
    from experiments.refine_v5.part_head import propose as fit_head

    path = root / 'manifest.json'
    rows = []
    for case in load_json(path)['cases']:
        candidate = fit_head(case)
        if candidate is not None:
            rows.append(candidate)
    save_json(root / 'head-proposals.json', dict(manifest_sha256=digest(path), results=rows))


def propose_upper(root):
    from experiments.refine_v5.part_upper import propose as fit_upper

    path = root / 'manifest.json'
    rows = []
    for case in load_json(path)['cases']:
        for label, spine, clavicle in [('spine', True, False), ('clavicle', False, True), ('both', True, True)]:
            candidate = fit_upper(case, spine=spine, clavicle=clavicle)
            if candidate is not None:
                rows.append(dict(candidate, method=label))
    save_json(root / 'upper-proposals.json', dict(manifest_sha256=digest(path), results=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    auxiliary = parser.add_mutually_exclusive_group()
    auxiliary.add_argument('--head-only', action='store_true')
    auxiliary.add_argument('--upper-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seeds', type=Path)
    args = parser.parse_args()
    if (args.head_only or args.upper_only) and (args.checkpoint or args.seeds):
        parser.error('checkpoint and seeds apply only to the limb comparison')
    if args.head_only:
        propose_heads(args.root)
    elif args.upper_only:
        propose_upper(args.root)
    else:
        propose(args.root, args.checkpoint, args.seeds)


if __name__ == '__main__':
    main()
