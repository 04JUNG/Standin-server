"""Masked ridge supervision for segment corrections on recorded source poses."""

import argparse
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.goal_probe_v2 import prepare
from experiments.refine_v5.learning_data import digest, frame, source_camera
from experiments.refine_v5.part_vlm_observe import CHAINS


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def sample(case, source, seed, limb):
    chain, indices = CHAINS[limb]
    _, lookup, _, _, world = prepare(source)
    _, _, _, body = frame(source['bones'])
    camera = source_camera(source['bones'], case['selection']['selected_view'])
    base = unit(np.diff(world[[lookup[n] for n in chain], :3, 3], axis=0) @ body)
    kp = np.asarray(seed['keypoints'], float)[list(indices)]
    score = np.clip(np.asarray(seed['scores'], float)[list(indices)], 0., 1.)
    valid = np.minimum(score[:-1], score[1:]) >= .40
    xy = unit(np.diff(kp, axis=0))
    projected = unit((base @ camera.T)[:, :2])
    features = np.r_[base.ravel(), (xy * valid[:, None]).ravel(),
                     ((xy - projected) * valid[:, None]).ravel(), score, camera[2]]
    return features, base


def labels(case, gt, limb):
    chain, _ = CHAINS[limb]
    names, _, _, axes = frame(gt['bones'])
    points = np.asarray([names[name]['head'] for name in chain])
    side, kind = limb.split('_')
    parts = ('upper_arm', 'forearm') if kind == 'arm' else ('thigh', 'shin')
    mask = [case['part_visibility'][f'{side}_{part}'] == 'visible' for part in parts]
    return unit(np.diff(points, axis=0) @ axes), mask


def ridge(x, y, alpha=12.):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .15)
    y_mean = y.mean(axis=0)
    norm = (x - mean) / scale
    coef = np.linalg.solve(norm.T @ norm + alpha * np.eye(norm.shape[1]), norm.T @ (y - y_mean))
    return dict(mean=mean.tolist(), scale=scale.tolist(), bias=y_mean.tolist(), coef=coef.tolist(), samples=len(x))


def predict(case, source, seed, model):
    result = {}
    for limb in CHAINS:
        x, base = sample(case, source, seed, limb)
        vectors = []
        for index, row in enumerate(model['parts'][limb]):
            if not row:
                vectors.append(None)
                continue
            delta = np.asarray(row['bias']) + ((x - row['mean']) / row['scale']) @ row['coef']
            # Bound the learned target shift before using it as a weak prior.
            shift = delta * min(1., .65 / max(np.linalg.norm(delta), 1e-8))
            vectors.append(unit(base[index] + shift).tolist())
        result[limb] = vectors
    return result


def checked_seed(case, seeds):
    seed = seeds.get(case['case_id'])
    if (not seed or seed.get('image_sha256') != case['rough_sha256']
            or np.asarray(seed.get('keypoints')).shape != (17, 2)
            or np.asarray(seed.get('scores')).shape != (17,)
            or not np.isfinite(np.asarray(seed['keypoints'], float)).all()
            or not np.isfinite(np.asarray(seed['scores'], float)).all()):
        return None
    return seed


def train(p1, out, seeds_path, gt_audit):
    cohort = load_json(out / 'cohort.json')
    if digest(p1 / 'manifest.json') != cohort['p1_manifest_sha256']:
        raise ValueError('P0 binding changed')
    rows = {r['case_id']: r for r in load_json(p1 / 'manifest.json')['rows']}
    seeds = {r['case_id']: r for r in load_json(seeds_path)}
    targets = {r['case_id']: r for r in load_json(gt_audit)}
    if set(cohort['development']) & set(cohort['validation']):
        raise ValueError('training overlaps validation')
    by_part = {limb: [[], []] for limb in CHAINS}
    used, skipped = [], []
    for cid in cohort['development']:
        row = rows[cid]
        if row['split'] != 'development':
            raise ValueError('non-development target in training')
        seed = checked_seed(row, seeds)
        if seed is None:
            skipped.append(cid)
            continue
        source = load_json(p1 / 'p1-native' / f'{cid}-source.json')
        gt = targets[cid]
        if digest(p1 / source['source']) != source['source_sha256'] or gt['sha256'] != row['ground_truth_sha256']:
            raise ValueError('training pose binding changed')
        used.append(cid)
        for limb in CHAINS:
            x, base = sample(row, source, seed, limb)
            target, mask = labels(row, gt, limb)
            for segment, visible in enumerate(mask):
                if visible:
                    by_part[limb][segment].append((x, target[segment] - base[segment]))
    model = {}
    for limb, parts in by_part.items():
        model[limb] = []
        for part in parts:
            x, y = (map(np.asarray, zip(*part)) if part else ([], []))
            model[limb].append(ridge(x, y) if len(x) >= 10 else None)
    save_json(out / 'model.json', dict(parts=model, development_used=used, development_skipped=skipped,
                                      cohort_sha256=digest(out / 'cohort.json'), method='masked ridge alpha 12'))
    print('ORIGINAL_PRIOR', len(used), 'training rows; missing detector', skipped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('p1', 'out', 'seeds', 'gt-audit'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    train(args.p1, args.out, args.seeds, args.gt_audit)
