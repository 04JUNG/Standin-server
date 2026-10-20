"""Known-data camera ablation with frozen Human-Art XY and dual labels.

Ranks never read answers. All source sets have already been examined; this is
mechanism diagnosis, not a new holdout, policy selection, or deployment.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_camera_gate import DATASETS as OLD_DATASETS
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.dual_label_search import alternate_labels, merged_order
from standin_eval.perspective_search import (
    camera_grid, build_camera_index, score_camera_index, normalize_batch,
)

DATASETS = OLD_DATASETS + ['dual-label-confirm-balanced-20261012']
GRIDS = {4: 'baseline_4', 13: 'yaw_13', 39: 'yaw_pitch_39',
         117: 'yaw_pitch_perspective_117'}
ARMS = [f'pooled_{n}' for n in GRIDS]
SOURCES = ['scripts/diagnose_dual_camera_coverage.py',
           'standin_eval/dual_label_search.py', 'standin_eval/perspective_search.py',
           'scripts/experiment_z_axis_search.py', 'src/bvh.py']


def read(path):
    return json.loads(path.read_text())


def valid_query(prediction):
    if prediction.get('error'):
        return None
    xy = np.asarray(prediction['keypoints2d'], dtype=np.float32)
    mask = np.asarray(prediction['scores'], dtype=np.float32) >= .3
    if xy.shape != (17, 2) or mask.shape != (17,):
        raise ValueError('invalid prediction shape')
    if not np.isfinite(xy).all() or mask[5:].sum() < 6 or not mask[[5, 6, 11, 12]].all():
        return None
    return xy, mask


def summarize(rows, truth):
    keys = [(r['dataset'], r['image']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(truth):
        raise ValueError('rankings and answers must match one-to-one')
    records = []
    for row in rows:
        target = truth[(row['dataset'], row['image'])]
        record = {k: target[k] for k in ('group', 'family_id', 'pose_id')}
        record.update(dataset=row['dataset'], image=row['image'],
                      failed=bool(row.get('error')), arms={})
        for arm in ARMS:
            candidates = [] if record['failed'] else row[arm]
            record['arms'][arm] = {
                'top1': bool(candidates and candidates[0] == target['pose_id']),
                'top5': target['pose_id'] in candidates,
            }
        records.append(record)
    groups = {}
    for group in ('perspective', 'orthographic_control'):
        subset = [r for r in records if r['group'] == group]
        counts = {arm: {metric: sum(r['arms'][arm][metric] for r in subset)
                        for metric in ('top1', 'top5')} for arm in ARMS}
        contrasts = {}
        for old, new in zip(ARMS, ARMS[1:]):
            contrasts[new + '_vs_' + old] = {}
            for metric in ('top1', 'top5'):
                gains = [r['dataset'] + '/' + r['image'] for r in subset
                         if not r['arms'][old][metric] and r['arms'][new][metric]]
                losses = [r['dataset'] + '/' + r['image'] for r in subset
                          if r['arms'][old][metric] and not r['arms'][new][metric]]
                contrasts[new + '_vs_' + old][metric] = {
                    'gains': len(gains), 'losses': len(losses),
                    'net': len(gains) - len(losses),
                    'gained_images': gains, 'lost_images': losses,
                }
        groups[group] = {'n': len(subset), 'failures': sum(r['failed'] for r in subset),
                         'counts': counts, 'contrasts': contrasts}
    return {'groups': groups, 'records': records}


def verify_sources(out):
    policy = read(out / 'protocol.json')
    for name, expected in policy['frozen_sources'].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in policy['input_hashes'].items():
        assert sha256(ROOT / name) == expected, name
    return policy


def rank(args):
    # Atomic directory creation refuses duplicate or partially completed runs.
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for dataset in DATASETS:
        root = ROOT / 'out' / dataset
        for filename in ('protocol.json', 'manifest.json', 'answers.json', 'rankings.json'):
            path = root / filename
            inputs[str(path.relative_to(ROOT))] = sha256(path)
        for item in read(root / 'manifest.json'):
            image = root / 'images' / item['image']
            assert sha256(image) == item['sha256'], str(image)
            path = root / 'predictions' / (Path(item['image']).stem + '.json')
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    dev_path = ROOT / 'out/dual-label-dev-20261012/rankings.json'
    inputs[str(dev_path.relative_to(ROOT))] = sha256(dev_path)
    poses = load_library(args.library_root)
    policy = {
        'stage': 'post-hoc mechanism diagnosis; all 340 families already examined',
        'datasets': DATASETS, 'expected_images': 680, 'expected_families': 340,
        'arms': ARMS, 'grid_names': GRIDS,
        'question': 'Separate yaw, pitch, and perspective effects under the same dual-label rule',
        'fixed': 'Human-Art raw XY, score>=.3 mask, hip/torso normalization, dual-label min distance, exact source-ID scoring',
        'failure': 'all original failures retained in denominator; >=6 body joints and all shoulder/hip anchors',
        'selection': 'none; no tuning, new holdout, significance test, or service change',
        'answer_access': 'hash files for integrity; parse answers only in evaluate after rankings saved',
        'limitations': ['procedural line drawings, not human roughs',
                       'earlier datasets have first-family-member selection bias',
                       '340 pose families each have two paired images, not 680 independent poses'],
        'frozen_sources': {name: sha256(ROOT / name) for name in SOURCES},
        'input_hashes': inputs,
        'library_geometry': geometry_hash(poses),
        'library_db_sha256': sha256(args.library_root / 'poses.db'),
    }
    save_json(args.out / 'protocol.json', policy)  # Before new rankings or metrics.
    indices = {n: build_camera_index(poses, camera_grid()[name]) for n, name in GRIDS.items()}
    prior_dev = {(r['dataset'], r['image']): r for r in read(dev_path)}
    records = []
    times = {arm: [] for arm in ARMS}
    verified = 0
    for dataset in DATASETS:
        root = ROOT / 'out' / dataset
        previous = {r['image']: r for r in read(root / 'rankings.json')}
        for item in read(root / 'manifest.json'):
            name = item['image']
            path = root / 'predictions' / (Path(name).stem + '.json')
            row = {'dataset': dataset, 'image': name, 'prediction_sha256': sha256(path)}
            query = valid_query(read(path))
            if query is None:
                assert previous[name].get('error')
                row['error'] = 'original Human-Art extraction failure'
                records.append(row)
                continue
            xy, mask = query
            row['shared_xy_sha256'] = hashlib.sha256(xy.tobytes()).hexdigest()
            row['shared_mask'] = mask.tolist()
            q = normalize_batch(xy)
            alt, alt_mask = alternate_labels(q, mask)
            prior_cost = None
            expected_pooled = prior_dev.get((dataset, name), previous[name])
            for n, index in indices.items():
                start = time.perf_counter()
                direct = score_camera_index(index, q, mask)
                swapped = score_camera_index(index, alt, alt_mask)
                cost = np.minimum(direct, swapped)
                if prior_cost is not None:
                    assert np.all(cost <= prior_cost + 1e-6), 'nested camera costs must not increase'
                prior_cost = cost
                order = merged_order(direct, swapped)
                row[f'pooled_{n}'] = [poses[int(i)].pose_id for i in order[:5]]
                times[f'pooled_{n}'].append(time.perf_counter() - start)
                if n in (4, 117):
                    old_key = f'anchor_{n}' if f'anchor_{n}' in previous[name] else GRIDS[n]
                    direct_top5 = [poses[int(i)].pose_id for i in np.argsort(direct, kind='stable')[:5]]
                    assert direct_top5 == previous[name][old_key][:5]
                    if f'pooled_{n}' in expected_pooled:
                        assert row[f'pooled_{n}'] == expected_pooled[f'pooled_{n}'][:5]
            records.append(row)
            verified += 1
        print('ranked', dataset, len(records), flush=True)
    assert len(records) == 680
    verify_sources(args.out)
    assert geometry_hash(poses) == policy['library_geometry']
    assert sha256(args.library_root / 'poses.db') == policy['library_db_sha256']
    save_json(args.out / 'rankings.json', records)
    save_json(args.out / 'timing.json', times)
    save_json(args.out / 'audit.json', {
        'images': len(records), 'baseline_and_pooled117_verified': verified,
        'failures_preserved': len(records) - verified,
        'images_and_prediction_hashes_checked': True,
        'nested_costs_checked': True, 'ranking_sha256': sha256(args.out / 'rankings.json'),
        'answers_parsed_during_rank': False,
    })


def evaluate(args):
    policy = verify_sources(args.out)
    assert sha256(args.out / 'rankings.json') == read(args.out / 'audit.json')['ranking_sha256']
    rows = read(args.out / 'rankings.json')
    truth = {(d, r['image']): r for d in DATASETS
             for r in read(ROOT / 'out' / d / 'answers.json')}
    assert len({r['family_id'] for r in truth.values()}) == policy['expected_families']
    result = summarize(rows, truth)
    result['stage'] = policy['stage']
    result['by_dataset'] = {d: summarize([r for r in rows if r['dataset'] == d],
        {k: v for k, v in truth.items() if k[0] == d})['groups'] for d in DATASETS}
    save_json(args.out / 'results.json', result)
    for group, values in result['groups'].items():
        print(group, 'n=', values['n'], 'failures=', values['failures'], values['counts'], flush=True)
    for d, values in result['by_dataset'].items():
        print(d, {g: v['counts'] for g, v in values.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('rank', 'evaluate'))
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args)
