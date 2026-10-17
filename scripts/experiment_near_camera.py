"""Development-only expansion to nearer gallery cameras; no new holdout."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import Camera, camera_grid, build_camera_index, score_camera_index, normalize_batch
from standin_eval.dual_label_search import alternate_labels, merged_order

SOURCE = ROOT / 'out/perspective-isolation-confirm-20261013'
ARMS = ('pooled_117', 'expanded_195', 'keepfirst_195')
GROUPS = ('perspective', 'strong_perspective', 'orthographic_control')
CODE = ('scripts/experiment_near_camera.py', 'standin_eval/perspective_search.py',
        'standin_eval/dual_label_search.py', 'scripts/diagnose_dual_camera_coverage.py',
        'scripts/experiment_z_axis_search.py', 'src/bvh.py')


def read(path):
    return json.loads(path.read_text())


def near_cameras():
    return [Camera(c.yaw, c.pitch, distance) for c in camera_grid()['yaw_pitch_39'] for distance in (2., 3.)]


def select_policy(counts):
    baseline = counts['pooled_117']
    eligible = []
    for arm in ARMS[1:]:
        current = counts[arm]
        if (current['strong_perspective']['top5'] - baseline['strong_perspective']['top5'] >= 6
                and all(current[g]['top1'] >= baseline[g]['top1'] for g in GROUPS)
                and all(current[g]['top5'] >= baseline[g]['top5'] for g in ('perspective', 'orthographic_control'))):
            eligible.append(arm)
    eligible.sort(key=lambda arm: (
        counts[arm]['strong_perspective']['top5'],
        sum(counts[arm][g]['top5'] for g in GROUPS),
        sum(counts[arm][g]['top1'] for g in GROUPS), arm == 'keepfirst_195'), reverse=True)
    return eligible


def verify(out):
    policy = read(out / 'protocol.json')
    for name, digest in policy['frozen_sources'].items():
        assert sha256(ROOT / name) == digest
    for name, digest in policy['input_hashes'].items():
        assert sha256(ROOT / name) == digest
    return policy


def rank(args):
    args.out.mkdir(parents=True, exist_ok=False)
    poses = load_library(args.library_root)
    source_policy = read(SOURCE / 'protocol.json')
    assert geometry_hash(poses) == source_policy['hashes']['library_geometry']
    assert sha256(args.library_root / 'poses.db') == source_policy['hashes']['library_db']
    manifest = read(SOURCE / 'manifest.json')
    previous = {r['image']: r for r in read(SOURCE / 'rankings.json')}
    inputs = {str((SOURCE / name).relative_to(ROOT)): sha256(SOURCE / name) for name in
              ('manifest.json', 'protocol.json', 'answers.json', 'rankings.json', 'results.json', 'audit.json')}
    for item in manifest:
        assert sha256(SOURCE / 'images' / item['image']) == item['sha256']
        path = SOURCE / 'predictions' / (Path(item['image']).stem + '.json')
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        assert sha256(path) == previous[item['image']]['prediction_sha256']
    policy = {
        'stage': 'development only; all120 families/360 renders previously examined, retired as holdout for this hypothesis',
        'source': str(SOURCE.relative_to(ROOT)), 'images': 360, 'families': 120,
        'arms': ARMS, 'groups': GROUPS,
        'hypothesis': 'same angles and XY but add nearer distances2,3 to original None,4,8;195 total cameras',
        'near_cameras': [asdict(c) for c in near_cameras()],
        'fixed': 'cached Human-Art XY and >=.3 mask; >=6 body plus four anchors; same hip/torso normalization and dual-label min',
        'keepfirst': 'preserve original pooled117 first pose, fill remaining ranks from expanded195 order without duplication',
        'invalid_projection': 'same all17 joint depth>.1 rule; invalid near views cannot score; no all-surface collision guarantee for retrieval',
        'selection': 'strong Top5 gain>=6/120; all3 group Top1 nondecreasing; moderate/control Top5 nondecreasing; maximize strong Top5 then total Top5 then total Top1; ties prefer keepfirst',
        'scoring': 'exact source pose ID; failures stay in denominator; no mirror credit',
        'answer_access': 'rank only hashes answers; evaluate parses them after rankings are saved',
        'frozen_sources': {p: sha256(ROOT / p) for p in CODE}, 'input_hashes': inputs,
        'library_geometry': geometry_hash(poses),
        'library_db_sha256': sha256(args.library_root / 'poses.db'),
    }
    save_json(args.out / 'protocol.json', policy)
    indices = {117: build_camera_index(poses, camera_grid()['yaw_pitch_perspective_117']),
               78: build_camera_index(poses, near_cameras())}
    rows, timings = [], []
    for item in manifest:
        name = item['image']
        path = SOURCE / 'predictions' / (Path(name).stem + '.json')
        row = {'image': name, 'prediction_sha256': sha256(path)}
        query = valid_query(read(path))
        if query is None:
            assert previous[name].get('error')
            row['error'] = 'original Human-Art extraction failure'
            rows.append(row)
            continue
        xy, mask = query
        row['shared_xy_sha256'] = hashlib.sha256(xy.tobytes()).hexdigest()
        row['shared_mask'] = mask.tolist()
        assert row['shared_xy_sha256'] == previous[name]['shared_xy_sha256']
        assert row['shared_mask'] == previous[name]['shared_mask']
        q = normalize_batch(xy)
        alternative, alt_mask = alternate_labels(q, mask)
        start = time.perf_counter()
        costs = {n: np.minimum(score_camera_index(index, q, mask),
                               score_camera_index(index, alternative, alt_mask)) for n, index in indices.items()}
        orders = {'pooled_117': merged_order(costs[117], costs[117]),
                  'expanded_195': merged_order(costs[117], costs[78]),
                  'keepfirst_195': merged_order(costs[117], costs[78], preserve_first=True)}
        timings.append(time.perf_counter()-start)
        for arm, order in orders.items():
            row[arm] = [poses[int(i)].pose_id for i in order[:5]]
        assert row['pooled_117'] == previous[name]['pooled_117']
        assert row['pooled_117'][0] == row['keepfirst_195'][0]
        assert all(len(set(row[arm])) == 5 for arm in ARMS)
        rows.append(row)
        if len(rows) % 40 == 0:
            print('ranked', len(rows), '/360', flush=True)
    assert len(rows) == 360
    verify(args.out)
    save_json(args.out / 'rankings.json', rows)
    save_json(args.out / 'timing.json', {'all_arms_query_seconds': timings})
    save_json(args.out / 'audit.json', {
        'baseline_top5_verified': sum('error' not in r for r in rows),
        'failures_preserved': sum('error' in r for r in rows),
        'images_and_predictions_verified': len(rows), 'xy_and_mask_preserved': True,
        'keepfirst_top1_preserved': True, 'near_projection_invalid_count': int((~indices[78][1]).sum()),
        'near_projection_total': int(indices[78][1].size),
        'ranking_sha256': sha256(args.out / 'rankings.json'),
    })


def evaluate(args):
    verify(args.out)
    assert not (args.out / 'results.json').exists()
    assert sha256(args.out / 'rankings.json') == read(args.out / 'audit.json')['ranking_sha256']
    rows = read(args.out / 'rankings.json')
    answers = {r['image']: r for r in read(SOURCE / 'answers.json')}
    assert len(rows) == len(answers) == len({r['image'] for r in rows}) == 360
    assert set(answers) == {r['image'] for r in rows}
    records = []
    for row in rows:
        target = answers[row['image']]
        record = {k: target[k] for k in ('image', 'pose_id', 'family_id', 'group')}
        record['error'] = row.get('error')
        record['arms'] = {}
        for arm in ARMS:
            ids = row.get(arm, [])
            record['arms'][arm] = {'top1': bool(ids and ids[0] == target['pose_id']),
                                   'top5': target['pose_id'] in ids}
        records.append(record)
    counts = {arm: {g: {metric: sum(r['arms'][arm][metric] for r in records if r['group'] == g)
                        for metric in ('top1', 'top5')} for g in GROUPS} for arm in ARMS}
    original = read(SOURCE / 'results.json')
    for g in GROUPS:
        assert sum(r['group'] == g for r in records) == 120
        assert counts['pooled_117'][g] == original[g]['counts']['pooled_117']
    changes = {}
    for arm in ARMS[1:]:
        changes[arm] = {}
        for g in GROUPS:
            subset = [r for r in records if r['group'] == g]
            gains = sum(r['arms'][arm]['top5'] and not r['arms']['pooled_117']['top5'] for r in subset)
            losses = sum(not r['arms'][arm]['top5'] and r['arms']['pooled_117']['top5'] for r in subset)
            changes[arm][g] = {'gains': gains, 'losses': losses, 'net': gains-losses}
    eligible = select_policy(counts)
    result = {'stage': 'development only, not a new validation', 'families': 120, 'images': 360,
              'counts': counts, 'top5_changes': changes, 'eligible': eligible,
              'selected': eligible[0] if eligible else None,
              'failures': {g: sum(bool(r['error']) for r in records if r['group'] == g) for g in GROUPS},
              'records': records}
    save_json(args.out / 'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('rank', 'evaluate'))
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args)
