"""Fresh-family test of perspective projection with identical dual-label XY.

All setup is fixed before inference. A stronger-perspective stress condition
uses source geometry ONLY when generating images, never when ranking them.
"""
import argparse
import fcntl
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
from scripts.experiment_z_rendered import render_pose, save_json, sha256
from scripts.experiment_perspective_images import infer as image_infer
from scripts.confirm_camera_fit import geometry_hash, paired_comparison
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import (
    Camera, camera_points, project, normalize_batch, camera_grid,
    build_camera_index, score_camera_index,
)
from standin_eval.dual_label_search import alternate_labels, merged_order

SEED = 20261031
COUNT = 120
GROUPS = ('perspective', 'strong_perspective', 'orthographic_control')
ARMS = ('anchor_4', 'pooled_39', 'pooled_117')
PRIOR = ('z-rough-20261012/experiment', 'z-diagnosis-20261012-heldout',
         'perspective-images-20261012', 'perspective-replication-20261012',
         'camera-fit-dev-20261012', 'camera-fit-confirm-20261012',
         'dual-label-confirm-balanced-20261012')
SOURCE_FILES = (
    'scripts/confirm_perspective_isolation.py', 'scripts/diagnose_dual_camera_coverage.py',
    'scripts/experiment_z_axis_search.py', 'scripts/experiment_z_rendered.py',
    'scripts/experiment_perspective_images.py', 'scripts/confirm_camera_fit.py',
    'standin_eval/perspective_search.py', 'standin_eval/dual_label_search.py',
    'standin_eval/z_diagnostics.py', 'src/bvh.py',
)


def read(path):
    return json.loads(path.read_text())


def select(poses, excluded, seed=SEED, count=COUNT):
    families = {}
    for pose in poses:
        if pose.family_id not in excluded:
            families.setdefault(pose.family_id, []).append(pose)
    pool = list(families.values())
    rng = np.random.default_rng(seed)
    selected = [pool[int(i)] for i in rng.choice(len(pool), count, replace=False)]
    return [members[int(rng.integers(len(members)))] for members in selected]


def projection_strength(joints, camera):
    if camera.distance is None:
        return 1.
    depths = camera_points(joints, camera)[5:, 2] + camera.distance
    if np.min(depths) <= .1:
        raise ValueError('invalid depth')
    return float(np.max(depths) / np.min(depths))


def strong_camera(joints, yaw, pitch):
    points = camera_points(joints, Camera(yaw, pitch))
    z = points[5:, 2]
    # Solve (d+zmax)/(d+zmin)=2; maintain 0.65 torso lengths from all joints.
    # The unchanged renderer additionally checks its full surface volume.
    distance = max(float(np.max(z) - 2 * np.min(z)), .65 - float(np.min(points[:, 2])))
    return Camera(yaw, pitch, distance)


def sources():
    return {p: sha256(ROOT / p) for p in SOURCE_FILES}


def verify(args):
    policy = read(args.out / 'protocol.json')
    assert sources() == policy['frozen_sources']
    assert sha256(args.model) == policy['hashes']['model']
    assert sha256(args.library_root / 'poses.db') == policy['hashes']['library_db']
    for path, digest in policy['prior_answers'].items():
        assert sha256(ROOT / path) == digest
    assert sha256(ROOT / 'out/dual-camera-ablation-dev-20261013/results.json') == policy['hashes']['development']
    return policy


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'images').mkdir()
    poses = load_library(args.library_root)
    prior_paths = [ROOT / 'out' / name / 'answers.json' for name in PRIOR]
    excluded = {r['family_id'] for p in prior_paths for r in read(p)}
    assert len(excluded) == 400
    selected = select(poses, excluded)
    policy = {
        'stage': 'fresh-family fixed-size confirmation of projection effect',
        'seed': SEED, 'samples_families': COUNT, 'groups': GROUPS, 'arms': ARMS,
        'primary': 'pooled117 versus pooled39, perspective group; single primary contrast',
        'secondary': 'strong-perspective stress and orthographic control; anchor4 descriptive only',
        'fixed': 'same Human-Art M raw XY, >=.3 mask, >=6 body joints and four shoulder/hip anchors; hip/torso norm; dual-label min for both primary arms',
        'projection': '39 orthographic yaw/pitch versus same 39 angles with distances None,4,8 (117); no estimated Z',
        'query_camera': 'yaw 15/75/135/225; pitch -25/20; moderate distance 4.5/6.5; control yaw0/45/90/180 orthographic',
        'stress_camera': 'same yaw/pitch as moderate; source geometry solves body depth max/min=2, clamped so all joints remain >=.65 torso lengths ahead of camera; no pose substitution; report achieved ratio',
        'stress_preflight': 'at least 100/120 stress images must have max/min depth ratio>=1.5 before inference; otherwise stop, do not relabel stress as successful',
        'scoring': 'exact original pose ID; mirror IDs are not interchangeable; all extraction failures count as misses',
        'success_gate': 'moderate Top5 delta>=.05, paired bootstrap95 lower>0, exact two-sided McNemar p<.05; moderate Top1 and both control Top1/Top5 nondecreasing; stress Top5 nondecreasing',
        'statistics': '20000 paired resamples within group by family, seed20261019; 360 renders are only120 families, groups never pooled as independent poses',
        'sampling': 'exclude all 400 prior query families; uniformly choose120 families then a uniform member in each; no resampling from outcomes',
        'limits': ['procedural mannequin contours, not human roughs', 'closed-set gallery',
                   'family holdout is not guaranteed motion-sequence or semantic holdout',
                   'depth magnification ratio is not a perceptual quality score'],
        'excluded_families': sorted(excluded),
        'test_families': [p.family_id for p in selected],
        'test_pose_ids': [p.pose_id for p in selected],
        'frozen_sources': sources(),
        'prior_answers': {str(p.relative_to(ROOT)): sha256(p) for p in prior_paths},
        'hashes': {'model': sha256(args.model), 'library_db': sha256(args.library_root / 'poses.db'),
                   'library_geometry': geometry_hash(poses),
                   'development': sha256(ROOT / 'out/dual-camera-ablation-dev-20261013/results.json')},
    }
    save_json(args.out / 'protocol.json', policy)
    manifest, answers = [], []
    for i, pose in enumerate(selected):
        yaw, pitch = (15, 75, 135, 225)[i % 4], (-25, 20)[(i // 4) % 2]
        cameras = (Camera(yaw, pitch, (4.5, 6.5)[(i // 8) % 2]),
                   strong_camera(pose.joints, yaw, pitch), Camera((0, 45, 90, 180)[i % 4]))
        for group, camera in zip(GROUPS, cameras):
            name = f'{group}_{i:03d}.png'
            image, gt = render_pose(pose.joints, camera.yaw, style='rough', seed=SEED+i,
                                    pitch=camera.pitch, distance=camera.distance)
            image.save(args.out / 'images' / name)
            manifest.append({'image': name, 'sha256': sha256(args.out / 'images' / name)})
            answers.append({'image': name, 'group': group, 'family_id': pose.family_id,
                            'pose_id': pose.pose_id, 'camera': asdict(camera),
                            'body_depth_ratio': projection_strength(pose.joints, camera),
                            'ground_truth_xy': gt.tolist()})
        if (i+1) % 10 == 0:
            print('rendered', i+1, '/', COUNT, flush=True)
    save_json(args.out / 'manifest.json', manifest)
    save_json(args.out / 'answers.json', answers)
    strong_count = sum(r['body_depth_ratio'] >= 1.5 for r in answers if r['group'] == 'strong_perspective')
    save_json(args.out / 'prepared.json', {
        'manifest_sha256': sha256(args.out / 'manifest.json'),
        'answers_sha256': sha256(args.out / 'answers.json'),
        'stress_images_ratio_ge_1_5': strong_count,
        'stress_preflight_passed': strong_count >= 100,
    })
    assert strong_count >= 100, 'stress preflight failed, no inference allowed'
    print('prepared', len(manifest), 'images; strong >=1.5:', strong_count, flush=True)


def prepared(args):
    verify(args)
    record = read(args.out / 'prepared.json')
    assert record['stress_preflight_passed']
    assert sha256(args.out / 'manifest.json') == record['manifest_sha256']
    assert sha256(args.out / 'answers.json') == record['answers_sha256']


def infer(args):
    prepared(args)
    image_infer(args)  # Reads manifest, images, model only, no source answers.


def rank(args):
    prepared(args)
    assert not (args.out / 'rankings.json').exists(), 'rankings already exist'
    poses = load_library(args.library_root)
    policy = verify(args)
    assert geometry_hash(poses) == policy['hashes']['library_geometry']
    keys = {4: 'baseline_4', 39: 'yaw_pitch_39', 117: 'yaw_pitch_perspective_117'}
    indices = {n: build_camera_index(poses, camera_grid()[key]) for n, key in keys.items()}
    rows, timing = [], {arm: [] for arm in ARMS}
    for item in read(args.out / 'manifest.json'):
        name = item['image']
        assert sha256(args.out / 'images' / name) == item['sha256']
        path = args.out / 'predictions' / (Path(name).stem + '.json')
        row = {'image': name, 'prediction_sha256': sha256(path)}
        prediction = read(path)
        query = valid_query(prediction)
        if query is None:
            row['error'] = prediction.get('error', 'insufficient Human-Art body joints/anchors')
            rows.append(row)
            continue
        xy, mask = query
        row['shared_xy_sha256'] = hashlib.sha256(xy.tobytes()).hexdigest()
        row['shared_mask'] = mask.tolist()
        q = normalize_batch(xy)
        alternate, alternate_mask = alternate_labels(q, mask)
        for n, index in indices.items():
            start = time.perf_counter()
            direct = score_camera_index(index, q, mask)
            if n == 4:
                order = np.argsort(direct, kind='stable')
                arm = 'anchor_4'
            else:
                swapped = score_camera_index(index, alternate, alternate_mask)
                order = merged_order(direct, swapped)
                arm = f'pooled_{n}'
            row[arm] = [poses[int(i)].pose_id for i in order[:5]]
            timing[arm].append(time.perf_counter()-start)
        rows.append(row)
        if len(rows) % 40 == 0:
            print('ranked', len(rows), '/360', flush=True)
    assert len(rows) == 360
    verify(args)
    save_json(args.out / 'rankings.json', rows)
    save_json(args.out / 'timing.json', timing)
    save_json(args.out / 'ranked.json', {'rankings_sha256': sha256(args.out / 'rankings.json'),
                                        'answers_parsed_during_rank': False})


def success_gate(groups):
    p = groups['perspective']
    c = groups['orthographic_control']
    s = groups['strong_perspective']
    contrast = p['projection_vs_orthographic']
    return bool(contrast['delta'] >= .05 and contrast['paired_bootstrap_95ci'][0] > 0
                and contrast['exact_mcnemar_p'] < .05
                and p['counts']['pooled_117']['top1'] >= p['counts']['pooled_39']['top1']
                and all(c['counts']['pooled_117'][k] >= c['counts']['pooled_39'][k] for k in ('top1', 'top5'))
                and s['counts']['pooled_117']['top5'] >= s['counts']['pooled_39']['top5'])


def evaluate(args):
    prepared(args)
    assert not (args.out / 'results.json').exists(), 'already evaluated'
    assert sha256(args.out / 'rankings.json') == read(args.out / 'ranked.json')['rankings_sha256']
    rows = read(args.out / 'rankings.json')
    answers = {r['image']: r for r in read(args.out / 'answers.json')}
    assert len(rows) == len(answers) == 360
    assert len({r['image'] for r in rows}) == 360
    result = {}
    for group in GROUPS:
        records = []
        for row in rows:
            answer = answers[row['image']]
            if answer['group'] != group:
                continue
            record = {k: answer[k] for k in ('image', 'pose_id', 'family_id', 'body_depth_ratio')}
            if row.get('error'):
                record['error'] = row['error']
            for arm in ARMS:
                candidates = row.get(arm, [])
                record[arm+'_rank'] = candidates.index(answer['pose_id'])+1 if answer['pose_id'] in candidates else None
            records.append(record)
        assert len(records) == len({r['family_id'] for r in records}) == COUNT
        counts = {arm: {f'top{k}': sum(r[arm+'_rank'] is not None and r[arm+'_rank'] <= k for r in records)
                        for k in (1, 5)} for arm in ARMS}
        result[group] = {'n': COUNT, 'extraction_failures': sum('error' in r for r in records),
                         'counts': counts, 'projection_vs_orthographic': paired_comparison(records, 'pooled_117', 'pooled_39'),
                         'depth_ratio_min_median_max': np.quantile([r['body_depth_ratio'] for r in records], [0, .5, 1]).tolist(),
                         'records': records}
        print(group, {k: v for k, v in result[group].items() if k != 'records'}, flush=True)
    result['gate_passed'] = success_gate(result)
    save_json(args.out / 'results.json', result)
    print('gate_passed', result['gate_passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'rank', 'evaluate'))
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / '.perspective-isolation-confirm.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.stage](args)
