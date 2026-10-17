"""Read-only recomputation of prepared identities, projections and rankings."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.confirm_perspective_isolation import read, verify, select, GROUPS, ARMS
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import (
    Camera, camera_points, project, normalize_batch, camera_grid, build_camera_index, score_camera_index,
)
from standin_eval.z_axis_search import build_search_index, distances


def audit(args):
    policy = verify(args)
    prepared = read(args.out / 'prepared.json')
    assert prepared['stress_preflight_passed']
    assert sha256(args.out / 'manifest.json') == prepared['manifest_sha256']
    assert sha256(args.out / 'answers.json') == prepared['answers_sha256']
    assert sha256(args.out / 'rankings.json') == read(args.out / 'ranked.json')['rankings_sha256']
    poses = load_library(args.library_root)
    assert geometry_hash(poses) == policy['hashes']['library_geometry']
    by_id = {p.pose_id: p for p in poses}
    excluded = {r['family_id'] for path in policy['prior_answers'] for r in read(ROOT / path)}
    assert excluded == set(policy['excluded_families']) and len(excluded) == 400
    chosen = select(poses, excluded, seed=policy['seed'], count=policy['samples_families'])
    assert [p.pose_id for p in chosen] == policy['test_pose_ids']
    assert [p.family_id for p in chosen] == policy['test_families']
    assert not {p.family_id for p in chosen} & excluded
    answers = {r['image']: r for r in read(args.out / 'answers.json')}
    manifest = read(args.out / 'manifest.json')
    rankings = {r['image']: r for r in read(args.out / 'rankings.json')}
    expected_names = {f'{group}_{i:03d}.png' for group in GROUPS for i in range(120)}
    assert len(manifest) == len(answers) == len(rankings) == 360
    assert {m['image'] for m in manifest} == set(answers) == set(rankings) == expected_names
    indices = {n: build_camera_index(poses, camera_grid()[key]) for n, key in
               ((39, 'yaw_pitch_39'), (117, 'yaw_pitch_perspective_117'))}
    original = build_search_index(poses)
    permutation = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
    checked, failures, ratios = 0, 0, {g: [] for g in GROUPS}
    for item in manifest:
        name = item['image']
        answer = answers[name]
        i = int(Path(name).stem.rsplit('_', 1)[1])
        assert answer['pose_id'] == chosen[i].pose_id
        assert answer['family_id'] == chosen[i].family_id
        assert name == f"{answer['group']}_{i:03d}.png"
        camera = Camera(**answer['camera'])
        joints = by_id[answer['pose_id']].joints
        yaw, pitch = (15, 75, 135, 225)[i % 4], (-25, 20)[(i // 4) % 2]
        if answer['group'] == 'orthographic_control':
            assert camera == Camera((0, 45, 90, 180)[i % 4])
        elif answer['group'] == 'perspective':
            assert camera == Camera(yaw, pitch, (4.5, 6.5)[(i // 8) % 2])
        else:
            points = camera_points(joints, Camera(yaw, pitch))
            expected_distance = max(float(points[5:, 2].max() - 2*points[5:, 2].min()), .65-float(points[:, 2].min()))
            assert camera.yaw == yaw and camera.pitch == pitch
            assert abs(camera.distance - expected_distance) < 1e-6
        xy, valid = project(joints, camera)
        assert valid and np.allclose(normalize_batch(xy), normalize_batch(answer['ground_truth_xy']), atol=1e-5)
        ratio = 1.
        if camera.distance is not None:
            depth = camera_points(joints, camera)[5:, 2] + camera.distance
            ratio = float(depth.max()/depth.min())
        assert abs(ratio-answer['body_depth_ratio']) < 1e-6
        ratios[answer['group']].append(ratio)
        assert sha256(args.out / 'images' / name) == item['sha256']
        path = args.out / 'predictions' / (Path(name).stem + '.json')
        row = rankings[name]
        assert sha256(path) == row['prediction_sha256']
        prediction = read(path)
        if prediction.get('error'):
            assert row.get('error') and all(arm not in row for arm in ARMS)
            failures += 1
            continue
        xy = np.asarray(prediction['keypoints2d'], dtype=np.float32)
        mask = np.asarray(prediction['scores'], dtype=np.float32) >= .3
        if not np.isfinite(xy).all() or mask[5:].sum() < 6 or not mask[[5, 6, 11, 12]].all():
            assert row.get('error') and all(arm not in row for arm in ARMS)
            failures += 1
            continue
        assert not row.get('error')
        assert hashlib.sha256(xy.tobytes()).hexdigest() == row['shared_xy_sha256']
        assert mask.tolist() == row['shared_mask']
        q = normalize_batch(xy)
        cost = distances(original, q, np.zeros(17), valid_mask=mask, z_weight=0)
        expected = []
        for j in np.argsort(cost, kind='stable'):
            pid = original.pose_ids[int(j)]
            if pid not in expected:
                expected.append(pid)
            if len(expected) == 5:
                break
        assert expected == row['anchor_4']
        for n, index in indices.items():
            direct = score_camera_index(index, q, mask)
            alternate = score_camera_index(index, q[permutation], mask[permutation])
            order = np.argsort(np.minimum(direct, alternate), kind='stable')
            assert row[f'pooled_{n}'] == [poses[int(j)].pose_id for j in order[:5]]
        checked += 1
        if checked % 60 == 0:
            print('audited', checked, flush=True)
    result = {
        'frozen_sources_and_inputs_verified': True,
        'random_family_member_selection_replayed': True,
        'heldout_families': 120, 'excluded_prior_families': 400, 'family_overlap': 0,
        'mirror_source_count': sum(p.pose_id.endswith('_mirror') for p in chosen),
        'images_and_projected_answers_verified': 360,
        'prediction_hashes_checked': 360, 'all_three_rankings_recomputed': checked,
        'extraction_failures_preserved': failures,
        'strong_ratio_ge_1_5': sum(v >= 1.5 for v in ratios['strong_perspective']),
        'depth_ratio_min_median_max': {g: np.quantile(v, [0, .5, 1]).tolist() for g, v in ratios.items()},
        'audit_script_sha256': sha256(Path(__file__)),
        'ranking_sha256': sha256(args.out / 'rankings.json'),
    }
    assert result['strong_ratio_ge_1_5'] >= 100
    save_json(args.out / 'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    audit(parser.parse_args())
