"""Recompute all expanded rankings using one full195-camera index."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_near_camera import SOURCE, ARMS, near_cameras, read, verify
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import camera_grid, build_camera_index, score_camera_index, normalize_batch


def audit(args):
    policy = verify(args.out)
    poses = load_library(args.library_root)
    assert geometry_hash(poses) == policy['library_geometry']
    assert sha256(args.library_root/'poses.db') == policy['library_db_sha256']
    main = read(args.out/'audit.json')
    assert sha256(args.out/'rankings.json') == main['ranking_sha256']
    rows = {r['image']: r for r in read(args.out/'rankings.json')}
    source_rows = {r['image']: r for r in read(SOURCE/'rankings.json')}
    manifest = read(SOURCE/'manifest.json')
    assert len(rows) == len(manifest) == 360
    assert set(rows) == {r['image'] for r in manifest}
    index = build_camera_index(poses, camera_grid()['yaw_pitch_perspective_117'] + near_cameras())
    permutation = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
    checked = failures = 0
    for item in manifest:
        name = item['image']
        row = rows[name]
        path = SOURCE/'predictions'/(Path(name).stem+'.json')
        assert sha256(SOURCE/'images'/name) == item['sha256']
        assert sha256(path) == row['prediction_sha256']
        pred = read(path)
        if pred.get('error'):
            assert row.get('error') and all(a not in row for a in ARMS)
            failures += 1
            continue
        xy = np.asarray(pred['keypoints2d'], dtype=np.float32)
        mask = np.asarray(pred['scores'], dtype=np.float32) >= .3
        if not np.isfinite(xy).all() or mask[5:].sum() < 6 or not mask[[5, 6, 11, 12]].all():
            assert row.get('error') and all(a not in row for a in ARMS)
            failures += 1
            continue
        assert hashlib.sha256(xy.tobytes()).hexdigest() == row['shared_xy_sha256']
        assert mask.tolist() == row['shared_mask']
        assert row['pooled_117'] == source_rows[name]['pooled_117']
        q = normalize_batch(xy)
        scores = np.minimum(score_camera_index(index, q, mask),
                            score_camera_index(index, q[permutation], mask[permutation]))
        order = np.argsort(scores, kind='stable')
        expected = [poses[int(i)].pose_id for i in order if np.isfinite(scores[int(i)])]
        assert row['expanded_195'] == expected[:5]
        first = source_rows[name]['pooled_117'][0]
        assert row['keepfirst_195'] == ([first] + [p for p in expected if p != first])[:5]
        checked += 1
        if checked % 60 == 0:
            print('audited', checked, flush=True)
    assert checked == main['baseline_top5_verified'] and failures == main['failures_preserved']
    result = {'full195_index_ranking_matches': checked, 'keepfirst_ranking_matches': checked,
              'failures_retained': failures, 'images_and_predictions_checked': len(manifest),
              'geometry_and_source_hashes_verified': True,
              'audit_script_sha256': sha256(Path(__file__)), 'ranking_sha256': main['ranking_sha256']}
    save_json(args.out/'independent-audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library-root', type=Path, required=True)
    audit(parser.parse_args())
