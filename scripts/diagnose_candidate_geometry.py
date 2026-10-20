"""Measure stored retrieval errors against source 3D; never rewrite rankings."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.pose_geometry_diagnostic import normalize_body, yaw_aligned_rms, body_label_errors
from standin_eval.perspective_search import normalize_batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'p25': float(np.quantile(values, .25)) if values else None,
            'p75': float(np.quantile(values, .75)) if values else None}


def run(a):
    paths = [a.source / f for f in ('protocol.json', 'rankings.json', 'answers.json', 'results.json')]
    hashes = {str(p): sha(p) for p in paths}
    policy, rankings, answers, original = [json.loads(p.read_text()) for p in paths]
    poses = load_library(a.library_root)
    assert geometry_hash(poses) == policy['hashes']['library_geometry']
    assert all(np.all(p.scores[5:] > 0) for p in poses), 'missing gallery body joints'
    bodies = normalize_body(np.stack([p.joints for p in poses]))
    ids = {p.pose_id: i for i, p in enumerate(poses)}
    ranking_map = {r['image']: r for r in rankings}
    thresholds = [1e-5, .05, .1, .2]
    distances = {}; neighbors = []
    for target in sorted({r['pose_id'] for r in answers}):
        ti = ids[target]
        d = yaw_aligned_rms(bodies[ti], bodies)
        assert d[ti] < 1e-10
        distances[target] = d
        other = d.copy(); other[ti] = np.inf
        ni = int(np.argmin(other))
        neighbors.append({'pose_id': target, 'nearest_other_id': poses[ni].pose_id,
                          'nearest_other_rms': float(other[ni]),
                          'other_count_within': {str(t): int((other <= t).sum()) for t in thresholds}})
    rows = []
    for answer in answers:
        target = answer['pose_id']; stored = ranking_map[answer['image']]
        row = {'image': answer['image'], 'group': answer['group'], 'pose_id': target,
               'error': stored.get('error')}
        if not row['error']:
            prediction = a.source / 'predictions' / (Path(answer['image']).stem+'.json')
            assert sha(prediction) == stored['prediction_sha256']
            pred = json.loads(prediction.read_text())
            mask = np.asarray(pred['scores']) >= .3
            assert mask.tolist() == stored['shared_mask']
            direct, swapped = body_label_errors(normalize_batch(pred['keypoints2d']),
                normalize_batch(answer['ground_truth_xy']), mask)
            row['oracle_xy_labels'] = {'direct_error_torso': direct, 'swapped_error_torso': swapped,
                                      'swap_reduces_error_by_quarter': swapped <= direct * .75 and direct > 0}
        for arm in ('anchor_4', 'anchor_117'):
            candidates = stored.get(arm, [])[:5]
            if not candidates:
                assert row['error']; row[arm] = None; continue
            assert len(candidates) == len(set(candidates)) == 5
            ci = [ids[k] for k in candidates]; d = distances[target][ci]
            pair_d = [float(yaw_aligned_rms(bodies[ci[i]], bodies[ci[j]])) for i in range(5) for j in range(i+1, 5)]
            row[arm] = {'ids': candidates, 'rms_to_source': d.tolist(),
                        'best_rms': float(d.min()), 'best_rms_id': candidates[int(d.argmin())],
                        'exact_top1': candidates[0] == target, 'exact_top5': target in candidates,
                        'same_family_top5': any(poses[k].family_id == poses[ids[target]].family_id for k in ci),
                        'explicit_mirror_top5': (target[:-7] if target.endswith('_mirror') else target+'_mirror') in candidates,
                        'closest_candidate_pair_rms': min(pair_d)}
        rows.append(row)
    summary = {}
    for group in ('perspective', 'orthographic_control'):
        subset = [r for r in rows if r['group'] == group]
        group_result = {'n': len(subset), 'failed': sum(bool(r['error']) for r in subset)}
        for arm in ('anchor_4', 'anchor_117'):
            valid = [r[arm] for r in subset if r[arm] is not None]
            missed = [r for r in valid if not r['exact_top5']]
            counts = {f'top{k}': sum(r[f'exact_top{k}'] for r in valid) for k in (1, 5)}
            assert all(counts[k] == original[group]['counts'][arm][k] for k in counts)
            group_result[arm] = {'exact_counts_unchanged': counts,
                'searchable_top5_misses': len(missed),
                'miss_best_rms': describe([r['best_rms'] for r in missed]),
                'misses_with_candidate_within': {str(t): sum(r['best_rms'] <= t for r in missed) for t in thresholds},
                'misses_with_same_family': sum(r['same_family_top5'] for r in missed),
                'misses_with_explicit_mirror': sum(r['explicit_mirror_top5'] for r in missed),
                'top1_rms': describe([r['rms_to_source'][0] for r in valid]),
                'top5_closest_pair_rms': describe([r['closest_candidate_pair_rms'] for r in valid]),
                'top5_contains_pair_within': {str(t): sum(r['closest_candidate_pair_rms'] <= t for r in valid) for t in thresholds}}
        summary[group] = group_result
        for arm in ('anchor_4', 'anchor_117'):
            missed = [r for r in subset if r[arm] and not r[arm]['exact_top5']]
            mirror = [r for r in missed if r[arm]['explicit_mirror_top5']]
            group_result[arm]['oracle_xy_label_swap'] = {
                'all_searchable_n': sum('oracle_xy_labels' in r for r in subset),
                'all_searchable_quarter_reduction': sum(r.get('oracle_xy_labels', {}).get('swap_reduces_error_by_quarter', False) for r in subset),
                'mirror_misses_n': len(mirror),
                'mirror_misses_quarter_reduction': sum(r['oracle_xy_labels']['swap_reduces_error_by_quarter'] for r in mirror),
                'other_misses_quarter_reduction': sum(r['oracle_xy_labels']['swap_reduces_error_by_quarter'] for r in missed if not r[arm]['explicit_mirror_top5'])}
    assert hashes == {str(p): sha(p) for p in paths}, 'source artifact mutated'
    report = {'scope': 'post-hoc diagnosis of previously evaluated synthetic 120 families / 240 images, not a new test',
        'metric': 'body12 RMS 3D distance in torso lengths; hip center; optimal yaw only; no reflection or joint swaps',
        'limits': ['not human relevance or acceptability', 'ignores face and hands',
                   'left/right XY test uses GT after retrieval, not an available correction or independent accuracy',
                   'thresholds describe geometry only, never replace exact-ID success',
                   'failures counted separately; geometry summaries conditional on searchable images'],
        'source_hashes': hashes, 'library_geometry_hash': geometry_hash(poses),
        'source_counts_verified': True, 'thresholds': thresholds, 'summary': summary,
        'source_neighborhoods': neighbors, 'records': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--library-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
