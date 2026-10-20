"""Independent scalar checks of saved joint diagnostics; no rank changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(out, source, rankings_path):
    protocol, result = read(out / 'protocol.json'), read(out / 'results.json')
    for p, h in protocol['input_hashes'].items():
        assert digest(Path(p)) == h
    targets = {r['image']: r for r in read(source / 'answers.json')}
    ranks = {r['image']: r for r in read(rankings_path)}
    rows = result['records']
    assert len(rows) == len(targets) == 360
    assert len({r['image'] for r in rows}) == 360
    checked = 0
    max_normalized_difference = 0.
    for row in rows:
        target = targets[row['image']]
        pred = read(source / 'predictions' / (Path(row['image']).stem + '.json'))
        assert row['top5_hit'] == (target['pose_id'] in ranks[row['image']].get('keepfirst_blend75', []))
        gt, xy, scores = target['ground_truth_xy'], pred['keypoints2d'], pred['scores']
        torso = math.hypot((gt[5][0]+gt[6][0]-gt[11][0]-gt[12][0])/2,
                           (gt[5][1]+gt[6][1]-gt[11][1]-gt[12][1])/2)
        flip = lambda j: j + 1 if j % 2 else j - 1
        distance = lambda j, k: math.hypot(xy[j][0]-gt[k][0], xy[j][1]-gt[k][1])/torso
        mask = [j for j in range(5, 17) if scores[j] >= .3]
        assert row['searchable'] == (len(mask) >= 6 and all(j in mask for j in (5, 6, 11, 12)))
        swap = sum(distance(j, flip(j)) for j in mask) < sum(distance(j, j) for j in mask)
        assert row['oracle_global_swap'] == swap
        for joint in row['joints']:
            j = joint['index']; k = flip(j) if swap else j
            assert math.isclose(joint['direct_error'], distance(j, j), abs_tol=1e-10)
            assert math.isclose(joint['global_error'], distance(j, k), abs_tol=1e-10)
            assert joint['score'] == scores[j] and joint['valid'] == (j in mask)
            if row['searchable']:
                def normalized(points, index):
                    hip = [(points[11][c]+points[12][c])/2 for c in (0, 1)]
                    shoulder = [(points[5][c]+points[6][c])/2 for c in (0, 1)]
                    length = math.hypot(shoulder[0]-hip[0], shoulder[1]-hip[1])
                    return [(points[index][c]-hip[c])/(1. if length < 1e-6 else length) for c in (0, 1)]
                a, b = normalized(xy, j), normalized(gt, k)
                difference = abs(joint['normalized_error'] - math.dist(a, b))
                max_normalized_difference = max(max_normalized_difference, difference)
                # Production normalization is float32; this check uses scalar float64.
                assert math.isclose(joint['normalized_error'], math.dist(a, b), rel_tol=1e-6, abs_tol=3e-6), (row['image'], j, joint['normalized_error'], math.dist(a, b))
            else:
                assert joint['normalized_error'] is None
            checked += 1
        for pair in row['pairs']:
            part_names = ('shoulders', 'elbows', 'wrists', 'hips', 'knees', 'ankles')
            a = 5 + 2 * part_names.index(pair['part']); b = a + 1
            ka, kb = (b, a) if swap else (a, b)
            gain = (distance(a, ka)+distance(b, kb)-distance(a, kb)-distance(b, ka))/2
            assert math.isclose(pair['mean_error_reduction_if_pair_swapped'], gain, abs_tol=1e-10)
            assert pair['material_reduction'] == (gain > .25)

    def check_stats(saved, joints):
        assert saved['n'] == len(joints)
        for key in ('direct_error', 'global_error', 'normalized_error'):
            values = np.array([j[key] for j in joints if j[key] is not None])
            if not joints:
                continue
            assert saved[key]['n'] == len(values)
            if len(values):
                for quantile, label in ((.5, 'median'), (.9, 'p90')):
                    assert np.isclose(saved[key][label], np.quantile(values, quantile), atol=1e-12)
                for threshold, label in ((.25, 'over_025'), (.5, 'over_05')):
                    assert saved[key][label] == int((values > threshold).sum())

    for name, saved in result['groups'].items():
        group = rows if name == 'all' else [r for r in rows if r['group'] == name]
        assert saved['n'] == len(group)
        assert saved['searchable'] == sum(r['searchable'] for r in group)
        assert saved['top5_hits'] == sum(r['top5_hit'] for r in group)
        assert saved['oracle_global_swaps'] == sum(r['oracle_global_swap'] for r in group)
        joints = [j for r in group for j in r['joints']]
        check_stats(saved['all_finite_joints'], joints)
        check_stats(saved['valid_joints'], [j for j in joints if j['valid']])
        bins = [('lt03', -math.inf, .3), ('03_to_05', .3, .5), ('05_to_07', .5, .7), ('ge07', .7, math.inf)]
        for label, lo, hi in bins:
            check_stats(saved['by_score'][label], [j for j in joints if lo <= j['score'] < hi])
        for part in saved['by_part']:
            check_stats(saved['by_part'][part], [j for j in joints if j['part'] == part and j['valid']])
            for label, lo, hi in bins:
                check_stats(saved['by_part_score'][part][label], [j for j in joints if j['part'] == part and lo <= j['score'] < hi])
            pairs = [p for r in group for p in r['pairs'] if p['part'] == part]
            assert saved['local_pair_swap'][part] == {'eligible': len(pairs), 'material_reduction': sum(p['material_reduction'] for p in pairs)}
        for label, hit in [('hit', True), ('miss', False)]:
            check_stats(saved['by_search_outcome'][label], [j for r in group if r['searchable'] and r['top5_hit'] == hit for j in r['joints'] if j['valid']])
    record = {'images_checked': len(rows), 'scalar_joint_errors_checked': checked,
              'all_summary_cells_checked': True, 'source_hashes_unchanged': True,
              'max_normalized_float32_vs_scalar64_difference': max_normalized_difference,
              'results_sha256': digest(out / 'results.json'), 'auditor_sha256': digest(Path(__file__))}
    with (out / 'audit.json').open('x') as f:
        json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('out', 'source', 'rankings'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); audit(a.out, a.source, a.rankings)
