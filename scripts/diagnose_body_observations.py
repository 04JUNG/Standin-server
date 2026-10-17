"""Development-only joint error diagnosis. Oracle relabeling never enters search."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_rendered import save_json, sha256
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.dual_label_search import LEFT_RIGHT
from standin_eval.perspective_search import normalize_batch

PARTS = ('shoulders', 'elbows', 'wrists', 'hips', 'knees', 'ankles')


def diagnose(xy, scores, gt):
    xy, scores, gt = np.asarray(xy, float), np.asarray(scores, float), np.asarray(gt, float)
    assert xy.shape == gt.shape == (17, 2) and scores.shape == (17,)
    assert np.isfinite(gt).all()
    torso = np.linalg.norm(gt[[5, 6]].mean(0) - gt[[11, 12]].mean(0))
    assert torso > 1e-6
    finite = np.isfinite(xy).all(1) & np.isfinite(scores)
    valid = finite & (scores >= .3)
    valid[:5] = False
    direct = np.linalg.norm(xy - gt, axis=1) / torso
    flipped = np.linalg.norm(xy - gt[LEFT_RIGHT], axis=1) / torso
    # One whole-body mapping; never pick a separate best label for each joint.
    swap = bool(valid.any() and flipped[valid].mean() < direct[valid].mean())
    target = gt[LEFT_RIGHT] if swap else gt
    error = flipped if swap else direct
    normalized = None
    if valid_query({'keypoints2d': xy, 'scores': scores}) is not None:
        normalized = np.linalg.norm(normalize_batch(xy) - normalize_batch(target), axis=1)
    joints, pairs = [], []
    for j in range(5, 17):
        if not finite[j]:
            continue
        joints.append({'index': j, 'part': PARTS[(j - 5) // 2], 'score': float(scores[j]),
                       'valid': bool(valid[j]), 'direct_error': float(direct[j]),
                       'global_error': float(error[j]),
                       'normalized_error': None if normalized is None else float(normalized[j])})
    for i, part in enumerate(PARTS):
        a, b = 5 + 2 * i, 6 + 2 * i
        if valid[a] and valid[b]:
            alternative = (np.linalg.norm(xy[a] - target[b]) + np.linalg.norm(xy[b] - target[a])) / (2 * torso)
            gain = float(error[[a, b]].mean() - alternative)
            pairs.append({'part': part, 'mean_error_reduction_if_pair_swapped': gain,
                          'material_reduction': gain > .25})
    return {'gt_torso_pixels': float(torso), 'oracle_global_swap': swap,
            'valid_body_joints': int(valid.sum()), 'joints': joints, 'pairs': pairs}


def stats(joints):
    result = {'n': len(joints)}
    if not joints:
        return result
    for key in ('direct_error', 'global_error', 'normalized_error'):
        values = [j[key] for j in joints if j[key] is not None]
        result[key] = {'n': len(values)}
        if values:
            values = np.asarray(values)
            result[key].update(median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                               over_025=int((values > .25).sum()), over_05=int((values > .5).sum()))
    return result


def summarize(rows):
    joints = [j for r in rows for j in r.get('joints', [])]
    valid = [j for j in joints if j['valid']]
    bins = [('lt03', -np.inf, .3), ('03_to_05', .3, .5), ('05_to_07', .5, .7), ('ge07', .7, np.inf)]
    return {'n': len(rows), 'searchable': sum(r['searchable'] for r in rows),
            'top5_hits': sum(r['top5_hit'] for r in rows),
            'oracle_global_swaps': sum(r.get('oracle_global_swap', False) for r in rows),
            'all_finite_joints': stats(joints), 'valid_joints': stats(valid),
            'by_part': {p: stats([j for j in valid if j['part'] == p]) for p in PARTS},
            'by_score': {name: stats([j for j in joints if lo <= j['score'] < hi]) for name, lo, hi in bins},
            'by_part_score': {p: {name: stats([j for j in joints if j['part'] == p and lo <= j['score'] < hi])
                                 for name, lo, hi in bins} for p in PARTS},
            'by_search_outcome': {label: stats([j for r in rows if r['searchable'] and r['top5_hit'] == hit
                                              for j in r.get('joints', []) if j['valid']])
                                  for label, hit in [('hit', True), ('miss', False)]},
            'local_pair_swap': {p: {'eligible': sum(q['part'] == p for r in rows for q in r.get('pairs', [])),
                                    'material_reduction': sum(q['part'] == p and q['material_reduction']
                                                              for r in rows for q in r.get('pairs', []))} for p in PARTS}}


def read(path):
    return json.loads(path.read_text())


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [args.source / n for n in ('answers.json', 'manifest.json', 'audit.json')]
    paths += sorted((args.source / 'predictions').glob('*.json'))
    paths += [args.rankings, Path(__file__), ROOT / 'scripts/diagnose_dual_camera_coverage.py',
              ROOT / 'standin_eval/dual_label_search.py', ROOT / 'standin_eval/perspective_search.py']
    hashes = {str(p): sha256(p) for p in paths}
    save_json(args.out / 'protocol.json', {
        'stage': 'post-hoc known development diagnosis; no search changes or new accuracy claim',
        'sample': 'all 360 images / 120 previously used families, including extraction failures',
        'error': 'raw Euclidean pixel error / GT projected shoulder-midpoint to hip-midpoint length',
        'global_mapping': 'one direct or whole-body LR GT mapping minimizes mean error on original score>=.3 body joints; ties direct; diagnostic oracle only',
        'score_bins': ['<.3', '[.3,.5)', '[.5,.7)', '>=.7'],
        'error_thresholds': [.25, .5],
        'local_pair': 'after global mapping, both scores>=.3; material if pair mean error reduced >.25 GT torso by swapping target pair; diagnostic not proof of label error',
        'secondary': 'predicted-anchor normalized error and baseline keepfirst_blend75 Top5 hit vs miss; no causal claim',
        'limits': 'scores not calibrated probabilities; joints/images within family correlated; renderer joint centers need not equal visible contour landmarks; no human rough accuracy',
        'input_hashes': hashes})
    targets = read(args.source / 'answers.json')
    manifest = {r['image']: r for r in read(args.source / 'manifest.json')}
    rankings = {r['image']: r for r in read(args.rankings)}
    assert len(targets) == len(manifest) == len(rankings) == 360
    assert len({r['family_id'] for r in targets}) == 120
    assert {r['image'] for r in targets} == set(manifest) == set(rankings)
    rows = []
    for target in targets:
        name = target['image']
        assert sha256(args.source / 'images' / name) == manifest[name]['sha256']
        path = args.source / 'predictions' / (Path(name).stem + '.json')
        pred, rank = read(path), rankings[name]
        assert sha256(path) == rank['prediction_sha256']
        searchable = valid_query(pred) is not None
        assert searchable == (not bool(rank.get('error')))
        row = {k: target[k] for k in ('image', 'group', 'family_id', 'pose_id')}
        row.update(searchable=searchable, top5_hit=target['pose_id'] in rank.get('keepfirst_blend75', []))
        if not pred.get('error'):
            row.update(diagnose(pred['keypoints2d'], pred['scores'], target['ground_truth_xy']))
        rows.append(row)
    assert all(sha256(Path(p)) == h for p, h in hashes.items())
    groups = {g: summarize([r for r in rows if r['group'] == g]) for g in sorted({r['group'] for r in rows})}
    groups['all'] = summarize(rows)
    save_json(args.out / 'results.json', {'groups': groups, 'records': rows, 'source_hashes_unchanged': True})
    print(json.dumps({g: {k: v for k, v in s.items() if k in ('n', 'searchable', 'top5_hits', 'oracle_global_swaps')}
                      for g, s in groups.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'rankings', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
