"""Compare confidence camera fitting, frozen rerank and equal-budget uniform fitting."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_confidence_cost import verify as verify_prior, read, BASE, GROUPS
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.experiment_near_camera import near_cameras
from standin_eval.perspective_search import normalize_batch, camera_grid, build_camera_index
from standin_eval.camera_refinement import normalized_joints
from standin_eval.confidence_camera_fit import fit_confidence

RERANK = 'confidence100'
CONTROL = 'uniform_refit'
JOINT = 'confidence_refit'
ARMS = (BASE, RERANK, CONTROL, JOINT)
CODE = ('scripts/experiment_confidence_camera.py', 'standin_eval/confidence_camera_fit.py',
        'standin_eval/confidence_cost.py', 'standin_eval/camera_refinement.py',
        'standin_eval/dual_label_search.py', 'standin_eval/perspective_search.py',
        'scripts/experiment_refinement_blend.py', 'scripts/experiment_head_hint.py')


def prior_args(args):
    return SimpleNamespace(source=args.source, refinement_out=args.refinement_out, blend_out=args.blend_out,
                           out=args.confidence_out, library_root=args.library_root, model=args.model)


def gate(counts):
    return bool(counts[JOINT]['strong_perspective']['top5'] >= max(counts[b]['strong_perspective']['top5'] for b in (BASE, CONTROL)) + 3
                and counts[JOINT]['strong_perspective']['top5'] >= counts[RERANK]['strong_perspective']['top5']
                and all(counts[JOINT][g]['top5'] >= max(counts[b][g]['top5'] for b in ARMS[:3]) for g in (GROUPS[0], GROUPS[2]))
                and all(counts[JOINT][g]['top1'] == counts[BASE][g]['top1'] for g in GROUPS))


def verify(args):
    verify_prior(prior_args(args))
    p = read(args.out / 'protocol.json')
    assert all(sha256(ROOT / f) == h for f, h in p['frozen_sources'].items())
    assert all(sha256(Path(f)) == h for f, h in p['input_hashes'].items())


def rank(args):
    verify_prior(prior_args(args))
    assert read(args.confidence_out / 'audit.json')['ranking_sha256'] == sha256(args.confidence_out / 'rankings.json')
    args.out.mkdir(parents=True, exist_ok=False); (args.out / 'costs').mkdir()
    paths = [args.confidence_out / n for n in ('protocol.json', 'rankings.json', 'results.json', 'audit.json')]
    paths += [args.source / 'answers.json', args.refinement_out / 'rankings.json']
    save_json(args.out / 'protocol.json', {
        'stage': 'development only, same examined 120 families / 360 images',
        'arms': ARMS, 'hypothesis': 'confidence should affect camera fitting, not only frozen-camera reranking',
        'weight': '.25+.75*clip((score-.3)/.4,0,1); strength1 fixed, no new strength search',
        'objective': 'weighted mean body Euclidean residual; final .25*117cost+.75*optimized objective',
        'control': 'uniform positive weights in identical optimizer, on all searchable images',
        'starts': 'three best195 coarse cameras per label/objective plus previous pooled body-fitted camera in BOTH labels; all1981 poses',
        'budget': 'four fixed step scales from camera_refinement, two sweeps each, yawperiodic360, pitch+-60, inverse distance0..1.25; identical for control and weighted arm',
        'fixed': 'raw Human-Art XY, score>=.3 mask, hip/torso normalization and first117 result; no head, no queryZ, no extra scale/translation fit',
        'selection': 'strongTop5>=max(original,uniform_refit)+3 and >=frozenconfidence100; moderate/controlTop5>=all3comparators; allTop1 preserved',
        'failure': 'retain all original extraction failures in120 per group, no replacement',
        'ranking_inputs': 'predictions, gallery, old unlabelled costs/cameras; no parsed answers/group/GT camera',
        'limits': 'same repeatedly examined synthetic development; scores not probabilities; no human rough accuracy claim',
        'frozen_sources': {p: sha256(ROOT / p) for p in CODE}, 'input_hashes': {str(p): sha256(p) for p in paths}})
    poses = load_library(args.library_root)
    assert geometry_hash(poses) == read(args.source / 'protocol.json')['hashes']['library_geometry']
    points = normalized_joints(np.stack([p.joints for p in poses]))
    cameras = camera_grid()['yaw_pitch_perspective_117'] + near_cameras()
    index = build_camera_index(poses, cameras)
    previous = {r['image']: r for r in read(args.refinement_out / 'rankings.json')}
    rows = []
    for old in read(args.confidence_out / 'rankings.json'):
        name = old['image']; path = args.source / 'predictions' / (Path(name).stem + '.json')
        assert sha256(path) == old['prediction_sha256']; pred = read(path); query = valid_query(pred)
        row = {k: old[k] for k in ('image', 'prediction_sha256')}
        if query is None:
            assert old.get('error'); row['error'] = old['error']; rows.append(row); continue
        xy, mask = query; q = normalize_batch(xy)
        row['shared_xy_sha256'] = hashlib.sha256(xy.tobytes()).hexdigest(); row['shared_mask'] = mask.tolist()
        assert row['shared_xy_sha256'] == old['shared_xy_sha256'] and row['shared_mask'] == old['shared_mask']
        path = args.refinement_out / 'costs' / (Path(name).stem + '.npz')
        assert sha256(path) == previous[name]['costs_sha256']
        with np.load(path, allow_pickle=False) as d:
            base, fit, warm = [d[k] for k in ('baseline', 'refined', 'parameters')]
        path = args.confidence_out / 'costs' / (Path(name).stem + '.npz')
        assert sha256(path) == old['costs_sha256']
        with np.load(path, allow_pickle=False) as d: frozen = d['weighted']
        for arm, cost in ((BASE, fit), (RERANK, frozen)):
            row[arm] = [poses[i].pose_id for i in order(base, blend_cost(base, cost, .75))[:5]]
            assert row[arm] == old[arm]
        output = {}
        for arm, weighted in ((CONTROL, False), (JOINT, True)):
            cost, params, labels = fit_confidence(points, index, cameras, q, mask, warm, pred['scores'], weighted=weighted)
            assert np.all(cost <= (frozen if weighted else fit) + 2e-6)
            row[arm] = [poses[i].pose_id for i in order(base, blend_cost(base, cost, .75))[:5]]
            assert row[arm][0] == row[BASE][0]
            output[arm + '_cost'] = cost; output[arm + '_parameters'] = params; output[arm + '_swapped'] = labels
        path = args.out / 'costs' / (Path(name).stem + '.npz'); np.savez_compressed(path, **output)
        row['costs_sha256'] = sha256(path); rows.append(row)
        if len(rows) % 30 == 0: print('fitted', len(rows), '/360', flush=True)
    assert len(rows) == 360; verify(args)
    save_json(args.out / 'rankings.json', rows)
    save_json(args.out / 'ranked.json', {'sha256': sha256(args.out / 'rankings.json'), 'answers_parsed': False})


def evaluate(args):
    verify(args); assert not (args.out / 'results.json').exists()
    assert sha256(args.out / 'rankings.json') == read(args.out / 'ranked.json')['sha256']
    rows = read(args.out / 'rankings.json'); answers = {r['image']: r for r in read(args.source / 'answers.json')}
    assert len(rows) == len(answers) == 360 and {r['image'] for r in rows} == set(answers)
    counts = {a: {g: {'top1': 0, 'top5': 0} for g in GROUPS} for a in ARMS}
    changes = {b: {g: {'gains': 0, 'losses': 0} for g in GROUPS} for b in ARMS[:3]}
    failures = {g: 0 for g in GROUPS}
    for row in rows:
        t = answers[row['image']]; g, pid = t['group'], t['pose_id']; failures[g] += bool(row.get('error'))
        for arm in ARMS:
            ids = row.get(arm, []); counts[arm][g]['top1'] += bool(ids and ids[0] == pid); counts[arm][g]['top5'] += pid in ids
        for b in changes:
            before, after = pid in row.get(b, []), pid in row.get(JOINT, [])
            changes[b][g]['gains'] += after and not before; changes[b][g]['losses'] += before and not after
    result = {'stage': 'development only', 'counts': counts, 'joint_vs_baselines': changes, 'failures': failures,
              'gate_passed': gate(counts), 'selected': JOINT if gate(counts) else None}
    save_json(args.out / 'results.json', result); print(result, flush=True)


def parse():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('rank', 'evaluate'))
    for name in ('source', 'refinement-out', 'blend-out', 'confidence-out', 'out', 'library-root', 'model'):
        p.add_argument('--' + name, type=Path, required=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse(); args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / '.confidence-camera-dev.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); globals()[args.stage](args)
