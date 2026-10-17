"""Frozen-camera confidence reranking on already examined development data."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_camera_refinement import verify as verify_source, GROUPS
from scripts.experiment_near_camera import read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import normalize_batch
from standin_eval.camera_refinement import normalized_joints, project_parameters
from standin_eval.confidence_cost import frozen_costs

BASE = 'keepfirst_blend75'
STRENGTHS = {'confidence025': .25, 'confidence050': .5, 'confidence100': 1.}
CODE = ('scripts/experiment_confidence_cost.py', 'standin_eval/confidence_cost.py',
        'standin_eval/camera_refinement.py', 'standin_eval/perspective_search.py',
        'standin_eval/dual_label_search.py', 'scripts/experiment_head_hint.py',
        'scripts/experiment_refinement_blend.py')


def select_policy(counts):
    choices = [a for a in STRENGTHS
               if counts[a]['strong_perspective']['top5'] >= counts[BASE]['strong_perspective']['top5'] + 3
               and all(counts[a][g][k] >= counts[BASE][g][k] for g in GROUPS for k in ('top1', 'top5'))]
    return sorted(choices, key=lambda a: (-counts[a]['strong_perspective']['top5'],
                  -sum(counts[a][g]['top5'] for g in GROUPS), STRENGTHS[a]))


def verify(args):
    verify_source(SimpleNamespace(source=args.source, out=args.refinement_out,
                                  library_root=args.library_root, model=args.model))
    policy = read(args.out / 'protocol.json')
    assert all(sha256(ROOT / p) == h for p, h in policy['frozen_sources'].items())
    assert all(sha256(Path(p)) == h for p, h in policy['input_hashes'].items())


def rank(args):
    verify_source(SimpleNamespace(source=args.source, out=args.refinement_out,
                                  library_root=args.library_root, model=args.model))
    for folder in (args.refinement_out, args.blend_out):
        assert read(folder / 'audit.json')['ranking_sha256'] == sha256(folder / 'rankings.json')
    args.out.mkdir(parents=True, exist_ok=False); (args.out / 'costs').mkdir()
    paths = [args.source / n for n in ('answers.json', 'manifest.json')]
    paths += [d / n for d in (args.refinement_out, args.blend_out) for n in ('rankings.json', 'audit.json')]
    paths += sorted((args.source / 'predictions').glob('*.json'))
    paths += [ROOT / 'out/body-observation-diagnostic-20261013' / n for n in ('results.json', 'audit.json')]
    save_json(args.out / 'protocol.json', {
        'stage': 'development on already examined 120 families / 360 images; not new holdout',
        'hypothesis': 'confidence limits misleading joint residuals at unchanged fitted cameras',
        'joint_weight': '.25+.75*clip((score-.3)/.4,0,1); all valid joints retain positive weight',
        'strengths': STRENGTHS,
        'formula': '.25*pooled117+.75*((1-strength)*old_fitted+strength*confidence_weighted)',
        'confidence_weighted': 'weighted mean Euclidean body residual at SAME stored camera and original label per pose; infer original label by closest uniform residual to stored fit, tie direct; weights and mask permuted with observations',
        'fixed': 'Human-Art XY, score>=.3 mask, hip/torso normalization, gallery1981, one stored camera per pose, original pooled117 first result; no refit, no queryZ, no extra head cue',
        'uniform_control': 'recomputed uniform cost at frozen camera must match stored fit within2e-5; baseline ranks must reproduce exactly',
        'selection': 'strongTop5>=baseline+3; all3 Top1/Top5 nondecreasing; maximize strongTop5,totalTop5 then smaller strength',
        'ranking_inputs': 'predictions, old unlabelled costs/cameras and library only; answers hashed not parsed until evaluate',
        'failure': 'all existing failures retained; no sample replacement',
        'limits': 'scores are not probabilities; same development used for diagnosis; only one original winning camera retained per pose, not a new weighted-camera optimizer',
        'frozen_sources': {p: sha256(ROOT / p) for p in CODE},
        'input_hashes': {str(p): sha256(p) for p in paths}})
    poses = load_library(args.library_root)
    assert geometry_hash(poses) == read(args.source / 'protocol.json')['hashes']['library_geometry']
    points = normalized_joints(np.stack([p.joints for p in poses]))
    originals = {r['image']: r for r in read(args.blend_out / 'rankings.json')}
    manifest = {r['image']: r for r in read(args.source / 'manifest.json')}
    rows = []; max_delta = 0.
    for old in read(args.refinement_out / 'rankings.json'):
        name = old['image']; path = args.source / 'predictions' / (Path(name).stem + '.json')
        assert sha256(path) == old['prediction_sha256']
        assert sha256(args.source / 'images' / name) == manifest[name]['sha256']
        pred = read(path); query = valid_query(pred)
        row = {'image': name, 'prediction_sha256': sha256(path)}
        if query is None:
            assert old.get('error'); row['error'] = old['error']; rows.append(row); continue
        xy, mask = query; q = normalize_batch(xy)
        row['shared_xy_sha256'] = hashlib.sha256(xy.tobytes()).hexdigest(); row['shared_mask'] = mask.tolist()
        assert row['shared_xy_sha256'] == old['shared_xy_sha256'] and row['shared_mask'] == old['shared_mask']
        path = args.refinement_out / 'costs' / (Path(name).stem + '.npz')
        assert sha256(path) == old['costs_sha256']
        with np.load(path, allow_pickle=False) as d:
            base, fit, params = [d[k] for k in ('baseline', 'refined', 'parameters')]
        features, valid = project_parameters(points, params); assert valid.all()
        uniform, weighted = frozen_costs(features, q, mask, pred['scores'], fit)
        delta = float(np.max(np.abs(uniform - fit))); assert delta < 2e-5
        max_delta = max(max_delta, delta)
        for arm, alpha in ((BASE, 0.), *STRENGTHS.items()):
            score = blend_cost(base, blend_cost(fit, weighted, alpha), .75)
            indices = order(base, score)
            row[arm] = [poses[i].pose_id for i in indices[:5]]
            assert row[arm][0] == row[BASE][0]
        assert row[BASE] == originals[name][BASE]
        dest = args.out / 'costs' / (Path(name).stem + '.npz')
        np.savez_compressed(dest, uniform=uniform, weighted=weighted)
        row['costs_sha256'] = sha256(dest); row['original_costs_sha256'] = sha256(path); rows.append(row)
    assert len(rows) == 360; verify(args)
    save_json(args.out / 'rankings.json', rows)
    save_json(args.out / 'ranked.json', {'sha256': sha256(args.out / 'rankings.json'),
              'answers_parsed': False, 'max_uniform_control_delta': max_delta})


def evaluate(args):
    verify(args); assert not (args.out / 'results.json').exists()
    assert sha256(args.out / 'rankings.json') == read(args.out / 'ranked.json')['sha256']
    rows = read(args.out / 'rankings.json'); answers = {r['image']: r for r in read(args.source / 'answers.json')}
    assert len(rows) == len(answers) == 360 and {r['image'] for r in rows} == set(answers)
    counts = {a: {g: {'top1': 0, 'top5': 0} for g in GROUPS} for a in (BASE, *STRENGTHS)}
    changes = {a: {g: {'gains': 0, 'losses': 0} for g in GROUPS} for a in STRENGTHS}
    failed = {g: 0 for g in GROUPS}
    for row in rows:
        truth = answers[row['image']]; g, target = truth['group'], truth['pose_id']
        failed[g] += bool(row.get('error'))
        for arm in counts:
            ids = row.get(arm, []); counts[arm][g]['top1'] += bool(ids and ids[0] == target)
            counts[arm][g]['top5'] += target in ids
            if arm != BASE:
                before, after = target in row.get(BASE, []), target in ids
                changes[arm][g]['gains'] += after and not before; changes[arm][g]['losses'] += before and not after
    eligible = select_policy(counts)
    result = {'stage': 'development only', 'counts': counts, 'failures': failed, 'top5_changes': changes,
              'eligible': eligible, 'selected': eligible[0] if eligible else None}
    save_json(args.out / 'results.json', result); print(result, flush=True)


def parse():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('rank', 'evaluate'))
    for name in ('source', 'refinement-out', 'blend-out', 'out', 'library-root', 'model'):
        p.add_argument('--' + name, type=Path, required=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse(); args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / '.confidence-cost-dev.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); globals()[args.stage](args)
