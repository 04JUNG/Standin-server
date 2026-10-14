"""Development only; rankings use image-derived XY, answers opened in evaluate."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_camera_gate import DATASETS
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import camera_grid, build_camera_index, score_camera_index, normalize_batch
from standin_eval.dual_label_search import alternate_labels, merged_order

ARMS = [f'{method}_{n}' for n in (4, 117) for method in ('anchor', 'pooled', 'keepfirst')]


def rank(a):
    a.out.mkdir(parents=True, exist_ok=False)
    poses = load_library(a.library_root)
    policy = {'stage': 'development only; all 220 families previously examined, no new holdout',
        'datasets': DATASETS, 'arms': ARMS,
        'hypotheses': 'same Human-Art XY: original labels plus global left/right label permutation; mask follows observations',
        'selection': '117-view policy only: perspective Top5 gain >=11/220 vs anchor117; control Top5 not lower; both groups Top1 not lower; maximize total Top5 then total Top1; ties prefer keepfirst',
        'gallery': 'same 1981 BVH poses; no mirror credit, no source-ID input, no family collapse',
        'frozen_sources': {p: sha256(ROOT/p) for p in ('scripts/experiment_dual_labels.py', 'standin_eval/dual_label_search.py', 'standin_eval/perspective_search.py')},
        'library_geometry': geometry_hash(poses),
        'source_hashes': {d: {f:sha256(ROOT/'out'/d/f) for f in ('protocol.json','manifest.json','rankings.json')} for d in DATASETS}}
    save_json(a.out/'protocol.json', policy)
    indices = {n: build_camera_index(poses, camera_grid()['baseline_4' if n == 4 else 'yaw_pitch_perspective_117']) for n in (4,117)}
    records = []; timings = {n: [] for n in (4,117)}; verified = 0
    for dataset in DATASETS:
        root = ROOT/'out'/dataset
        previous = {r['image']:r for r in json.loads((root/'rankings.json').read_text())}
        for item in json.loads((root/'manifest.json').read_text()):
            name = item['image']; path = root/'predictions'/(Path(name).stem+'.json')
            p = json.loads(path.read_text())
            row = {'dataset': dataset, 'image': name, 'prediction_sha256': sha256(path)}
            if p.get('error'):
                row['error'] = p['error']; records.append(row); continue
            xy = np.array(p['keypoints2d'], dtype=np.float32)
            mask = np.array(p['scores'], dtype=np.float32) >= .3
            row['shared_xy_sha256'] = hashlib.sha256(xy.tobytes()).hexdigest()
            row['shared_mask'] = mask.tolist()
            if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                row['error'] = 'insufficient Human-Art body joints/anchors'
                assert previous[name].get('error')
                records.append(row); continue
            query = normalize_batch(xy); alternate, alt_mask = alternate_labels(query, mask)
            for n in (4,117):
                start = time.perf_counter()
                direct = score_camera_index(indices[n], query, mask)
                swapped = score_camera_index(indices[n], alternate, alt_mask)
                orders = {'anchor': np.argsort(direct, kind='stable'),
                          'pooled': merged_order(direct, swapped),
                          'keepfirst': merged_order(direct, swapped, preserve_first=True)}
                timings[n].append(time.perf_counter()-start)
                for method, order in orders.items():
                    row[f'{method}_{n}'] = [poses[int(i)].pose_id for i in order[:5]]
                old_key = f'anchor_{n}' if f'anchor_{n}' in previous[name] else ('baseline_4' if n==4 else 'yaw_pitch_perspective_117')
                assert row[f'anchor_{n}'] == previous[name][old_key][:5]
                assert row[f'keepfirst_{n}'][0] == row[f'anchor_{n}'][0]
            verified += 1; records.append(row)
        print('ranked', dataset, 'total', len(records), flush=True)
    save_json(a.out/'rankings.json', records)
    save_json(a.out/'audit.json', {'baseline_top5_verified_images': verified,
        'failures': sum('error' in r for r in records), 'no_source_answers_read_by_rank': True,
        'ranking_sha256': sha256(a.out/'rankings.json')})
    save_json(a.out/'timing.json', timings)


def evaluate(a):
    policy = json.loads((a.out/'protocol.json').read_text())
    for path, h in policy['frozen_sources'].items(): assert sha256(ROOT/path) == h
    records = json.loads((a.out/'rankings.json').read_text())
    assert sha256(a.out/'rankings.json') == json.loads((a.out/'audit.json').read_text())['ranking_sha256']
    truth = {(d,r['image']): r for d in DATASETS for r in json.loads((ROOT/'out'/d/'answers.json').read_text())}
    assert len(records) == len(truth) == 440
    assert len({r['family_id'] for r in truth.values()}) == 220
    scores = []
    for row in records:
        answer = truth[(row['dataset'],row['image'])]
        r = {k: answer[k] for k in ('image','group','pose_id','family_id')}
        r['dataset'] = row['dataset']; r['error'] = row.get('error'); r['arms'] = {}
        for arm in ARMS:
            candidates = row.get(arm, [])
            r['arms'][arm] = {'top1': bool(candidates and candidates[0]==answer['pose_id']),
                              'top5': answer['pose_id'] in candidates}
        scores.append(r)
    counts = {}
    for arm in ARMS:
        counts[arm] = {g: {k:sum(r['arms'][arm][k] for r in scores if r['group']==g) for k in ('top1','top5')} for g in ('perspective','orthographic_control')}
    eligible = []
    for arm in ('keepfirst_117','pooled_117'):
        p,c = counts[arm]['perspective'], counts[arm]['orthographic_control']
        bp,bc = counts['anchor_117']['perspective'], counts['anchor_117']['orthographic_control']
        if p['top5']-bp['top5'] >= 11 and c['top5']>=bc['top5'] and p['top1']>=bp['top1'] and c['top1']>=bc['top1']:
            eligible.append(arm)
    eligible.sort(key=lambda arm: (sum(g['top5'] for g in counts[arm].values()),sum(g['top1'] for g in counts[arm].values()),arm.startswith('keepfirst')), reverse=True)
    result = {'stage':'development only, not independently validated', 'families':220, 'images':440,
        'failures':sum(bool(r['error']) for r in scores), 'counts':counts, 'eligible':eligible,
        'selected':eligible[0] if eligible else None, 'records':scores}
    save_json(a.out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('rank','evaluate'))
    p.add_argument('--library-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a = p.parse_args(); globals()[a.stage](a)
