"""Post-test counterfactual diagnostics, never an image-accuracy improvement.

Both exact XY and per-query source camera are oracle information. Outputs stay
separate from frozen confirmation results and may only guide later development.
"""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.confirm_near_camera import prepared, verify, GROUPS
from scripts.experiment_near_camera import near_cameras, read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json, sha256
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import Camera, camera_grid, build_camera_index, score_camera_index, normalize_batch
from standin_eval.dual_label_search import alternate_labels, merged_order


def pooled_cost(index, xy, mask):
    alt, alt_mask = alternate_labels(xy, mask)
    return np.minimum(score_camera_index(index, xy, mask), score_camera_index(index, alt, alt_mask))


def rank_of(order, target):
    positions = np.flatnonzero(np.asarray(order) == target)
    return int(positions[0]) + 1 if len(positions) else None


def hit(rank, k=5):
    return rank is not None and rank <= k


def failure_bucket(row):
    # Ordered diagnostic buckets, not exclusive causal attribution.
    if row.get('error'):
        return 'extraction_failure'
    if hit(row['keepfirst_195_rank']):
        return 'observed_success'
    if hit(row['raw_195_rank']):
        return 'keepfirst_displacement'
    if hit(row['oracle_xy_grid195_rank']):
        return 'exact_xy_recovers_on_same_grid'
    if hit(row['oracle_xy_exact_camera_rank']):
        return 'exact_xy_needs_source_camera'
    return 'not_recovered_even_with_both_oracles'


def summarize(rows):
    arms = ('pooled_117', 'keepfirst_195', 'raw_195', 'observed_xy_exact_camera',
            'oracle_xy_grid195', 'oracle_xy_exact_camera')
    result = {}
    for group in GROUPS:
        subset = [r for r in rows if r['group'] == group]
        assert len(subset) == len({r['family_id'] for r in subset}) == 120
        buckets = {}
        for row in subset:
            bucket = failure_bucket(row)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        valid = [r for r in subset if not r.get('error')]
        result[group] = {'n': len(subset), 'searchable': len(valid),
            'counts': {a: {f'top{k}': sum(hit(r.get(a+'_rank'), k) for r in subset)
                           for k in (1, 5)} for a in arms},
            'failure_buckets': buckets,
            'label_best_xy_error_median_torso': float(np.median([r['label_best_xy_error_torso'] for r in valid])),
            'exact_camera_recovers_observed_misses': sum(not hit(r['keepfirst_195_rank']) and hit(r['observed_xy_exact_camera_rank']) for r in valid),
            'raw195_vs_keepfirst_top5': {
                'raw_only': sum(hit(r['raw_195_rank']) and not hit(r['keepfirst_195_rank']) for r in valid),
                'keepfirst_only': sum(hit(r['keepfirst_195_rank']) and not hit(r['raw_195_rank']) for r in valid)}}
    return result


def run(args):
    source = SimpleNamespace(out=args.source, library_root=args.library_root, model=args.model)
    prepared(source)
    policy = verify(source)
    audit = read(args.source/'audit.json')
    assert audit['both_rankings_recomputed'] == 357
    assert audit['ranking_sha256'] == sha256(args.source/'rankings.json')
    args.out.mkdir(parents=True, exist_ok=False)
    poses = load_library(args.library_root)
    assert geometry_hash(poses) == policy['hashes']['library_geometry']
    paths = [args.source/name for name in ('answers.json','manifest.json','rankings.json','results.json','audit.json','protocol.json')]
    paths += sorted((args.source/'predictions').glob('*.json'))
    digests = {str(p):sha256(p) for p in paths}
    manifest = read(args.source/'manifest.json')
    for item in manifest:
        assert sha256(args.source/'images'/item['image']) == item['sha256']
    save_json(args.out/'protocol.json', {
        'stage': 'post-test diagnostic on already evaluated120 families/360 images',
        'warning': 'Exact XY and source camera are oracle information; no new heldout accuracy or policy selection',
        'fixed': 'same raw XY, mask, >=6 body and all anchors, torso normalization, dual labels,1981 gallery',
        'arms': 'actual117, actual195, actual195 with117 first; oracleXY195 same mask; add exact source camera to ALL gallery poses for actualXY and oracleXY',
        'failures': 'all3 extraction failures retained in denominator; no GT replacement for failed extractions',
        'classification': 'ordered buckets: extraction, observed hit, keepfirst displacement, exactXY recovery, exactXY+sourcecamera recovery, unresolved; not independent causal counts',
        'no_leakage_claim': 'This script intentionally reads GT after completed confirmation; do not call oracle ranks deployable',
        'input_hashes': digests, 'source_script_sha256':sha256(Path(__file__))})
    answers = read(args.source/'answers.json')
    old = {r['image']:r for r in read(args.source/'rankings.json')}
    assert len(answers) == len(old) == len(manifest) == 360
    assert {r['image'] for r in answers} == set(old) == {r['image'] for r in manifest}
    locations = {p.pose_id:i for i,p in enumerate(poses)}
    index = build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117']+near_cameras())
    base = (index[0][:117],index[1][:117])
    records=[]
    for target in answers:
        row = {k:target[k] for k in ('image','group','pose_id','family_id')}
        saved = old[row['image']]
        path = args.source/'predictions'/(Path(row['image']).stem+'.json')
        assert sha256(path) == saved['prediction_sha256']
        query = valid_query(read(path))
        if query is None:
            assert saved.get('error')
            row['error'] = saved['error']
            records.append(row)
            continue
        xy,mask=query
        assert mask.tolist() == saved['shared_mask']
        assert hashlib.sha256(xy.tobytes()).hexdigest() == saved['shared_xy_sha256']
        xy = normalize_batch(xy)
        gt = normalize_batch(target['ground_truth_xy'])
        exact = build_camera_index(poses,[Camera(**target['camera'])])
        baseline = pooled_cost(base,xy,mask)
        observed = pooled_cost(index,xy,mask)
        oracle = pooled_cost(index,gt,mask)
        orders = {'pooled_117':merged_order(baseline,baseline),
            'keepfirst_195':merged_order(baseline,observed,preserve_first=True),
            'raw_195':merged_order(observed,observed),
            'observed_xy_exact_camera':merged_order(observed,pooled_cost(exact,xy,mask)),
            'oracle_xy_grid195':merged_order(oracle,oracle),
            'oracle_xy_exact_camera':merged_order(oracle,pooled_cost(exact,gt,mask))}
        for arm in ('pooled_117','keepfirst_195'):
            assert [poses[int(i)].pose_id for i in orders[arm][:5]] == saved[arm]
        target_i=locations[row['pose_id']]
        for arm,order in orders.items():
            row[arm+'_rank']=rank_of(order,target_i)
        row['oracle_exact_source_cost']=float(pooled_cost(exact,gt,mask)[target_i])
        assert row['oracle_exact_source_cost'] < 1e-4
        bodymask=mask.copy();bodymask[:5]=False
        direct=float(np.linalg.norm(xy[bodymask]-gt[bodymask],axis=-1).mean())
        alt,altmask=alternate_labels(xy,bodymask)
        swapped=float(np.linalg.norm(alt[altmask]-gt[altmask],axis=-1).mean())
        row['direct_xy_error_torso']=direct
        row['label_best_xy_error_torso']=min(direct,swapped)
        row['bucket']=failure_bucket(row)
        records.append(row)
        if len(records)%60==0:
            print('diagnosed',len(records),'/360',flush=True)
    for path,digest in digests.items():
        assert sha256(Path(path)) == digest
    prepared(source)
    summary=summarize(records)
    save_json(args.out/'results.json',{'warning':'post-test oracle diagnosis, not new image accuracy',
        'summary':summary,'records':records,'source_results_unchanged':True,
        'observed_rankings_reproduced':sum(not r.get('error') for r in records)})
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--library-root',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.near-failure-diagnostic.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
