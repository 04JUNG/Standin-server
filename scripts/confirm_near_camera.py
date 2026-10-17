"""Fresh120-family confirmation of dev-selected near-camera keep-first policy."""
import argparse
import fcntl
import hashlib
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.confirm_perspective_isolation import select, strong_camera, projection_strength, PRIOR
from scripts.experiment_near_camera import near_cameras, read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose, save_json, sha256
from scripts.experiment_perspective_images import infer as image_infer
from scripts.confirm_camera_fit import geometry_hash, paired_comparison
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import Camera, camera_grid, build_camera_index, score_camera_index, normalize_batch
from standin_eval.dual_label_search import alternate_labels, merged_order

SEED = 20261101
GROUPS = ('perspective', 'strong_perspective', 'orthographic_control')
ARMS = ('pooled_117', 'keepfirst_195')
DEV = ROOT/'out/near-camera-dev-20261013'
SOURCE_FILES = ('scripts/confirm_near_camera.py', 'scripts/audit_near_confirmation.py',
    'scripts/confirm_perspective_isolation.py', 'scripts/experiment_near_camera.py',
    'scripts/experiment_z_axis_search.py', 'scripts/experiment_z_rendered.py',
    'scripts/experiment_perspective_images.py', 'scripts/confirm_camera_fit.py',
    'scripts/diagnose_dual_camera_coverage.py', 'standin_eval/perspective_search.py',
    'standin_eval/dual_label_search.py', 'standin_eval/z_diagnostics.py', 'src/bvh.py')


def sources():
    return {name: sha256(ROOT/name) for name in SOURCE_FILES}


def verify(args):
    policy = read(args.out/'protocol.json')
    assert sources() == policy['frozen_sources']
    assert sha256(args.model) == policy['hashes']['model']
    assert sha256(args.library_root/'poses.db') == policy['hashes']['library_db']
    assert sha256(DEV/'results.json') == policy['hashes']['selection']
    assert sha256(DEV/'independent-audit.json') == policy['hashes']['development_audit']
    for path, digest in policy['prior_answers'].items():
        assert sha256(ROOT/path) == digest
    return policy


def prepared(args):
    verify(args)
    snapshot = read(args.out/'prepared.json')
    assert snapshot['stress_preflight_passed']
    assert sha256(args.out/'manifest.json') == snapshot['manifest_sha256']
    assert sha256(args.out/'answers.json') == snapshot['answers_sha256']


def prepare(args):
    assert read(DEV/'results.json')['selected'] == 'keepfirst_195'
    audit = read(DEV/'independent-audit.json')
    assert audit['ranking_sha256'] == sha256(DEV/'rankings.json')
    assert audit['full195_index_ranking_matches'] == 346 and audit['failures_retained'] == 14
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'images').mkdir()
    poses = load_library(args.library_root)
    paths = [ROOT/'out'/d/'answers.json' for d in (*PRIOR, 'perspective-isolation-confirm-20261013')]
    excluded = {r['family_id'] for path in paths for r in read(path)}
    assert len(excluded) == 520
    chosen = select(poses, excluded, seed=SEED, count=120)
    policy = {
        'stage': 'fresh fixed-size family confirmation, no optional stopping',
        'seed': SEED, 'samples_families': 120, 'groups': GROUPS, 'arms': ARMS,
        'selected': 'keepfirst_195',
        'primary': 'strong_perspective Top5: keepfirst195 versus pooled117',
        'gallery': 'same1981 poses;117 views None,4,8 plus78 views distances2,3; same yaw13 and pitch3',
        'fixed': 'Human-Art raw XY and >=.3 mask shared; >=6 body joints and all shoulder/hip anchors; same normalization, dual-label distance; no input Z',
        'selection_policy': 'retain pooled117 first pose, then expanded195 order without duplicates; no per-image GT, group routing or weight changes',
        'scoring': 'exact original pose_id, mirror not interchangeable, all extraction failures kept',
        'success_gate': 'strong Top5 delta>=.05, paired95CI lower>0, exact McNemar p<.05; moderate and control Top5 nondecreasing; all3 groups Top1 exactly equal by keepfirst contract',
        'statistics': '20000 paired bootstrap resamples by family within group seed20261019; single primary contrast; groups not pooled as360 independent poses',
        'sampling': 'exclude520 used families; uniformly choose120 families then one uniform member per family; seed fixed before inference',
        'cameras': 'moderate off-grid yaw15/75/135/225 pitch-25/20 distance4.5/6.5; strong same orientation targetdepthratio2 with .65 near-joint safety; control old4 yaw orthographic',
        'stress_preflight': 'at least100/120 strong images achieve body depth ratio>=1.5 before inference; no replacement of failed poses',
        'limits': ['synthetic mannequin line drawings, not real roughs', 'family not sequence/semantic holdout',
                   'closed-set gallery includes source pose', 'cannot improve Top1 under keepfirst'],
        'excluded_families': sorted(excluded), 'test_families': [p.family_id for p in chosen],
        'test_pose_ids': [p.pose_id for p in chosen],
        'frozen_sources': sources(),
        'prior_answers': {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        'hashes': {'model': sha256(args.model), 'library_db': sha256(args.library_root/'poses.db'),
                   'library_geometry': geometry_hash(poses), 'selection': sha256(DEV/'results.json'),
                   'development_audit': sha256(DEV/'independent-audit.json')},
    }
    save_json(args.out/'protocol.json', policy)
    manifest, answers = [], []
    for i, pose in enumerate(chosen):
        yaw, pitch = (15,75,135,225)[i%4], (-25,20)[(i//4)%2]
        cameras = (Camera(yaw,pitch,(4.5,6.5)[(i//8)%2]), strong_camera(pose.joints,yaw,pitch),
                   Camera((0,45,90,180)[i%4]))
        for group, camera in zip(GROUPS, cameras):
            name = f'{group}_{i:03d}.png'
            image, xy = render_pose(pose.joints,camera.yaw,style='rough',seed=SEED+i,
                                    pitch=camera.pitch,distance=camera.distance)
            image.save(args.out/'images'/name)
            manifest.append({'image':name,'sha256':sha256(args.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                            'camera':asdict(camera),'body_depth_ratio':projection_strength(pose.joints,camera),
                            'ground_truth_xy':xy.tolist()})
        if (i+1)%20 == 0:
            print('rendered', i+1, '/120', flush=True)
    save_json(args.out/'manifest.json',manifest)
    save_json(args.out/'answers.json',answers)
    strong_count = sum(r['body_depth_ratio']>=1.5 for r in answers if r['group']=='strong_perspective')
    save_json(args.out/'prepared.json',{'manifest_sha256':sha256(args.out/'manifest.json'),
              'answers_sha256':sha256(args.out/'answers.json'),'stress_preflight_passed':strong_count>=100,
              'strong_ratio_ge_1_5':strong_count})
    assert strong_count>=100
    print('prepared360; strong>=1.5',strong_count,flush=True)


def infer(args):
    prepared(args)
    image_infer(args)


def rank(args):
    prepared(args)
    assert not (args.out/'rankings.json').exists()
    poses = load_library(args.library_root)
    policy = verify(args)
    assert geometry_hash(poses)==policy['hashes']['library_geometry']
    indices = {117:build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117']),
               78:build_camera_index(poses,near_cameras())}
    rows, times = [], []
    for item in read(args.out/'manifest.json'):
        name = item['image']
        assert sha256(args.out/'images'/name)==item['sha256']
        path = args.out/'predictions'/(Path(name).stem+'.json')
        row = {'image':name,'prediction_sha256':sha256(path)}
        pred = read(path)
        query = valid_query(pred)
        if query is None:
            row['error']=pred.get('error','insufficient Human-Art body joints/anchors')
            rows.append(row)
            continue
        xy,mask=query
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest()
        row['shared_mask']=mask.tolist()
        q=normalize_batch(xy)
        alt,alt_mask=alternate_labels(q,mask)
        start=time.perf_counter()
        costs={n:np.minimum(score_camera_index(index,q,mask),score_camera_index(index,alt,alt_mask)) for n,index in indices.items()}
        orders={'pooled_117':merged_order(costs[117],costs[117]),
                'keepfirst_195':merged_order(costs[117],costs[78],preserve_first=True)}
        times.append(time.perf_counter()-start)
        for arm,order in orders.items():
            row[arm]=[poses[int(i)].pose_id for i in order[:5]]
        assert row['pooled_117'][0]==row['keepfirst_195'][0]
        assert all(len(set(row[arm]))==5 for arm in ARMS)
        rows.append(row)
        if len(rows)%60==0:
            print('ranked',len(rows),'/360',flush=True)
    assert len(rows)==360
    verify(args)
    save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'ranking_sha256':sha256(args.out/'rankings.json'),'answers_parsed_during_rank':False})
    save_json(args.out/'timing.json',{'combined_query_seconds':times})


def success_gate(groups):
    contrast=groups['strong_perspective']['near_vs_baseline']
    return bool(contrast['delta']>=.05 and contrast['paired_bootstrap_95ci'][0]>0
        and contrast['exact_mcnemar_p']<.05
        and all(groups[g]['counts']['keepfirst_195']['top5']>=groups[g]['counts']['pooled_117']['top5'] for g in ('perspective','orthographic_control'))
        and all(groups[g]['counts']['keepfirst_195']['top1']==groups[g]['counts']['pooled_117']['top1'] for g in GROUPS))


def evaluate(args):
    prepared(args)
    assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['ranking_sha256']
    rows=read(args.out/'rankings.json')
    answers={r['image']:r for r in read(args.out/'answers.json')}
    assert len(rows)==len(answers)==len({r['image'] for r in rows})==360
    assert set(answers)=={r['image'] for r in rows}
    result={}
    for group in GROUPS:
        records=[]
        for row in rows:
            target=answers[row['image']]
            if target['group']!=group:
                continue
            r={k:target[k] for k in ('image','pose_id','family_id','body_depth_ratio')}
            if row.get('error'):
                r['error']=row['error']
            for arm in ARMS:
                ids=row.get(arm,[])
                r[arm+'_rank']=ids.index(target['pose_id'])+1 if target['pose_id'] in ids else None
            records.append(r)
        assert len(records)==len({r['family_id'] for r in records})==120
        counts={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records) for k in (1,5)} for arm in ARMS}
        result[group]={'n':120,'extraction_failures':sum('error' in r for r in records),'counts':counts,
                       'near_vs_baseline':paired_comparison(records,'keepfirst_195','pooled_117'),
                       'depth_ratio_min_median_max':np.quantile([r['body_depth_ratio'] for r in records],[0,.5,1]).tolist(),
                       'records':records}
        print(group,{k:v for k,v in result[group].items() if k!='records'},flush=True)
    result['gate_passed']=success_gate(result)
    save_json(args.out/'results.json',result)
    print('gate_passed',result['gate_passed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--library-root',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.near-camera-confirm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        globals()[args.stage](args)
