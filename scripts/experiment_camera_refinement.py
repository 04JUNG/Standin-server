"""Development experiment: fixed XY, coarse-to-fine camera search, no GT ranking."""
import argparse
import fcntl
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_near_camera import prepared,verify as verify_source,GROUPS
from scripts.experiment_near_camera import read,near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from standin_eval.perspective_search import camera_grid,build_camera_index,normalize_batch,score_camera_index
from standin_eval.dual_label_search import alternate_labels,merged_order
from standin_eval.camera_refinement import normalized_joints,refine_costs,DEFAULT_STEPS

ARMS=('pooled_117','keepfirst_195','refined','keepfirst_refined')
CODE=('scripts/experiment_camera_refinement.py','standin_eval/camera_refinement.py',
      'standin_eval/perspective_search.py','standin_eval/dual_label_search.py')


def select_policy(counts):
    eligible=[]
    for arm in ARMS[2:]:
        if (counts[arm]['strong_perspective']['top5']>=counts['keepfirst_195']['strong_perspective']['top5']+6
            and all(counts[arm][g]['top1']>=counts['pooled_117'][g]['top1'] for g in GROUPS)
            and all(counts[arm][g]['top5']>=max(counts['pooled_117'][g]['top5'],counts['keepfirst_195'][g]['top5'])
                    for g in ('perspective','orthographic_control'))):
            eligible.append(arm)
    eligible.sort(key=lambda a:(counts[a]['strong_perspective']['top5'],
        sum(counts[a][g]['top5'] for g in GROUPS),sum(counts[a][g]['top1'] for g in GROUPS),a=='keepfirst_refined'),reverse=True)
    return eligible


def verify(args):
    prepared(SimpleNamespace(out=args.source,library_root=args.library_root,model=args.model))
    policy=read(args.out/'protocol.json')
    assert {p:sha256(ROOT/p) for p in CODE}==policy['frozen_sources']
    for p,digest in policy['input_hashes'].items():
        assert sha256(Path(p))==digest
    return policy


def rank(args):
    source=SimpleNamespace(out=args.source,library_root=args.library_root,model=args.model)
    prepared(source)
    source_policy=verify_source(source)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'costs').mkdir()
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==source_policy['hashes']['library_geometry']
    manifest=read(args.source/'manifest.json')
    paths=[args.source/name for name in ('manifest.json','answers.json','rankings.json','results.json','audit.json')]
    paths+=sorted((args.source/'predictions').glob('*.json'))
    save_json(args.out/'protocol.json',{
        'stage':'development on already evaluated120 families,360 images; no new inference/holdout',
        'arms':ARMS,'selection':'strong Top5 >= keepfirst195+6; all3 Top1 >=pooled117; moderate/control Top5 >=both baselines; maximize strongTop5,totalTop5,totalTop1; ties prefer keepfirst',
        'optimizer':{'starts_per_pose_per_label':3,'steps_yaw_pitch_inverse_distance':DEFAULT_STEPS,'sweeps_per_scale':2,
                     'bounds':{'yaw':'periodic360','pitch':[-60,60],'inverse_distance':[0,1.25]}},
        'fixed':'all1981 gallery poses optimized; no target shortlist; same HumanArt XY/mask/anchors/normalization/dual labels; seed from195 fixed cameras',
        'ranking_inputs':'image-derived XY and gallery only, no target ID or source camera; answers only hashed',
        'first_preserved':'pooled117 first pose, not operating four-view baseline',
        'failure':'all extraction failures retained in denominator; no sample replacement',
        'frozen_sources':{p:sha256(ROOT/p) for p in CODE},
        'input_hashes':{str(p):sha256(p) for p in paths}})
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cameras)
    base=(index[0][:117],index[1][:117])
    points=normalized_joints(np.stack([p.joints for p in poses]))
    original={r['image']:r for r in read(args.source/'rankings.json')}
    rows=[]
    for item in manifest:
        name=item['image']
        assert sha256(args.source/'images'/name)==item['sha256']
        path=args.source/'predictions'/(Path(name).stem+'.json')
        row={'image':name,'prediction_sha256':sha256(path)}
        assert row['prediction_sha256']==original[name]['prediction_sha256']
        query=valid_query(read(path))
        if query is None:
            assert original[name].get('error')
            row['error']=original[name]['error'];rows.append(row);continue
        xy,mask=query
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest()
        row['shared_mask']=mask.tolist()
        q=normalize_batch(xy);alt,altmask=alternate_labels(q,mask)
        baseline=np.minimum(score_camera_index(base,q,mask),score_camera_index(base,alt,altmask))
        coarse=np.minimum(score_camera_index(index,q,mask),score_camera_index(index,alt,altmask))
        start=time.perf_counter()
        refined,params=refine_costs(points,index,cameras,q,mask)
        row['refinement_seconds']=time.perf_counter()-start
        assert np.all(refined<=coarse+1e-6)
        orders={'pooled_117':merged_order(baseline,baseline),
                'keepfirst_195':merged_order(baseline,coarse,preserve_first=True),
                'refined':merged_order(refined,refined),
                'keepfirst_refined':merged_order(baseline,refined,preserve_first=True)}
        for arm,order in orders.items():
            row[arm]=[poses[int(i)].pose_id for i in order[:5]]
        for arm in ARMS[:2]:
            assert row[arm]==original[name][arm]
        dest=args.out/'costs'/(Path(name).stem+'.npz')
        np.savez_compressed(dest,baseline=baseline,coarse=coarse,refined=refined,parameters=params)
        row['costs_sha256']=sha256(dest)
        rows.append(row)
        if len(rows)%30==0:
            print('refined',len(rows),'/360',flush=True)
    assert len(rows)==360
    verify(args)
    save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed_during_ranking':False})


def evaluate(args):
    verify(args)
    assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json')
    answers={r['image']:r for r in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==len({r['image'] for r in rows})==360
    assert set(answers)=={r['image'] for r in rows}
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in ARMS}
    failures={g:0 for g in GROUPS}
    for row in rows:
        target=answers[row['image']];g=target['group']
        failures[g]+=bool(row.get('error'))
        for arm in ARMS:
            ids=row.get(arm,[])
            counts[arm][g]['top1']+=bool(ids and ids[0]==target['pose_id'])
            counts[arm][g]['top5']+=target['pose_id'] in ids
    eligible=select_policy(counts)
    result={'stage':'development only','counts':counts,'failures':failures,
            'eligible':eligible,'selected':eligible[0] if eligible else None}
    save_json(args.out/'results.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','out','library-root','model'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.camera-refinement-dev.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        globals()[args.stage](args)
