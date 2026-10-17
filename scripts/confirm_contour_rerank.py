"""Fresh120-family confirmation of frozen contour100, one primary contrast."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_perspective_isolation import select,strong_camera,projection_strength,PRIOR
from scripts.experiment_near_camera import read,near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_perspective_images import infer as image_infer
from scripts.experiment_humanart_z import read_rgb
from scripts.confirm_camera_fit import geometry_hash,paired_comparison
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.experiment_refinement_blend import blend_cost
from scripts.experiment_head_hint import order
from standin_eval.camera_refinement import normalized_joints,refine_costs,DEFAULT_STEPS
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.dual_label_search import alternate_labels
from standin_eval.contour_rerank import prepare_query,candidate_evidence,rerank

SEED=20261104
GROUPS=('perspective','strong_perspective','orthographic_control')
BASE='keepfirst_blend75';SELECTED='contour100';ARMS=(BASE,SELECTED)
DEV=ROOT/'out/contour-rerank-dev-20261013'
CODE=('scripts/confirm_contour_rerank.py','scripts/audit_contour_confirmation.py',
      'scripts/experiment_contour_rerank.py','standin_eval/contour_rerank.py',
      'scripts/confirm_perspective_isolation.py','scripts/experiment_near_camera.py',
      'scripts/experiment_z_axis_search.py','scripts/experiment_z_rendered.py',
      'scripts/experiment_perspective_images.py','scripts/experiment_humanart_z.py',
      'scripts/confirm_camera_fit.py','scripts/diagnose_dual_camera_coverage.py',
      'scripts/experiment_refinement_blend.py','scripts/experiment_head_hint.py',
      'standin_eval/camera_refinement.py','standin_eval/perspective_search.py',
      'standin_eval/dual_label_search.py','standin_eval/z_diagnostics.py','src/bvh.py','src/features.py')


def verify(args):
    p=read(args.out/'protocol.json')
    assert all(sha256(ROOT/n)==h for n,h in p['code'].items())
    assert all(sha256(Path(n))==h for n,h in p['inputs'].items())
    assert sha256(args.model)==p['hashes']['model']
    assert sha256(args.library_root/'poses.db')==p['hashes']['library_db']
    return p


def prepared(args):
    p=verify(args);s=read(args.out/'prepared.json')
    assert s['strong_ratio_ge_1_5']>=100
    assert s['manifest_sha256']==sha256(args.out/'manifest.json')
    assert s['answers_sha256']==sha256(args.out/'answers.json')
    return p


def prepare(args):
    assert read(DEV/'results.json')['selected']==SELECTED
    assert read(DEV/'audit.json')['ranking_sha256']==sha256(DEV/'rankings.json')
    dp=read(DEV/'protocol.json');assert all(sha256(ROOT/n)==h for n,h in dp['code'].items())
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('images','costs','records','masks'): (args.out/name).mkdir()
    poses=load_library(args.library_root)
    prior=[ROOT/'out'/p/'answers.json' for p in (*PRIOR,'perspective-isolation-confirm-20261013',
        'near-camera-confirm-20261013','refinement-blend-confirm-20261013','camera-regularization-confirm-20261013')]
    excluded={r['family_id'] for p in prior for r in read(p)};assert len(excluded)==880
    chosen=select(poses,excluded,seed=SEED,count=120)
    inputs=prior+[DEV/n for n in ('protocol.json','results.json','audit.json','REPORT.md')]
    save_json(args.out/'protocol.json',{
        'stage':'fresh fixed120 query families; selected policy only; no optional stopping',
        'seed':SEED,'selected':SELECTED,'weight':1.,'arms':ARMS,'groups':GROUPS,
        'primary':'strong Top5 contour100 versus keepfirst_blend75, exact sourceID',
        'gate':'strong delta>=.05, paired95CI lower>0, exactMcNemar p<.05; moderate/control Top5>=baseline; all3 Top1 identical',
        'statistics':'20000 family-paired resamples,seed20261019; one primary contrast; groups not pooled as360 independent poses',
        'frozen_contour':dp,'baseline':'(1-.75)*pooled117+.75*refined, same arithmetic, stable ties; first pooled117 fixed',
        'optimizer':{'starts':3,'steps':DEFAULT_STEPS,'sweeps':2,'pitch_bounds':[-60,60],'inverse_distance_bounds':[0,1.25]},
        'fixed':'same HumanArt XY,>=.3 mask,>=6body plus shoulders/hips,hip/torso,dual labels; all1981 candidates fit; only ranks2..20 get fixed contour score; tail/first unchanged; entire-query fallback on any render failure',
        'sampling':'exclude880 used families,uniform120 families then uniform member,no replacement',
        'cameras':'moderate yaw15/75/135/225,pitch-25/20,d4.5/6.5; strong depthratio2 with.65torso safety; old4 ortho controls',
        'limits':['same renderer for source and candidate favors this feature, not humanrough accuracy',
                  'full boundary effect not isolated occlusion','family not semantic/sequence independence',
                  'closed-set gallery includes source','Top1 fixed by design'],
        'excluded_families':sorted(excluded),'test_families':[p.family_id for p in chosen],
        'test_pose_ids':[p.pose_id for p in chosen],'prior_answers':[str(p) for p in prior],
        'inputs':{str(p):sha256(p) for p in inputs},'code':{n:sha256(ROOT/n) for n in CODE},
        'hashes':{'model':sha256(args.model),'library_db':sha256(args.library_root/'poses.db'),'geometry':geometry_hash(poses)}})
    manifest=[];answers=[]
    for i,pose in enumerate(chosen):
        yaw,pitch=(15,75,135,225)[i%4],(-25,20)[(i//4)%2]
        cameras=(Camera(yaw,pitch,(4.5,6.5)[(i//8)%2]),strong_camera(pose.joints,yaw,pitch),Camera((0,45,90,180)[i%4]))
        for group,cam in zip(GROUPS,cameras):
            name=f'{group}_{i:03d}.png'
            image,xy=render_pose(pose.joints,cam.yaw,style='rough',seed=SEED+i,pitch=cam.pitch,distance=cam.distance)
            image.save(args.out/'images'/name);manifest.append({'image':name,'sha256':sha256(args.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                            'camera':asdict(cam),'body_depth_ratio':projection_strength(pose.joints,cam),'ground_truth_xy':xy.tolist()})
        if (i+1)%30==0:print('rendered',i+1,'/120',flush=True)
    save_json(args.out/'manifest.json',manifest);save_json(args.out/'answers.json',answers)
    strong=sum(r['group']==GROUPS[1] and r['body_depth_ratio']>=1.5 for r in answers)
    save_json(args.out/'prepared.json',{'manifest_sha256':sha256(args.out/'manifest.json'),
              'answers_sha256':sha256(args.out/'answers.json'),'strong_ratio_ge_1_5':strong})
    assert strong>=100;print('strong preflight',strong,'/120',flush=True)


def infer(args):
    prepared(args);image_infer(args)
    paths=sorted((args.out/'predictions').glob('*.json'));assert len(paths)==360
    save_json(args.out/'inferred.json',{p.name:sha256(p) for p in paths})


def rank(args):
    prepared(args);assert not (args.out/'rankings.json').exists()
    inferred=read(args.out/'inferred.json')
    assert all(sha256(args.out/'predictions'/n)==h for n,h in inferred.items())
    poses=load_library(args.library_root);assert geometry_hash(poses)==verify(args)['hashes']['geometry']
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cameras);bi=(index[0][:117],index[1][:117])
    points=normalized_joints(np.stack([p.joints for p in poses]));rows=[]
    for item in read(args.out/'manifest.json'):
        name=item['image'];assert sha256(args.out/'images'/name)==item['sha256']
        path=args.out/'predictions'/(Path(name).stem+'.json');pred=read(path)
        row={'image':name,'prediction_sha256':sha256(path)};query=valid_query(pred)
        if query is None:
            row['error']=pred.get('error','insufficient Human-Art body joints/anchors');rows.append(row);continue
        xy,mask=query;q=normalize_batch(xy);alt,am=alternate_labels(q,mask)
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        base=np.minimum(score_camera_index(bi,q,mask),score_camera_index(bi,alt,am))
        start=time.perf_counter();fit,params=refine_costs(points,index,cameras,q,mask)
        row['refinement_seconds']=time.perf_counter()-start
        scores=blend_cost(base,fit,.75);original=order(base,scores);short=original[:20]
        row[BASE]=[poses[i].pose_id for i in original[:5]]
        row['shortlist']=[poses[i].pose_id for i in short];row['shortlist_indices']=short.tolist()
        row['contour_costs']=[0.]*20;row['candidate_errors']={};row['affines']={};masks=[]
        start=time.perf_counter()
        try:
            evidence=prepare_query(read_rgb(args.out/'images'/name),xy)
            row['query_torso']=evidence['torso'];row['query_line_pixels']=int(evidence['mask'].sum())
            for j,i in enumerate(short[1:],1):
                try:
                    cost,affine,pixels=candidate_evidence(evidence,poses[i].joints,params[i])
                    row['contour_costs'][j]=cost;row['affines'][str(j)]=affine.tolist()
                    if len(rows)<3:masks.append((j,pixels))
                except Exception as exc:row['candidate_errors'][str(j)]=type(exc).__name__+': '+str(exc)
            if len(rows)<3:
                np.savez_compressed(args.out/'masks'/(Path(name).stem+'.npz'),query=evidence['mask'],
                                    **{f'candidate_{j}':pixels for j,pixels in masks})
        except Exception as exc:row['query_error']=type(exc).__name__+': '+str(exc)
        row['contour_seconds']=time.perf_counter()-start;row['fallback']=bool(row['candidate_errors'] or row.get('query_error'))
        selected=original if row['fallback'] else rerank(original,scores,row['contour_costs'],1.)
        assert selected[0]==original[0] and np.array_equal(selected[20:],original[20:])
        row[SELECTED]=[poses[i].pose_id for i in selected[:5]]
        dest=args.out/'costs'/(Path(name).stem+'.npz')
        np.savez_compressed(dest,baseline=base,refined=fit,parameters=params)
        row['costs_sha256']=sha256(dest);rows.append(row)
        save_json(args.out/'records'/(Path(name).stem+'.json'),row)
        if len(rows)%30==0:print('ranked',len(rows),'/360',flush=True)
    assert len(rows)==360;prepared(args)
    save_json(args.out/'rankings.json',rows)
    save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def success_gate(groups):
    primary=groups[GROUPS[1]]['comparison']
    return bool(primary['delta']>=.05 and primary['paired_bootstrap_95ci'][0]>0 and primary['exact_mcnemar_p']<.05
        and all(groups[g]['counts'][SELECTED]['top5']>=groups[g]['counts'][BASE]['top5'] for g in (GROUPS[0],GROUPS[2]))
        and all(groups[g]['counts'][SELECTED]['top1']==groups[g]['counts'][BASE]['top1'] for g in GROUPS))


def evaluate(args):
    prepared(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.out/'answers.json')}
    assert len(rows)==len(answers)==len({r['image'] for r in rows})==360
    groups={}
    for group in GROUPS:
        records=[]
        for row in rows:
            a=answers[row['image']]
            if a['group']!=group:continue
            rec={k:a[k] for k in ('image','pose_id','family_id','body_depth_ratio')}
            if row.get('error'):rec['error']=row['error']
            rec['fallback']=bool(row.get('fallback'));rec['target_in_original20']=a['pose_id'] in row.get('shortlist',[])
            for arm in ARMS:
                ids=row.get(arm,[]);rec[arm+'_rank']=ids.index(a['pose_id'])+1 if a['pose_id'] in ids else None
            records.append(rec)
        assert len(records)==len({r['family_id'] for r in records})==120
        counts={arm:{f'top{k}':sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in records)
                     for k in (1,5)} for arm in ARMS}
        groups[group]={'n':120,'counts':counts,'comparison':paired_comparison(records,SELECTED,BASE),
                       'extraction_failures':sum('error' in r for r in records),
                       'fallback_images':sum(r['fallback'] for r in records),
                       'target_in_original20':sum(r['target_in_original20'] for r in records),'records':records}
        print(group,{k:v for k,v in groups[group].items() if k!='records'},flush=True)
    groups['gate_passed']=success_gate(groups)
    groups['timing']={k:float(np.median([r[k] for r in rows if not r.get('error')])) for k in ('refinement_seconds','contour_seconds')}
    save_json(args.out/'results.json',groups);print('gate_passed',groups['gate_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    for name in ('out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.contour-confirm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
