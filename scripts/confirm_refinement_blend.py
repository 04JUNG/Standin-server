"""Fresh family confirmation of frozen keepfirst 75 percent camera-fit blend."""
import argparse
import fcntl
import hashlib
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_perspective_isolation import select,strong_camera,projection_strength,PRIOR
from scripts.experiment_near_camera import read,near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,save_json,sha256
from scripts.experiment_perspective_images import infer as image_infer
from scripts.confirm_camera_fit import geometry_hash,paired_comparison
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.experiment_refinement_blend import blend_cost
from standin_eval.camera_refinement import normalized_joints,refine_costs,DEFAULT_STEPS
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.dual_label_search import alternate_labels,merged_order

SEED=20261102
GROUPS=('perspective','strong_perspective','orthographic_control')
ARMS=('pooled_117','keepfirst_195','keepfirst_blend75')
SELECTED=ARMS[-1]
DEV=ROOT/'out/refinement-blend-dev-20261013'
CODE=('scripts/confirm_refinement_blend.py','scripts/audit_refinement_confirmation.py',
      'scripts/experiment_refinement_blend.py','standin_eval/camera_refinement.py',
      'scripts/confirm_perspective_isolation.py','scripts/experiment_near_camera.py',
      'scripts/experiment_z_axis_search.py','scripts/experiment_z_rendered.py',
      'scripts/experiment_perspective_images.py','scripts/confirm_camera_fit.py',
      'scripts/diagnose_dual_camera_coverage.py','standin_eval/perspective_search.py',
      'standin_eval/dual_label_search.py','standin_eval/z_diagnostics.py','src/bvh.py')


def sources():return {p:sha256(ROOT/p) for p in CODE}


def verify(args):
    p=read(args.out/'protocol.json')
    assert sources()==p['frozen_sources']
    assert sha256(args.model)==p['hashes']['model']
    assert sha256(args.library_root/'poses.db')==p['hashes']['library_db']
    for path,digest in p['input_hashes'].items():assert sha256(Path(path))==digest
    return p


def prepared(args):
    verify(args);snap=read(args.out/'prepared.json')
    assert snap['strong_ratio_ge_1_5']>=100
    assert sha256(args.out/'manifest.json')==snap['manifest_sha256']
    assert sha256(args.out/'answers.json')==snap['answers_sha256']


def prepare(args):
    assert read(DEV/'results.json')['selected']==SELECTED
    audit=read(DEV/'audit.json')
    assert audit['ranking_sha256']==sha256(DEV/'rankings.json')
    assert audit['all_six_policy_top5_replayed']==357 and audit['selection_same_under_alternative_arithmetic']
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'images').mkdir();(args.out/'costs').mkdir()
    poses=load_library(args.library_root)
    prior=[ROOT/'out'/p/'answers.json' for p in (*PRIOR,'perspective-isolation-confirm-20261013','near-camera-confirm-20261013')]
    excluded={r['family_id'] for p in prior for r in read(p)}
    assert len(excluded)==640
    chosen=select(poses,excluded,seed=SEED,count=120)
    inputs=prior+[DEV/'results.json',DEV/'audit.json']
    save_json(args.out/'protocol.json',{
        'stage':'fresh fixed-size120 family confirmation, no optional stopping',
        'seed':SEED,'selected':SELECTED,'alpha':.75,'arms':ARMS,'groups':GROUPS,
        'primary':'strong Top5 keepfirst_blend75 versus keepfirst195; exact original ID',
        'success_gate':'primary delta>=.05, paired95CI lower>0, exact McNemar p<.05; moderate/control Top5>=both baselines; all3 Top1 equal pooled117',
        'statistics':'20000 paired family bootstrap,seed20261019; groups not pooled as360 independent poses; single primary contrast',
        'formula':'(1-.75)*pooled117_cost+.75*refined_cost; stable gallery-order ties; first pose preserved from pooled117',
        'optimizer':{'starts_per_label_per_pose':3,'steps':DEFAULT_STEPS,'sweeps':2,'pitch_bounds':[-60,60],'inverse_distance_bounds':[0,1.25]},
        'fixed':'same HumanArt XY,>=.3 mask,>=6 body and four anchors,hip/torso normalization,dual labels,all1981 poses fitted; no input Z',
        'sampling':'exclude640 used families, sample120 uniformly then uniform member,seed fixed before inference',
        'cameras':'moderate offgrid yaw15/75/135/225 pitch-25/20 distance4.5/6.5; strong targetbodydepthratio2 with nearjoint safety.65; control old4 ortho',
        'limits':['synthetic mannequin line drawings,not real roughs','family not sequence/semantic independence','closed-set gallery includes source pose','Top1 cannot improve under first preservation'],
        'excluded_families':sorted(excluded),'test_families':[p.family_id for p in chosen],'test_pose_ids':[p.pose_id for p in chosen],
        'prior_answers':[str(p) for p in prior],'input_hashes':{str(p):sha256(p) for p in inputs},
        'frozen_sources':sources(),'hashes':{'model':sha256(args.model),'library_db':sha256(args.library_root/'poses.db'),'geometry':geometry_hash(poses)}})
    manifest=[];answers=[]
    for i,pose in enumerate(chosen):
        yaw,pitch=(15,75,135,225)[i%4],(-25,20)[(i//4)%2]
        cameras=(Camera(yaw,pitch,(4.5,6.5)[(i//8)%2]),strong_camera(pose.joints,yaw,pitch),Camera((0,45,90,180)[i%4]))
        for group,cam in zip(GROUPS,cameras):
            name=f'{group}_{i:03d}.png'
            im,xy=render_pose(pose.joints,cam.yaw,style='rough',seed=SEED+i,pitch=cam.pitch,distance=cam.distance)
            im.save(args.out/'images'/name)
            manifest.append({'image':name,'sha256':sha256(args.out/'images'/name)})
            answers.append({'image':name,'group':group,'pose_id':pose.pose_id,'family_id':pose.family_id,
                            'camera':asdict(cam),'body_depth_ratio':projection_strength(pose.joints,cam),'ground_truth_xy':xy.tolist()})
        if (i+1)%30==0:print('rendered',i+1,'/120',flush=True)
    save_json(args.out/'manifest.json',manifest);save_json(args.out/'answers.json',answers)
    strong=sum(r['group']=='strong_perspective' and r['body_depth_ratio']>=1.5 for r in answers)
    save_json(args.out/'prepared.json',{'manifest_sha256':sha256(args.out/'manifest.json'),
        'answers_sha256':sha256(args.out/'answers.json'),'strong_ratio_ge_1_5':strong})
    assert strong>=100


def infer(args):prepared(args);image_infer(args)


def rank(args):
    prepared(args);assert not (args.out/'rankings.json').exists()
    poses=load_library(args.library_root);assert geometry_hash(poses)==verify(args)['hashes']['geometry']
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras()
    index=build_camera_index(poses,cameras);base=(index[0][:117],index[1][:117])
    points=normalized_joints(np.stack([p.joints for p in poses]));rows=[]
    for item in read(args.out/'manifest.json'):
        name=item['image'];assert sha256(args.out/'images'/name)==item['sha256']
        path=args.out/'predictions'/(Path(name).stem+'.json');pred=read(path)
        row={'image':name,'prediction_sha256':sha256(path)};query=valid_query(pred)
        if query is None:
            row['error']=pred.get('error','insufficient Human-Art body joints/anchors');rows.append(row);continue
        xy,mask=query;row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        q=normalize_batch(xy);alt,am=alternate_labels(q,mask)
        baseline=np.minimum(score_camera_index(base,q,mask),score_camera_index(base,alt,am))
        coarse=np.minimum(score_camera_index(index,q,mask),score_camera_index(index,alt,am))
        start=time.perf_counter();fit,params=refine_costs(points,index,cameras,q,mask)
        row['refinement_seconds']=time.perf_counter()-start
        score=blend_cost(baseline,fit,.75);first=int(np.argmin(baseline))
        order=merged_order(score,score);order=np.concatenate(([first],order[order!=first]))
        orders={'pooled_117':merged_order(baseline,baseline),
                'keepfirst_195':merged_order(baseline,coarse,preserve_first=True),SELECTED:order}
        for arm,indices in orders.items():row[arm]=[poses[int(i)].pose_id for i in indices[:5]]
        assert len({row[a][0] for a in ARMS})==1
        dest=args.out/'costs'/(Path(name).stem+'.npz');np.savez_compressed(dest,baseline=baseline,coarse=coarse,refined=fit,parameters=params)
        row['costs_sha256']=sha256(dest);rows.append(row)
        if len(rows)%30==0:print('ranked',len(rows),'/360',flush=True)
    assert len(rows)==360;verify(args)
    save_json(args.out/'rankings.json',rows);save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def success_gate(groups):
    contrast=groups['strong_perspective']['selected_vs_near']
    return bool(contrast['delta']>=.05 and contrast['paired_bootstrap_95ci'][0]>0 and contrast['exact_mcnemar_p']<.05
        and all(groups[g]['counts'][SELECTED]['top5']>=max(groups[g]['counts'][b]['top5'] for b in ARMS[:2]) for g in (GROUPS[0],GROUPS[2]))
        and all(groups[g]['counts'][SELECTED]['top1']==groups[g]['counts']['pooled_117']['top1'] for g in GROUPS))


def evaluate(args):
    prepared(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.out/'answers.json')}
    assert len(rows)==len(answers)==len({r['image'] for r in rows})==360
    assert set(answers)=={r['image'] for r in rows}
    groups={}
    for g in GROUPS:
        records=[]
        for row in rows:
            target=answers[row['image']]
            if target['group']!=g:continue
            rec={k:target[k] for k in ('image','pose_id','family_id','body_depth_ratio')}
            if row.get('error'):rec['error']=row['error']
            for a in ARMS:
                ids=row.get(a,[]);rec[a+'_rank']=ids.index(target['pose_id'])+1 if target['pose_id'] in ids else None
            records.append(rec)
        assert len(records)==len({r['family_id'] for r in records})==120
        counts={a:{f'top{k}':sum(r[a+'_rank'] is not None and r[a+'_rank']<=k for r in records) for k in (1,5)} for a in ARMS}
        groups[g]={'n':120,'extraction_failures':sum('error' in r for r in records),'counts':counts,
                   'selected_vs_near':paired_comparison(records,SELECTED,'keepfirst_195'),
                   'selected_vs_117':paired_comparison(records,SELECTED,'pooled_117'),'records':records}
        print(g,{k:v for k,v in groups[g].items() if k!='records'},flush=True)
    groups['gate_passed']=success_gate(groups);save_json(args.out/'results.json',groups);print('gate_passed',groups['gate_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    for name in ('out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.refinement-confirm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
