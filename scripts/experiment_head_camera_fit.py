"""Development: joint fitting versus frozen head rerank and equal-budget control."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_head_hint import verify as verify_head,read,BASE,GROUPS,order,load_library,save_json,sha256,valid_query
from scripts.experiment_refinement_blend import blend_cost
from scripts.confirm_camera_fit import geometry_hash
from scripts.experiment_near_camera import near_cameras
from standin_eval.perspective_search import normalize_batch,camera_grid,build_camera_index
from standin_eval.camera_refinement import normalized_joints
from standin_eval.head_camera_fit import fit_joint

RERANK='head010';CONTROL='body_refit';JOINT='joint_head010'
ARMS=(BASE,RERANK,CONTROL,JOINT)
CODE=('scripts/experiment_head_camera_fit.py','standin_eval/head_camera_fit.py',
      'standin_eval/head_hint.py','standin_eval/camera_refinement.py','standin_eval/perspective_search.py',
      'standin_eval/dual_label_search.py','scripts/experiment_refinement_blend.py')


def gate(counts):
    return bool(counts[JOINT]['strong_perspective']['top5']>=max(counts[b]['strong_perspective']['top5'] for b in (BASE,CONTROL))+3
        and all(counts[JOINT][g]['top5']>=max(counts[b][g]['top5'] for b in ARMS[:3]) for g in (GROUPS[0],GROUPS[2]))
        and all(counts[JOINT][g]['top1']==counts[BASE][g]['top1'] for g in GROUPS))


def verify(args):
    verify_head(SimpleNamespace(source=args.source,refinement_out=args.refinement_out,blend_out=args.blend_out,
                                out=args.head_out,library_root=args.library_root,model=args.model))
    p=read(args.out/'protocol.json')
    assert all(sha256(ROOT/f)==h for f,h in p['frozen_sources'].items())
    assert all(sha256(Path(f))==h for f,h in p['input_hashes'].items())


def rank(args):
    verify_head(SimpleNamespace(source=args.source,refinement_out=args.refinement_out,blend_out=args.blend_out,
                                out=args.head_out,library_root=args.library_root,model=args.model))
    assert read(args.head_out/'audit.json')['ranking_sha256']==sha256(args.head_out/'rankings.json')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'costs').mkdir()
    paths=[args.head_out/n for n in ('protocol.json','rankings.json','results.json','audit.json')]
    paths+=[args.source/'answers.json',args.refinement_out/'rankings.json']
    save_json(args.out/'protocol.json',{
        'stage':'development only, same examined120 families/360 images',
        'arms':ARMS,'weight':.1,'threshold':.7,'tolerance':.25,'cap':1.,
        'hypothesis':'head information must affect fitted cameras rather than only rerank body-fitted cameras',
        'objective':'bodyXY + (.1/.75)*clipped_head_error; final .25*117cost+.75*objective, equivalent to prior rerank weight .1 at fixed camera',
        'control':'body_refit uses identical four-start optimizer and same confidence eligibility but head weight zero; isolates extra optimization from extra information',
        'starts':'three best195 coarse cameras under each objective plus previous body-fitted camera; all1981 poses and both labels',
        'steps':'same4 scales,2 sweeps,pitch+-60,inverse_distance0..1.25; no distance regularization',
        'fixed':'raw Human-Art XY/bodymask/normalization/first117 result; confidence<.7 preserves complete original cost/parameters/order',
        'selection':'joint strongTop5>=max(original,matched body refit)+3; moderate/controlTop5>=all three baselines; every Top1 preserved',
        'failure':'all3 existing extraction failures retained in denominator',
        'ranking_inputs':'observed XY/confidence, gallery, prior unlabelled cameras/costs; no answers or group routing',
        'frozen_sources':{p:sha256(ROOT/p) for p in CODE},'input_hashes':{str(p):sha256(p) for p in paths}})
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    points=normalized_joints(np.stack([p.joints for p in poses]))
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras();index=build_camera_index(poses,cameras)
    previous={r['image']:r for r in read(args.refinement_out/'rankings.json')};rows=[]
    for old in read(args.head_out/'rankings.json'):
        name=old['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==old['prediction_sha256'];pred=read(path);query=valid_query(pred)
        row={k:old[k] for k in ('image','prediction_sha256')}
        if query is None:
            assert old.get('error');row['error']=old['error'];rows.append(row);continue
        xy,mask=query;q=normalize_batch(xy)
        row['shared_xy_sha256']=hashlib.sha256(xy.tobytes()).hexdigest();row['shared_mask']=mask.tolist()
        assert row['shared_xy_sha256']==old['shared_xy_sha256'] and row['shared_mask']==old['shared_mask']
        op=args.refinement_out/'costs'/(Path(name).stem+'.npz');assert sha256(op)==previous[name]['costs_sha256']
        with np.load(op,allow_pickle=False) as d:base,fit,warm=[d[k] for k in ('baseline','refined','parameters')]
        hp=args.head_out/'costs'/(Path(name).stem+'.npz');assert sha256(hp)==old['costs_sha256']
        with np.load(hp,allow_pickle=False) as d:head=d['head_penalty']
        bo=order(base,blend_cost(base,fit,.75));ho=order(base,blend_cost(base,fit,.75)+.1*head)
        row[BASE]=[poses[i].pose_id for i in bo[:5]];row[RERANK]=[poses[i].pose_id for i in ho[:5]]
        assert row[BASE]==old[BASE] and row[RERANK]==old[RERANK]
        row['enabled']=old['head_hint_enabled'];output={}
        for arm,w in ((CONTROL,0.),(JOINT,.1)):
            if row['enabled']:
                cost,params=fit_joint(points,index,cameras,q,mask,warm,w)
                limit=fit.astype(float)+(w/.75)*head
                assert np.all(cost<=limit+2e-6)
            else:cost,params=fit.copy(),warm.copy()
            ordered=order(base,blend_cost(base,cost,.75));row[arm]=[poses[i].pose_id for i in ordered[:5]]
            assert row[arm][0]==row[BASE][0]
            if not row['enabled']:assert row[arm]==row[BASE]
            output[arm+'_cost']=cost;output[arm+'_parameters']=params
        path=args.out/'costs'/(Path(name).stem+'.npz');np.savez_compressed(path,**output)
        row['costs_sha256']=sha256(path);rows.append(row)
        if len(rows)%30==0:print('fitted',len(rows),'/360',flush=True)
    assert len(rows)==360;verify(args)
    save_json(args.out/'rankings.json',rows);save_json(args.out/'ranked.json',{'sha256':sha256(args.out/'rankings.json'),'answers_parsed':False})


def evaluate(args):
    verify(args);assert not (args.out/'results.json').exists()
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    rows=read(args.out/'rankings.json');answers={r['image']:r for r in read(args.source/'answers.json')}
    assert len(rows)==len(answers)==360 and set(answers)=={r['image'] for r in rows}
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in ARMS}
    changes={b:{g:{'gains':0,'losses':0} for g in GROUPS} for b in ARMS[:3]}
    failures={g:0 for g in GROUPS};applied={g:0 for g in GROUPS}
    for row in rows:
        truth=answers[row['image']];g=truth['group'];target=truth['pose_id']
        failures[g]+=bool(row.get('error'));applied[g]+=row.get('enabled',False)
        for arm in ARMS:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=bool(ids and ids[0]==target);counts[arm][g]['top5']+=target in ids
        for b in changes:
            before=target in row.get(b,[]);after=target in row.get(JOINT,[])
            changes[b][g]['gains']+=after and not before;changes[b][g]['losses']+=before and not after
    result={'stage':'development only','counts':counts,'joint_vs_baselines':changes,'failures':failures,'applied':applied,
            'gate_passed':gate(counts),'selected':JOINT if gate(counts) else None}
    save_json(args.out/'results.json',result);print(result,flush=True)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rank','evaluate'))
    for name in ('source','refinement-out','blend-out','head-out','out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.head-camera-fit-dev.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
