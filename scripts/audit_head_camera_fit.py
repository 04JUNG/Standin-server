"""Audit warm-start fitting, equal-budget control, no-change branch and outcomes."""
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_head_camera_fit import (parse,verify,read,BASE,RERANK,CONTROL,JOINT,ARMS,GROUPS,
    load_library,sha256,save_json,valid_query,geometry_hash,normalized_joints,near_cameras)
from standin_eval.perspective_search import Camera,project,normalize_batch,camera_grid,build_camera_index
from standin_eval.dual_label_search import alternate_labels
from standin_eval.head_camera_fit import objective,fit_joint


def audit(args):
    verify(args);rows=read(args.out/'rankings.json')
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    points=normalized_joints(np.stack([p.joints for p in poses]))
    cams=camera_grid()['yaw_pitch_perspective_117']+near_cameras();index=build_camera_index(poses,cams)
    old={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    prior={r['image']:r for r in read(args.head_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    checked=failed=enabled=disabled=physical=replayed=0;maxerror=0.
    for row in rows:
        name=row['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==row['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        pred=read(path);query=valid_query(pred)
        if query is None:
            assert row.get('error')==old[name]['error'] and all(a not in row for a in ARMS)
            failed+=1;continue
        xy,mask=query;q=normalize_batch(xy);alt,am=alternate_labels(q,mask)
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256'] and mask.tolist()==row['shared_mask']
        active=bool(np.isfinite(pred['scores'][0]) and pred['scores'][0]>=.7)
        assert row['enabled']==active
        path=args.refinement_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==old[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit,warm=[d[k] for k in ('baseline','refined','parameters')]
        path=args.head_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==prior[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:head=d['head_penalty']
        assert row[BASE]==prior[name][BASE] and row[RERANK]==prior[name][RERANK]
        path=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==row['costs_sha256']
        with np.load(path,allow_pickle=False) as d:
            for arm,weight in ((CONTROL,0.),(JOINT,.1)):
                cost=d[arm+'_cost'];params=d[arm+'_parameters'];w=weight if active else 0.
                assert np.isfinite(cost).all() and (cost>=0).all()
                if not active:assert np.array_equal(cost,fit) and np.array_equal(params,warm) and row[arm]==row[BASE]
                assert np.all(cost<=fit.astype(float)+(w/.75)*head+2e-6)
                dc=objective(points,params,q,mask,w);ac=objective(points,params,alt,am,w)
                assert np.minimum(abs(cost-dc),abs(cost-ac)).max()<2e-6
                score=.25*base.astype(float)+.75*cost.astype(float);first=int(np.argmin(base))
                ordered=np.argsort(score,kind='stable').tolist();ordered=[first]+[i for i in ordered if i!=first]
                assert row[arm]==[poses[i].pose_id for i in ordered[:5]] and row[arm][0]==row[BASE][0]
                for i in sorted(set(ordered[:5]+list(range(0,len(poses),600)))):
                    yaw,pitch,inv=map(float,params[i]);assert 0<=yaw<360 and -60<=pitch<=60 and 0<=inv<=1.25
                    px,v=project(poses[i].joints,Camera(yaw,pitch,None if inv==0 else 1/inv));assert v
                    px=normalize_batch(px);md=mask.copy();md[:5]=False;ma=am.copy();ma[:5]=False
                    head_error=float(np.linalg.norm(px[0]-q[0]));hp=min(1,max(0,head_error-.25))
                    dc=float(np.linalg.norm(px[md]-q[md],axis=-1).mean())+(w/.75)*hp
                    ac=float(np.linalg.norm(px[ma]-alt[ma],axis=-1).mean())+(w/.75)*hp
                    delta=min(abs(float(cost[i])-dc),abs(float(cost[i])-ac));assert delta<3e-5
                    maxerror=max(maxerror,delta);physical+=1
                if active and enabled<3:
                    fc,fp=fit_joint(points,index,cams,q,mask,warm,weight)
                    assert np.array_equal(fc,cost) and np.array_equal(fp,params);replayed+=1
        enabled+=int(active);disabled+=int(not active);checked+=1
    assert len(rows)==checked+failed==360
    answers={r['image']:r for r in read(args.source/'answers.json')};reported=read(args.out/'results.json')
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in ARMS}
    changes={b:{g:{'gains':0,'losses':0} for g in GROUPS} for b in ARMS[:3]}
    failures={g:0 for g in GROUPS};applied={g:0 for g in GROUPS}
    for row in rows:
        truth=answers[row['image']];g=truth['group'];target=truth['pose_id']
        failures[g]+=bool(row.get('error'));applied[g]+=row.get('enabled',False)
        for arm in ARMS:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=int(bool(ids) and ids[0]==target);counts[arm][g]['top5']+=int(target in ids)
        for b in changes:
            before=target in row.get(b,[]);after=target in row.get(JOINT,[])
            changes[b][g]['gains']+=int(after and not before);changes[b][g]['losses']+=int(before and not after)
    assert counts==reported['counts'] and changes==reported['joint_vs_baselines']
    assert failures==reported['failures'] and applied==reported['applied']
    gate=(counts[JOINT][GROUPS[1]]['top5']>=max(counts[b][GROUPS[1]]['top5'] for b in (BASE,CONTROL))+3
          and all(counts[JOINT][g]['top5']>=max(counts[b][g]['top5'] for b in ARMS[:3]) for g in (GROUPS[0],GROUPS[2]))
          and all(counts[JOINT][g]['top1']==counts[BASE][g]['top1'] for g in GROUPS))
    assert reported['gate_passed']==gate and reported['selected']==(JOINT if gate else None)
    verify(args)
    result={'inputs_verified':360,'full_gallery_costs_and_rankings_checked':checked,'failures_retained':failed,
            'applied':enabled,'low_confidence_exactly_unchanged':disabled,'optimizer_replays':replayed,
            'independent_scalar_projections':physical,'max_scalar_error':maxerror,'outcomes_recomputed':True,
            'ranking_sha256':sha256(args.out/'rankings.json'),'audit_script_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':audit(parse())
