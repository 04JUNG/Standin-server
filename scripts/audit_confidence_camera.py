"""Check objective/camera/label consistency, matched optimizer, ranks and outcomes."""
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_confidence_camera import (parse,verify,read,BASE,RERANK,CONTROL,JOINT,ARMS,GROUPS,
    load_library,sha256,save_json,valid_query,geometry_hash,normalized_joints,near_cameras)
from standin_eval.perspective_search import Camera,project,normalize_batch,camera_grid,build_camera_index
from standin_eval.camera_refinement import project_parameters
from standin_eval.confidence_camera_fit import fit_confidence


def audit(args):
    verify(args);assert not (args.out/'audit.json').exists()
    rows=read(args.out/'rankings.json')
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root)
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    points=normalized_joints(np.stack([p.joints for p in poses]))
    cams=camera_grid()['yaw_pitch_perspective_117']+near_cameras();index=build_camera_index(poses,cams)
    old={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    prior={r['image']:r for r in read(args.confidence_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    checked=failed=physical=replayed=0;maxerror=0.;maxvector=0.
    for row in rows:
        name=row['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==row['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        pred=read(path);query=valid_query(pred)
        if query is None:
            assert row.get('error')==old[name]['error'] and all(a not in row for a in ARMS)
            failed+=1;continue
        xy,mask=query;q=normalize_batch(xy)
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256'] and mask.tolist()==row['shared_mask']
        path=args.refinement_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==old[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit,warm=[d[k] for k in ('baseline','refined','parameters')]
        path=args.confidence_out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==prior[name]['costs_sha256']
        with np.load(path,allow_pickle=False) as d:frozen=d['weighted']
        assert row[BASE]==prior[name][BASE] and row[RERANK]==prior[name][RERANK]
        path=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==row['costs_sha256']
        with np.load(path,allow_pickle=False) as d:
            for arm,weighted in ((CONTROL,False),(JOINT,True)):
                cost=d[arm+'_cost'];params=d[arm+'_parameters'];labels=d[arm+'_swapped']
                assert labels.dtype==bool and labels.shape==(len(poses),)
                assert np.isfinite(cost).all() and (cost>=0).all()
                assert np.all(cost<=(frozen if weighted else fit)+2e-6)
                assert np.all((params[:,0]>=0)&(params[:,0]<360))
                assert np.all(abs(params[:,1])<=60) and np.all((params[:,2]>=0)&(params[:,2]<=1.25))
                features,valid=project_parameters(points,params);assert valid.all()
                # Explicit independent slot mapping/weight arithmetic; not production objective.
                costs=[]
                for swapped in (False,True):
                    total=np.zeros(len(poses));denom=0.
                    for j in range(5,17):
                        k=(j+1 if j%2 else j-1) if swapped else j
                        if not mask[k]:continue
                        w=.25+.75*min(1.,max(0.,(pred['scores'][k]-.3)/.4)) if weighted else 1.
                        total+=np.linalg.norm(features[:,j]-q[k],axis=1).astype(float)*w;denom+=w
                    costs.append(total/denom)
                delta=float(np.max(abs(cost-np.where(labels,costs[1],costs[0]))))
                maxvector=max(maxvector,delta);assert delta<1e-10
                score=.25*base.astype(float)+.75*cost.astype(float);first=int(np.argmin(base))
                ordered=np.argsort(score,kind='stable').tolist();ordered=[first]+[i for i in ordered if i!=first]
                assert row[arm]==[poses[i].pose_id for i in ordered[:5]] and row[arm][0]==row[BASE][0]
                for i in sorted(set(ordered[:5]+list(range(0,len(poses),600)))):
                    yaw,pitch,inv=map(float,params[i])
                    px,v=project(poses[i].joints,Camera(yaw,pitch,None if inv==0 else 1/inv));assert v
                    px=normalize_batch(px);num=den=0.
                    for j in range(5,17):
                        k=(j+1 if j%2 else j-1) if labels[i] else j
                        if not mask[k]:continue
                        w=.25+.75*min(1.,max(0.,(pred['scores'][k]-.3)/.4)) if weighted else 1.
                        num+=float(np.linalg.norm(px[j]-q[k]))*w;den+=w
                    delta=abs(float(cost[i])-num/den);assert delta<3e-5
                    maxerror=max(maxerror,delta);physical+=1
                if checked<3:
                    fc,fp,fl=fit_confidence(points,index,cams,q,mask,warm,pred['scores'],weighted=weighted)
                    assert np.array_equal(fc,cost) and np.array_equal(fp,params) and np.array_equal(fl,labels)
                    replayed+=1
        checked+=1
    assert len(rows)==checked+failed==360
    answers={r['image']:r for r in read(args.source/'answers.json')};reported=read(args.out/'results.json')
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in ARMS}
    changes={b:{g:{'gains':0,'losses':0} for g in GROUPS} for b in ARMS[:3]};failures={g:0 for g in GROUPS}
    for row in rows:
        truth=answers[row['image']];g=truth['group'];target=truth['pose_id'];failures[g]+=bool(row.get('error'))
        for arm in ARMS:
            ids=row.get(arm,[]);counts[arm][g]['top1']+=int(bool(ids) and ids[0]==target);counts[arm][g]['top5']+=int(target in ids)
        for b in changes:
            before=target in row.get(b,[]);after=target in row.get(JOINT,[])
            changes[b][g]['gains']+=int(after and not before);changes[b][g]['losses']+=int(before and not after)
    assert counts==reported['counts'] and changes==reported['joint_vs_baselines'] and failures==reported['failures']
    gate=(counts[JOINT]['strong_perspective']['top5']>=max(counts[b]['strong_perspective']['top5'] for b in (BASE,CONTROL))+3
          and counts[JOINT]['strong_perspective']['top5']>=counts[RERANK]['strong_perspective']['top5']
          and all(counts[JOINT][g]['top5']>=max(counts[b][g]['top5'] for b in ARMS[:3]) for g in (GROUPS[0],GROUPS[2]))
          and all(counts[JOINT][g]['top1']==counts[BASE][g]['top1'] for g in GROUPS))
    assert reported['gate_passed']==gate and reported['selected']==(JOINT if gate else None)
    verify(args)
    result={'inputs_verified':360,'full_gallery_costs_and_rankings_checked':checked,'failures_retained':failed,
            'optimizer_replays':replayed,'independent_scalar_projections':physical,'max_scalar_error':maxerror,
            'max_independent_vector_error':maxvector,'outcomes_recomputed':True,
            'ranking_sha256':sha256(args.out/'rankings.json'),'audit_script_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':audit(parse())
