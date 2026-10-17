"""Reconstruct every regularized ranking and independently check camera costs."""
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_camera_regularization import (parse,verify,read,WEIGHTS,BASE,order,
    load_library,sha256,save_json,valid_query,normalize_batch,normalized_joints,geometry_hash,
    camera_grid,near_cameras,build_camera_index,GROUPS)
from standin_eval.camera_regularization import regularized_costs,penalty
from standin_eval.camera_refinement import parameter_cost
from standin_eval.perspective_search import Camera,project
from standin_eval.dual_label_search import alternate_labels


def audit(args):
    verify(args)
    rows=read(args.out/'rankings.json')
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root);points=normalized_joints(np.stack([p.joints for p in poses]))
    assert geometry_hash(poses)==read(args.source/'protocol.json')['hashes']['library_geometry']
    cams=camera_grid()['yaw_pitch_perspective_117']+near_cameras();index=build_camera_index(poses,cams)
    old={r['image']:r for r in read(args.refinement_out/'rankings.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    oldblend={r['image']:r for r in read(args.blend_out/'rankings.json')}
    checked=failures=physical=replays=0;maxerror=0.
    for number,row in enumerate(rows):
        name=row['image'];path=args.source/'predictions'/(Path(name).stem+'.json')
        assert sha256(path)==row['prediction_sha256']
        assert sha256(args.source/'images'/name)==manifest[name]['sha256']
        query=valid_query(read(path))
        if query is None:
            assert row.get('error')==old[name]['error'] and all(a not in row for a in WEIGHTS)
            failures+=1;continue
        xy,mask=query;q=normalize_batch(xy);alt,amask=alternate_labels(q,mask)
        assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']
        assert mask.tolist()==row['shared_mask']
        path=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(path)==row['costs_sha256']
        oldpath=args.refinement_out/'costs'/(Path(name).stem+'.npz')
        assert sha256(oldpath)==old[name]['costs_sha256']
        with np.load(oldpath,allow_pickle=False) as source:
            base=source['baseline'];bo=order(base,source['refined'])[:5]
            assert row[BASE]==oldblend[name][BASE]==[poses[i].pose_id for i in bo]
            assert row['inverse_distance_squared'][BASE]==float(np.sum(source['parameters'][bo[1:],2].astype(float)**2))
        with np.load(path,allow_pickle=False) as data:
            assert np.array_equal(data['baseline'],base)
            for arm,weight in WEIGHTS.items():
                cost=data[arm+'_cost'];params=data[arm+'_parameters']
                assert cost.shape==(len(poses),) and params.shape==(len(poses),3)
                assert np.isfinite(cost).all() and (cost>=0).all()
                assert ((params[:,0]>=0)&(params[:,0]<360)).all()
                assert ((params[:,1]>=-60)&(params[:,1]<=60)).all()
                assert ((params[:,2]>=0)&(params[:,2]<=1.25)).all()
                direct=parameter_cost(points,params,q,mask)+penalty(params,weight)
                swapped=parameter_cost(points,params,alt,amask)+penalty(params,weight)
                error=np.minimum(abs(direct-cost),abs(swapped-cost))
                assert np.max(error)<1e-6
                selected=order(base,cost)[:5]
                assert row[arm]==[poses[i].pose_id for i in selected]
                assert row[arm][0]==row[BASE][0]
                assert row['inverse_distance_squared'][arm]==float(np.sum(params[selected[1:],2].astype(float)**2))
                for i in sorted(set(selected.tolist()+list(range(0,len(poses),600)))):
                    yaw,pitch,inv=map(float,params[i]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
                    px,valid=project(poses[i].joints,cam);assert valid
                    px=normalize_batch(px);dmask=mask.copy();dmask[:5]=False
                    smask=amask.copy();smask[:5]=False
                    dc=np.linalg.norm(px[dmask]-q[dmask],axis=-1).mean()+weight*inv**2
                    sc=np.linalg.norm(px[smask]-alt[smask],axis=-1).mean()+weight*inv**2
                    error=min(abs(float(cost[i])-dc),abs(float(cost[i])-sc))
                    assert error<3e-5
                    maxerror=max(maxerror,error);physical+=1
                if number in (0,60,119,120,180,239,240,300,359):
                    fresh,fp=regularized_costs(points,index,cams,q,mask,weight)
                    assert np.array_equal(fresh,cost) and np.array_equal(fp,params)
                    replays+=1
        checked+=1
    assert checked+failures==len(rows)==360
    # Answers enter only this post-ranking scoring audit, not optimization.
    answer={a['image']:a for a in read(args.source/'answers.json')}
    reported=read(args.out/'results.json');arms=(BASE,*WEIGHTS)
    counts={a:{g:{'top1':0,'top5':0} for g in GROUPS} for a in arms}
    energy={a:0. for a in arms};failed={g:0 for g in GROUPS}
    transitions={a:{g:{'gains':0,'losses':0} for g in GROUPS} for a in WEIGHTS}
    for row in rows:
        truth=answer[row['image']];g=truth['group'];target=truth['pose_id']
        failed[g]+=bool(row.get('error'))
        for arm in arms:
            ids=row.get(arm,[])
            counts[arm][g]['top1']+=int(bool(ids) and ids[0]==target)
            counts[arm][g]['top5']+=int(target in ids)
            energy[arm]+=row.get('inverse_distance_squared',{}).get(arm,0.)
            if arm!=BASE:
                before=target in row.get(BASE,[]);after=target in ids
                transitions[arm][g]['gains']+=int(after and not before)
                transitions[arm][g]['losses']+=int(before and not after)
    assert counts==reported['counts'] and failed==reported['failures']
    assert energy==reported['sum_inverse_distance_squared_ranks2to5']
    eligible=[a for a in WEIGHTS if energy[a]<=.9*energy[BASE] and
              all(counts[a][g][m]>=counts[BASE][g][m] for g in GROUPS for m in ('top1','top5'))]
    eligible.sort(key=lambda a:(-counts[a]['strong_perspective']['top5'],
                  -sum(counts[a][g]['top5'] for g in GROUPS),energy[a],WEIGHTS[a]))
    assert eligible==reported['eligible'] and reported['selected']==(eligible[0] if eligible else None)
    verify(args)
    result={'ranking_sha256':sha256(args.out/'rankings.json'),'inputs_checked':360,
            'all_gallery_camera_costs_checked_queries':checked,'failures_retained':failures,
            'optimizer_replays':replays,'independent_scalar_checks':physical,
            'maximum_scalar_error':maxerror,'baseline_and_first_results_preserved':checked,
            'counts_and_selection_independently_recomputed':True,
            'top5_transitions_vs_baseline':transitions,
            'audit_script_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',result);print(result)


if __name__=='__main__':audit(parse())
