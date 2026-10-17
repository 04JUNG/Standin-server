"""Independent sampling, projection, score and outcome audit of frozen prior."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_camera_regularization import (prepared,verify,read,select,GROUPS,ARMS,BASE,SELECTED,
    load_library,geometry_hash,sha256,save_json,energy_stats,success_gate)
from scripts.experiment_near_camera import near_cameras
from standin_eval.perspective_search import Camera,project,camera_points,normalize_batch,camera_grid,build_camera_index,score_camera_index
from standin_eval.camera_refinement import parameter_cost,normalized_joints
from scripts.confirm_camera_fit import paired_comparison


def audit(args):
    prepared(args);policy=verify(args)
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root);assert geometry_hash(poses)==policy['hashes']['geometry']
    excluded={r['family_id'] for path in policy['prior_answers'] for r in read(Path(path))}
    assert len(excluded)==760 and excluded==set(policy['excluded_families'])
    chosen=select(poses,excluded,seed=policy['seed'],count=120)
    assert [p.pose_id for p in chosen]==policy['test_pose_ids']
    assert [p.family_id for p in chosen]==policy['test_families']
    assert not {p.family_id for p in chosen}&excluded
    manifest=read(args.out/'manifest.json');answers={r['image']:r for r in read(args.out/'answers.json')}
    rows={r['image']:r for r in read(args.out/'rankings.json')}
    expected={f'{g}_{i:03d}.png' for g in GROUPS for i in range(120)}
    assert len(manifest)==len(answers)==len(rows)==360
    assert {r['image'] for r in manifest}==set(answers)==set(rows)==expected
    index=build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117'])
    points=normalized_joints(np.stack([p.joints for p in poses]))
    permutation=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])
    checked=failures=physical=0;maxerror=0.;ratios={g:[] for g in GROUPS}
    energies={f:{a:0. for a in ARMS} for f in policy['test_families']}
    records={g:[] for g in GROUPS}
    for item in manifest:
        name=item['image'];target=answers[name];row=rows[name];g=target['group']
        i=int(Path(name).stem.rsplit('_',1)[1]);pose=chosen[i]
        assert name==f'{g}_{i:03d}.png' and target['pose_id']==pose.pose_id and target['family_id']==pose.family_id
        cam=Camera(**target['camera']);yaw,pitch=(15,75,135,225)[i%4],(-25,20)[(i//4)%2]
        if g==GROUPS[0]:assert cam==Camera(yaw,pitch,(4.5,6.5)[(i//8)%2])
        elif g==GROUPS[2]:assert cam==Camera((0,45,90,180)[i%4])
        else:
            p=camera_points(pose.joints,Camera(yaw,pitch));d=max(float(p[5:,2].max()-2*p[5:,2].min()),.65-float(p[:,2].min()))
            assert cam.yaw==yaw and cam.pitch==pitch and abs(cam.distance-d)<1e-6
        xy,v=project(pose.joints,cam)
        assert v and np.allclose(normalize_batch(xy),normalize_batch(target['ground_truth_xy']),atol=1e-5)
        ratio=1.
        if cam.distance is not None:
            depth=camera_points(pose.joints,cam)[5:,2]+cam.distance;ratio=float(depth.max()/depth.min())
        assert abs(ratio-target['body_depth_ratio'])<1e-6;ratios[g].append(ratio)
        assert sha256(args.out/'images'/name)==item['sha256']
        path=args.out/'predictions'/(Path(name).stem+'.json');assert sha256(path)==row['prediction_sha256']
        pred=read(path);rec={k:target[k] for k in ('image','pose_id','family_id','body_depth_ratio')}
        for arm in ARMS:
            ids=row.get(arm,[]);rec[arm+'_rank']=ids.index(target['pose_id'])+1 if target['pose_id'] in ids else None
        if row.get('error'):rec['error']=row['error']
        records[g].append(rec)
        if not pred.get('error'):
            xy=np.asarray(pred['keypoints2d'],dtype=np.float32);mask=np.asarray(pred['scores'],dtype=np.float32)>=.3
        if pred.get('error') or not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            assert row.get('error') and all(a not in row for a in ARMS);failures+=1;continue
        assert mask.tolist()==row['shared_mask'] and hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']
        q=normalize_batch(xy);alt=q[permutation];am=mask[permutation]
        costpath=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(costpath)==row['costs_sha256']
        with np.load(costpath,allow_pickle=False) as data:
            base=data['baseline'];fit=data['refined'];fp=data['parameters'];reg=data['regularized'];rp=data['regularized_parameters']
        recomputed=np.minimum(score_camera_index(index,q,mask),score_camera_index(index,alt,am))
        assert np.array_equal(base,recomputed)
        first=int(np.argmin(base))
        for arm,cost,params,weight in ((BASE,fit,fp,0.),(SELECTED,reg,rp,.05)):
            assert np.isfinite(cost).all() and np.all(cost>=0)
            dc=parameter_cost(points,params,q,mask)+weight*params[:,2].astype(float)**2
            ac=parameter_cost(points,params,alt,am)+weight*params[:,2].astype(float)**2
            assert np.min(np.stack([abs(dc-cost),abs(ac-cost)]),axis=0).max()<2e-6
            score=.25*base.astype(float)+.75*cost.astype(float)
            ordered=np.argsort(score,kind='stable').tolist();ordered=[first]+[j for j in ordered if j!=first]
            assert row[arm]==[poses[j].pose_id for j in ordered[:5]]
            energy=float(np.sum(params[ordered[1:5],2].astype(float)**2))
            assert energy==row['inverse_distance_squared'][arm];energies[target['family_id']][arm]+=energy
            for j in sorted(set(ordered[:5]+list(range(0,len(poses),600)))):
                cy,cp,inv=map(float,params[j]);assert 0<=cy<360 and -60<=cp<=60 and 0<=inv<=1.25
                candidate,v=project(poses[j].joints,Camera(cy,cp,None if inv==0 else 1/inv));assert v
                candidate=normalize_batch(candidate);m=mask.copy();m[:5]=False;ma=am.copy();ma[:5]=False
                dc=float(np.linalg.norm(candidate[m]-q[m],axis=-1).mean())+weight*inv**2
                ac=float(np.linalg.norm(candidate[ma]-alt[ma],axis=-1).mean())+weight*inv**2
                error=min(abs(float(cost[j])-dc),abs(float(cost[j])-ac));assert error<3e-5
                maxerror=max(maxerror,error);physical+=1
        checked+=1
    assert checked+failures==360 and sum(r>=1.5 for r in ratios[GROUPS[1]])>=100
    reported=read(args.out/'results.json')
    for g,rr in records.items():
        counts={a:{f'top{k}':sum(r[a+'_rank'] is not None and r[a+'_rank']<=k for r in rr) for k in (1,5)} for a in ARMS}
        assert counts==reported['groups'][g]['counts']
        assert sum('error' in r for r in rr)==reported['groups'][g]['extraction_failures']
        assert paired_comparison(rr,SELECTED,BASE)==reported['groups'][g]['selected_vs_baseline']
    stats=energy_stats([v[BASE] for v in energies.values()],[v[SELECTED] for v in energies.values()])
    assert stats==reported['energy']
    assert reported['gate_passed']==success_gate(reported['groups'],stats)
    verify(args)
    result={'family_selection_replayed':True,'family_overlap':0,'excluded_families':760,'new_families':120,
        'mirror_sources':sum(p.pose_id.endswith('_mirror') for p in chosen),'images_answers_and_predictions_verified':360,
        'both_rankings_and_full_gallery_costs_checked':checked,'baseline_cost_vectors_recomputed':checked,'failures_retained':failures,
        'independent_scalar_projection_checks':physical,'max_scalar_cost_error':maxerror,
        'depth_ratio_min_median_max':{g:np.quantile(v,[0,.5,1]).tolist() for g,v in ratios.items()},
        'outcome_recomputed':True,'audit_script_sha256':sha256(Path(__file__)),'ranking_sha256':sha256(args.out/'rankings.json')}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())
