"""Frozen confirmation audit: sampling, render geometry, scores and candidate order."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_refinement_blend import prepared,verify,read,select,GROUPS,ARMS,SELECTED
from scripts.experiment_near_camera import near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json
from scripts.confirm_camera_fit import geometry_hash
from standin_eval.perspective_search import Camera,project,camera_points,normalize_batch,camera_grid,build_camera_index,score_camera_index


def audit(args):
    prepared(args);policy=verify(args)
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root);assert geometry_hash(poses)==policy['hashes']['geometry']
    excluded={r['family_id'] for p in policy['prior_answers'] for r in read(Path(p))}
    assert len(excluded)==640 and excluded==set(policy['excluded_families'])
    chosen=select(poses,excluded,seed=policy['seed'],count=120)
    assert [p.pose_id for p in chosen]==policy['test_pose_ids']
    assert [p.family_id for p in chosen]==policy['test_families']
    assert not {p.family_id for p in chosen}&excluded
    manifest=read(args.out/'manifest.json');answers={r['image']:r for r in read(args.out/'answers.json')}
    rows={r['image']:r for r in read(args.out/'rankings.json')}
    expected={f'{g}_{i:03d}.png' for g in GROUPS for i in range(120)}
    assert len(manifest)==len(answers)==len(rows)==360
    assert {r['image'] for r in manifest}==set(answers)==set(rows)==expected
    index=build_camera_index(poses,camera_grid()['yaw_pitch_perspective_117']+near_cameras())
    baseindex=(index[0][:117],index[1][:117])
    permutation=np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])
    checked=failures=physical=0;maxerror=0.;ratios={g:[] for g in GROUPS}
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
        pred=read(path)
        if pred.get('error'):
            assert row.get('error') and all(a not in row for a in ARMS);failures+=1;continue
        xy=np.asarray(pred['keypoints2d'],dtype=np.float32);mask=np.asarray(pred['scores'],dtype=np.float32)>=.3
        if not np.isfinite(xy).all() or mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            assert row.get('error') and all(a not in row for a in ARMS);failures+=1;continue
        assert mask.tolist()==row['shared_mask'] and hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']
        q=normalize_batch(xy);alt=q[permutation];am=mask[permutation]
        costpath=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(costpath)==row['costs_sha256']
        with np.load(costpath,allow_pickle=False) as data:
            base,coarse,fit,params=[data[k] for k in ('baseline','coarse','refined','parameters')]
        for ix,cached in ((baseindex,base),(index,coarse)):
            got=np.minimum(score_camera_index(ix,q,mask),score_camera_index(ix,alt,am))
            assert np.array_equal(got,cached)
        assert np.all(fit<=coarse+1e-6) and np.isfinite(fit).all() and np.all(fit>=0)
        score=.25*base.astype(float)+.75*fit.astype(float)
        bo=np.argsort(base,kind='stable').tolist();co=np.argsort(coarse,kind='stable').tolist();so=np.argsort(score,kind='stable').tolist()
        first=bo[0];orders={'pooled_117':bo,'keepfirst_195':[first]+[j for j in co if j!=first],SELECTED:[first]+[j for j in so if j!=first]}
        for a,order in orders.items():assert row[a]==[poses[j].pose_id for j in order[:5]]
        selected=set(so[:5]+bo[:5]+list(range(0,len(poses),400)))
        for j in selected:
            cy,cp,inv=map(float,params[j]);assert 0<=cy<360 and -60<=cp<=60 and 0<=inv<=1.25
            candidate,v=project(poses[j].joints,Camera(cy,cp,None if inv==0 else 1./inv));assert v
            candidate=normalize_batch(candidate);m=mask.copy();m[:5]=False;ma=am.copy();ma[:5]=False
            dc=float(np.linalg.norm(candidate[m]-q[m],axis=-1).mean());ac=float(np.linalg.norm(candidate[ma]-alt[ma],axis=-1).mean())
            error=min(abs(float(fit[j])-dc),abs(float(fit[j])-ac));assert error<3e-5
            maxerror=max(maxerror,error);physical+=1
        checked+=1
        if checked%60==0:print('audited',checked,flush=True)
    assert checked+failures==360 and sum(r>=1.5 for r in ratios[GROUPS[1]])>=100
    verify(args)
    result={'family_selection_replayed':True,'family_overlap':0,'excluded_families':640,'new_families':120,
        'mirror_sources':sum(p.pose_id.endswith('_mirror') for p in chosen),'images_answers_and_predictions_verified':360,
        'all_three_rankings_replayed':checked,'baseline_cost_vectors_recomputed':checked,'failures_retained':failures,
        'independent_scalar_projection_checks':physical,'max_scalar_cost_error':maxerror,
        'depth_ratio_min_median_max':{g:np.quantile(v,[0,.5,1]).tolist() for g,v in ratios.items()},
        'audit_script_sha256':sha256(Path(__file__)),'ranking_sha256':sha256(args.out/'rankings.json')}
    save_json(args.out/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())
