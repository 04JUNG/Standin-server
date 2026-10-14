"""Fresh sampling, geometry, frozen XY/camera, scalar boundary and score audit."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.confirm_contour_rerank import prepared,SEED,BASE,SELECTED,GROUPS
from scripts.confirm_perspective_isolation import select
from scripts.experiment_near_camera import read,near_cameras
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json,render_pose
from scripts.experiment_humanart_z import read_rgb
from scripts.confirm_camera_fit import geometry_hash
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.audit_contour_rerank import brute_cost
from standin_eval.perspective_search import camera_grid,build_camera_index,score_camera_index,normalize_batch
from standin_eval.camera_refinement import normalized_joints,refine_costs,project_parameters
from standin_eval.dual_label_search import alternate_labels,LEFT_RIGHT
from standin_eval.contour_rerank import prepare_query,candidate_evidence


def independent_stats(records):
    delta=[]
    for row in records:
        a=row[BASE+'_rank'];b=row[SELECTED+'_rank']
        delta.append(int(b is not None and b<=5)-int(a is not None and a<=5))
    gain=delta.count(1);loss=delta.count(-1);n=gain+loss
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(gain,loss)+1))/(2**n)) if n else 1.
    rng=np.random.default_rng(20261019);sample=np.asarray(delta)[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    return {'gains':gain,'losses':loss,'delta':float(np.mean(delta)),
            'paired_bootstrap_95ci':np.quantile(sample,[.025,.975]).tolist(),'exact_mcnemar_p':p}


def main(args):
    policy=prepared(args)
    assert sha256(args.out/'rankings.json')==read(args.out/'ranked.json')['sha256']
    poses=load_library(args.library_root);assert geometry_hash(poses)==policy['hashes']['geometry']
    by_id={p.pose_id:p for p in poses}
    excluded={r['family_id'] for name in policy['prior_answers'] for r in read(Path(name))}
    assert len(excluded)==880 and sorted(excluded)==policy['excluded_families']
    selected=select(poses,excluded,seed=SEED,count=120)
    assert [p.pose_id for p in selected]==policy['test_pose_ids']
    assert [p.family_id for p in selected]==policy['test_families']
    assert len(set(policy['test_families']))==120 and not (set(policy['test_families'])&excluded)
    answers=read(args.out/'answers.json');manifest=read(args.out/'manifest.json');rows=read(args.out/'rankings.json')
    results=read(args.out/'results.json');assert len(answers)==len(manifest)==len(rows)==360
    assert len({r['image'] for r in rows})==360
    truth={a['image']:a for a in answers};items={r['image']:r for r in manifest}
    max_pixel=0.;ratios=[]
    for a in answers:
        p=by_id[a['pose_id']];assert p.family_id==a['family_id']
        position=int(Path(a['image']).stem.rsplit('_',1)[1]);assert p.pose_id==selected[position].pose_id
        j=p.joints.astype(float);hip=(j[11]+j[12])/2;shoulder=(j[5]+j[6])/2
        points=(j-hip)/np.linalg.norm(shoulder-hip);yaw,pitch=np.deg2rad([a['camera']['yaw'],a['camera']['pitch']])
        ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        xyz=points@ry.T@rx.T;plane=xyz[:,:2].copy();plane[:,1]*=-1;distance=a['camera']['distance']
        ratio=1.
        if distance is not None:
            assert np.min(xyz[:,2]+distance)>.1
            plane/=xyz[:,2,None]+distance
            dep=xyz[5:,2]+distance;ratio=float(dep.max()/dep.min())
        assert abs(ratio-a['body_depth_ratio'])<2e-6
        if a['group']==GROUPS[1]:ratios.append(ratio)
        target=np.asarray(a['ground_truth_xy']);pc=plane-plane.mean(0);tc=target-target.mean(0)
        scale=float((pc*tc).sum()/(pc*pc).sum());assert scale>0
        error=float(abs(pc*scale-tc).max());max_pixel=max(max_pixel,error);assert error<1e-4
    assert sum(r>=1.5 for r in ratios)>=100
    cameras=camera_grid()['yaw_pitch_perspective_117']+near_cameras();index=build_camera_index(poses,cameras)
    bi=(index[0][:117],index[1][:117]);points=normalized_joints(np.stack([p.joints for p in poses]))
    checked=scalar=render_replays=optimizer_replays=0;max_boundary=0.;max_fit=0.;candidate_errors={}
    inferred=read(args.out/'inferred.json');assert len(inferred)==360
    for row in rows:
        name=row['image'];path=args.out/'predictions'/(Path(name).stem+'.json')
        assert sha256(args.out/'images'/name)==items[name]['sha256']
        assert sha256(path)==row['prediction_sha256']==inferred[path.name]
        query=valid_query(read(path))
        if row.get('error'):assert query is None;continue
        xy,mask=query;assert row['shared_mask']==mask.tolist()
        assert row['shared_xy_sha256']==hashlib.sha256(xy.tobytes()).hexdigest()
        costpath=args.out/'costs'/(Path(name).stem+'.npz');assert sha256(costpath)==row['costs_sha256']
        with np.load(costpath,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
        q=normalize_batch(xy);alt,am=alternate_labels(q,mask)
        recomputed=np.minimum(score_camera_index(bi,q,mask),score_camera_index(bi,alt,am))
        assert np.array_equal(recomputed,base)
        features,valid=project_parameters(points,params);assert valid.all();uniform=[]
        for perm in (np.arange(17),LEFT_RIGHT):
            keep=np.flatnonzero(mask[perm][5:])+5
            uniform.append(np.linalg.norm(features[:,keep]-q[perm][keep],axis=2).mean(1))
        err=float(np.min(abs(np.stack(uniform)-fit),axis=0).max());max_fit=max(max_fit,err);assert err<2e-5
        score=(1-.75)*base.astype(float)+.75*fit.astype(float);first=int(np.argmin(base))
        ordered=sorted(range(len(poses)),key=lambda k:(score[k],k));ordered=[first]+[k for k in ordered if k!=first]
        assert row['shortlist_indices']==ordered[:20]
        assert row['shortlist']==[poses[k].pose_id for k in ordered[:20]]
        assert row[BASE]==[poses[k].pose_id for k in ordered[:5]]
        assert row['fallback']==bool(row['candidate_errors'] or row.get('query_error'))
        for text in row['candidate_errors'].values():candidate_errors[text]=candidate_errors.get(text,0)+1
        assert len(row['contour_costs'])==20 and row['contour_costs'][0]==0 and all(0<=c<=.5 for c in row['contour_costs'])
        reordered=ordered
        if not row['fallback']:
            head=sorted(range(1,20),key=lambda k:(score[ordered[k]]+row['contour_costs'][k],k))
            reordered=[first]+[ordered[k] for k in head]+ordered[20:]
        assert row[SELECTED]==[poses[k].pose_id for k in reordered[:5]]
        assert reordered[0]==ordered[0] and reordered[20:]==ordered[20:]
        if optimizer_replays<2:
            rf,rp=refine_costs(points,index,cameras,q,mask)
            assert np.array_equal(rf,fit) and np.array_equal(rp,params);optimizer_replays+=1
        maskpath=args.out/'masks'/(Path(name).stem+'.npz')
        if maskpath.exists():
            evidence=prepare_query(read_rgb(args.out/'images'/name),xy)
            assert evidence['torso']==row['query_torso'] and evidence['mask'].sum()==row['query_line_pixels']
            with np.load(maskpath,allow_pickle=False) as masks:
                assert np.array_equal(evidence['mask'],masks['query'])
                for key in masks.files:
                    if key=='query':continue
                    rank=int(key.split('_')[1]);cost=brute_cost(masks['query'],masks[key],row['query_torso'])
                    delta=abs(cost-row['contour_costs'][rank]);max_boundary=max(max_boundary,delta);assert delta<2e-7;scalar+=1
                    if rank in (1,19):
                        replay,affine,pixels=candidate_evidence(evidence,poses[ordered[rank]].joints,params[ordered[rank]])
                        assert replay==row['contour_costs'][rank] and np.array_equal(pixels,masks[key])
                        assert affine.tolist()==row['affines'][str(rank)];render_replays+=1
        checked+=1
    for group in GROUPS:
        rr=[r for r in rows if truth[r['image']]['group']==group];assert len(rr)==120
        expected=[]
        for row in rr:
            a=truth[row['image']];rec={k:a[k] for k in ('image','pose_id','family_id','body_depth_ratio')}
            if row.get('error'):rec['error']=row['error']
            rec['fallback']=bool(row.get('fallback'));rec['target_in_original20']=a['pose_id'] in row.get('shortlist',[])
            for arm in (BASE,SELECTED):
                ids=row.get(arm,[]);rec[arm+'_rank']=ids.index(a['pose_id'])+1 if a['pose_id'] in ids else None
            expected.append(rec)
        out=results[group];assert expected==out['records']
        for arm in (BASE,SELECTED):
            for k in (1,5):assert out['counts'][arm]['top'+str(k)]==sum(r[arm+'_rank'] is not None and r[arm+'_rank']<=k for r in expected)
        assert out['comparison']==independent_stats(expected)
        assert out['extraction_failures']==sum('error' in r for r in expected)
        assert out['fallback_images']==sum(r['fallback'] for r in expected)
        assert out['target_in_original20']==sum(r['target_in_original20'] for r in expected)
    primary=results[GROUPS[1]]['comparison']
    gate=(primary['delta']>=.05 and primary['paired_bootstrap_95ci'][0]>0 and primary['exact_mcnemar_p']<.05
          and all(results[g]['counts'][SELECTED]['top5']>=results[g]['counts'][BASE]['top5'] for g in (GROUPS[0],GROUPS[2]))
          and all(results[g]['counts'][SELECTED]['top1']==results[g]['counts'][BASE]['top1'] for g in GROUPS))
    assert gate==results['gate_passed']
    report={'previous_families':880,'new_disjoint_families':120,'used_families_total':1000,
            'images_and_truth_projection_checked':360,'full_gallery_baselines_and_rankings':checked,
            'max_projection_pixel_error':max_pixel,'strong_depth_ratio_min_median_max':np.quantile(ratios,[0,.5,1]).tolist(),
            'max_stored_camera_fit_error':max_fit,'optimizer_replays':optimizer_replays,
            'bruteforce_boundary_costs':scalar,'max_boundary_difference':max_boundary,'render_replays':render_replays,
            'candidate_errors':candidate_errors,'gate_independently_recomputed':True,
            'ranking_sha256':sha256(args.out/'rankings.json'),'results_sha256':sha256(args.out/'results.json'),
            'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('out','library-root','model'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
