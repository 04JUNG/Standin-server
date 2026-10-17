"""Independent offline replay and immutable geometry snapshots for this study."""
import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bvh import load_coco17
from scripts.experiment_z_axis_search import load_library,_family_id
from scripts.experiment_z_rendered import sha256,save_json
from standin_eval.z_axis_search import Pose3D
from standin_eval.perspective_search import Camera,camera_grid,project,normalize_batch
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.dual_label_search import alternate_labels


def read(p):return json.loads(p.read_text())


def geometry_hash(poses):
    h=hashlib.sha256()
    for p in poses:h.update(p.pose_id.encode());h.update(np.asarray(p.joints,np.float32).tobytes())
    return h.hexdigest()


def historical_gallery(backup,current):
    """Read approved external backup without restoring/modifying shared library."""
    con=sqlite3.connect('file:'+str(backup/'poses.db')+'?mode=ro',uri=True)
    rows=con.execute('SELECT pose_id,bvh_path,meta_json FROM poses ORDER BY pose_id').fetchall();con.close()
    poses=[]
    for pid,bvh,meta in rows:
        paths=(Path(bvh),current/'bvh'/Path(bvh).name,backup/'removed'/Path(bvh).name)
        path=next(p for p in paths if p.is_file())
        joints,scores=load_coco17(str(path))
        poses.append(Pose3D(pid,_family_id(pid,json.loads(meta or '{}')),np.asarray(joints,np.float32),np.asarray(scores,np.float32)))
    return poses


def snapshot(path,poses):
    if path.exists():
        with np.load(path,allow_pickle=False) as d:
            assert np.array_equal(d['joints'],np.stack([p.joints for p in poses]))
            assert d['pose_ids'].tolist()==[p.pose_id for p in poses]
    else:np.savez_compressed(path,pose_ids=[p.pose_id for p in poses],family_ids=[p.family_id for p in poses],
                             joints=np.stack([p.joints for p in poses]),scores=np.stack([p.scores for p in poses]))


def brute_contour_cost(query,candidate,torso):
    """Pixel nearest-neighbor distance without OpenCV distance transforms."""
    a=np.argwhere(query);b=np.argwhere(candidate)
    def direction(x,y):
        values=[]
        for start in range(0,len(x),128):
            delta=x[start:start+128,None,:].astype(float)-y[None,:,:]
            values.extend(np.minimum(np.sqrt(np.square(delta).sum(-1).min(1))/torso,.5))
        return np.mean(values)
    return float((direction(a,b)+direction(b,a))/2)


def audit(a):
    policy=read(a.fresh/'protocol.json');prepared=read(a.fresh/'prepared.json')
    assert sha256(a.library_root/'poses.db')==policy['library_db_sha256']
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h,n
    for n,h in policy['inputs'].items():assert sha256(Path(n))==h,n
    assert sha256(a.fresh/'manifest.json')==prepared['manifest_sha256']
    assert sha256(a.fresh/'answers.json')==prepared['answers_sha256']
    poses=load_library(a.library_root);old=historical_gallery(a.backup,a.library_root)
    old_policy=read(a.base/'out/refinement-blend-confirm-20261013/protocol.json')
    assert sha256(a.backup/'poses.db')==old_policy['hashes']['library_db']
    assert geometry_hash(old)==old_policy['hashes']['geometry'],'historical BVH geometry drift'
    assert len(old)==1981 and len(poses)==1995
    snapshot(a.fresh/'gallery-1995.npz',poses);snapshot(a.fresh/'historical-gallery-1981.npz',old)
    byid={p.pose_id:p for p in poses};truth=read(a.fresh/'answers.json');manifest=read(a.fresh/'manifest.json')
    assert len(truth)==len(manifest)==360
    assert len(set(policy['test_families']))==120 and not set(policy['test_families'])&set(policy['excluded_families'])
    assert len(set(policy['test_pose_ids']))==120
    assert all(byid[r['pose_id']].family_id==r['family_id'] for r in truth)
    ratios=[]
    for r in truth:
        cam=Camera(**r['camera']);xy,valid=project(byid[r['pose_id']].joints,cam);assert valid
        assert np.allclose(normalize_batch(xy),normalize_batch(r['ground_truth_xy']),atol=1e-5)
        if cam.distance is not None:
            from standin_eval.perspective_search import camera_points
            z=camera_points(byid[r['pose_id']].joints,cam)[5:,2]+cam.distance
            assert np.isclose(z.max()/z.min(),r['body_depth_ratio'])
        if r['group']=='strong_perspective':ratios.append(r['body_depth_ratio'])
    assert len(ratios)==120 and all(r>=1.5 for r in ratios)
    for item in manifest:assert sha256(a.fresh/'images'/item['image'])==item['sha256']
    assert sha256(a.fresh/'rankings.json')==read(a.fresh/'ranked.json')['sha256']
    records=read(a.fresh/'rankings.json');points=normalized_joints(np.stack([p.joints for p in poses]))
    cameras=camera_grid()['yaw_pitch_perspective_117']
    features=np.stack([normalize_batch(project(np.stack([p.joints for p in poses]),c)[0]) for c in cameras])
    validity=np.stack([project(np.stack([p.joints for p in poses]),c)[1] for c in cameras])
    replayed=0;failed=0;brute_masks=0
    counts={arm:{g:{'top1':0,'top5':0} for g in ('perspective','strong_perspective','orthographic_control')} for arm in ('prior','unlock')}
    answers={r['image']:r for r in truth}
    for row in records:
        name=row['image'];predpath=a.fresh/'predictions'/(Path(name).stem+'.json')
        assert sha256(predpath)==row['prediction_sha256']
        pred=read(predpath);xy=np.asarray(pred['keypoints2d'],np.float32);mask=np.asarray(pred['scores'])>=.3
        valid=mask[5:].sum()>=6 and mask[[5,6,11,12]].all() and np.isfinite(xy).all()
        if row.get('error'):assert not valid;failed+=1
        else:
            assert valid and mask.tolist()==row['shared_mask']
            assert hashlib.sha256(xy.tobytes()).hexdigest()==row['shared_xy_sha256']
            costpath=a.fresh/'costs'/(Path(name).stem+'.npz');assert sha256(costpath)==row['costs_sha256']
            with np.load(costpath,allow_pickle=False) as d:base,fit,params=[d[k] for k in ('baseline','refined','parameters')]
            q=normalize_batch(xy);alt,am=alternate_labels(q,mask);choices=[]
            for query,m in ((q,mask),(alt,am)):
                selected=np.flatnonzero(m[5:])+5
                raw=np.linalg.norm(features[:,:,selected]-query[selected],axis=-1).mean(-1)
                choices.append(np.where(validity,raw,np.inf).min(0))
            assert np.allclose(base,np.minimum(*choices),atol=2e-5)
            fitted,ok=project_parameters(points,params);assert ok.all()
            fitted_costs=[]
            for query,m in ((q,mask),(alt,am)):
                selected=np.flatnonzero(m[5:])+5
                fitted_costs.append(np.linalg.norm(fitted[:,selected]-query[selected],axis=-1).mean(-1))
            # The optimizer chooses the better independently fitted hypothesis,
            # not necessarily the cheaper opposite label at its chosen camera.
            assert np.all(np.minimum(np.abs(fit-fitted_costs[0]),np.abs(fit-fitted_costs[1]))<3e-5)
            assert np.all(fit<=base+3e-5)
            score=.25*base.astype(float)+.75*fit.astype(float)
            first=int(base.argmin());order=np.argsort(score,kind='stable');order=np.r_[first,order[order!=first]]
            short=order[:20];assert short.tolist()==row['shortlist_indices']
            contour=np.asarray(row['contour_costs']);errors=row['contour_errors']
            if row.get('masks_sha256'):
                path=a.fresh/'masks'/(Path(name).stem+'.npz');assert sha256(path)==row['masks_sha256']
                with np.load(path,allow_pickle=False) as d:
                    for key in d.files:
                        if key=='query':continue
                        k=int(key.split('_')[1])
                        assert np.isclose(brute_contour_cost(d['query'],d[key],row['query_torso']),contour[k],atol=1e-6)
                        brute_masks+=1
            pscore=score[short].copy();pscore[1:]+=contour[1:]
            prior=short if any(k!='0' for k in errors) else np.r_[short[0],short[1:][np.argsort(pscore[1:],kind='stable')]]
            unlock=short[np.argsort(score[short]+(np.zeros(20) if errors else contour),kind='stable')]
            assert prior.tolist()==row['prior_order20'] and unlock.tolist()==row['unlock_order20']
            for arm,order20 in (('prior',prior),('unlock',unlock)):
                assert [poses[j].pose_id for j in order20[:5]]==row[arm]
            replayed+=1
        target=answers[name]['pose_id'];group=answers[name]['group']
        for arm in counts:
            ids=row.get(arm,[]);counts[arm][group]['top1']+=bool(ids and ids[0]==target)
            counts[arm][group]['top5']+=target in ids
        if (replayed+failed)%60==0:print('audited',replayed+failed,'/360',flush=True)
    assert counts==read(a.fresh/'results.json')['counts']
    results=read(a.fresh/'results.json')
    for group,stats in results['top1_pairs'].items():
        matching=[r for r in records if answers[r['image']]['group']==group]
        before=np.array([bool(r.get('prior') and r['prior'][0]==answers[r['image']]['pose_id']) for r in matching])
        after=np.array([bool(r.get('unlock') and r['unlock'][0]==answers[r['image']]['pose_id']) for r in matching])
        gains=int((~before&after).sum());losses=int((before&~after).sum());n=gains+losses
        exact=min(1.,math.ldexp(float(sum(math.comb(n,k) for k in range(min(gains,losses)+1))),1-n)) if n else 1.
        assert gains==stats['gains'] and losses==stats['losses'] and np.isclose(exact,stats['mcnemar_p'],atol=1e-16)
        rng=np.random.default_rng(20261019);delta=after.astype(int)-before.astype(int)
        boot=delta[rng.integers(0,120,(20000,120))].mean(1)
        assert np.allclose(np.quantile(boot,[.025,.975]),stats['ci95'])
    audit={'ranking_sha256':sha256(a.fresh/'rankings.json'),'results_sha256':sha256(a.fresh/'results.json'),
           'replayed':replayed,'failures_retained':failed,'truth_projection_replays':360,'fresh_families':120,
           'gallery_geometry_sha256':geometry_hash(poses),'historical_geometry_sha256':geometry_hash(old),
           'gallery_snapshot_sha256':sha256(a.fresh/'gallery-1995.npz'),
           'historical_snapshot_sha256':sha256(a.fresh/'historical-gallery-1981.npz'),
           'external_change':'15 additions and removal webtoon_combat_47; exact1981 recovered read-only from approved backup',
           'brute_pixel_contour_checks':brute_masks,'paired_statistics_replayed':True,
           'limit':'all346 rankings/camera costs replayed; brute contour checks limited to saved masks of first3 queries; no real-rough accuracy claim'}
    save_json(a.fresh/'audit.json',audit);print(audit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','backup','fresh','library-root'):p.add_argument('--'+n,type=Path,required=True)
    audit(p.parse_args())
