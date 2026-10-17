"""Fresh-family validation of free first-rank selection, with fixed fair contours."""
import argparse
import fcntl
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import render_pose,sha256,save_json
from scripts.review_perspective_roughs import rgb,keep_first_order,best_views
from standin_eval.perspective_search import Camera,camera_grid,camera_points,build_camera_index,normalize_batch
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.dual_label_search import alternate_labels
from standin_eval.z_diagnostics import infer_pose
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence,rerank
from standin_eval.joint_shape import reorder_shortlist

SEED=20261107
GROUPS=('perspective','strong_perspective','orthographic_control')
ARMS=('prior','unlock')
CODE=('scripts/confirm_first_rank_release.py','scripts/experiment_z_axis_search.py','scripts/experiment_z_rendered.py',
      'scripts/review_perspective_roughs.py','standin_eval/perspective_search.py','standin_eval/camera_refinement.py',
      'standin_eval/dual_label_search.py','standin_eval/z_diagnostics.py','standin_eval/contour_letterbox.py',
      'standin_eval/contour_rerank.py','standin_eval/joint_shape.py','src/features.py','src/bvh.py')


def read(p):return json.loads(p.read_text())


def verify(a):
    p=read(a.out/'protocol.json')
    for n,h in p['code'].items():assert sha256(ROOT/n)==h,n
    assert sha256(a.model)==p['model_sha256'] and sha256(a.library_root/'poses.db')==p['library_db_sha256']
    for n,h in p['inputs'].items():assert sha256(Path(n))==h,n
    return p


def strong_camera(joints,yaw,pitch):
    p=camera_points(joints,Camera(yaw,pitch));z=p[5:,2]
    return Camera(yaw,pitch,max(float(z.max()-2*z.min()),.65-float(p[:,2].min())))


def ratio(joints,camera):
    if camera.distance is None:return 1.
    depths=camera_points(joints,camera)[5:,2]+camera.distance
    assert depths.min()>.1
    return float(depths.max()/depths.min())


def prepare(a):
    old=read(a.prior/'protocol.json');excluded=set(old['excluded_families'])|set(old['test_families']);assert len(excluded)==1000
    poses=load_library(a.library_root);families={}
    for p in poses:
        if p.family_id not in excluded:families.setdefault(p.family_id,[]).append(p)
    assert len(families)>=120
    rng=np.random.default_rng(SEED);pool=list(families.values())
    groups=[pool[int(i)] for i in rng.choice(len(pool),120,replace=False)]
    chosen=[members[int(rng.integers(len(members)))] for members in groups]
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('images','predictions','records','costs','masks'):(a.out/name).mkdir()
    save_json(a.out/'protocol.json',{'seed':SEED,'stage':'fresh120 families excluded1000 previous; fixed free-first policy only',
        'primary':'strong perspective exact sourceID Top1 unlock minus previous fixed-first contour',
        'gate':'strong Top1delta>=.10, paired95CI lower>0, exactMcNemar p<.05; other2Top1>=prior; all3Top5>=prior-3/120',
        'statistics':'20000 family-paired bootstrap seed20261019; primary strong Top1; no sample replacement or weight selection',
        'selected':'unlock only; broad/tie/focused joint feature policies failed development and are not adopted',
        'fixed':'same Human-Art XY/mask,dual hypotheses,cameras/fit,blend.25/.75; all20 fair contour costs allow former1 to move; tail unchanged',
        'fallback':'prior retains fixed-first blend if any ranks2..20 fail; unlock removes all contours on ANY20 failure but still ranks by blend',
        'camera_groups':'moderate offgrid15/75/135/225,-25/20,d4.5/6.5; strong body depthratio2 with.65joint clearance; old4 ortho',
        'failures':'generation/inference/anchor failures retained in all denominators; no replacing images',
        'limits':['same renderer and closed-set gallery','not actual human rough accuracy','family not sequence/semantic independence',
                  'Top5 tolerance allows up to3 fewer hits per group, report actual regressions'],
        'excluded_families':sorted(excluded),'test_families':[p.family_id for p in chosen],'test_pose_ids':[p.pose_id for p in chosen],
        'inputs':{str(a.prior/'protocol.json'):sha256(a.prior/'protocol.json'),str(a.dev/'results.json'):sha256(a.dev/'results.json')},
        'code':{n:sha256(ROOT/n) for n in CODE},'model_sha256':sha256(a.model),'library_db_sha256':sha256(a.library_root/'poses.db')})
    manifest=[];answers=[]
    for i,p in enumerate(chosen):
        yaw,pitch=(15,75,135,225)[i%4],(-25,20)[(i//4)%2]
        cameras=(Camera(yaw,pitch,(4.5,6.5)[(i//8)%2]),strong_camera(p.joints,yaw,pitch),Camera((0,45,90,180)[i%4]))
        for group,cam in zip(GROUPS,cameras):
            name=f'{group}_{i:03d}.png';entry={'image':name}
            truth={'image':name,'group':group,'pose_id':p.pose_id,'family_id':p.family_id,'camera':asdict(cam),'body_depth_ratio':ratio(p.joints,cam)}
            try:
                im,xy=render_pose(p.joints,cam.yaw,style='rough',seed=SEED+i,pitch=cam.pitch,distance=cam.distance)
                im.save(a.out/'images'/name);entry['sha256']=sha256(a.out/'images'/name);truth['ground_truth_xy']=xy.tolist()
            except ValueError as exc:entry['error']=str(exc)
            manifest.append(entry);answers.append(truth)
        if (i+1)%30==0:print('prepared',i+1,'/120',flush=True)
    strong=sum(r['group']==GROUPS[1] and r['body_depth_ratio']>=1.5 for r in answers)
    save_json(a.out/'manifest.json',manifest);save_json(a.out/'answers.json',answers)
    save_json(a.out/'prepared.json',{'manifest_sha256':sha256(a.out/'manifest.json'),'answers_sha256':sha256(a.out/'answers.json'),'strong_ratio_ge1_5':strong})
    assert strong>=100;verify(a)


def infer(a):
    verify(a);assert not (a.out/'inferred.json').exists()
    from rtmlib import RTMPose
    model=RTMPose(str(a.model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    for entry in read(a.out/'manifest.json'):
        path=a.out/'predictions'/(Path(entry['image']).stem+'.json')
        if path.exists():continue
        if entry.get('error'):prediction={'error':'image generation: '+entry['error']}
        else:
            assert sha256(a.out/'images'/entry['image'])==entry['sha256']
            try:prediction=infer_pose(rgb(a.out/'images'/entry['image']),model,crop=True,turns=(0,1,2,3),is_3d=False)
            except Exception as exc:prediction={'error':type(exc).__name__+': '+str(exc)}
        save_json(path,prediction)
    paths=sorted((a.out/'predictions').glob('*.json'));assert len(paths)==360
    save_json(a.out/'inferred.json',{p.name:sha256(p) for p in paths});print('inferred360',flush=True)


def rank(a):
    verify(a);assert not (a.out/'rankings.json').exists()
    inferred=read(a.out/'inferred.json');assert all(sha256(a.out/'predictions'/n)==h for n,h in inferred.items())
    poses=load_library(a.library_root); points=normalized_joints(np.stack([p.joints for p in poses]))
    cams117=camera_grid()['yaw_pitch_perspective_117'];cams=cams117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index=build_camera_index(poses,cams);index117=(index[0][:117],index[1][:117]);rows=[]
    for entry in read(a.out/'manifest.json'):
        name=entry['image'];path=a.out/'predictions'/(Path(name).stem+'.json');pred=read(path)
        row={'image':name,'prediction_sha256':sha256(path)}
        if pred.get('error'):row['error']=pred['error'];rows.append(row);continue
        xy=np.asarray(pred['keypoints2d'],np.float32);mask=np.asarray(pred['scores'],np.float32)>=.3
        if mask[5:].sum()<6 or not mask[[5,6,11,12]].all() or not np.isfinite(xy).all():
            row['error']='insufficient Human-Art body/anchors';rows.append(row);continue
        q=normalize_batch(xy);alt,am=alternate_labels(q,mask)
        direct,_=best_views(index117,q,mask);other,_=best_views(index117,alt,am);base=np.minimum(direct,other)
        fit,params=refine_costs(points,index,cams,q,mask);score=.25*base.astype(float)+.75*fit.astype(float)
        order=keep_first_order(base,score);short=order[:20];penalties=np.zeros(20);errors={};masks={}
        query,_=prepare_letterboxed_query(rgb(a.out/'images'/name),xy)
        if len(rows)<3:masks['query']=query['mask']
        for k,j in enumerate(short):
            try:
                penalties[k],_,pixels=candidate_evidence(query,poses[j].joints,params[j])
                if len(rows)<3:masks['candidate_'+str(k)]=pixels
            except ValueError as exc:errors[str(k)]=str(exc)
        previous_errors={k:v for k,v in errors.items() if k!='0'}
        previous_penalties=penalties.copy();previous_penalties[0]=0
        old=order if previous_errors else rerank(order,score,previous_penalties,1.)
        new=reorder_shortlist(order,score,np.zeros(20) if errors else penalties,1.)
        assert np.array_equal(old[20:],order[20:]) and np.array_equal(new[20:],order[20:])
        row.update(shared_xy_sha256=hashlib.sha256(xy.tobytes()).hexdigest(),shared_mask=mask.tolist(),
                   shortlist_indices=short.tolist(),contour_costs=penalties.tolist(),contour_errors=errors,
                   query_torso=query['torso'],prior=[poses[j].pose_id for j in old[:5]],unlock=[poses[j].pose_id for j in new[:5]],
                   prior_order20=old[:20].tolist(),unlock_order20=new[:20].tolist())
        costs=a.out/'costs'/(Path(name).stem+'.npz');np.savez_compressed(costs,baseline=base,refined=fit,parameters=params);row['costs_sha256']=sha256(costs)
        if masks:
            path=a.out/'masks'/(Path(name).stem+'.npz');np.savez_compressed(path,**masks);row['masks_sha256']=sha256(path)
        rows.append(row);save_json(a.out/'records'/(Path(name).stem+'.json'),row)
        if len(rows)%30==0:print('ranked',len(rows),'/360',flush=True)
    assert len(rows)==360;save_json(a.out/'rankings.json',rows);save_json(a.out/'ranked.json',{'sha256':sha256(a.out/'rankings.json'),'answers_parsed':False});verify(a)


def paired(before,after):
    b=np.asarray(before,bool);a=np.asarray(after,bool);delta=a.astype(int)-b.astype(int)
    rng=np.random.default_rng(20261019);boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    gains=int((~b&a).sum());losses=int((b&~a).sum());discordant=gains+losses
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.
    return {'delta':float(delta.mean()),'gains':gains,'losses':losses,'ci95':np.quantile(boot,[.025,.975]).tolist(),'mcnemar_p':p}


def evaluate(a):
    verify(a);assert not (a.out/'results.json').exists()
    assert read(a.out/'ranked.json')['sha256']==sha256(a.out/'rankings.json')
    rows=read(a.out/'rankings.json');answers={r['image']:r for r in read(a.out/'answers.json')};counts={arm:{g:dict(top1=0,top5=0) for g in GROUPS} for arm in ARMS};pairs={g:[] for g in GROUPS}
    for r in rows:
        t=answers[r['image']];g,target=t['group'],t['pose_id'];hits=[]
        for arm in ARMS:
            ids=r.get(arm,[]);one=bool(ids and ids[0]==target);counts[arm][g]['top1']+=one;counts[arm][g]['top5']+=target in ids;hits.append(one)
        pairs[g].append(hits)
    stats={g:paired(np.asarray(pairs[g])[:,0],np.asarray(pairs[g])[:,1]) for g in GROUPS};s=stats[GROUPS[1]]
    gate=(s['delta']>=.10 and s['ci95'][0]>0 and s['mcnemar_p']<.05
          and all(counts['unlock'][g]['top1']>=counts['prior'][g]['top1'] for g in (GROUPS[0],GROUPS[2]))
          and all(counts['unlock'][g]['top5']>=counts['prior'][g]['top5']-3 for g in GROUPS))
    result={'counts':counts,'top1_pairs':stats,'gate_passed':gate,'failures':{g:sum(bool(r.get('error')) for r in rows if answers[r['image']]['group']==g) for g in GROUPS}}
    save_json(a.out/'results.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','infer','rank','evaluate'))
    for n in ('prior','dev','model','library-root','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/'.first-rank-confirm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[a.stage](a)
