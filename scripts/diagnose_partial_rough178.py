"""One reviewed occluded rough: partial observation diagnostic, no GT accuracy."""
import argparse
import fcntl
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.z_axis_search import Pose3D
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index
from standin_eval.camera_refinement import normalized_joints
from standin_eval.partial_similarity import refine_partial


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    sources=(a.gallery,a.crop/'results.json',a.original/'predictions/rough_178.json',a.full/'rankings.json')
    policy={'stage':'rough178 examined development diagnostic, NOT new holdout',
        'hypothesis':'uncertain hidden torso anchors should not determine observed-arm fit',
        'query':'previous confidence-selected crop Human-Art XY reused; no inference rerun, GT or query depth',
        'arms':{'upper6':[5,6,7,8,9,10],'forearm4':[7,8,9,10]},
        'mask':'intersect declared observation subset with same cached confidence>=.3; intentionally different partial masks; original fixed-XY baseline preserved',
        'scoring':'translation/positive-scale least-squares RMS on selected observed joints;195 camera seeds,3 starts,original4 step scales/2 sweeps,dual global labels',
        'ranking':'partial fitted costs only, stable sort exact pose IDs; no contour with uncertain hips',
        'units':'pixels in query; projected gallery torso units removed by fitted scale; RMS normalized by selected query RMS; +cameraZ away',
        'criterion':'mechanical invariance to excluded hip/shoulder/leg perturbation; no quality adoption criterion without source-ID or human GT',
        'limits':['masks are manually declared observation hypotheses, not automatic occlusion inference','fewer joints increase ambiguity of unseen lower body',
                  'shoulders in upper6 may themselves be inaccurate','no source poseID, posture class or depth GT available'],
        'inputs':{str(p):sha256(p) for p in sources},
        'code':{n:sha256(ROOT/n) for n in ('scripts/diagnose_partial_rough178.py','standin_eval/partial_similarity.py')}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:
        poses=[Pose3D(pid,fam,j,s) for pid,fam,j,s in zip(d['pose_ids'],d['family_ids'],d['joints'],d['scores'])]
    assert len(poses)==1981
    pred=read(a.crop/'results.json')['confidence_selected'];xy=np.asarray(pred['keypoints2d'],np.float32);confidence=np.asarray(pred['scores'],np.float32)
    points=normalized_joints(np.stack([p.joints for p in poses]));cams117=camera_grid()['yaw_pitch_perspective_117']
    cams=cams117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)];index=build_camera_index(poses,cams)
    rows={}
    for name,selected in policy['arms'].items():
        mask=np.zeros(17,bool);mask[selected]=True;mask&=confidence>=.3
        assert mask.sum()==len(selected)
        cost,params,coarse,swap=refine_partial(points,index,cams,xy,mask)
        order=np.argsort(cost,kind='stable')
        assert np.all(cost<=coarse+2e-5)
        np.savez_compressed(a.out/(name+'-costs.npz'),cost=cost,parameters=params,coarse=coarse,global_swap=swap,order=order,mask=mask)
        candidates=[]
        for j in order[:20]:
            yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
            candidates.append({'pose_id':str(poses[j].pose_id),'pose_index':int(j),'cost':float(cost[j]),'camera':asdict(cam),'global_swap':bool(swap[j])})
        rows[name]={'mask':mask.tolist(),'top20':candidates,'unique_families_top20':len({poses[j].family_id for j in order[:20]}),
                    'near_best_cost_le_best_plus_1e_3':int((cost<=cost[order[0]]+1e-3).sum()),
                    'xy_sha256':hashlib.sha256(xy.tobytes()).hexdigest(),'costs_sha256':sha256(a.out/(name+'-costs.npz'))}
    full=next(r for r in read(a.full/'rankings.json') if r['id']=='rough_178')
    original=read(a.original/'predictions/rough_178.json');om=np.asarray(original['scores'])>=.3
    result={'arms':rows,'original_valid_body':int(om[5:].sum()),'original_baseline_search_failed':True,
            'crop_full_top1':full['candidates'][0]['pose_id'],'accuracy':None,'inference_rerun':False,'existing_review_changed':False,
            'warning':'no successful178 repair claim; only observed arms are compared, not table contact or occluded whole pose'}
    save_json(a.out/'results.json',result)
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print({name:{'top1':r['top20'][0]['pose_id'],'score':r['top20'][0]['cost'],'near_best':r['near_best_cost_le_best_plus_1e_3']} for name,r in rows.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('gallery','crop','original','full','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
