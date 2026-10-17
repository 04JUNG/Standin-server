"""Separate cached image-plane evidence from manually requested3D semantics."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from src.features import normalize_skeleton
from standin_eval.joint_shape import elbow_angles,leg_crossing
from standin_eval.dual_label_search import LEFT_RIGHT
from standin_eval.perspective_search import Camera,project,normalize_batch
from standin_eval.body_part_evidence import body_part_errors

NAMES=('rough_315','rough_301','rough_291')


def observations(q,mask):
    p=np.asarray(q,float);m=np.asarray(mask,bool)
    result={'elbows_2d_degrees':elbow_angles(p,m),'projected_leg_crossing':leg_crossing(p,m)}
    anchors=[5,6,11,12]
    if not m[anchors].all():return {**result,'torso_frame_available':False}
    hip=p[[11,12]].mean(0);up=p[[5,6]].mean(0)-hip;length=np.linalg.norm(up)
    if length<1e-8:return {**result,'torso_frame_available':False}
    up/=length;left=p[5]-p[6];left-=left.dot(up)*up
    if np.linalg.norm(left)<1e-8:return {**result,'torso_frame_available':False}
    left/=np.linalg.norm(left)
    result.update(torso_frame_available=True,wrists_height_2d_torso_units=[float((p[j]-hip).dot(up)/length) if m[j] else None for j in (9,10)],
                  wrists_side_2d_torso_units=[float((p[j]-hip).dot(left)/length) if m[j] else None for j in (9,10)])
    distances=[]
    for wrist,opposite in ((9,[6,8]),(10,[5,7])):
        distances.append(float(np.linalg.norm(p[wrist]-p[opposite].mean(0))/length) if m[[wrist,*opposite]].all() else None)
    result['wrist_opposite_upperarm_2d_torso_distances']=distances
    return result


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    paths=(a.gallery,a.source/'rankings.json',a.manual/'rankings.json',
           *[a.source/'predictions'/(n+'.json') for n in NAMES],*[a.source/'costs'/(n+'.npz') for n in NAMES])
    hashes={str(p):sha256(p) for p in paths}
    code=('scripts/diagnose_cached_arm_observations.py','standin_eval/body_part_evidence.py','standin_eval/joint_shape.py')
    policy={'purpose':'post-review image-plane observation diagnosis; not new search, correction, GT or validation',
            'queries':list(NAMES),'predictions':'original cached Human-Art XY/.3mask, no coordinate or threshold changes',
            'measurements':'2D elbows/leg crossing with existing .05 bone-length guard; normalized wrist heights/sides/opposite-upperarm distances',
            'comparison':'cached unconstrained blend first versus existing manual contour first; one global label alignment, then same mapping for every point',
            'limits':['2D angle not3D angle','projected proximity/crossing cannot prove arm contact or depth',
                      'Human-Art left/right not verified character-side annotation','deviation from model XY not human relevance error',
                      'no inferred semantic contradiction or accuracy from these measurements alone'],
            'inputs':hashes,'code':{n:sha256(ROOT/n) for n in code}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy()
    source={r['id']:r for r in read(a.source/'rankings.json')};manual={r['id']:r for r in read(a.manual/'rankings.json')};records=[]
    for name in NAMES:
        pred=read(a.source/'predictions'/(name+'.json'));xy=np.asarray(pred['keypoints2d'],np.float32);scores=np.asarray(pred['scores'],np.float32);mask=scores>=.3
        q=normalize_skeleton(xy,scores,valid_mask=mask).reshape(17,2)
        assert hashlib.sha256(q.tobytes()).hexdigest()==source[name]['shared_xy_sha256']
        query=observations(q,mask)
        # Independent atan2(cross,dot) check instead of existing arccos.
        for side,(s,e,w) in enumerate(((5,7,9),(6,8,10))):
            angle=query['elbows_2d_degrees'][side]
            if angle is not None:
                u,v=q[s].astype(float)-q[e],q[w].astype(float)-q[e]
                alternate=float(np.degrees(np.arctan2(abs(u[0]*v[1]-u[1]*v[0]),u.dot(v))))
                assert abs(alternate-angle)<1e-5
        with np.load(a.source/'costs'/(name+'.npz'),allow_pickle=False) as d:blend=.25*d['base'].astype(float)+.75*d['fit'].astype(float);params=d['parameters'].copy()
        first=int(np.argmin(blend));yaw,pitch,inv=map(float,params[first]);original_cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
        new=manual[name]['candidates'][0];views=[]
        for kind,index,cam in (('unconstrained_xy_first',first,original_cam),('manual_current_first',ids.index(new['pose_id']),Camera(**new['camera']))):
            projected,valid=project(joints[index],cam);assert valid
            c=normalize_batch(projected);part=body_part_errors(q,mask,c);aligned=c[LEFT_RIGHT] if part['global_swap'] else c
            errors=np.linalg.norm(aligned.astype(float)-q,axis=1);selected=np.flatnonzero(mask[5:])+5
            assert abs(errors[selected].mean()-part['body'])<1e-8
            views.append({'type':kind,'pose_id':ids[index],'global_label_swap':part['global_swap'],
                          'query_aligned_projected_observations':observations(aligned,mask),
                          'per_joint_xy_deviation_torso_units':[float(errors[j]) if mask[j] else None for j in range(17)],'part_deviations':part})
        records.append({'query':name,'cached_xy_observations':query,'mask':mask.tolist(),'confidence_scores':scores.tolist(),'candidates':views})
    save_json(a.out/'records.json',records)
    save_json(a.out/'results.json',{'queries':len(records),'accuracy':None,'original_xy_and_rankings_unchanged':True,
                                  'next':'inspect source artwork and cached joints before treating projected-feature mismatch as3D pose error; verified keypoint labels needed for quantitative correction'})
    for p,h in hashes.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print([{r['query']:r['cached_xy_observations']} for r in records],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('source','manual','gallery','out'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
