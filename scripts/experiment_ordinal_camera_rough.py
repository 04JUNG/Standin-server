"""Manually constrained camera fitting for reviewed rough288, development only."""
import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.review_perspective_roughs import rgb,preview,data_url
from standin_eval.z_axis_search import Pose3D
from standin_eval.ordinal_camera_fit import refine_ordinal,feasible
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,normalize_batch
from standin_eval.manual_pose_cues import describe
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    save_json(a.out/'protocol.json',{'stage':'rough288 reviewed development only; fixed user ordinal annotation',
        'user_clarification':'anatomical left/right; forward means bending body forward, not facing camera',
        'constraints':'camera-depth mean(right elbow/wrist)-mean(left elbow/wrist)>=.1torso; hips-shoulders>=.1torso',
        'limit':'latter is only a camera-relative lean proxy, not intrinsic anatomical forward bend; user validation required',
        'scoring':'same Human-Art XY/mask; feasible117 and195-start fit,.25/.75; fair contour top20; no source/preferred pose IDs',
        'files':{str(p):sha256(p) for p in (a.gallery,a.source/'rankings.json',a.previous/'rankings.json')},
        'code':{n:sha256(ROOT/n) for n in ('scripts/experiment_ordinal_camera_rough.py','standin_eval/ordinal_camera_fit.py')}})
    with np.load(a.gallery,allow_pickle=False) as d:
        poses=[Pose3D(pid,fam,j,s) for pid,fam,j,s in zip(d['pose_ids'],d['family_ids'],d['joints'],d['scores'])]
    pred=read(a.source/'predictions/rough_288.json');xy=np.asarray(pred['keypoints2d'],np.float32);mask=np.asarray(pred['scores'])>=.3
    assert mask[5:].sum()>=6 and mask[[5,6,11,12]].all()
    q=normalize_batch(xy);points=normalized_joints(np.stack([p.joints for p in poses]));cams117=camera_grid()['yaw_pitch_perspective_117']
    cams=cams117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index=build_camera_index(poses,cams)
    base,_=refine_ordinal(points,(index[0][:117],index[1][:117]),cams117,q,mask,steps=())
    fit,params=refine_ordinal(points,index,cams,q,mask);score=.25*base.astype(float)+.75*fit.astype(float)
    eligible=np.flatnonzero(np.isfinite(score));assert len(eligible)>=20
    short=eligible[np.argsort(score[eligible],kind='stable')][:20]
    old=next(r for r in read(a.source/'rankings.json') if r['id']=='rough_288');query,_=prepare_letterboxed_query(rgb(Path(old['path'])),xy)
    contour=np.zeros(20);errors={}
    for k,j in enumerate(short):
        try:contour[k]=candidate_evidence(query,poses[j].joints,params[j])[0]
        except ValueError as exc:errors[str(k)]=str(exc)
    order=short[np.argsort(score[short]+(0 if errors else contour),kind='stable')]
    projected,valid=project_parameters(points,params)
    data=copy.deepcopy(read(a.previous/'review-data-private.json'));rows=copy.deepcopy(read(a.previous/'rankings.json'))
    data['study_id']='manual-ordinal-camera-cues-20261014'
    item=next(q for q in data['queries'] if q['id']=='rough_288');row=next(q for q in rows if q['id']=='rough_288')
    item['after']=[];row['candidates']=[];row.update(ordinal_eligible_count=len(eligible),ordinal_contour_errors=errors)
    for j in order[:5]:
        assert valid[j] and feasible(points[j:j+1],params[j:j+1])[0]
        yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
        im,kind,err=preview(poses[j].joints,cam);canonical,ckind,_=preview(poses[j].joints,Camera(45))
        features=describe(poses[j].joints,cam,projected[j])
        assert features['left_arm_nearer_than_right']>=.1-1e-6 and features['shoulders_nearer_than_hips']>=.1-1e-6
        c={'pose_id':str(poses[j].pose_id),'pose_index':int(j),'camera':asdict(cam),'kind':kind,'features':features,'render_error':err}
        row['candidates'].append(c);item['after'].append(dict(c,matched=data_url(im),canonical=data_url(canonical),canonical_kind=ckind))
    item['contour_fallback']=bool(errors)
    item['status_text']='rough_288 · 팔 앞뒤 조건을 지키며 카메라 재탐색 · 몸 숙임은 시점 기준 근사, 검수 필요'
    data['summary_label']='검수 특징 수동 입력 · 288은 조건부 카메라 재탐색 · 178만 영역 재추론 · 머리 방향과 벤치 접촉 미지원'
    np.savez_compressed(a.out/'ordinal-costs.npz',base=base,fit=fit,parameters=params,short=short,contour=contour,order=order)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'review-data-private.json',data)
    save_json(a.out/'summary.json',{'queries':10,'searchable':10,'accuracy':None,'manual_annotation':True,
                                  'rankings_sha256':sha256(a.out/'rankings.json')})
    save_json(a.out/'presentation-policy.json',{'files':{'scripts/templates/perspective_rough_review.html':sha256(ROOT/'scripts/templates/perspective_rough_review.html')}})
    print('ordinal camera eligible',len(eligible),'top5',[str(poses[j].pose_id) for j in order[:5]],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','previous','gallery','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
