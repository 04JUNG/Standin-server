"""Separate image-input recovery arm; never changes the original failure record."""
import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.review_perspective_roughs import rgb,preview,data_url,best_views
from standin_eval.z_axis_search import Pose3D
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index,normalize_batch
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.dual_label_search import alternate_labels
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    policy={'stage':'development input-crop recovery, NOT fixed-XY baseline experiment',
        'selection':'use previously prespecified crop sweep winner by median model confidence, no ranking/GT selection',
        'search':'same body.3/6+anchors gate; dual labels117+78 fit,.25/.75; free first; fair contour top20',
        'files':{str(p):sha256(p) for p in (a.gallery,a.crop/'results.json',a.manual/'rankings.json')},
        'code_sha256':sha256(Path(__file__)),
        'limits':['crop sweep explored36 hypotheses on one reviewed image','confidence not correctness','not human validated',
                  'manual feature variants on other4 images are inherited, not automatic recognition']}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:
        poses=[Pose3D(pid,fam,j,s) for pid,fam,j,s in zip(d['pose_ids'],d['family_ids'],d['joints'],d['scores'])]
    prediction=read(a.crop/'results.json')['confidence_selected']
    xy=np.asarray(prediction['keypoints2d'],np.float32);mask=np.asarray(prediction['scores'])>=.3
    assert mask[5:].sum()>=6 and mask[[5,6,11,12]].all()
    case=next(r for r in read(a.source/'protocol.json')['cases'] if r['id']=='rough_178');image=rgb(Path(case['path']))
    q=normalize_batch(xy);alt,am=alternate_labels(q,mask);cams117=camera_grid()['yaw_pitch_perspective_117']
    cams=cams117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index=build_camera_index(poses,cams);idx117=(index[0][:117],index[1][:117])
    direct,_=best_views(idx117,q,mask);swapped,_=best_views(idx117,alt,am);base=np.minimum(direct,swapped)
    fit,params=refine_costs(normalized_joints(np.stack([p.joints for p in poses])),index,cams,q,mask)
    score=.25*base.astype(float)+.75*fit.astype(float);short=np.argsort(score,kind='stable')[:20]
    contour=np.zeros(20);errors={};query,_=prepare_letterboxed_query(image,xy)
    for k,j in enumerate(short):
        try:contour[k]=candidate_evidence(query,poses[j].joints,params[j])[0]
        except ValueError as exc:errors[str(k)]=str(exc)
    order=short[np.argsort(score[short]+(0 if errors else contour),kind='stable')]
    np.savez_compressed(a.out/'costs.npz',base=base,fit=fit,parameters=params,short=short,contour=contour,order=order)
    data=copy.deepcopy(read(a.manual/'review-data-private.json'));rows=copy.deepcopy(read(a.manual/'rankings.json'))
    data['study_id']='manual-cues-with-crop-recovery-20261013'
    data['summary_label']='178만 인물 영역 재추론 · 나머지는 동일 관절 · 검수 특징은 수동 입력 · 3D 팔 각도 표시 · 벤치 접촉 미지원'
    item=next(q for q in data['queries'] if q['id']=='rough_178');row=next(r for r in rows if r['id']=='rough_178')
    item.pop('error',None);row.pop('error',None)
    item['status_text']='rough_178 · Before 추출 실패 / After 인물 영역 재추론 · 관절과 후보 검수 필요'
    item['after']=[];row['candidates']=[];row['crop_prediction']=prediction
    row['contour_errors']=errors;item['contour_fallback']=bool(errors)
    for j in order[:5]:
        yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
        im,kind,err=preview(poses[j].joints,cam);canonical,ckind,_=preview(poses[j].joints,Camera(45))
        c={'pose_id':str(poses[j].pose_id),'pose_index':int(j),'camera':asdict(cam),'kind':kind,'render_error':err}
        row['candidates'].append(c);item['after'].append(dict(c,matched=data_url(im),canonical=data_url(canonical),canonical_kind=ckind))
    overlay=Image.fromarray(image);draw=ImageDraw.Draw(overlay);width=max(2,int(max(image.shape[:2])/200))
    for s,e in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)):
        if mask[s] and mask[e]:draw.line([tuple(xy[s]),tuple(xy[e])],fill='#008aa6',width=width)
    for j,p in enumerate(xy):
        if mask[j]:draw.ellipse((p[0]-width,p[1]-width,p[0]+width,p[1]+width),fill='#df8034')
    overlay.save(a.out/'rough178-overlay.png');item['overlay']=data_url(overlay,480)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'review-data-private.json',data)
    save_json(a.out/'summary.json',{'queries':10,'searchable':10,'accuracy':None,'manual_annotation':True,
                                  'crop_recovery':True,'rankings_sha256':sha256(a.out/'rankings.json')})
    save_json(a.out/'presentation-policy.json',{'files':{'scripts/templates/perspective_rough_review.html':sha256(ROOT/'scripts/templates/perspective_rough_review.html')}})
    print('178 recovered12 observed body joints, top1',row['candidates'][0]['pose_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','manual','crop','gallery','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
