"""Fit predeclared joint-feature candidates to cached rough315 XY, diagnostic only."""
import argparse
import fcntl
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json
from scripts.review_perspective_roughs import best_views,preview
from standin_eval.z_axis_search import Pose3D
from standin_eval.perspective_search import Camera,camera_grid,build_camera_index
from standin_eval.camera_refinement import normalized_joints,refine_costs
from standin_eval.dual_label_search import alternate_labels
from src.features import normalize_skeleton


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    files=(a.gallery,a.coverage/'protocol.json',a.coverage/'results.json',a.coverage/'extended1995-metrics.json',a.coverage/'audit.json',a.source/'predictions/rough_315.json')
    policy={'stage':'reviewed315 diagnostic subset from previously frozen mechanical coverage; NOT full-gallery rank or new validation',
        'selection':'all extended1995 poses satisfying bilateral-arm-proximity AND deep-crouch predicates; no names/semantic tags/preferred pose IDs used',
        'query':'original Human-Art XY/.3 mask unchanged; no new inference or depth',
        'camera':'dual global labels117+78 starts and original refinement; .25coarse117+.75fit; no contour or lower-body manual edits',
        'limits':['geometric predicates may not mean actual folded arms','whole-pose and facing direction require human verification',
                  'costs compared only within selected candidate set, not a Top1/5 accuracy result','new extension poses may themselves derive from reviewed materials'],
        'inputs':{str(p):sha256(p) for p in files},
        'code':{n:sha256(ROOT/n) for n in ('scripts/fit_joint_requirement_candidates.py','standin_eval/camera_refinement.py','standin_eval/perspective_search.py')}}
    save_json(a.out/'protocol.json',policy)
    selected=[r for r in read(a.coverage/'extended1995-metrics.json') if r['joint_requirements_geometry']]
    assert len(selected)>0
    with np.load(a.gallery,allow_pickle=False) as d:
        ids=d['pose_ids'].tolist();poses=[Pose3D(r['pose_id'],r['family_id'],d['joints'][ids.index(r['pose_id'])],d['scores'][ids.index(r['pose_id'])]) for r in selected]
    pred=read(a.source/'predictions/rough_315.json');xy=np.asarray(pred['keypoints2d'],np.float32);scores=np.asarray(pred['scores'],np.float32);mask=scores>=.3
    assert mask[5:].sum()>=6 and mask[[5,6,11,12]].all()
    q=normalize_skeleton(xy,scores,valid_mask=mask).reshape(17,2);alt,am=alternate_labels(q,mask)
    cams117=camera_grid()['yaw_pitch_perspective_117'];cams=cams117+[Camera(c.yaw,c.pitch,d) for c in camera_grid()['yaw_pitch_39'] for d in (2.,3.)]
    index=build_camera_index(poses,cams);idx117=(index[0][:117],index[1][:117]);direct,_=best_views(idx117,q,mask);other,_=best_views(idx117,alt,am)
    base=np.minimum(direct,other);fit,params=refine_costs(normalized_joints(np.stack([p.joints for p in poses])),index,cams,q,mask)
    mixed=.25*base.astype(float)+.75*fit.astype(float);order=np.argsort(mixed,kind='stable');rows=[]
    sheet=Image.new('RGB',(640,405),'#f5f5f5');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',17)
    draw.text((8,8),'315 · 두 기하 조건 후보 · 정확도/팔짱 의미 미검증',font=font,fill='#202124')
    for k,j in enumerate(order):
        yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
        im,kind,error=preview(poses[j].joints,cam,size=300);sheet.paste(im,(k*320,35))
        draw.text((k*320+5,342),poses[j].pose_id[:30],font=font,fill='#202124')
        rows.append({'pose_id':poses[j].pose_id,'cost':float(mixed[j]),'coarse':float(base[j]),'fit':float(fit[j]),'camera':asdict(cam),'render_kind':kind,'render_error':error})
    sheet.save(a.out/'candidate-contact-sheet.png');np.savez_compressed(a.out/'costs.npz',base=base,fit=fit,parameters=params,order=order,mask=mask,query=q,pose_ids=[p.pose_id for p in poses])
    save_json(a.out/'results.json',{'candidates':rows,'accuracy':None,'source_query_sha256':sha256(a.source/'predictions/rough_315.json'),
                                  'original_top5_and_review_unchanged':True,'no_gt_id_used_in_scoring':True})
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('gallery','coverage','source','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
