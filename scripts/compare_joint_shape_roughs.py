"""Apply fixed candidate cues to reviewed roughs without interpreting them as GT."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.review_perspective_roughs import rgb,preview,data_url
from scripts.experiment_joint_shape import read
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import sha256,save_json
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence
from standin_eval.joint_shape import shape_penalties,reorder_shortlist
from standin_eval.joint_focus import focused_costs
from standin_eval.joint_ties import reorder_near_ties
from src.features import normalize_skeleton
from standin_eval.perspective_search import Camera
from dataclasses import asdict


def rank(a):
    assert read(a.source/'audit.json')['ranking_sha256']==sha256(a.source/'rankings.json')
    # Cached indices belong to the exact source gallery, not just its count.
    # This preflight was added after the historical study; at-run source and
    # its original hash are preserved beside those immutable local outputs.
    expected=read(a.source/'protocol.json')['library_db_sha256']
    if sha256(a.library_root/'poses.db')!=expected:
        raise ValueError('gallery version differs from cached source; use the verified historical snapshot workflow')
    a.out.mkdir(parents=True,exist_ok=False)
    policy={'stage':'reviewed10 real roughs are development feedback, not source3D validation',
        'weights':.15,'margin':.025,'variants':['unlock','joint_shape','focused'],
        'fixed':'same observed XY/mask and cached camera costs/parameters; same20 shortlist; no human poseIDs in scoring',
        'inputs':{str(a.source/p):sha256(a.source/p) for p in ('protocol.json','rankings.json','review-data-private.json')},
        'code':{n:sha256(ROOT/n) for n in ('scripts/compare_joint_shape_roughs.py','standin_eval/joint_shape.py','standin_eval/joint_focus.py','standin_eval/joint_ties.py')}}
    save_json(a.out/'protocol.json',policy)
    poses=load_library(a.library_root);points=normalized_joints(np.stack([p.joints for p in poses]));rows=[]
    for old in read(a.source/'rankings.json'):
        row={'id':old['id'],'old_after':[c['pose_id'] for c in old.get('after',[])]}
        if old.get('error'):row['error']=old['error'];rows.append(row);continue
        pred=read(a.source/'predictions'/(old['id']+'.json'))
        xy=np.asarray(pred['keypoints2d'],np.float32);confidence=np.asarray(pred['scores'],np.float32);mask=confidence>=.3
        q=normalize_skeleton(xy,confidence,valid_mask=mask).reshape(17,2)
        assert hashlib.sha256(q.tobytes()).hexdigest()==old['shared_xy_sha256']
        path=a.source/'costs'/(old['id']+'.npz');assert sha256(path)==old['costs_sha256']
        with np.load(path,allow_pickle=False) as d:base,fit,params,blend=[d[k] for k in ('base','fit','parameters','blend_order')]
        score=.25*base.astype(float)+.75*fit.astype(float);short=blend[:20]
        projected,valid=project_parameters(points[short],params[short]);assert valid.all()
        angle,cross,meta=shape_penalties(q,mask,projected);focused_angle,focused_cross,focused_meta=focused_costs(meta,q,confidence)
        contour=np.zeros(20);errors={}
        try:
            query,_=prepare_letterboxed_query(rgb(Path(old['path'])),xy)
            for k,j in enumerate(short):
                try:contour[k]=candidate_evidence(query,poses[j].joints,params[j])[0]
                except ValueError as exc:errors[str(k)]=str(exc)
        except ValueError as exc:errors['query']=str(exc)
        used=np.zeros(20) if errors else contour
        unlocked=reorder_shortlist(blend,score,used,1.)
        full=reorder_shortlist(blend,score,used+.15*(angle+cross),1.)
        fair=score.copy();fair[short]+=used;focused=np.zeros(len(poses));focused[short]=.15*(focused_angle+focused_cross)
        focused_order=reorder_near_ties(unlocked,fair,focused)
        orders={'unlock':unlocked,'joint_shape':full,'focused':focused_order}
        row.update(metadata=meta,focused_metadata=focused_meta,contour_errors=errors,
                   orders20={v:order[:20].tolist() for v,order in orders.items()},
                   top5={v:[poses[j].pose_id for j in order[:5]] for v,order in orders.items()},
                   top20_ids={v:[poses[j].pose_id for j in order[:20]] for v,order in orders.items()})
        row['candidates']={}
        for j in set(j for order in orders.values() for j in order[:5]):
            yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
            im,kind,error=preview(poses[j].joints,cam);canonical,ckind,_=preview(poses[j].joints,Camera(45))
            row['candidates'][poses[j].pose_id]={'pose_id':poses[j].pose_id,'pose_index':int(j),'camera':asdict(cam),'kind':kind,
                 'render_error':error,'canonical_kind':ckind,'matched':data_url(im),'canonical':data_url(canonical)}
        rows.append(row);print('shape rough',row['id'],row['top5']['unlock'][0],row['top5']['joint_shape'][0],flush=True)
    save_json(a.out/'rankings.json',rows)
    # Human preference IDs enter AFTER ranks are saved, as development diagnostics.
    comments=read(a.source/'human-feedback.json')['records'];diagnostics=[]
    for c in comments:
        row=next(r for r in rows if r['id']==c['query_id'])
        diagnostics.append({'id':c['query_id'],'preferred_id':c['pose_id'],'old_rank':c['rank'],
            'new_ranks':{v:(row['top20_ids'][v].index(c['pose_id'])+1 if c['pose_id'] in row['top20_ids'][v] else None) for v in ('unlock','joint_shape','focused')}})
    summary={'queries':10,'searchable':sum(not r.get('error') for r in rows),'accuracy':None,
             'human_preference_development_diagnostics':diagnostics,'rankings_sha256':sha256(a.out/'rankings.json')}
    save_json(a.out/'summary.json',summary);print(summary,flush=True)
    olddata=read(a.source/'review-data-private.json')
    for variant in ('unlock','joint_shape','focused'):
        folder=a.out/variant;folder.mkdir()
        data=copy.deepcopy(olddata);data['study_id']='rough-joint-shape-'+variant+'-20261013'
        data['before_label']='Before · 지난 개선 검색';data['after_label']='After · '+{'unlock':'1위 고정 해제','joint_shape':'팔 각도·다리 교차 실험','focused':'비슷한 후보의 특징 비교'}[variant]
        data['summary_label']='동일 관절·카메라 · 이전 검색과 새 정렬 비교 · 특징 점수는 개발 실험 · 실제 정확도 미확인'
        for item,row in zip(data['queries'],rows):
            item['before']=item['after']
            if row.get('error'):item['after']=[];continue
            item['after']=[row['candidates'][pose] for pose in row['top5'][variant]]
            item['contour_fallback']=bool(row['contour_errors'])
        save_json(folder/'review-data-private.json',data)
        save_json(folder/'rankings.json',rows)
        save_json(folder/'summary.json',dict(summary,rankings_sha256=sha256(folder/'rankings.json')))
        save_json(folder/'protocol.json',{'files':{'scripts/templates/perspective_rough_review.html':sha256(ROOT/'scripts/templates/perspective_rough_review.html')}})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','library-root','out'):p.add_argument('--'+n,type=Path,required=True)
    rank(p.parse_args())
