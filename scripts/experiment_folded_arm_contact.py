"""Development follow-up: distinguish cross-body hands from true arm proximity."""
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
from standin_eval.folded_arm_contact import folded_arm_contact
from standin_eval.manual_pose_cues import describe
from standin_eval.perspective_search import Camera
from standin_eval.camera_refinement import project_parameters,normalized_joints
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    policy={'stage':'reviewed315 development follow-up; not new independent validation',
        'requirement':'user provided folded arms; both wrists cross body midline and approach opposite upper-arm midpoint<=.5torso, height.35..1.1torso',
        'motivation':'previous elbow-only/manual folded heuristic admitted typing poses; contact proximity distinguishes that failure',
        'rank':'same pinned1981/cached XY and fit; eligible whole gallery,.25base+.75fit; fair contour all eligible up to20',
        'code':{n:sha256(ROOT/n) for n in ('scripts/experiment_folded_arm_contact.py','standin_eval/folded_arm_contact.py')},
        'files':{str(p):sha256(p) for p in (a.gallery,a.source/'rankings.json',a.previous/'rankings.json')},
        'limits':['manual annotation','threshold developed on reviewed library, no new holdout','no chest/back determination or finger/contact verification']}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy()
    old=next(r for r in read(a.source/'rankings.json') if r['id']=='rough_315')
    with np.load(a.source/'costs/rough_315.npz',allow_pickle=False) as d:base,fit,params=[d[k] for k in ('base','fit','parameters')]
    projected,valid=project_parameters(normalized_joints(joints),params);assert valid.all()
    descriptions=[folded_arm_contact(p) for p in joints];eligible=np.array([j for j,(yes,f) in enumerate(descriptions) if yes],int)
    scores=.25*base.astype(float)+.75*fit.astype(float);short=eligible[np.argsort(scores[eligible],kind='stable')][:20]
    pred=read(a.source/'predictions/rough_315.json');query,_=prepare_letterboxed_query(rgb(Path(old['path'])),pred['keypoints2d'])
    contour=np.zeros(len(short));errors={}
    for k,j in enumerate(short):
        try:contour[k]=candidate_evidence(query,joints[j],params[j])[0]
        except ValueError as exc:errors[str(k)]=str(exc)
    order=short[np.argsort(scores[short]+(0 if errors else contour),kind='stable')]
    data=copy.deepcopy(read(a.previous/'review-data-private.json'));rows=copy.deepcopy(read(a.previous/'rankings.json'))
    data['study_id']='manual-cues-contact-crop-20261013';data['after_label']='After · 검수 특징 조건 실험'
    item=next(q for q in data['queries'] if q['id']=='rough_315');row=next(q for q in rows if q['id']=='rough_315')
    item['after']=[];row['candidates']=[];row.update(contact_eligible_count=len(eligible),contact_contour_errors=errors)
    for j in order[:5]:
        yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
        im,kind,err=preview(joints[j],cam);canonical,ckind,_=preview(joints[j],Camera(45))
        features=describe(joints[j],cam,projected[j]);features.update(descriptions[j][1]);features['folded_arms']=False
        c={'pose_id':ids[j],'pose_index':int(j),'camera':asdict(cam),'kind':kind,'features':features,'render_error':err}
        row['candidates'].append(c);item['after'].append(dict(c,matched=data_url(im),canonical=data_url(canonical),canonical_kind=ckind))
    item['contour_fallback']=bool(errors)
    item['status_text']='rough_315 · 양손과 반대쪽 팔의 근접 조건 · 전체 자세와 팔짱 검수 필요'
    save_json(a.out/'rankings.json',rows);save_json(a.out/'review-data-private.json',data)
    save_json(a.out/'summary.json',{'queries':10,'searchable':10,'accuracy':None,'manual_annotation':True,
                                  'crop_recovery':True,'rankings_sha256':sha256(a.out/'rankings.json')})
    save_json(a.out/'presentation-policy.json',{'files':{'scripts/templates/perspective_rough_review.html':sha256(ROOT/'scripts/templates/perspective_rough_review.html')}})
    np.savez_compressed(a.out/'contact-costs.npz',eligible=eligible,short=short,contour=contour,order=order)
    print('contact eligible',len(eligible),'top5',[ids[j] for j in order[:5]],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','previous','gallery','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
