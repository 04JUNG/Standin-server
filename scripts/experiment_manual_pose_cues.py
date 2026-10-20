"""Development-only human-annotated pose requirements on a pinned1981 gallery."""
import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.review_perspective_roughs import rgb,preview,data_url
from standin_eval.camera_refinement import project_parameters,normalized_joints
from standin_eval.perspective_search import Camera
from standin_eval.manual_pose_cues import describe,satisfies
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence

CUES={
    'rough_315':{'folded_arms':True},
    'rough_288':{'left_arm_nearer_min':.1},
    'rough_301':{'elbow_degrees':{'right':[155,180]},'left_hand_hip_max':.6},
    'rough_291':{'elbow_degrees':{'left':[70,110],'right':[70,110]},'projected_leg_crossing':True},
}


def read(p):return json.loads(p.read_text())


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    # Freeze user-visible requirements, tolerances and source hashes before ranks.
    policy={'scope':'development manual feature annotation; NOT automatic recognition or accuracy validation',
            'sides':'character anatomical left/right, confirmed directly by user',
            'cues':CUES,'query178':'retain extraction failure, no threshold lowering or invented missing joints',
            'query288_forward':'measure shoulders relative hips in camera depth; ambiguous forward posture not imposed',
            'rank':'eligible poses across whole pinned gallery; .25base+.75fit; fair contour top20; any render failure disables all contour',
            'original_gallery':'1981 exact recovered geometry, no newly added15 poses or removed-pose replacement',
            'limitations':['torso-derived front','forearm/wrist geometry not detailed hand contact','bench contact not modeled',
                           'manual features change input information; not a fair unassisted-search accuracy claim'],
            'files':{str(p):sha256(p) for p in (a.gallery,a.source/'rankings.json',a.previous/'rankings.json')},
            'code':{n:sha256(ROOT/n) for n in ('scripts/experiment_manual_pose_cues.py','standin_eval/manual_pose_cues.py')}}
    save_json(a.out/'protocol.json',policy)
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy()
    points=normalized_joints(joints);oldrows=read(a.source/'rankings.json');previous=read(a.previous/'rankings.json')
    data=read(a.previous/'joint_shape/review-data-private.json');data=copy.deepcopy(data)
    data['study_id']='manual-pose-cues-20261013';data['before_label']='Before · 지난 팔 각도·다리 교차'
    data['after_label']='After · 검수 특징 조건 검색';data['summary_label']='검수자가 준 특징 사용 · 자동 인식 아님 · 실제 3D 팔 각도와 앞뒤 관계 · 벤치 접촉은 미지원'
    rows=[]
    for old,prev,item in zip(oldrows,previous,data['queries']):
        assert old['id']==prev['id']==item['id'];name=old['id'];item['before']=item['after']
        row={'id':name,'cues':CUES.get(name,{})}
        if old.get('error'):row['error']=old['error'];item['after']=[];rows.append(row);continue
        pred=read(a.source/'predictions'/(name+'.json'));xy=np.asarray(pred['keypoints2d'],np.float32)
        with np.load(a.source/'costs'/(name+'.npz'),allow_pickle=False) as d:base,fit,params=[d[k] for k in ('base','fit','parameters')]
        assert len(ids)==len(base)
        projected,valid=project_parameters(points,params);assert valid.all()
        features=[];eligible=[];failcounts={}
        for j,p in enumerate(joints):
            yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
            f=describe(p,cam,projected[j]);features.append(f)
            ok,fail=satisfies(f,row['cues'])
            if ok:eligible.append(j)
            for reason in fail:failcounts[reason]=failcounts.get(reason,0)+1
        row.update(eligible_count=len(eligible),failed_requirements=failcounts)
        if not row['cues']:
            order=[ids.index(pid) for pid in prev['top5']['joint_shape']];errors={}
        elif not eligible:
            row['error']='라이브러리에 검수 조건을 모두 만족하는 후보 없음';item['error']=row['error'];item['after']=[];rows.append(row);continue
        else:
            scores=.25*base.astype(float)+.75*fit.astype(float);eligible=np.asarray(eligible,int)
            short=eligible[np.argsort(scores[eligible],kind='stable')][:20];contour=np.zeros(len(short));errors={}
            query,_=prepare_letterboxed_query(rgb(Path(old['path'])),xy)
            for k,j in enumerate(short):
                try:contour[k]=candidate_evidence(query,joints[j],params[j])[0]
                except ValueError as exc:errors[str(k)]=str(exc)
            order=short[np.argsort(scores[short]+(0 if errors else contour),kind='stable')]
            row['shortlist_ids']=[ids[j] for j in short];row['contour_costs']=contour.tolist()
        row['contour_errors']=errors;row['candidates']=[];item['after']=[];item['contour_fallback']=bool(errors)
        for j in order[:5]:
            yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
            im,kind,err=preview(joints[j],cam);canonical,ckind,_=preview(joints[j],Camera(45))
            record={'pose_id':ids[j],'pose_index':int(j),'camera':asdict(cam),'features':features[j],'kind':kind,'render_error':err}
            if row['cues']:assert satisfies(features[j],row['cues'])[0]
            row['candidates'].append(record);item['after'].append(dict(record,matched=data_url(im),canonical=data_url(canonical),canonical_kind=ckind))
        rows.append(row);print(name,'eligible',row['eligible_count'],'top1',row['candidates'][0]['pose_id'],flush=True)
    save_json(a.out/'rankings.json',rows);save_json(a.out/'review-data-private.json',data)
    save_json(a.out/'summary.json',{'queries':10,'searchable':sum(not r.get('error') for r in rows),'accuracy':None,
                                  'manual_annotation':True,'rankings_sha256':sha256(a.out/'rankings.json')})
    # Presentation adapter only; scoring protocol remains a separate frozen file.
    save_json(a.out/'presentation-policy.json',{'files':{'scripts/templates/perspective_rough_review.html':sha256(ROOT/'scripts/templates/perspective_rough_review.html')}})
    for path,h in policy['files'].items():assert sha256(Path(path))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','previous','gallery','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
