"""Replay four manually annotated requirements using a pinned historical gallery."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json
from scripts.review_perspective_roughs import rgb
from standin_eval.camera_refinement import normalized_joints,project_parameters
from standin_eval.perspective_search import Camera
from standin_eval.manual_pose_cues import describe,satisfies
from standin_eval.contour_letterbox import prepare_letterboxed_query
from standin_eval.contour_rerank import candidate_evidence


def read(p):return json.loads(p.read_text())


def run(a):
    policy=read(a.manual/'protocol.json')
    for path,h in policy['files'].items():assert sha256(Path(path))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy()
    points=normalized_joints(joints);source={r['id']:r for r in read(a.source/'rankings.json')};checked=0;contours=0
    for row in read(a.manual/'rankings.json'):
        if not row['cues']:continue
        name=row['id'];old=source[name]
        with np.load(a.source/'costs'/(name+'.npz'),allow_pickle=False) as d:base,fit,params=[d[k] for k in ('base','fit','parameters')]
        projected,valid=project_parameters(points,params);assert valid.all();eligible=[];features=[]
        for j,p in enumerate(joints):
            yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
            f=describe(p,cam,projected[j]);features.append(f)
            if satisfies(f,row['cues'])[0]:eligible.append(j)
        assert len(eligible)==row['eligible_count']
        scores=.25*base.astype(float)+.75*fit.astype(float);eligible=np.asarray(eligible,int)
        short=eligible[np.argsort(scores[eligible],kind='stable')][:20]
        assert [ids[j] for j in short]==row['shortlist_ids']
        pred=read(a.source/'predictions'/(name+'.json'));q,_=prepare_letterboxed_query(rgb(Path(old['path'])),pred['keypoints2d'])
        costs=np.zeros(len(short));errors={}
        for k,j in enumerate(short):
            try:costs[k]=candidate_evidence(q,joints[j],params[j])[0];contours+=1
            except ValueError as exc:errors[str(k)]=str(exc)
        assert errors==row['contour_errors'] and np.allclose(costs,row['contour_costs'],atol=1e-8)
        order=short[np.argsort(scores[short]+(0 if errors else costs),kind='stable')]
        assert [ids[j] for j in order[:5]]==[c['pose_id'] for c in row['candidates']]
        for c,j in zip(row['candidates'],order[:5]):
            assert c['features']==features[j] and satisfies(features[j],row['cues'])[0]
            for side,(s,e,w) in enumerate(((5,7,9),(6,8,10))):
                u,v=joints[j,s]-joints[j,e],joints[j,w]-joints[j,e]
                expected=np.degrees(np.arctan2(np.linalg.norm(np.cross(u,v)),np.dot(u,v)))
                assert np.isclose(expected,c['features']['elbows_3d_degrees'][side],atol=1e-4)
        checked+=1
    save_json(a.manual/'audit.json',{'ranking_sha256':sha256(a.manual/'rankings.json'),'gallery_sha256':sha256(a.gallery),
        'full_gallery_cue_replays':checked,'successful_contour_replays':contours,'top5_all_requirements_checked':True,
        'accuracy':None,'limit':'geometric predicate replay, NOT independent human verification of folded arms or furniture contact'})
    crop=read(a.crop/'results.json');selected=max(crop['records'],key=lambda r:r['median_body_confidence'])
    assert selected==crop['confidence_selected']
    assert selected['valid_body']==12 and selected['full_searchable']
    save_json(a.recovered/'audit.json',{'ranking_sha256':sha256(a.recovered/'rankings.json'),
        'inherited_manual_audit_sha256':sha256(a.manual/'audit.json'),'crop_confidence_selection_replayed':True,
        'observed_body_joints':12,'visual_limit':'shoulder and hip placement still uncertain on occluded/chibi torso; candidates are not a proven repair'})
    print('manual4 full-gallery filters,',contours,'contour replays,20 candidate3D angles, crop confidence selection PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','manual','crop','recovered','gallery'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
