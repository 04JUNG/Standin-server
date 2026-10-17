"""Search camera → API → real refine solver, projection and safety parity."""
import copy
import hashlib
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from src.bvh import load_coco17, parse_bvh, fk
from src.config import CFG
from src.features import normalize_skeleton
from src.library import pose_to_feature, build_entries_from_pose
from src.refine import refine_bvh
from src.refine_projection import (
    FixedProjection, LEFT_RIGHT, ProjectionDomainError, active_projection,
    projection_scope, query_sha256, search_camera, search_rotation,
)
from src.search_geometry import Camera, project, normalize_batch
from api.candidate_presentation import refine_in_camera
from api.models import CandidateCamera, RefineRequest
from tests.test_smoke import _synthetic_bvh, _bvh_with_rotation


def camera_payload(base, query, parameters=(30.,20.,.25), swapped=False):
    return search_camera('test',hashlib.sha256(Path(base).read_bytes()).hexdigest(),query,
                         np.ones(17,bool),parameters,swapped,'front',.01)


@pytest.mark.parametrize('yaw,pitch,distance',[(0,0,None),(30,20,4.),(135,-30,2.)])
def test_fixed_projection_matches_search_with_translated_world_origin(yaw,pitch,distance):
    from src.library import _synthetic_pose
    # Feed the same float32 source geometry as the search index; do not compare
    # an unquantized translated double source against a quantized search input.
    points=(_synthetic_pose('sitting')+[35,92,-18]).astype(np.float32)
    rotation=search_rotation(yaw,pitch)
    # Same root-pivot alignment as rotate_bvh_text, not a reflection.
    aligned=(points-points[0])@rotation.T+points[0]
    projection=FixedProjection.from_aligned_base(aligned,distance)
    expected,valid=project(points,Camera(yaw,pitch,distance))
    assert valid and np.linalg.det(rotation)==pytest.approx(1)
    with projection_scope(projection):
        actual=pose_to_feature(aligned,'front').reshape(17,2)
    assert np.allclose(actual,normalize_batch(expected),atol=2e-6)
    assert active_projection() is None


@pytest.mark.parametrize('yaw,pitch,distance',[(0,0,None),(30,20,4.),(135,-30,2.)])
@pytest.mark.parametrize('swapped',[False,True])
@pytest.mark.parametrize('mode',['conservative','aggressive'])
def test_real_refine_improves_same_target_with_fixed_search_camera_and_labels(tmp_path,yaw,pitch,distance,swapped,mode):
    base=_synthetic_bvh(tmp_path,'base.bvh')
    target=_bvh_with_rotation(tmp_path,'target.bvh','LeftArm',15.)
    source_hash=hashlib.sha256(Path(base).read_bytes()).hexdigest()
    query,_=project(load_coco17(target)[0],Camera(yaw,pitch,distance))
    query=query*150+[256,330]
    if swapped: query=query[LEFT_RIGHT]
    scores=np.ones(17); old_query=query.copy();old_scores=scores.copy()
    camera=CandidateCamera(**camera_payload(base,query,(yaw,pitch,0 if distance is None else 1/distance),swapped))
    cfg=copy.copy(CFG);cfg.refine_enabled=True;cfg.refine_v2_enabled=True
    result=refine_in_camera(refine_bvh,base,query,scores,'back',camera=camera,
                            allowed_limbs=['right_arm' if swapped else 'left_arm'],
                            lower_body_observed=True,refine_mode=mode,cfg=cfg)
    assert result.refined, result.to_dict()
    assert result.loss_final<result.loss_base and 'left_arm' in result.limbs
    final=tmp_path/'result.bvh';final.write_text(result.bvh_text)
    original_joints,original_frames=parse_bvh(base)
    final_joints,final_frames=parse_bvh(str(final))
    assert np.allclose(fk(original_joints,original_frames[0])[0],fk(final_joints,final_frames[0])[0],atol=1e-7)
    assert all(np.array_equal(a[2],b[2]) for a,b in zip(original_joints,final_joints))
    before,_=project(load_coco17(base)[0],Camera(yaw,pitch,distance))
    after,_=project(load_coco17(str(final))[0],Camera(yaw,pitch,distance))
    measured=normalize_batch(query[LEFT_RIGHT] if swapped else query)
    assert np.linalg.norm(normalize_batch(after)[5:]-measured[5:]) < np.linalg.norm(normalize_batch(before)[5:]-measured[5:])
    assert np.array_equal(query,old_query) and np.array_equal(scores,old_scores)
    assert hashlib.sha256(Path(base).read_bytes()).hexdigest()==source_hash
    assert active_projection() is None
    assert result.diagnostics['search_camera']['labels_swapped']==swapped


def test_projection_scope_resets_on_exception_and_is_thread_local():
    from src.library import _synthetic_pose
    points=_synthetic_pose('sitting'); baseline=pose_to_feature(points,'front')
    a=FixedProjection.from_aligned_base(points,2.)
    b=FixedProjection.from_aligned_base(points,8.)
    with pytest.raises(RuntimeError):
        with projection_scope(a):
            with projection_scope(b): assert active_projection() is b
            assert active_projection() is a
            raise RuntimeError('solver failed')
    def work(projection):
        with projection_scope(projection): return pose_to_feature(points,'front')
    with ThreadPoolExecutor(max_workers=2) as executor:
        result=list(executor.map(work,[a,b]))
    assert not np.allclose(result[0],result[1])
    assert active_projection() is None and np.array_equal(pose_to_feature(points,'front'),baseline)


def test_near_plane_does_not_clamp_an_invalid_pose():
    points=np.zeros((17,3));points[[5,6],1]=1
    projection=FixedProjection.from_aligned_base(points,1.)
    points[9,2]=-1
    with projection_scope(projection),pytest.raises(ProjectionDomainError):
        pose_to_feature(points,'front')
    assert active_projection() is None


def test_v2_camera_requires_complete_parameters_and_binds_pose_and_query(tmp_path):
    base=_synthetic_bvh(tmp_path,'base.bvh');query=np.zeros((17,2))
    payload=camera_payload(base,query)
    for key in ('projection','labels_swapped','distance_torso','pose_id','query_sha256','observed_joint_mask'):
        broken=payload.copy();broken.pop(key)
        with pytest.raises(ValidationError): CandidateCamera(**broken)
    request=dict(pose_id='test',view='front',keypoints=query.tolist(),scores=[1.]*17,
                 camera=payload,distance_metric='perspective_pos')
    assert RefineRequest(**request).camera.version=='candidate-camera-v2'
    for change in ({'pose_id':'other'},{'keypoints':(query+1).tolist()},{'camera':None}):
        with pytest.raises(ValidationError): RefineRequest(**dict(request,**change))
    broken=payload.copy();broken['observed_joint_mask']=[True]*17;broken['observed_joint_mask'][9]=False
    with pytest.raises(ValidationError): RefineRequest(**dict(request,camera=broken))
    broken=payload.copy();broken['version']='candidate-camera-v1';broken['reference']='pelvis-torso-yaw'
    with pytest.raises(ValidationError): CandidateCamera(**broken)


def test_swapped_scores_and_allowed_limbs_follow_same_correspondence(tmp_path):
    from src.refine import RefineResult
    base=_synthetic_bvh(tmp_path,'base.bvh');query=np.arange(34).reshape(17,2)
    scores=np.linspace(.1,1.,17)
    camera=CandidateCamera(**camera_payload(base,query,swapped=True))
    def solver(path,keypoints,values,view,**kwargs):
        assert np.array_equal(keypoints,query[LEFT_RIGHT])
        assert np.array_equal(values,scores[LEFT_RIGHT])
        assert kwargs['allowed_limbs']==['right_arm']
        assert active_projection() is not None
        return RefineResult(False,'no_gain',path,.1,.1,0,'test')
    result=refine_in_camera(solver,base,query,scores,'front',camera=camera,allowed_limbs=['left_arm'])
    assert not result.refined and result.bvh_path==base and active_projection() is None


def test_changed_source_rejected_before_solver(tmp_path):
    base=_synthetic_bvh(tmp_path,'base.bvh');query=np.zeros((17,2))
    payload=camera_payload(base,query);payload['source_bvh_sha256']='0'*64
    with pytest.raises(ValueError,match='source changed'):
        refine_in_camera(lambda *a,**k:pytest.fail('must not solve'),base,query,np.ones(17),'front',camera=CandidateCamera(**payload))


def test_analyze_to_refine_roundtrip_keeps_real_search_camera(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as api
    from src.pipeline import Pipeline
    from src.repo import build_db
    from src.schema import Action,BBox,Relationship,Shot,Skeleton,VLMAnalysis,View
    base=_synthetic_bvh(tmp_path,'base.bvh')
    target=_bvh_with_rotation(tmp_path,'target.bvh','LeftArm',15.)
    points,scores=load_coco17(base)
    entries=build_entries_from_pose('test',points,{},base,scores)
    db=tmp_path/'poses.db';build_db(entries,str(db))
    query,_=project(load_coco17(target)[0],Camera(30,20,4.));query=query*150+[256,330]
    class VLM:
        def analyze(self,*args):
            return VLMAnalysis(1,Shot.FULL_HALF,Action.OTHER,View.FRONT,Relationship.SOLO,
                               approx_boxes=[BBox(0,0,512,768)],lower_body_visible=[True],lower_body_visibility_known=[True])
    class Detector:
        def detect(self,*args): return [BBox(0,0,512,768)]
    class Pose:
        def estimate(self,*args): return [Skeleton(query.copy(),np.full(17,.9))]
    for name,value in [('pose_search_mode','perspective'),('refine_v2_enabled',True),('refine_enabled',True)]:
        monkeypatch.setattr(CFG,name,value)
    pipe=Pipeline(entries,vlm_client=VLM(),detector=Detector(),pose_model=Pose())
    image=io.BytesIO();Image.new('RGB',(512,768),'white').save(image,format='PNG')
    with patch.dict(api.STATE,{'pipeline':pipe,'db_path':str(db)},clear=True):
        client=TestClient(api.app)
        response=client.post('/analyze',files={'file':('rough.png',image.getvalue(),'image/png')})
        assert response.status_code==200,response.text
        person=response.json()['people'][0];candidate=person['candidates'][0]
        assert person['refine_allowed'] and any(person['scores'])
        assert candidate['camera']['version']=='candidate-camera-v2'
        trace=person['quality_trace']['perspective_search']['cameras']['test']['refine_camera']
        assert candidate['camera']==CandidateCamera(**trace).model_dump()
        payload={k:person[k] for k in ('keypoints','scores','refine_allowed','refinable_limbs',
                 'lower_body_observed','skeleton_state','coverage_class','slot_origin','skeleton_source','distance_metric')}
        payload.update(pose_id='test',view=candidate['view'],camera=candidate['camera'],
                       search_distance=candidate['distance'],render_thumbnail=False,refine_mode='conservative')
        refined=client.post('/refine',json=payload)
        assert refined.status_code==200,refined.text
        result=refined.json()
        assert result['reason']!='skeleton_policy'
        assert result['diagnostics']['search_camera']['labels_swapped']==candidate['camera']['labels_swapped']
        assert result['diagnostics']['search_camera']['distance_torso']==candidate['camera']['distance_torso']
        assert active_projection() is None
        with patch.object(api,'refine_bvh',side_effect=ProjectionDomainError('unsafe proposal')):
            failed=client.post('/refine',json=payload)
        assert failed.status_code==200 and failed.json()['reason']=='camera_projection_invalid'
        assert failed.json()['refined'] is False and failed.json()['bvh'] is None
        assert failed.json()['refine_outcome']=='reverted'
        assert active_projection() is None
        # Lost binding must fail rather than silently use the nearest four-view label.
        payload.pop('camera')
        assert client.post('/refine',json=payload).status_code==422


def test_final_selector_uses_same_perspective_not_orthographic_metric(tmp_path):
    from src.refine import RefineResult
    from src.refine_selector import _target_evidence,common_metrics
    base=_synthetic_bvh(tmp_path,'base.bvh')
    points,_=load_coco17(base)
    query,_=project(points,Camera(30,20,2.));query=query*150+[256,330]
    camera=CandidateCamera(**camera_payload(base,query,(30,20,.5)))
    def solver(path,keypoints,scores,view,**kwargs):
        evidence=_target_evidence(keypoints,scores,path,view,CFG)
        metrics=common_metrics(path,evidence,view,CFG)
        assert metrics['joint_nme']['value']<1e-6
        with projection_scope(None):
            wrong=common_metrics(path,evidence,view,CFG)
        assert wrong['joint_nme']['value']>.01
        return RefineResult(False,'already_matched',path,0.,0.,0,'test')
    result=refine_in_camera(solver,base,query,np.ones(17),'front',camera=camera)
    assert not result.refined and result.bvh_path==base


def test_v1_configuration_preserves_original_search_and_refine_path():
    from tests.test_perspective_runtime import descriptor,gallery,observations
    from src.pipeline import Pipeline
    from src.library import build_synthetic_index
    from src.perspective_search import PerspectiveSearchIndex
    index=gallery()
    with patch.object(CFG,'refine_v2_enabled',False),patch.object(CFG,'pose_search_mode','perspective'), \
         patch.object(PerspectiveSearchIndex,'build',return_value=index), \
         patch.object(index,'search',side_effect=AssertionError('v1 must retain legacy search gates')):
        pipe=Pipeline(build_synthetic_index());desc=descriptor()
        result=pipe._search_one(desc,desc.skeleton,search_image=observations()[2])
        with patch.object(CFG,'pose_search_mode','legacy'):
            legacy=pipe._search_one(desc,desc.skeleton,search_image=observations()[2])
    assert [(c.pose_id,c.distance) for c in result[0]]==[(c.pose_id,c.distance) for c in legacy[0]]
    assert result[1:]==legacy[1:]


def test_search_candidate_uses_selected_camera_without_refitting_and_rejects_stale_source(tmp_path):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from src.schema import Skeleton
    from api.candidate_presentation import candidate_camera
    base=_synthetic_bvh(tmp_path,'base.bvh');query=np.zeros((17,2))
    payload=camera_payload(base,query,swapped=True)
    descriptor=SimpleNamespace(skeleton=Skeleton(query,np.ones(17)),quality_trace={
        'perspective_search':{'applied':True,'cameras':{'test':{'refine_camera':payload}}}})
    with patch('api.candidate_presentation.get_bvh_path',return_value=base), \
         patch('api.candidate_presentation.fit_camera',side_effect=AssertionError('must not re-fit')):
        assert candidate_camera('unused',SimpleNamespace(pose_id='test'),descriptor)==payload
        Path(base).write_text(Path(base).read_text()+'\n')
        with pytest.raises(HTTPException) as failure:
            candidate_camera('unused',SimpleNamespace(pose_id='test'),descriptor)
        assert failure.value.status_code==409


def test_fixed_camera_distance_does_not_follow_solver_torso_changes():
    from src.library import _synthetic_pose
    base=_synthetic_pose('sitting').astype(float)
    fixed=FixedProjection.from_aligned_base(base,4.)
    proposal=base.copy();proposal[[5,6],1]+=.1
    origin=base[[11,12]].mean(0)
    torso=np.linalg.norm(base[[5,6]].mean(0)-origin)
    centered=proposal-origin
    expected=centered[:,:2]/(centered[:,2,None]+4*torso);expected[:,1]*=-1
    assert np.allclose(fixed.project(proposal),expected)
    assert fixed.distance_world==pytest.approx(4*torso)


def test_query_binding_survives_json_negative_zero_normalization():
    query=np.zeros((17,2));query[0,0]=-0.
    canonical=np.zeros((17,2))
    assert query_sha256(query)==query_sha256(canonical)
