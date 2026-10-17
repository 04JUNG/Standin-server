"""Regression contracts from set4 feedback, without per-cut exceptions."""
from types import SimpleNamespace
import numpy as np
from src.pose_requirements import requirements,measure,WholePoseValidator
from src.pose_query import prepare_pose_query
from src.semantic_search import SemanticQuery
from src.library import _synthetic_pose
from src.vlm.client import parse_pose_descriptions

LEX=SimpleNamespace(labels={('object','phone'):'휴대폰',('object','pillow'):'베개'})


def test_independent_count_survives_unknown_laterality_and_incidental_props():
    q=SemanticQuery(objects=['phone'],held_object_hands={'phone':'unknown'},text='양손으로 휴대폰을 잡고 두 팔을 앞으로 뻗음')
    r=requirements(q,prepare_pose_query(q,LEX))
    assert r['hand_counts']=={'phone':'two'}
    q=SemanticQuery(postures=['lying'],objects=['phone','pillow'],body_details='양팔을 벌리고 등을 바닥에 대고 누워 있다')
    prepared=prepare_pose_query(q,LEX)
    assert prepared['objects']==[] and requirements(q,prepared)['posture']=='lying'


def test_both_arms_are_constraints_without_a_prop():
    q=SemanticQuery(body_details='왼손을 허리에 얹고 오른팔은 자연스럽게 내린다')
    assert requirements(q,prepare_pose_query(q,LEX))['arms']=={'left':'waist','right':'lowered'}
    q=SemanticQuery(body_details='양팔을 골반 옆으로 내린 자세')
    assert requirements(q,prepare_pose_query(q,LEX))['arms']=={'left':'lowered','right':'lowered'}


def test_parser_keeps_count_separate_from_side():
    desc=parse_pose_descriptions({'people':[{'visible':True,'objects':['phone'],'context_evidence':'휴대폰을 잡음',
        'held_object_hands':{'phone':'both'},'arm_states':{'left':'forward','right':'screen_right'}}]},1)[0]
    desc.held_object_hands['phone']='unknown'
    assert desc.held_object_hand_counts=={'phone':'two'} and desc.arm_states=={'left':'forward'}


def test_measured_posture_is_gravity_sensitive_and_arm_states_are_body_relative():
    p=_synthetic_pose('standing');scores=np.ones(17)
    upright=measure(p,scores)
    # Rotate the same shape onto the ground: a camera change cannot make a
    # standing BVH eligible as a supine pose.
    a=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    lying=measure(p@a.T,scores)
    assert upright['posture']=='standing' and lying['posture']=='lying'
    assert upright['arms']==lying['arms']


def test_both_phone_uses_bound_metadata_and_far_wrists_are_rejected():
    p=_synthetic_pose('standing');scores=np.ones(17)
    cam=SimpleNamespace(pose_ids=['one'],points=[p],scores=[scores],hashes={'one':'sha'})
    record={'meta':{'bvh_sha256':'sha','body_parts':{'hands':{'right':'phone','left':'relaxed'}}}}
    validator=WholePoseValidator(cam,{'one':record},{})
    req=dict(active=True,posture=None,arms={},hand_counts={'phone':'two'})
    checked=validator.check(req,'one')
    assert 'hand_count:phone' in checked['conflicts']
    record['meta']['bvh_sha256']='stale'
    checked=validator.check(req,'one')
    assert checked['checks']['hand_count:phone']['status']=='unknown'


def test_unknown_arm_does_not_become_a_pass():
    p=_synthetic_pose('standing');scores=np.ones(17);scores[9]=0
    cam=SimpleNamespace(pose_ids=['a'],points=[p],scores=[scores],hashes={'a':'sha'})
    validator=WholePoseValidator(cam,{},{});req=dict(active=True,posture=None,arms={'left':'lowered'},hand_counts={})
    assert validator.check(req,'a')['status']=='needs_review'


def test_filtered_empty_geometry_reaches_generation_instead_of_wrong_top5():
    from src.camera_search import CameraIndex
    from src.fullbody_routing import FullBodySearch
    from tests.test_fullbody_routing import decision,evidence,Generator
    from src.semantic_search import SemanticSearchResult
    p=_synthetic_pose('standing')
    camera=CameraIndex(['a'],[p],[np.ones(17)],hashes={'a':'sha'})
    class Text:
        core_records={};body={};service=SimpleNamespace(index=SimpleNamespace(lexicon=LEX))
        def search(self,q,top_k):return SemanticSearchResult('no_result')
    generator=Generator();engine=FullBodySearch([SimpleNamespace(pose_id='a',meta={})],Text(),camera,generator=generator)
    d=decision();d.semantic.description.update(postures=['lying'],body_details='등을 바닥에 대고 누워 있음',search_text='누운 자세',body_observations={})
    ev=evidence();ev['keypoints']=p[:,:2].copy();ev['keypoints'][:,1]*=-1
    result=engine.search(d,ev)
    assert not result['selected'] and generator.calls==1
    assert result['branches']['geometry']['whole_pose_validation']['counts']['conflict']==1
    assert result['generation']['status']=='rejected'  # malformed BVH from the stub never escapes
