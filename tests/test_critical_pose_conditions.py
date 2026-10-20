"""No feedback IDs: critical shape requirements must survive every branch."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.pose_requirements import (WholePoseValidator, audit_generated,
    knee_support_evidence, requirements)
from src.semantic_search import SemanticQuery, SemanticHit, SemanticSearchResult


def body(*, kneeling=False, both=False, folded=False):
    p=np.zeros((17,3),float)
    p[0]=(0,2.3,0);p[5]=(-.4,2,0);p[6]=(.4,2,0)
    p[7]=(-.6,1.5,0);p[8]=(.6,1.5,0)
    p[9]=(-.5,1,0);p[10]=(.5,1,0)
    p[11]=(-.2,1,0);p[12]=(.2,1,0)
    p[13]=(-.2,.5,0);p[14]=(.2,.5,0)
    p[15]=(-.2,0,0);p[16]=(.2,0,0)
    if kneeling:
        p[13]=(-.2,0,0);p[15]=(-.2,0,1)
        p[14]=(.2,0,0) if both else (.2,1,1);p[16]=(.2,0,1)
    if folded:p[9]=(.35,1.7,0);p[10]=(-.35,1.7,0)
    return p,np.ones(17)


def validator(p,s,*,posture='standing',bound=True,fold=None):
    record=dict(groups={'pose_action':[dict(field='posture',value=posture)]},
        meta={'bvh_sha256':'sha' if bound else 'stale'})
    camera=SimpleNamespace(pose_ids=['a'],points=[p],scores=[s],hashes={'a':'sha'})
    shapes={} if fold is None else {'a':dict(source_sha256='sha',relations={'fold_arms':fold})}
    return WholePoseValidator(camera,{'a':record},{},shapes=shapes)


def req(**overrides):
    return {**dict(active=True,posture=None,arms={},body_states={},hand_counts={}),**overrides}


@pytest.mark.parametrize('text,support',[('한쪽 무릎을 꿇고 반대 무릎을 세움','one'),
    ('두 무릎을 꿇고 앉음','both'),('무릎을 꿇고 있음',None)])
def test_kneeling_request_keeps_posture_and_explicit_support(text,support):
    r=requirements(SemanticQuery(postures=['kneeling'],body_details=text),{})
    assert r['posture']=='kneeling' and r['knee_support']==support and r['active']


@pytest.mark.parametrize('scope',['half','bust','shoulder_head','head'])
def test_crop_never_requires_hidden_knee_contact(scope):
    r=requirements(SemanticQuery(postures=['kneeling'],visible_scope=scope,
        body_details='한쪽 무릎을 꿇었다'),{})
    assert r['posture'] is None and r['knee_support'] is None


def test_fold_relation_overrides_coarse_waist_bins_not_object_intent():
    r=requirements(SemanticQuery(actions=['fold_arms'],text='팔짱을 낀 자세',
        arm_states={'left':'waist','right':'waist'},held_object_hands={'bag':'right'}),{})
    assert r['relations']==['fold_arms'] and r['arms']=={}
    assert r['object_hands']=={'bag':'right'}
    assert r['hand_counts']=={'bag':'one'}


def test_negated_fold_does_not_create_a_hard_relation():
    r=requirements(SemanticQuery(actions=['fold_arms'],text='팔짱을 끼지 않았다'),{})
    assert r['relations']==[]


@pytest.mark.parametrize('both,expected',[(False,'one'),(True,'both')])
def test_knee_support_requires_low_bent_knees_and_bound_posture(both,expected):
    p,s=body(kneeling=True,both=both)
    r=req(posture='kneeling',knee_support=expected)
    assert knee_support_evidence(p,s)['support']==expected
    assert validator(p,s,posture='kneeling').check(r,'a')['status']=='passed'
    assert validator(p,s,posture='kneeling',bound=False).check(r,'a')['status']=='needs_review'
    assert validator(p,s,posture='sitting').check(r,'a')['status']=='needs_review'


def test_two_knees_cannot_satisfy_one_knee_request():
    p,s=body(kneeling=True,both=True)
    result=validator(p,s,posture='kneeling').check(req(posture='kneeling',knee_support='one'),'a')
    assert result['status']=='conflict' and 'knee_support' in result['conflicts']


def test_bent_raised_knee_or_bad_label_cannot_certify_kneeling():
    p,s=body()
    p[13]=(-.2,.8,.4)  # bent but well above the ankle plane
    r=req(posture='kneeling')
    assert validator(p,s,posture='kneeling').check(r,'a')['status']=='conflict'


@pytest.mark.parametrize('mutation',['missing','nonfinite','degenerate'])
def test_missing_or_invalid_leg_evidence_is_unknown(mutation):
    p,s=body(kneeling=True)
    if mutation=='missing':s[15]=0
    elif mutation=='nonfinite':p[13,1]=np.nan
    else:p[13]=p[11]
    assert knee_support_evidence(p,s)['support'] is None
    if mutation!='nonfinite':
        assert validator(p,s,posture='kneeling').check(req(posture='kneeling'),'a')['status']=='needs_review'


def test_knee_support_is_scale_translation_yaw_invariant_but_not_camera_pitch():
    p,s=body(kneeling=True)
    angle=.71;rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    assert knee_support_evidence((p@rot.T)*7+[10,8,-4],s)['support']=='one'
    roll_to_ground=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    assert knee_support_evidence(p@roll_to_ground.T,s)['support']!='one'


@pytest.mark.parametrize('profile',[True,False,None,'stale'])
def test_fold_needs_bound_shape_profile_and_measured_chest_cross(profile):
    p,s=body(folded=True);v=validator(p,s,fold=profile if isinstance(profile,bool) else None)
    if profile=='stale':v.shapes={'a':dict(source_sha256='stale',relations={'fold_arms':True})}
    result=v.check(req(relations=['fold_arms']),'a')
    expected='passed' if profile is True else 'conflict' if profile is False else 'needs_review'
    assert result['status']==expected


def test_fold_tag_cannot_rescue_waist_hands_or_missing_wrist():
    p,s=body();assert validator(p,s,fold=True).check(req(relations=['fold_arms']),'a')['status']=='conflict'
    p,s=body(folded=True);s[9]=0
    assert validator(p,s,fold=True).check(req(relations=['fold_arms']),'a')['status']=='needs_review'


def test_alternative_semantic_adapter_cannot_bypass_the_common_gate():
    from src.camera_search import CameraIndex
    from src.fullbody_routing import FullBodySearch
    from src.search_routing import PoseDescription
    p,s=body();camera=CameraIndex(['a'],[p],[s],hashes={'a':'sha'})
    class Text:
        core_records={};body={};shapes={}
        def search(self,q,top_k):return SemanticSearchResult('ok',[SemanticHit('a',.9,{},{})])
    engine=FullBodySearch([SimpleNamespace(pose_id='a',meta={})],Text(),camera)
    result=engine.semantic_search(PoseDescription(),req(relations=['fold_arms']))
    assert result['status']=='no_result' and result['hits']==[]


def test_review_recovery_cannot_drop_a_critical_fold_requirement():
    from tests.test_integrated_recovery import constrained_scene
    engine,decision,evidence,_=constrained_scene()
    engine.generator=None
    decision.semantic.description.update(actions=['fold_arms'],objects=[],
        held_object_hand_counts={},search_text='서서 팔짱을 낀 자세',body_details='가슴 앞에서 팔짱')
    result=engine.search(decision,evidence,review_geometry=True)
    assert result['selected']==[]
    retained=result['geometry_review_fallback']['retained_hard_conditions']
    assert retained['relations']==['fold_arms']


@pytest.mark.parametrize('folded,action,expected',[(True,'fold_arms','passed'),
    (False,'fold_arms','conflict'),(True,'hold','conflict')])
def test_generated_bvh_is_rechecked_not_trusted_from_a_plan(monkeypatch,folded,action,expected):
    import src.bvh
    p,s=body(folded=folded)
    monkeypatch.setattr(src.bvh,'load_coco17',lambda path:(p,s))
    text='unit-only-frozen-bvh';digest=hashlib.sha256(text.encode()).hexdigest()
    candidate=dict(bvh=text,bvh_sha256=digest,search_v2={'pose_action':{'action':[action]}},meta_json={})
    assert audit_generated(req(relations=['fold_arms']),candidate)['status']==expected


def test_generated_bvh_hash_mismatch_is_rejected_before_shape_audit():
    with pytest.raises(ValueError,match='generated_bvh_hash_mismatch'):
        audit_generated(req(relations=['fold_arms']),dict(bvh='not-bound',bvh_sha256='stale'))


@pytest.mark.parametrize('kneeling,expected',[(True,'passed'),(False,'conflict')])
def test_generated_kneeling_label_alone_cannot_pass(monkeypatch,kneeling,expected):
    import src.bvh
    p,s=body(kneeling=kneeling)
    monkeypatch.setattr(src.bvh,'load_coco17',lambda path:(p,s))
    text='unit-only-knee-bvh';digest=hashlib.sha256(text.encode()).hexdigest()
    candidate=dict(bvh=text,bvh_sha256=digest,search_v2={'pose_action':{'posture':['kneeling']}},meta_json={})
    assert audit_generated(req(posture='kneeling',knee_support='one'),candidate)['status']==expected


def test_geometry_and_semantic_share_fold_eligibility_including_positive_case():
    from src.camera_search import CameraIndex
    from src.fullbody_routing import FullBodySearch
    from src.search_routing import PoseDescription
    good,s=body(folded=True);bad,_=body()
    camera=CameraIndex(['bad','good'],[bad,good],[s,s],hashes={'bad':'bad-sha','good':'good-sha'})
    class Text:
        core_records={};body={}
        shapes={pid:dict(source_sha256=pid+'-sha',relations={'fold_arms':True}) for pid in ('bad','good')}
        def search(self,q,top_k):
            return SemanticSearchResult('ok',[SemanticHit(pid,.9,{},{}) for pid in ('bad','good')])
    engine=FullBodySearch([SimpleNamespace(pose_id=pid,meta={}) for pid in ('bad','good')],Text(),camera)
    ev=dict(keypoints=good[:,:2]*[1,-1],joint_mask=np.ones(17,bool),coverage_class='full')
    for result in (engine.geometry_search(ev,req(relations=['fold_arms'])),
                   engine.semantic_search(PoseDescription(),req(relations=['fold_arms']))):
        assert [h['pose_id'] for h in result['hits']]==['good']
