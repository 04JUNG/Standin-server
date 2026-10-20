from types import SimpleNamespace
import numpy as np
from src.library import _synthetic_pose
from src.pose_requirements import WholePoseValidator,measure,compare_pose_details
from src.natural_pose_worker import independent_reasons


def test_intrinsic_leg_spacing_survives_side_lying_and_supine_rotation():
    p=_synthetic_pose('standing');s=np.ones(17)
    p[15,0]-=30;p[16,0]+=30
    wanted=measure(p,s)
    for rotation in (np.array([[1,0,0],[0,0,-1],[0,1,0]]),np.array([[0,-1,0],[1,0,0],[0,0,1]])):
        actual=measure(p@rotation.T,s)
        assert actual['body']['legs']==wanted['body']['legs']
        assert np.isclose(actual['metrics']['leg_separation_hip_widths'],wanted['metrics']['leg_separation_hip_widths'])


def test_detail_difference_is_near_not_wrong_core_or_success():
    p=_synthetic_pose('standing');s=np.ones(17)
    cam=SimpleNamespace(pose_ids=['a'],points=[p],scores=[s],hashes={'a':'sha'})
    validator=WholePoseValidator(cam,{},{});req=dict(active=True,posture='standing',arms={},hand_counts={},body_states={'legs':'apart'})
    checked=validator.check(req,'a')
    assert checked['status']=='near_match' and checked['detail_conflicts']==['body:legs']
    req['posture']='lying'
    assert validator.check(req,'a')['status']=='conflict'


def test_projected_detail_fit_does_not_rotate_away_lean_or_invent_hidden_legs():
    p=_synthetic_pose('standing')[:,:2].copy();p[:,1]*=-1;m=np.ones(17,bool)
    tilted=p@np.array([[.866,-.5],[.5,.866]]).T
    fit=compare_pose_details(p,tilted,m)
    assert np.isclose(fit['checks']['torso_lean_deg']['difference'],30,atol=.01)
    assert fit['cost']>0
    m[11:]=False
    assert compare_pose_details(p,tilted,m)['checks']=={}


def test_visible_phone_finger_overlap_is_not_softened_as_a_detail():
    metrics={'L':dict(bend=90,forearm_roll=0,wrist_flex=0,wrist_rad=0)}
    assert independent_reasons(metrics,{'Lhand~Rhand':1.18})
    assert not independent_reasons(metrics,{'Lhand~Rhand':.19})


def test_failed_generated_near_match_keeps_evidence_outside_success_candidates():
    from src.natural_pose import generate_loop
    from tests.test_natural_pose import proposal,TAXO
    result=generate_loop('자세',lambda *a:proposal(),lambda p:dict(passed=False,reasons=['다리 간격'],
        validation={'independent':True},near_candidate={'bvh':'evidence','bvh_sha256':'sha'}),TAXO,max_retries=0)
    assert result.status=='rejected' and not result.candidates
    assert result.diagnostics['near_candidates'][0]['not_exact_match']
    assert not result.diagnostics['near_candidates'][0]['release_allowed']


def test_only_near_library_hits_do_not_suppress_generation():
    from src.camera_search import CameraIndex
    from src.fullbody_routing import FullBodySearch
    from src.semantic_search import SemanticSearchResult
    from tests.test_fullbody_routing import decision,evidence,Generator
    from tests.test_whole_pose_conditions import LEX
    p=_synthetic_pose('standing');camera=CameraIndex(['a'],[p],[np.ones(17)],hashes={'a':'sha'})
    class Text:
        core_records={};body={};service=SimpleNamespace(index=SimpleNamespace(lexicon=LEX))
        def search(self,q,top_k):return SemanticSearchResult('no_result')
    generator=Generator();engine=FullBodySearch([SimpleNamespace(pose_id='a',meta={})],Text(),camera,generator=generator)
    d=decision();d.semantic.description.update(postures=['standing'],actions=[],
        search_text='서 있는 자세',body_details='서 있는 자세',body_observations={'knees':'bent'})
    ev=evidence();ev['keypoints']=p[:,:2].copy();ev['keypoints'][:,1]*=-1
    result=engine.search(d,ev)
    assert not result['selected'] and len(result['near_candidates'])==1
    assert generator.calls==1 and result['generation']['status']=='rejected'
