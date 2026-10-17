"""Offline contracts: original whole poses only, hybrid ranks, fail-closed fallback."""
from types import SimpleNamespace
import numpy as np
import pytest

from src.fullbody_routing import FullBodySearch, build_api_pipeline
from src.search_routing import (JointEvidence, PoseDescription, RoutingConfig,
    assess_semantic_evidence, decide_route)
from src.semantic_search import SemanticHit, SemanticSearchResult
from src.experimental.text_pose_generation import GenerationResult
from src.config import CFG


class Camera:
    cameras = [None] * 117
    def __init__(self, ids=('a', 'b'), distance=.2):
        self.ids, self.distance, self.calls = ids, distance, 0
    def search(self, points, mask, part):
        self.calls += 1
        return [dict(pose_id=p, distance=self.distance+i*.01,
                     camera={'yaw': 45.}) for i,p in enumerate(self.ids)]


class Semantic:
    def __init__(self, ids=('b', 'c'), status=None):
        self.ids, self.status, self.calls = ids, status, 0
    def search(self, query, top_k):
        self.calls += 1
        if self.status == 'raise':
            raise RuntimeError('offline provider error')
        return SemanticSearchResult(self.status or ('ok' if self.ids else 'no_result'),
            hits=[SemanticHit(p, .9-i*.01, {}, {}) for i,p in enumerate(self.ids)], backend='stub')


class Generator:
    def __init__(self): self.calls=0
    def generate(self, description):
        self.calls+=1
        return GenerationResult('preview_ready', candidates=[{'bvh':'test'}])


def decision(sufficient=True, text=True, error=None, block=None):
    desc = PoseDescription(visible=True, certainty='high', postures=['sitting'],
        actions=['fold_arms'], search_text='의자에 앉아 양팔을 가슴 앞에 모아 팔짱', body_details='양팔을 가슴 앞에 교차')
    sem=assess_semantic_evidence(desc if text else None, RoutingConfig(), vlm_error=error)
    return decide_route(JointEvidence(sufficient, block, [], {}, {}), sem)


def evidence():
    return dict(keypoints=np.zeros((17,2)), joint_mask=np.ones(17,bool), coverage_class='full')


def engine(camera=None, semantic=None):
    entries=[SimpleNamespace(pose_id=p,meta={}) for p in ('a','b','c')]
    return FullBodySearch(entries, semantic or Semantic(), camera or Camera(), generator=Generator())


def test_good_joints_also_search_text_and_promote_agreement():
    e=engine(); out=e.search(decision(), evidence())
    assert out['status']=='ok' and out['route']=='hybrid'
    assert e.camera.calls==e.semantic.calls==1
    assert out['selected'][0]['pose_id']=='b'
    assert set(out['selected'][0]['evidence'])=={'geometry','semantic'}
    assert out['selected'][0]['rank_score']==pytest.approx(.65/62+.35/61)
    assert not out['composition_enabled'] and not out['split_search_enabled']
    assert all(x['kind']=='library_original' for x in out['selected'])
    assert e.generator.calls==0


def test_insufficient_joints_only_search_text():
    e=engine(); out=e.search(decision(False), evidence())
    assert out['route']=='semantic' and out['status']=='ok'
    assert e.camera.calls==0 and e.semantic.calls==1


@pytest.mark.parametrize('weight', [0, 1, -1, float('nan'), float('inf')])
def test_invalid_semantic_weight_rejected(weight):
    with pytest.raises(ValueError, match='semantic_weight'):
        FullBodySearch([], None, None, semantic_weight=weight)


def test_higher_semantic_weight_changes_fusion_only():
    entries=[SimpleNamespace(pose_id=p,meta={}) for p in ('a','b','c')]
    a=FullBodySearch(entries, Semantic(('c',)), Camera(('a',)))
    b=FullBodySearch(entries, Semantic(('c',)), Camera(('a',)), semantic_weight=.6)
    before=a.search(decision(), evidence()); after=b.search(decision(), evidence())
    assert before['selected'][0]['pose_id']=='a'
    assert after['selected'][0]['pose_id']=='c'
    assert after['selected'][0]['rank_score']==pytest.approx(.6/61)
    assert after['ranking']=='weighted_rrf_k60_geometry_0.4_semantic_0.6'
    assert not after['composition_enabled'] and not after['split_search_enabled']
    for name in ('geometry','semantic'):
        assert before['branches'][name]['hits']==after['branches'][name]['hits']
    text_only=b.search(decision(False), evidence())
    assert text_only['selected'][0]['rank_score']==pytest.approx(1/61)


@pytest.mark.parametrize('sufficient', [True,False])
def test_semantic_error_never_becomes_empty_generation(sufficient):
    e=engine(semantic=Semantic((), 'raise'))
    out=e.search(decision(sufficient), evidence())
    assert out['status']==('partial_error' if sufficient else 'error')
    assert e.generator.calls==0


@pytest.mark.parametrize('sufficient', [True,False])
def test_description_provider_error_is_not_abstention(sufficient):
    e=engine(); out=e.search(decision(sufficient,error='timeout'), evidence())
    assert out['status']==('partial_error' if sufficient else 'error')
    assert e.semantic.calls==0 and e.generator.calls==0


@pytest.mark.parametrize('sufficient',[True,False])
def test_successful_empty_search_can_generate_only_separate_preview(sufficient):
    e=engine(Camera(()),Semantic(()))
    out=e.search(decision(sufficient), evidence())
    assert out['status']=='no_result' and not out['selected']
    assert e.generator.calls==1 and out['generation']['status']=='preview_ready'
    assert out['generation']['release_allowed'] is False


def test_geometry_error_blocks_generation_even_when_text_empty():
    e=engine(Camera(()),Semantic(())); e.camera.cameras=[]
    out=e.search(decision(), evidence())
    assert out['status']=='error' and e.generator.calls==0


def test_generic_or_cropped_does_not_invent_full_body():
    e=engine(Camera(()),Semantic(()))
    assert e.search(decision(text=False),evidence())['status']=='undetermined'
    assert e.search(decision(False,text=False),evidence())['status']=='undetermined'
    out=e.search(decision(False),evidence(),generation_block=True)
    assert out['generation']['reason']=='cropped_or_unsupported_scope'
    assert e.generator.calls==0


def test_policy_block_runs_nothing():
    e=engine(); out=e.search(decision(block='entangled'),evidence())
    assert out['status']=='blocked' and not out['selected']
    assert e.generator.calls==e.camera.calls==e.semantic.calls==0


@pytest.mark.parametrize('bad', [float('nan'),float('inf'), '0.2'])
def test_bad_distance_is_error(bad):
    e=engine(); e.camera.search=lambda *a:[dict(pose_id='a',distance=bad)]
    assert e.geometry_search(evidence())['status']=='error'


def test_missing_anchor_is_error_not_no_result():
    e=engine(); ev=evidence(); ev['joint_mask'][11]=False
    assert e.geometry_search(ev)['status']=='error'


def test_cutoff_and_original_safety_filters(monkeypatch):
    import src.fullbody_routing as mod
    e=engine(Camera(distance=2))
    assert e.geometry_search(evidence())['status']=='no_result'
    e.camera=Camera(); e.meta['b']['source']='generated_composition'
    e.include_stored_compositions=False  # Reproduce the previous restricted corpus.
    monkeypatch.setattr(mod,'excluded_pose_ids', lambda:{'a'})
    out=e.search(decision(),evidence())
    assert [r['pose_id'] for r in out['selected']]==['c']


@pytest.mark.parametrize('branch', ['geometry', 'semantic'])
def test_stored_compositions_are_searchable_but_quarantine_still_wins(monkeypatch, branch):
    import src.fullbody_routing as mod
    e=engine(); e.meta['b']['source']='generated_composition'
    hits=[dict(pose_id='b',distance=.1,score=.5)]
    assert e._clean(hits,branch)==hits
    monkeypatch.setattr(mod,'excluded_pose_ids',lambda:{'b'})
    assert e._clean(hits,branch)==[]


def test_inclusion_preserves_weights_and_never_enables_runtime_composition():
    e=engine(); e.meta['b']['source']='generated_composition'
    out=e.search(decision(),evidence())
    assert out['stored_compositions_included']
    assert not out['composition_enabled'] and not out['split_search_enabled']
    assert out['selected'][0]['library_source']=='generated_composition'
    assert out['selected'][0]['rank_score']==pytest.approx(.65/62+.35/61)
    assert not out['selected'][0]['release_allowed']


def test_unknown_pose_is_error():
    e=engine(semantic=Semantic(('external',)))
    out=e.search(decision(False),evidence())
    assert out['status']=='error' and e.generator.calls==0


def test_factory_fails_closed_without_humanart(monkeypatch):
    monkeypatch.setenv('ROUTING_FULLBODY_ENABLED','1')
    monkeypatch.setattr(CFG,'pose_backend','mock')
    with pytest.raises(ValueError,match='requires POSE_BACKEND'):
        build_api_pipeline([],db_path='missing.db')


@pytest.mark.parametrize('rough',[False,True])
def test_pipeline_and_api_contract_always_describe_without_split(monkeypatch, rough):
    from src.pipeline import Pipeline
    from src.library import build_synthetic_index
    from tests.test_search_routing import _Img, _RoughPose, _CountingVLM
    from src.search_routing import DescribeOutcome
    from api.app import _routing_fields
    from api.models import DynamicSearchOut
    entries=build_synthetic_index(); pid=entries[0].pose_id
    vlm=_CountingVLM(); calls=[]
    def describe(image, boxes):
        calls.append(boxes)
        return DescribeOutcome('ok',descriptions={i:PoseDescription(**decision().semantic.description) for i in range(len(boxes))})
    vlm.describe_people=describe
    e=FullBodySearch(entries,Semantic((pid,)),Camera((pid,)),generator=Generator())
    monkeypatch.setattr(CFG,'search_routing_enabled',False)
    pipe=Pipeline(entries,vlm_client=vlm,pose_model=_RoughPose(all_rough=rough),fullbody_search=e)
    result=pipe.process_cut(_Img('full_half standing front'+(' rough' if rough else '')))
    assert len(calls)==1
    report=DynamicSearchOut(**_routing_fields(result,0)['dynamic_search'])
    assert report.status=='ok' and report.route==('semantic' if rough else 'hybrid')
    assert len(report.selected)==1 and not report.selected[0].refine_allowed
    assert result.person_candidates==[[]] and result.person_semantic_candidates==[[]]
    assert e.camera.calls==(0 if rough else 1)


def test_real_camera_117_exact_projection():
    from src.camera_search import CameraIndex
    p=np.zeros((17,3))
    for i in range(5,17):p[i]=[(i%2-.5)*.8, (17-i)*.15, (i%3)*.1]
    idx=CameraIndex(['a'],[p],[np.ones(17)])
    hits=idx.search(idx.projections[0,0],np.ones(17,bool),'full')
    assert len(idx.cameras)==117 and hits[0]['distance']==pytest.approx(0.,abs=1e-6)


@pytest.mark.parametrize('rough', [False, True])
@pytest.mark.parametrize('hand_status', ['disabled', 'ok', 'failed'])
@pytest.mark.parametrize('people', [1, 2])
def test_http_analyze_exposes_only_whole_pose_rank_list(monkeypatch, rough, hand_status, people):
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    import api.app as api
    from src.pipeline import Pipeline
    from src.library import build_synthetic_index
    from tests.test_search_routing import _Img, _RoughPose, _CountingVLM
    from src.search_routing import DescribeOutcome
    entries=build_synthetic_index(); pid=entries[0].pose_id
    vlm=_CountingVLM()
    hand_calls=[]
    def read_hands(image, boxes):
        hand_calls.append(boxes)
        if hand_status == 'failed':
            raise RuntimeError('offline hand provider failure')
        return {'people': [
            {'left': {'visible': True, 'shape': 'hold_phone' if i == 0 else 'fist', 'confidence': 'high'},
             'right': {'visible': True, 'shape': 'open', 'confidence': 'high'}}
            for i, _ in enumerate(boxes)]}
    vlm.read_hands=read_hands
    monkeypatch.setattr(CFG, 'vlm_hand_shapes', hand_status != 'disabled')
    vlm.describe_people=lambda image,boxes:DescribeOutcome('ok',descriptions={
        i:PoseDescription(**decision().semantic.description) for i in range(len(boxes))})
    pipe=Pipeline(entries,vlm_client=vlm,pose_model=_RoughPose(all_rough=rough),
        fullbody_search=FullBodySearch(entries,Semantic((pid,)),Camera((pid,))))
    result=pipe.process_cut(_Img(f'full_half standing front {people}p'+(' rough' if rough else '')))
    monkeypatch.setitem(api.STATE,'pipeline',SimpleNamespace(process_cut=lambda *a,**k:result))
    monkeypatch.setitem(api.STATE,'provider','mock')
    monkeypatch.setitem(api.STATE,'pose_backend','mock')
    data=io.BytesIO(); Image.new('RGB',(128,128),'white').save(data,format='PNG')
    response=TestClient(api.app).post('/analyze',files={'file':('rough.png',data.getvalue(),'image/png')})
    assert response.status_code==200,response.text
    assert len(response.json()['people']) == people
    person=response.json()['people'][0]
    assert person['candidates']==person['semantic_candidates']==[]
    assert person['dynamic_search']['route']==('semantic' if rough else 'hybrid')
    assert person['dynamic_search']['selected'][0]['pose_id']==pid
    assert not person['refine_allowed']
    assert len(hand_calls) == (0 if hand_status == 'disabled' else 1)
    hands=person['hands']
    assert hands['status'] == {'disabled': 'not_requested', 'ok': 'vlm',
                               'failed': 'unavailable'}[hand_status]
    assert hands['left']['preset'] == ('hold_phone' if hand_status == 'ok' else 'relaxed')
    assert hands['right']['preset'] == ('open' if hand_status == 'ok' else 'relaxed')
    if people == 2:
        other=response.json()['people'][1]['hands']
        assert other['left']['preset'] == ('fist' if hand_status == 'ok' else 'relaxed')


def test_factory_disabled_keeps_legacy(monkeypatch):
    import src.pipeline as module
    sentinel=object()
    monkeypatch.setenv('ROUTING_FULLBODY_ENABLED','0')
    monkeypatch.setattr(module,'Pipeline',lambda entries:sentinel)
    assert build_api_pipeline([],db_path='not-needed.db') is sentinel


@pytest.mark.parametrize('review_status', ['ok', 'error', 'skipped'])
def test_fullbody_review_precedes_search_and_preserves_diagnostics(monkeypatch, review_status):
    from src.pipeline import Pipeline
    from src.library import build_synthetic_index
    from tests.test_search_routing import _Img, _RoughPose, _CountingVLM
    from src.search_routing import DescribeOutcome
    entries = build_synthetic_index(); pid = entries[0].pose_id
    vlm = _CountingVLM(); order = []
    vlm.describe_people = lambda image, boxes: DescribeOutcome('ok', descriptions={
        i: PoseDescription(**decision().semantic.description) for i in range(len(boxes))})
    class Review:
        def repair(self, image, slots, *args):
            order.append('review')
            return dict(status=review_status, called=review_status!='skipped', targets=[0],
                        elapsed_ms=1.0, repaired_slots=[])
    engine = FullBodySearch(entries, Semantic((pid,)), Camera((pid,)))
    original = engine.search
    def search(*args, **kwargs):
        assert order == ['review']
        order.append('search')
        return original(*args, **kwargs)
    engine.search = search
    monkeypatch.setattr(CFG, 'search_routing_enabled', False)
    pipe = Pipeline(entries, vlm_client=vlm, pose_model=_RoughPose(),
                    fullbody_search=engine, skeleton_repairer=Review())
    result = pipe.process_cut(_Img('full_half standing front'))
    assert order == ['review', 'search']
    assert result.routing_summary['skeleton_repair']['status'] == review_status
    assert not result.descriptors[0].refine_allowed
