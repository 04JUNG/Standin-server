from types import SimpleNamespace
import copy

import pytest

from src.semantic_adapter_285 import Semantic285Adapter, build_semantic285_searcher
from src.semantic_search import SemanticQuery, build_semantic_searcher, UnavailableSemanticSearcher
from src.semantic_search_285 import SemanticSearchService, SemanticSettings
from src.semantic_shapes import CAMERAS
from tests.test_semantic_search_285 import _index, _write_library


def profile(**relations):
    return {'relations':relations,'views':[{'camera':dict(c),'fold_visible':True,'fold_score':.5,
        'v_visible':True,'v_score':.5} for c in CAMERAS]}


def candidate(id,posture='sitting',score=.8):
    return {'pose_id':id,'score':score,'score_components':{'concept':.8},'evidence':{},
            'pose':{'pose_id':id,'tags':{'pose_action':[{'field':'posture','value':posture}]}}}


def stub(raw,shapes=None,body=None,family=None):
    index=SimpleNamespace(pose_count=35,family=family or {})
    service=SemanticSearchService(index,SemanticSettings())
    adapter=Semantic285Adapter(service,shape_metadata=shapes,body_metadata=body)
    assert adapter.pool_service.settings.max_top_k==35
    assert not adapter.pool_service.settings.collapse_mirrors
    calls=[]
    def search(text,**kwargs):calls.append((text,kwargs));return copy.deepcopy(raw)
    adapter.pool_service=SimpleNamespace(search=search)
    return adapter,calls


def raw(candidates,status='ok'):
    return {'status':status,'candidates':candidates,'query':{},'index':{},'threshold':.35}


def test_entire_pool_is_filtered_before_top_k_not_just_first_five_or_twenty():
    bad=[candidate('bad'+str(i)) for i in range(27)]
    adapter,calls=stub(raw(bad+[candidate('good')]),{'good':profile(fold_arms=True)})
    result=adapter.search(SemanticQuery(text='팔짱을 끼고 앉은 자세',postures=['sitting']),1)
    assert result.status=='ok' and result.hits[0].pose_id=='good'
    assert calls[0][1]['top_k']==35 and calls[0][1]['source']=='vlm'
    assert result.hits[0].score==.8
    assert not result.hits[0].pose['refine_allowed']


def test_mirror_dedup_happens_after_validation_not_before():
    shapes={'x':profile(fold_arms=False),'x_mirror':profile(fold_arms=True)}
    adapter,_=stub(raw([candidate('x'),candidate('x_mirror')]),shapes,family={'x_mirror':'x'})
    r=adapter.search(SemanticQuery(text='앉아서 팔짱',postures=['sitting']))
    assert [h.pose_id for h in r.hits]==['x_mirror']


def test_required_posture_and_camera_hint_still_apply():
    shapes={'standing':profile(fold_arms=True),'sitting':profile(fold_arms=True)}
    adapter,_=stub(raw([candidate('standing','standing'),candidate('sitting')]),shapes)
    r=adapter.search(SemanticQuery(text='팔짱을 끼고 앉은 자세',postures=['sitting'],context_evidence='부감'))
    assert [h.pose_id for h in r.hits]==['sitting']
    assert r.hits[0].pose['preview_camera']['pitch_deg']>0
    assert r.hits[0].matched['camera_comparison']['kind']=='qualitative_visibility_not_pixel_fit'


def test_required_relation_missing_index_is_error_not_old_backend_fallback():
    adapter,calls=stub(raw([candidate('x')]))
    result=adapter.search(SemanticQuery(text='양팔을 V자로 펼친 자세'))
    assert result.status=='error' and 'index_missing' in result.reason and not calls


def test_known_body_conflict_rejected_unknown_neutral_without_score_mix():
    body={'bad':{'observations':{'knees':'straight','legs':'together'}},
          'unknown':{'observations':{'knees':'unknown','legs':'unknown'}}}
    adapter,_=stub(raw([candidate('bad'),candidate('unknown')]),body=body)
    result=adapter.search(SemanticQuery(text='다리를 벌리고 무릎을 굽힌 자세',body_observations={'knees':'bent','legs':'apart'}))
    assert [h.pose_id for h in result.hits]==['unknown']
    assert result.hits[0].score==.8


def test_fold_relation_removes_coarse_raised_arm_conflict():
    adapter,_=stub(raw([candidate('fold')]),{'fold':profile(fold_arms=True)},
        {'fold':{'observations':{'arms':'lowered','knees':'bent'}}})
    r=adapter.search(SemanticQuery(text='앉아서 팔짱',postures=['sitting'],body_observations={'arms':'raised','knees':'bent'}))
    assert r.status=='ok'


def test_search_error_and_empty_stay_separate():
    adapter,_=stub({'status':'error','error':{'code':'encoder_failed'}})
    assert adapter.search(SemanticQuery(text='손을 드는 자세')).reason=='semantic285:encoder_failed'
    adapter,_=stub(raw([],status='no_result'))
    assert adapter.search(SemanticQuery(text='손을 드는 자세')).status=='no_result'


def test_cold_index_error_is_preserved_and_bad_top_k_is_error():
    adapter=Semantic285Adapter(SemanticSearchService(None,SemanticSettings()))
    assert adapter.search(SemanticQuery(text='야구공을 던지는 자세')).status=='error'
    assert adapter.search(SemanticQuery(text='야구'),0).reason=='semantic285:invalid_top_k'


def test_real_core_adapter_defaults_to_no_preferences_and_keeps_id_json_contract(tmp_path):
    paths=_write_library(str(tmp_path))
    adapter=build_semantic285_searcher(**{k:v for k,v in paths.items() if k!='thumbs_dir'})
    assert isinstance(adapter,Semantic285Adapter)
    assert adapter.service.index.review_preferences=={}
    r=adapter.search(SemanticQuery(text='야구공을 던지는 투수'))
    assert r.status=='ok'
    assert all(h.pose_id==h.pose['pose_id'] and not h.pose['refine_allowed'] for h in r.hits)


def test_factory_missing_tags_and_cache_fail_closed(tmp_path):
    paths=_write_library(str(tmp_path))
    adapter=build_semantic285_searcher(**{k:v for k,v in paths.items() if k!='thumbs_dir'},shape_index_path=str(tmp_path/'missing.json'))
    assert isinstance(adapter,UnavailableSemanticSearcher)
    assert adapter.search(SemanticQuery(text='야구')).status=='error'


def test_backend_switch_default_unchanged_unknown_configuration_rejected(monkeypatch,tmp_path):
    monkeypatch.delenv('SEMANTIC_BACKEND',raising=False)
    assert build_semantic_searcher(str(tmp_path/'missing')).reason=='semantic_index_missing'
    monkeypatch.setenv('SEMANTIC_BACKEND','typo')
    assert build_semantic_searcher(str(tmp_path/'missing')).reason=='semantic_backend_unknown'


def test_backend_switch_builds_with_existing_config_without_db_path_field(monkeypatch,tmp_path):
    paths=_write_library(str(tmp_path))
    monkeypatch.setenv('SEMANTIC_BACKEND','semantic285')
    monkeypatch.setenv('SEMANTIC_285_DB_PATH',paths['db_path'])
    monkeypatch.setenv('SEMANTIC_TAXONOMY_PATH',paths['taxonomy_path'])
    monkeypatch.setenv('SEMANTIC_DENSE_ENABLED','0')
    for key in ('SEMANTIC_BODY_INDEX_PATH','SEMANTIC_SHAPE_INDEX_PATH','SEMANTIC_LEXICON_PATH'):
        monkeypatch.delenv(key,raising=False)
    s=build_semantic_searcher(paths['tags_path'])
    assert isinstance(s,Semantic285Adapter)
    assert s.search(SemanticQuery(text='야구공을 던지는 투수')).status=='ok'
