from types import SimpleNamespace
import pytest
from src.pose_query import prepare_pose_query,candidate_conditions
from src.semantic_search import SemanticQuery
from tests.test_fullbody_routing import engine

LEX=SimpleNamespace(labels={('object','phone'):'휴대폰',('object','cup'):'컵',('action','hold'):'잡기',('action','talk'):'대화'})


def query():
    return SemanticQuery(objects=['phone'],actions=['hold','talk'],
        text='스마트폰을 손에 들고 미소 지으며 대화하는 상체 자세',
        body_details='가슴 앞쪽으로 한 손을 올려 스마트폰을 쥐고 있으며 상체만 화면에 드러나 있다.')


def record(text,objects=()):
    return {'description':text,'tags':{'objects':[{'field':'object','value':o} for o in objects]}}


def test_preserves_vlm_hold_and_lost_hand_location_without_social_penalties():
    p=prepare_pose_query(query(),LEX)
    assert p['hand_count']=='one' and p['hand_position']=='chest_front'
    assert '가슴' in p['core_text'] and '한 손' in p['core_text']
    assert '미소' not in p['core_text'] and '대화' not in p['core_text'] and '드러나' not in p['core_text']
    assert 'action:hold' in p['tags'] and 'action:talk' not in p['tags']


def test_no_named_object_keeps_existing_query_unchanged():
    q=SemanticQuery(text='쪼그려 앉아 팔짱을 낀 자세',actions=['fold_arms'])
    p=prepare_pose_query(q,LEX)
    assert not p['enabled'] and p['core_text']==q.text and not p['tags']


def test_unknown_hands_do_not_become_a_mismatch_or_a_confirmed_match():
    p=prepare_pose_query(query(),LEX)
    c=candidate_conditions(p,record('휴대폰을 보는 자세',['phone']),LEX)
    assert c['object_state']=='match' and not c['conflicts']
    assert c['relations']['hand_count']['status']=='unknown'


@pytest.mark.parametrize('description,expected',[
    ('한 손으로 휴대폰을 들고 보기','match'),
    ('양손으로 휴대폰을 쥐고 보기','conflict'),
    ('휴대폰을 보면서 한 손으로 베개를 받침','unknown'),
])
def test_hand_count_is_attached_to_the_requested_object(description,expected):
    c=candidate_conditions(prepare_pose_query(query(),LEX),record(description,['phone']),LEX)
    assert c['relations']['hand_count']['status']==expected


def test_ear_phone_is_explicitly_different_from_chest_phone():
    c=candidate_conditions(prepare_pose_query(query(),LEX),record('한 손으로 휴대폰을 귀에 대고 통화',['phone']),LEX)
    assert 'hand_position' in c['conflicts']


def test_missing_prop_is_unknown_and_another_named_prop_is_conflict():
    p=prepare_pose_query(query(),LEX)
    assert candidate_conditions(p,record('손을 가슴 앞에 모음'),LEX)['object_state']=='unknown'
    assert candidate_conditions(p,record('가슴 앞에 컵을 들고 있음',['cup']),LEX)['object_state']=='conflict'


def test_generic_handheld_object_works_for_cup_too():
    p=prepare_pose_query(SemanticQuery(text='한 손에 텀블러를 들고 서 있음',objects=['cup'],actions=['hold']),LEX)
    assert p['enabled'] and p['hand_count']=='one'
    assert candidate_conditions(p,record('한 손으로 컵을 들고 있음',['cup']),LEX)['relations']['hand_count']['status']=='match'


def test_fusion_checks_both_branches_and_keeps_unknowns_and_conflicts_reviewable():
    e=engine();p=prepare_pose_query(query(),LEX)
    branches={'geometry':{'status':'ok','hits':[{'pose_id':'a','distance':.01},{'pose_id':'b','distance':.2}]},
              'semantic':{'status':'ok','hits':[{'pose_id':'c','score':.5}]}}
    conditions={'a':candidate_conditions(p,record('컵을 든 자세',['cup']),LEX),
                'b':candidate_conditions(p,record('손을 모음'),LEX),
                'c':candidate_conditions(p,record('한 손으로 휴대폰을 듦',['phone']),LEX)}
    out=e._fuse(branches,conditions)
    assert [r['pose_id'] for r in out]==['c','b','a']
    assert out[0]['rank_score']<out[1]['rank_score']


def test_structured_tags_keep_synonym_coverage_and_actual_hold_in_real_index(tmp_path):
    from tests.test_semantic_search_285 import _write_library
    from src.semantic_adapter_285 import build_semantic285_searcher
    paths=_write_library(str(tmp_path),poses=[
        ('one', [('action','hold'),('object','phone')], '한 손으로 휴대폰을 들고 보는 자세'),
        ('two', [('action','hold'),('object','phone')], '양손으로 휴대폰을 쥐고 보는 자세'),
    ])
    adapter=build_semantic285_searcher(**{k:v for k,v in paths.items() if k!='thumbs_dir'})
    q=SemanticQuery(text='한 손으로 스마트폰을 쥐고 보는 자세',actions=['hold'],objects=['phone'])
    result=adapter.search(q)
    assert result.status=='ok' and result.hits[0].pose_id=='one'
    evidence=result.hits[0].matched['semantic285']
    assert '스마트폰을' not in evidence['evidence']['uncovered_words']
    assert 'action:hold' in evidence['query']['tags']
    assert not any(t['value']=='raise_hand' for c in evidence['query']['concepts'] for t in c['tags'])
    # Full-body condition checks use the same tagged corpus as text retrieval.
    assert adapter.core_evidence(q,['one'])['one']['object_state']=='match'


def test_api_keeps_condition_evidence_and_ordering_metadata():
    from api.models import DynamicSearchOut
    from tests.test_fullbody_routing import decision,evidence
    e=engine();out=e.search(decision(),evidence())
    out['condition_ranking']='held_object_and_relations_then_weighted_rrf'
    conditions=candidate_conditions(prepare_pose_query(query(),LEX),record('한 손으로 휴대폰을 들고 보기',['phone']),LEX)
    out['selected'][0]['core_conditions']=conditions
    encoded=DynamicSearchOut(**out).model_dump(mode='json')
    assert encoded['selected'][0]['core_conditions']['object_state']=='match'
    assert encoded['condition_ranking']==out['condition_ranking']


def test_laterality_uses_object_specific_bound_metadata_not_filename():
    q=query();q.held_object_hands={'phone':'left'}
    p=prepare_pose_query(q,LEX)
    r=record('한 손으로 휴대폰을 봄',['phone'])
    r['meta']={'bvh_sha256':'bound','is_mirror':True,
               'body_parts':{'hands':{'left':'phone','right':'flat'}}}
    evidence=candidate_conditions(p,r,LEX,{'source_sha256':'bound'})
    assert evidence['relations']['held_hand:phone']['status']=='match'
    # A mirror's metadata is already swapped: do not flip its left hand again.
    r['meta']['body_parts']['hands']={'left':'flat','right':'phone'}
    assert candidate_conditions(p,r,LEX,{'source_sha256':'bound'})['relations']['held_hand:phone']['status']=='conflict'
    assert candidate_conditions(p,r,LEX,{'source_sha256':'stale'})['relations']['held_hand:phone']['status']=='unknown'
    q.held_object_hands={'phone':'unknown'}
    assert 'held_hand:phone' not in candidate_conditions(prepare_pose_query(q,LEX),r,LEX)['relations']


def test_correct_side_survives_family_collapse_even_if_other_side_has_higher_score():
    e=engine();e.meta['a']['pose_family_id']=e.meta['b']['pose_family_id']='same'
    q=query();q.held_object_hands={'phone':'left'};p=prepare_pose_query(q,LEX)
    conditions={}
    for pid,side in [('a','right'),('b','left')]:
        r=record('한 손으로 휴대폰을 봄',['phone'])
        r['meta']={'bvh_sha256':'ok','body_parts':{'hands':{side:'phone'}}}
        conditions[pid]=candidate_conditions(p,r,LEX,{'source_sha256':'ok'})
    branches={'semantic':{'status':'ok','hits':[{'pose_id':'a','score':.9},{'pose_id':'b','score':.7}]}}
    assert e._fuse(branches,conditions)[0]['pose_id']=='b'


def test_description_to_query_preserves_sides_and_parser_rejects_screen_coordinates():
    from src.search_routing import PoseDescription
    from src.pose_query import validated_object_hands
    desc=PoseDescription(objects=['phone'],held_object_hands={'phone':'left'})
    assert SemanticQuery.from_description(desc).held_object_hands=={'phone':'left'}
    assert validated_object_hands({'phone':'screen_right','cup':'left'},['phone'])=={}
    from src.vlm.client import parse_pose_descriptions
    parsed=parse_pose_descriptions({'people':[{'visible':True,'objects':['phone'],
        'context_evidence':'스마트폰이 보임','held_object_hands':{'phone':'left','cup':'right'}}]},1)[0]
    assert parsed.held_object_hands=={'phone':'left'}
