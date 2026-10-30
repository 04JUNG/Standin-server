import numpy as np
from src.pose_relations import validate_relations, reconcile_sides
from src.pose_query import prepare_pose_query, candidate_conditions
from src.search_routing import PoseDescription, RoutingConfig, assess_semantic_evidence, DescribeOutcome
from src.semantic_search import SemanticQuery
from tests.test_pose_query import LEX, record
from tests.test_fullbody_routing import engine, decision, evidence, Camera, Semantic


def relation(side='left',visible=True):
    return dict(kind='touch_head',side=side,chain_visible=visible,side_confidence='high',evidence='어깨까지 팔 연결이 보임')


def test_unconnected_side_abstains_even_when_vlm_confident():
    assert validate_relations([relation(visible=False)])[0]['side']=='unknown'
    assert validate_relations([relation(side='screen_right')])[0]['side']=='unknown'


def test_empty_hand_action_is_preserved_and_bvh_checks_side():
    q=SemanticQuery(text='한 손으로 머리를 짚음',body_details='왼손을 머리에 얹음',pose_relations=[relation()])
    prepared=prepare_pose_query(q,LEX)
    assert prepared['enabled'] and '왼손' in prepared['core_text']
    left={'left':{'near_head':True,'hand_height':.4},'right':{'near_head':False,'hand_height':-.7}}
    right={'left':left['right'],'right':left['left']}
    a=candidate_conditions(prepared,record('머리를 짚음'),LEX,measured=left)
    b=candidate_conditions(prepared,record('머리를 짚음'),LEX,measured=right)
    assert a['relations']['action:touch_head']['status']=='match'
    assert b['relations']['action:touch_head']['status']=='conflict'
    assert a['priority']<b['priority']


def test_watch_action_precedes_incidental_cup_and_does_not_certify_side():
    q=SemanticQuery(text='컵을 들고 손목 기기를 확인한다',objects=['cup'])
    p=prepare_pose_query(q,LEX)
    watch=candidate_conditions(p,record('손목을 들어 시계를 확인하기'),LEX)
    cup=candidate_conditions(p,record('컵을 들고 창밖을 봄',['cup']),LEX)
    assert watch['priority']<cup['priority']
    assert not watch['relations']['action:check_wrist']['side_verified']


def test_closeup_evidence_is_valid_without_arms_or_legs():
    d=PoseDescription(visible=True,certainty='medium',visible_scope='shoulder_head',head_view='profile_right',
        body_details='목과 어깨 위 얼굴만 보이고 코는 화면 오른쪽을 향함',search_text='오른쪽 옆얼굴과 어깨가 보이는 자세')
    assert assess_semantic_evidence(d,RoutingConfig()).valid
    d.visible=False
    assert not assess_semantic_evidence(d,RoutingConfig()).valid


def test_joint_conflict_abstains_does_not_overwrite_with_opposite_claim():
    d=PoseDescription(pose_relations=[relation()]);p=np.zeros((17,2));p[9]=[100,100];p[10]=[2,2]
    checks=reconcile_sides(d,p,np.ones(17,bool))
    assert checks and d.pose_relations[0]['side']=='unknown'


def test_cropped_empty_search_triggers_separate_validated_generation():
    e=engine(Camera(()),Semantic(()));ev=evidence();ev['output_scope']='half'
    result=e.search(decision(),ev)
    assert e.generator.calls==1 and result['generation']['status']=='preview_ready'
    assert result['selected']==[] and result['generation']['diagnostics']['unobserved_body_is_default']


def test_provider_error_never_triggers_generation():
    e=engine();r=e.search(decision(sufficient=False,text=False,error='upstream_error'),evidence())
    assert e.generator.calls==0 and r['status']=='error'


def test_transient_describe_failure_retries_once(monkeypatch):
    from src.vlm.client import GeminiVLMClient
    client=GeminiVLMClient(client=object(), sleep=lambda _:None, jitter=lambda *_:.5)
    results=iter([DescribeOutcome('error',error_reason='upstream_error',retryable=True),DescribeOutcome('ok')])
    monkeypatch.setattr(client,'_describe_people_once',lambda *args:next(results))
    assert client.describe_people(None,[]).attempts==2


def test_nonretryable_describe_error_stays_error(monkeypatch):
    from src.vlm.client import GeminiVLMClient
    client=GeminiVLMClient(client=object(), sleep=lambda _:None, jitter=lambda *_:.5)
    monkeypatch.setattr(client,'_describe_people_once',lambda *args:DescribeOutcome('error',retryable=False))
    assert client.describe_people(None,[]).attempts==1


def test_coarse_arm_raise_does_not_override_pullup():
    from src.pose_relations import relations_for_query
    r=relation();r['kind']='arm_raised'
    assert relations_for_query(SemanticQuery(actions=['pull_up'],pose_relations=[r]))==[]


def test_watch_side_requires_bound_authored_metadata():
    from src.pose_relations import relation_evidence
    rec={'description':'손목의 시계를 확인한다','meta':{'body_parts':{'active_hand':'left'}}}
    required=[{'kind':'check_wrist','side':'right'}]
    assert relation_evidence(required,rec)['action:check_wrist']['candidate'] is None
    assert relation_evidence(required,rec,bound=True)['action:check_wrist']['status']=='conflict'


def test_framing_reference_does_not_block_generation():
    from src.semantic_search import SemanticHit,SemanticSearchResult
    class Reference(Semantic):
        def search(self,query,top_k):
            return SemanticSearchResult('ok',[SemanticHit('a',1.,{'visible_region_reference':{'exact_head_tilt_matched':False}}, {})])
    e=engine(Camera(()),Reference())
    result=e.search(decision(False),evidence())
    assert result['status']=='no_result' and result['selected']==[]
    assert result['framing_references'][0]['kind']=='framing_reference'
    assert e.generator.calls==1
