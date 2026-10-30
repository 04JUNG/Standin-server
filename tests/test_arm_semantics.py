import copy
import pytest
from src.arm_refine.semantics import pair_quality,parse_query,search_text,tag_part


def measurement():
    return {'broad_geometry':{'fold_geometry':True},'palm_upperarm_skin':{s:{'distance_mm_p10':5,'within_12mm_fraction':.3} for s in ['left','right']},'anatomy':{'flags':[]},'skin_collisions':{'flags':[]}}


def part(label='fold_contact_pair',selection='both',quality=None):
    p={'id':label+'--'+selection,'label':label,'selection':selection,'relation':'paired' if selection=='both' else 'independent',
       'arms':{s:{'wrist_position_torso':[0,.2,0],'elbow_flexion_deg':90} for s in (['left','right'] if selection=='both' else [selection])}}
    p['semantics']=tag_part(p,measurement=quality)
    return p


def test_pairs_need_skin_not_name_or_crossing_alone():
    assert not pair_quality(None)['eligible_strict_fold']
    m=measurement();m['skin_collisions']['flags']=['hand inside opposite arm']
    assert not pair_quality(m)['eligible_strict_fold']
    m=measurement();m['palm_upperarm_skin']['left']['distance_mm_p10']=50
    assert not pair_quality(m)['eligible_strict_fold']
    assert pair_quality(measurement())['eligible_strict_fold']


@pytest.mark.parametrize('query',['팔짱','가슴 앞에서 팔짱을 낀다','folded arms','arms crossed'])
def test_fold_never_substitutes_guard_stack_or_fragment(query):
    good=part(quality=measurement());guard=part('guard_pair');stack=part('fold_tight_pair');fragment=part(selection='left',quality=measurement())
    result=search_text([guard,stack,fragment,good],query)
    assert [h['part_id'] for h in result['hits']]==[good['id']]


def test_held_candidate_remains_held_and_tucked_is_unavailable():
    p=part()
    assert search_text([p],'팔짱')['status']=='held_for_pair_quality'
    assert search_text([p],'팔짱')['hits']==[]
    assert search_text([p],'팔짱',include_held=True)['hits'][0]['quality_status']=='unmeasured'
    assert search_text([part(quality=measurement())],'손을 안쪽에 숨긴 팔짱')['hits']==[]


def test_negation_and_subject_relative_side():
    assert parse_query('팔짱 말고 오른손을 들어')['exclude_fold']
    assert parse_query('왼팔 위로 뻗기')['selection']=='left'
    assert parse_query('팔짱을 풀고 왼손 올리기')['exclude_fold']
    assert search_text([part('guard_pair'),part(quality=measurement())],'팔짱 아닌 양팔 방어')['hits'][0]['action']=='cross_guard'


def test_selected_low_arm_does_not_claim_source_head_action():
    p=part('head_touch','left');p['arms']['left']['wrist_position_torso'][1]=-1
    p['semantics']=tag_part(p)
    assert search_text([p],'왼손 머리 짚기')['hits']==[]


def test_missing_family_queries_reject_lexically_similar_other_actions():
    parts=[part('hip_both'),part('head_back'),part('chin_low'),part('phone_pair'),part('hands_behind_pair')]
    for query,action in [('양손 허리 짚기','hands_on_hips'),('양손 머리 뒤','head_back'),
                         ('턱 괴기','chin_support'),('양손 휴대폰','phone_pair'),('양손 등 뒤로','hands_behind')]:
        hits=search_text(parts,query)['hits']
        assert hits and {h['action'] for h in hits}=={action}


def test_both_hands_on_target_not_one_contact_plus_support():
    assert search_text([part('hip_offer_pair'),part('hip_both')],'양손 허리 짚기')['hits'][0]['label']=='hip_both'
    assert len(search_text([part('hip_offer_pair'),part('hip_both')],'양손 허리 짚기')['hits'])==1
    assert not search_text([part('chin_supported_pair')],'양손으로 턱 괴기')['hits']


def test_full_extension_requires_measured_elbow_angle():
    bent=part('reach_forward','right');straight=part('reach_full_forward','right')
    straight['arms']['right']['elbow_flexion_deg']=25
    assert [h['part_id'] for h in search_text([bent,straight],'오른팔 길게 앞으로 뻗기')['hits']]==[straight['id']]
