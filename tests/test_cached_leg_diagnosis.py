import copy
import pytest
from src.experimental.cached_leg_diagnosis import targets_from_cached, diagnose


def description(**updates):
    value = {'visible': True, 'visible_scope': 'full', 'certainty': 'high',
             'body_details': '', 'context_evidence': '', 'laterality_review': {}}
    value.update(updates)
    return value


def state(left=100., right=115.):
    return {'legs': {'left': {'valid': True, 'flexion_deg': left, 'knee_above_other_torso': .85},
                     'right': {'valid': True, 'flexion_deg': right, 'knee_above_other_torso': -.85}}}


def test_explicit_knees_are_not_discarded_by_empty_review_or_claim_visible_chain():
    target = targets_from_cached(description(context_evidence='오른쪽 무릎과 발이 바닥에 닿아 있고 왼쪽 무릎이 높게 올라와 있다.'))
    assert target['legs']['left']['action'] == 'raised_knee'
    assert target['legs']['right']['action'] == 'kneeling_ground'
    assert all(not row['chain_visible'] for row in target['legs'].values())
    assert all(row['laterality_source'] == 'explicit_natural_language' for row in target['legs'].values())
    result = diagnose(target, state())
    assert result['candidate_sides'] == []
    assert result['legs'][0]['status'] == 'match'
    assert result['legs'][1]['status'] == 'shape_matched_contact_held'
    assert result['legs'][1]['support_status'] == 'held'
    assert not result['auto_apply_allowed']


def test_explicit_bilateral_extension_does_not_require_named_left_and_right():
    target = targets_from_cached(description(body_details='의자에 앉아 두 다리를 책상 위에 올려 뻗은 모습이다.'))
    assert all(row['action'] == 'straight_leg' for row in target['legs'].values())
    assert all(row['support_surface'] == 'table' for row in target['legs'].values())
    assert all(row['laterality_source'] == 'explicit_bilateral_natural_language' for row in target['legs'].values())
    result = diagnose(target, state(34., 48.))
    assert result['candidate_sides'] == ['left', 'right']
    assert all(row['status'] == 'mismatch' and not row['auto_apply_allowed'] for row in result['legs'])
    assert target['relations'] == []  # Cross order is absent, never copied from a later human correction.


@pytest.mark.parametrize('scope', ['bust', 'half', 'head', None])
def test_upper_body_descriptions_do_not_create_leg_goals(scope):
    target = targets_from_cached(description(visible_scope=scope, body_details='양다리를 곧게 편다'))
    assert all(row['action'] == 'unknown' for row in target['legs'].values())
    assert diagnose(target, state())['candidate_sides'] == []


@pytest.mark.parametrize('sentence', ['화면 왼쪽 다리를 뻗는다', '왼다리를 펴지 않는다',
                                      '오른무릎이 높게 올라온 듯하다', '한쪽 다리를 뻗는다'])
def test_uncertain_screen_or_unassigned_side_stays_held(sentence):
    target = targets_from_cached(description(body_details=sentence))
    assert all(row['action'] == 'unknown' for row in target['legs'].values())


def test_arm_contact_target_is_not_a_raised_knee_observation():
    target = targets_from_cached(description(body_details='왼손을 왼쪽 무릎 위에 올리고 오른팔은 내린다.'))
    assert all(row['action'] == 'unknown' for row in target['legs'].values())


def test_explicit_abstention_and_conflicting_actions_are_held():
    review = {'status': 'reviewed', 'changes': [{'status': 'abstain', 'action': 'raised_knee'}]}
    target = targets_from_cached(description(body_details='왼무릎을 높게 세운다', laterality_review=review))
    assert target['legs']['left']['confidence'] == 'low'
    assert diagnose(target, state())['candidate_sides'] == []
    target = targets_from_cached(description(body_details='왼다리를 곧게 편다', context_evidence='왼다리를 굽힌다'))
    assert target['legs']['left']['reasons'] == ['contradictory_explicit_leg_actions']


def test_crossed_pair_never_splits_into_independent_extension():
    target = targets_from_cached(description(body_details='두 다리를 뻗고 발목을 교차한다'))
    assert target['relations'][0]['top_side'] == 'unknown'
    result = diagnose(target, state(45., 45.))
    assert result['candidate_sides'] == []
    assert all(row['status'] == 'held' for row in result['legs'])


def test_missing_or_nonfinite_native_geometry_is_not_a_match():
    target = targets_from_cached(description(body_details='양다리를 곧게 편다'))
    value = state(float('nan'), 12.)
    result = diagnose(target, value)
    assert result['legs'][0]['shape_status'] == 'held'
    assert result['legs'][1]['shape_status'] == 'match'
    assert result['candidate_sides'] == []


def test_adapter_does_not_mutate_cached_evidence_or_native_state():
    evidence = description(context_evidence='왼무릎이 높게 올라와 있다')
    before = copy.deepcopy(evidence)
    value = state()
    state_before = copy.deepcopy(value)
    assert diagnose(targets_from_cached(evidence), value)
    assert evidence == before and value == state_before


def test_arm_specific_abstention_does_not_erase_explicit_leg_observation():
    target = targets_from_cached(description(context_evidence='왼무릎이 높게 올라와 있다',
        laterality_review={'status': 'reviewed', 'changes': [{'action': 'touch_face', 'status': 'abstain'}]}))
    assert target['legs']['left']['action'] == 'raised_knee'
    assert target['legs']['left']['confidence'] == 'high'
    assert target['legs']['left']['chain_visible'] is False


def test_upright_standing_is_not_an_explicit_straight_knee_goal():
    target = targets_from_cached(description(body_details='두 다리를 앞뒤로 딛고 곧게 서 있는 모습이다.'))
    assert all(row['action'] == 'unknown' for row in target['legs'].values())
    assert diagnose(target, state(22., 27.))['candidate_sides'] == []


def test_staggered_crossed_legs_require_a_coupled_checker():
    target = targets_from_cached(description(body_details='두 다리를 엇갈려 뻗고 있다.'))
    assert target['relations'][0]['requires_atomic_relation_checker']
    assert diagnose(target, state(45., 45.))['candidate_sides'] == []


def test_crossed_arms_do_not_hold_an_independent_bilateral_leg_goal():
    target = targets_from_cached(description(context_evidence='두 팔이 가슴 앞에서 교차하여 팔짱을 이루고 양다리가 곧게 땅을 짚고 서 있음.'))
    assert target['relations'] == []
    assert all(x['action'] == 'straight_leg' for x in target['legs'].values())


def test_dataset_mapping_preserves_missing_person_without_p1_substitution():
    from src.experimental.cached_limb_inventory import match_cached_people
    row = {'expected_person_ids': ['X-P1', 'X-P2'], 'source_person_boxes': [
        {'reference_person_id': 'X-P1', 'box': [0,0,10,20]},
        {'reference_person_id': 'X-P2', 'box': [30,0,40,20]}]}
    # The cache's only person is P2 even though its current response index is0.
    result = match_cached_people(row, [{'index':0,'box':[30,0,40,20]}])
    assert result['X-P1']['status'] == 'held'
    assert result['X-P2']['cache_person_index'] == 0
