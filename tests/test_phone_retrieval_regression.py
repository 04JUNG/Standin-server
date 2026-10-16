import numpy as np
import pytest

from src.semantic_relevance import relevance_constraints
from src.semantic_body_index import describe_library_body
from src.library import _synthetic_pose

QUERY = '엎드려 두 손으로 휴대폰을 턱 밑에 쥐고 웃는 자세'


@pytest.mark.parametrize('description', [
    '엎드려 팔꿈치로 상체를 받치고 휴대폰을 보며 다리를 든 자세',
    '엎드려 양손으로 휴대폰을 보는 자세',
    '엎드려 휴대폰을 보고 한 손으로 베개를 받친 자세',
])
def test_unknown_or_matching_hand_count_is_not_a_contradiction(description):
    assert 'twohand_phone_not_onehand' not in relevance_constraints(QUERY, description, set())


@pytest.mark.parametrize('description', [
    '엎드려 한 손으로 휴대폰을 보는 자세',
    '엎드려 휴대폰을 오른손으로 쥐는 자세',
])
def test_explicit_one_hand_phone_still_conflicts(description):
    assert 'twohand_phone_not_onehand' in relevance_constraints(QUERY, description, set())


def test_supine_is_not_made_eligible_by_unknown_hand_count():
    assert 'prone_not_generic_lying' in relevance_constraints(QUERY, '등을 대고 누워 휴대폰 보기', set())


@pytest.mark.parametrize('state', ['lowered', 'raised', 'spread', 'unknown'])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('degrees', [45, 90, 180])
def test_arm_state_uses_body_frame_under_pitch_roll_yaw(state, axis, degrees):
    p = _synthetic_pose('standing')
    for sign, sh, elbow, hand in [(-1, 5, 7, 9), (1, 6, 8, 10)]:
        if state == 'raised':
            p[elbow], p[hand] = p[sh]+[0,.15,0], p[sh]+[0,.4,0]
        elif state == 'spread':
            p[elbow], p[hand] = p[sh]+[sign*.3,0,0], p[sh]+[sign*.6,0,0]
        elif state == 'unknown':
            p[elbow], p[hand] = p[sh]+[0,-.05,.2], p[sh]+[0,-.03,.4]
    a = np.deg2rad(degrees)
    r = np.eye(3)
    i, j = [x for x in range(3) if x != axis]
    r[i,i] = r[j,j] = np.cos(a)
    r[i,j], r[j,i] = -np.sin(a), np.sin(a)
    result = describe_library_body(p @ r.T * 3 + [100,200,-50], np.ones(17))
    assert result['observations']['arms'] == state
