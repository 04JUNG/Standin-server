"""Regression: umbrella is not neck contact; both hands carry explicit roles."""
import pytest
from src.arm_refine.relational_parts import plan_arms


def test_umbrella_and_phone_have_distinct_sides_targets_and_finger_shapes():
    p=plan_arms({'left':'어깨 옆 우산대를 잡는다','right':'휴대폰을 든다'},skeleton_scores=[0]*17)
    roles={m['side']:m['part'] for m in p['members']}
    assert roles['left']['target']=='external_vertical_shaft'
    assert roles['right']['hand_preset']=='hold_phone'
    assert roles['left']['hand_preset']=='grip'
    assert not p['image_skeleton_reliable'] and not p['image_skeleton_used_for_targets']
    assert p['geometry_source']=='destination_3d_skeleton_and_actual_skin'


def test_mirrored_language_preserves_anatomical_ownership():
    p=plan_arms({'right':'우산을 쥐고 있다','left':'스마트폰을 들고 있다'})
    assert {m['side']:m['part']['id'] for m in p['members']}=={'right':'umbrella_shaft_grip','left':'phone_hold'}


def test_knee_does_not_drop_the_other_lowered_hand():
    p=plan_arms({'left':'무릎 위에 손을 올린다','right':'손을 아래로 내린다'})
    assert [m['part']['id'] for m in p['members']]==['raised_knee_rest','lowered_hand']
    assert all('fingers' in m['change'] for m in p['members'])


def test_back_of_hand_up_near_knee_is_not_knee_support():
    p=plan_arms({'left':'무릎 위에 손을 올린다',
                 'right':'손등이 위로 가게 주먹을 쥐고 손을 무릎 쪽 가까이 아래로 내린다'})
    assert p['members'][1]['part']['id']=='lowered_hand'
    assert p['members'][1]['part']['orientation_rule']=='hand_back_up_and_nearer_knee'
    p=plan_arms({'left':'엄지가 위로 가게 우산대를 잡는다','right':'휴대폰을 든다'})
    assert p['members'][0]['part']['orientation_rule']=='thumb_side_up'


@pytest.mark.parametrize('clauses',[
 {'left':'우산을 잡는다'},
 {'left':'우산과 휴대폰을 잡는다','right':'휴대폰을 든다'},
 {'left':'뒷목에 손','right':'휴대폰을 든다'},
 {'left':'무릎 위에 손','right':'그대로'},
 {'left':'오른손으로 우산을 잡는다','right':'휴대폰을 든다'},
 {'left':'우산을 잡지 않는다','right':'휴대폰을 든다'},
])
def test_missing_or_ambiguous_roles_do_not_silently_pick_parts(clauses):
    with pytest.raises(ValueError):plan_arms(clauses)
