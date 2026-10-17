import pytest
from src.experimental.automatic_output_scope import choose_output_scope


def person():
    return dict(bbox=[0,0,500,1000], keypoints=[[200,300]]*17, valid_mask=[True]*17)


def test_visible_gesture_is_not_amputated_by_bust_label():
    result=choose_output_scope(person(),dict(body_scope='bust',visible_regions=['head','hands']),[1000,1000])
    assert result['output_scope']=='half'
    assert result['raw_detected_scope']=='bust'


def test_face_framing_keeps_requested_shoulder_base():
    result=choose_output_scope(person(),dict(body_scope='head',visible_regions=['head','neck']),[1000,1000])
    assert result['output_scope']=='bust'


def test_full_body_and_observed_legs_are_preserved():
    assert choose_output_scope(person(),dict(body_scope='full',visible_regions=[]),[1000,1000])['output_scope']=='full'
    result=choose_output_scope(person(),dict(body_scope='half',visible_regions=['knees']),[1000,1000])
    assert result['output_scope']=='full' and result['dynamic_scope_source']=='observed_legs'


def test_fake_body_points_on_face_do_not_invent_full_body():
    result=choose_output_scope(person(),dict(body_scope='head',visible_regions=['head']),[1000,1000])
    assert result['output_scope']=='bust' and result['observed_legs']==[]


def test_unknown_or_neighbour_scope_fails_closed():
    with pytest.raises(ValueError): choose_output_scope(person(),{},[1000,1000])
    with pytest.raises(ValueError): choose_output_scope(person(),dict(body_scope='full',head_center=[900,200]),[1000,1000])
