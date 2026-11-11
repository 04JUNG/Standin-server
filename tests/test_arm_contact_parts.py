"""Observed errors: wrong contact family, floating palm, independent fold apply."""
import pytest
from src.arm_refine.contact_parts import select_contact_part, admission_reasons


def valid_metrics():
    return dict(patch_median_m=.003, patch_p90_m=.007, normal_error_deg=5, hand_frame_error_deg=0,
                wrist_axis_deg=35, wrist_swing_deg=35, wrist_twist_deg=15,
                elbow_flexion_deg=120, bone_length_error_m=1e-7,
                wrist_goal_error_m=1e-7, protected_matrix_delta=0,
                elbow_rule_margin_m=.02, new_skin=[], patch_samples=30)


def test_head_side_does_not_substitute_back_of_neck():
    assert select_contact_part('머리 옆에 손','left').target == 'head_side'
    assert select_contact_part('뒷목 잡기','left').target == 'neck_back'
    assert select_contact_part('이마에 손','left').target == 'forehead_front'
    with pytest.raises(ValueError): select_contact_part('머리 아무데나','left')


def test_paired_fold_cannot_enter_independent_apply():
    for query in ['팔짱','fold_arms','양손으로 턱']:
        with pytest.raises(ValueError): select_contact_part(query,'left')


def test_near_fingertips_do_not_certify_floating_palm():
    part = select_contact_part('head_side','right')
    good = valid_metrics()
    assert admission_reasons(good,part) == []
    assert 'patch_median_m' in admission_reasons({**good,'patch_median_m':.028},part)
    assert 'patch_p90_m' in admission_reasons({**good,'patch_p90_m':.025},part)


def test_clear_skin_does_not_override_bad_contact_frame_or_elbow():
    part = select_contact_part('턱 받침','right'); good = valid_metrics()
    for field,value in [('normal_error_deg',85),('wrist_twist_deg',90),('elbow_rule_margin_m',-.01)]:
        assert admission_reasons({**good,field:value},part)
    assert admission_reasons({**good,'new_skin':[{'max_depth_mm':3}]},part)
    assert admission_reasons({**good,'patch_median_m':float('nan')},part)
    assert admission_reasons({},part)
