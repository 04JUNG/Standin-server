"""Regressions from observed authoring failures, independent of solver status."""
from src.arm_refine.contact_validation import contact_is_usable,blockout_is_usable


def test_contact_without_skin_collision_still_rejects_broken_wrist():
    trial={'new_skin':[],'kinematics':{'elbow_flexion_deg':148.3,'wrist_axis_angle_deg':99.4}}
    assert not contact_is_usable(trial)


def test_contact_rejects_new_penetration_and_nonselected_motion():
    trial={'new_skin':[],'kinematics':{'elbow_flexion_deg':112.6,'wrist_axis_angle_deg':53.2,
                                    'wrist_goal_error_m':1e-7,'bone_length_error_m':1e-7}}
    assert contact_is_usable(trial)
    assert not contact_is_usable(trial,.001)
    assert not contact_is_usable({**trial,'new_skin':[{'max_depth_mm':19.1}]})
    assert not contact_is_usable({'kinematics':trial['kinematics']})
    assert not contact_is_usable({'new_skin':[],'kinematics':{'elbow_flexion_deg':112.6,'wrist_axis_angle_deg':53.2}})


def test_optimizer_success_cannot_override_nine_mm_overlap():
    receipt={'solver_success':True,'min_safety_m':-.00948,'max_bone_error_m':0,
             'joint_limit_min_dot_margin':.01,'gesture_axis_min_margin':.01,'roles_m':[.01]}
    assert not blockout_is_usable(receipt)


def test_nonoverlap_does_not_approve_wrong_hand_route_or_bone_length():
    receipt={'min_safety_m':.003,'max_bone_error_m':0,'joint_limit_min_dot_margin':0,
             'gesture_axis_min_margin':0,'roles_m':[.01]}
    assert blockout_is_usable(receipt)
    assert not blockout_is_usable({**receipt,'roles_m':[-.02]})
    assert not blockout_is_usable({**receipt,'max_bone_error_m':.15})
