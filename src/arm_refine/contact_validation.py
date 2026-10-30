"""Admission for offline contact probes; solver completion alone is not success."""


def contact_is_usable(trial, protected_matrix_delta=0):
    kin=trial.get('kinematics') or {}
    return bool(
        trial.get('new_skin') == []
        and 0 <= kin.get('elbow_flexion_deg',999) < 165
        and 0 <= kin.get('wrist_axis_angle_deg',999) <= 65
        and 0 <= kin.get('wrist_goal_error_m',float('inf')) < 2e-5
        and 0 <= kin.get('bone_length_error_m',float('inf')) < 2e-5
        and protected_matrix_delta < 2e-5
    )


def blockout_is_usable(receipt, require_roles=True):
    return bool(
        receipt.get('min_safety_m',-1) >= -.0005
        and receipt.get('max_bone_error_m',1) < .0001
        and receipt.get('joint_limit_min_dot_margin',-1) >= -1e-5
        and receipt.get('gesture_axis_min_margin',-1) >= -1e-5
        and (not require_roles or (receipt.get('roles_m') and min(receipt['roles_m']) >= -.0005))
    )
