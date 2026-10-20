"""Frame transport for authored limbs; independent of Blender and recipes."""

import numpy as np
from scipy.spatial.transform import Rotation


def distributed_spine_frames(pelvis, pitch_degrees):
    """Share a requested torso bend across three joints, keeping chest intent."""
    if not np.isfinite(pitch_degrees) or abs(pitch_degrees) > 60:
        raise ValueError("authored torso pitch must stay within 60 degrees")
    return {
        name: pelvis
        @ Rotation.from_euler("X", pitch_degrees * fraction, degrees=True).as_matrix()
        for name, fraction in [("Spine", 1 / 3), ("Spine1", 2 / 3), ("Spine2", 1.0)]
    }


def transported_rotation(parent, rest_edge, target_edge, align):
    """Bend a parent's transported frame instead of restarting world roll."""
    return align(parent @ rest_edge, target_edge) @ parent


def anatomical_arm_frames(
    upper_edge, lower_edge, side, align, fallback, rest_upper=None, rest_lower=None
):
    """Place a T-rest arm's +Z flexion axis in its measured elbow plane.

    Resolves axial roll without moving shoulder, elbow or wrist. Near-straight
    limbs retain the supplied stable frame because their bend plane is noisy.
    """
    upper = np.asarray(upper_edge, dtype=float)
    upper = upper / np.linalg.norm(upper)
    lower = np.asarray(lower_edge, dtype=float)
    lower = lower / np.linalg.norm(lower)
    forward = lower - upper * (lower @ upper)
    if np.linalg.norm(forward) < np.sin(np.deg2rad(5)):
        return fallback, align(upper, lower) @ fallback
    forward /= np.linalg.norm(forward)
    x = upper * (1 if side == "Left" else -1)
    target_frame = np.column_stack((x, np.cross(forward, x), forward))
    sign = 1 if side == "Left" else -1
    rest_upper = (
        np.array([sign, 0.0, 0.0])
        if rest_upper is None
        else np.asarray(rest_upper, dtype=float)
    )
    rest_lower = (
        np.array([sign, 0.0, 0.0])
        if rest_lower is None
        else np.asarray(rest_lower, dtype=float)
    )
    rest_x = rest_upper / np.linalg.norm(rest_upper) * sign
    rest_z = np.array([0.0, 0.0, 1.0])
    rest_z -= rest_x * (rest_z @ rest_x)
    rest_z /= np.linalg.norm(rest_z)
    rest_frame = np.column_stack((rest_x, np.cross(rest_z, rest_x), rest_z))
    upper_frame = target_frame @ rest_frame.T
    return upper_frame, align(upper_frame @ rest_lower, lower) @ upper_frame


def foot_rotation(shin, setting):
    """Follow airborne shins; bounded correction toward planted foot targets.

    Limits are conservative authoring guards, not a clinical range-of-motion
    model. A contact target outside the guard needs a different leg pose.
    """
    mode = setting.get("mode", "follow_shin")
    pitch = float(setting.get("pitch_degrees", 0))
    if abs(pitch) > 35:
        raise ValueError("authored ankle pitch must stay within 35 degrees")
    local = Rotation.from_euler("X", pitch, degrees=True).as_matrix()
    if mode == "follow_shin":
        return shin @ local, {
            "mode": mode,
            "requested_degrees": abs(pitch),
            "applied_degrees": abs(pitch),
        }
    if mode != "planted":
        raise ValueError("foot mode must be follow_shin or planted")
    target = (
        Rotation.from_euler(
            "Y", setting.get("yaw_degrees", 0), degrees=True
        ).as_matrix()
        @ local
    )
    vector = Rotation.from_matrix(shin.T @ target).as_rotvec()
    angle = np.linalg.norm(vector)
    limit = np.deg2rad(45)
    fraction = min(1.0, limit / max(angle, 1e-12))
    return shin @ Rotation.from_rotvec(vector * fraction).as_matrix(), {
        "mode": mode,
        "requested_degrees": round(float(np.rad2deg(angle)), 3),
        "applied_degrees": round(float(np.rad2deg(angle * fraction)), 3),
        "contact_limited": bool(angle > limit),
    }
