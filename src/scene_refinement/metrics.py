"""Shared automatic policy; no cut IDs, feedback labels or per-actor overrides."""

import re, math
import numpy as np

YAW = {
    "front": 0.0,
    "front_left": -45.0,
    "left": -90.0,
    "back_left": -135.0,
    "back": 180.0,
    "back_right": 135.0,
    "right": 90.0,
    "front_right": 45.0,
}


def wrap(x):
    return (x + 180) % 360 - 180


def direction_policy(a):
    a = dict(a or {})
    notes = []
    body = a.get("torso_heading", "unknown")
    head = a.get("head_heading", "unknown")
    surface = a.get("torso_surface")
    # Surface is the explicit anatomy judgment, independent of face direction.
    if surface == "back" and body in ("front", "front_left", "front_right"):
        body = "back"
        notes.append("torso_surface_heading_conflict")
    if surface == "front" and body in ("back", "back_left", "back_right"):
        body = "front"
        notes.append("torso_surface_heading_conflict")
    if body not in YAW:
        body = "front"
        notes.append("unknown_torso_front_reference")
    evidence = str(a.get("head_evidence", "")).lower()
    # Resolve explicit profile evidence contradicting a frontal categorical answer.
    if head == "front" and "profile" in evidence:
        if re.search(
            r"(?:looking|facing|directed|angled|toward)\s+(?:the\s+)?left", evidence
        ):
            head = "left"
            notes.append("head_label_reconciled_with_profile_evidence")
        elif re.search(
            r"(?:looking|facing|directed|angled|toward)\s+(?:the\s+)?right", evidence
        ):
            head = "right"
            notes.append("head_label_reconciled_with_profile_evidence")
    if head not in YAW:
        head = body
        notes.append("unknown_head_follows_torso_reference")
    if (
        body in ("back", "back_left", "back_right")
        and head in ("left", "right")
        and abs(wrap(YAW[head] - YAW[body])) >= 90
    ):
        body = "back_" + head
        notes.append("back_surface_with_profile_face_oblique_torso_prior")
    delta = wrap(YAW[head] - YAW[body])
    limited = float(np.clip(delta, -85, 85))
    if abs(delta - limited) > 1e-3:
        notes.append("head_relative_yaw_anatomical_limit")
    return dict(
        torso_label=body,
        torso_camera_yaw=YAW[body],
        head_label=head,
        head_camera_yaw=YAW[head],
        head_relative_yaw=limited,
        head_pitch_degrees={"up": 15.0, "down": -20.0}.get(a.get("head_pitch"), 0.0),
        notes=notes,
    )


def scope_policy(a):
    regions = set(a.get("visible_regions") or [])
    scope = a.get("body_scope")
    notes = []
    if regions & {"thighs", "knees", "shins", "feet"}:
        scope = "full"
        notes.append("visible_leg_regions_require_full")
    elif regions & {"forearms", "hands", "hips"} and scope in (
        "head",
        "bust",
        "unknown",
        None,
    ):
        scope = "half"
        notes.append("visible_forearms_or_hips_require_half")
    if scope == "head":
        scope = "bust"
        notes.append("minimum_neck_shoulder_reference")
    if scope not in ("full", "half", "bust"):
        scope = "half" if regions & {"hands", "forearms"} else "bust"
        notes.append("unknown_scope_reference")
    return scope, notes


def measures(points):
    p = np.asarray(points, float)

    def angle(a, b, c):
        x, y = a - b, c - b
        return float(
            np.degrees(
                np.arccos(
                    np.clip(
                        x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-9), -1, 1
                    )
                )
            )
        )

    knees = [angle(p[a], p[b], p[c]) for a, b, c in ((11, 13, 15), (12, 14, 16))]
    leg = (
        sum(
            np.linalg.norm(p[a] - p[b]) + np.linalg.norm(p[b] - p[c])
            for a, b, c in ((11, 13, 15), (12, 14, 16))
        )
        / 2
    )
    hips = (p[11] + p[12]) / 2
    shoulder = (p[5] + p[6]) / 2
    torso = max(np.linalg.norm(shoulder - hips), 1e-6)
    height = (hips[1] - (p[15, 1] + p[16, 1]) / 2) / max(leg, 1e-6)
    thigh_horizontal = np.mean(
        [
            abs(p[a, 1] - p[b, 1]) / max(np.linalg.norm(p[a] - p[b]), 1e-6)
            for a, b in ((11, 13), (12, 14))
        ]
    )
    head_hand = min(np.linalg.norm(p[0] - p[9]), np.linalg.norm(p[0] - p[10])) / torso
    return dict(
        knee_min=min(knees),
        knee_max=max(knees),
        hip_height_leg=height,
        thigh_vertical_fraction=float(thigh_horizontal),
        foot_height_difference=abs(p[15, 1] - p[16, 1]) / max(leg, 1e-6),
        head_hand_torso=head_hand,
    )


def posture_cost(m, posture, leg_action):
    if posture == "standing":
        # Actual joint geometry, not broad library tags, excludes kicks and crouches.
        okay = m["knee_min"] >= 140 and m["hip_height_leg"] >= 0.72
        if leg_action not in ("one_leg_raised", "crossed"):
            okay = okay and m["foot_height_difference"] < 0.20
        return (
            (0.0 if okay else 100.0)
            + max(0, 165 - m["knee_min"]) / 100
            + max(0, 0.85 - m["hip_height_leg"])
        )
    if posture in ("sitting", "bent_over_surface"):
        okay = (
            45 <= m["knee_min"] <= 140
            and m["thigh_vertical_fraction"] < 0.75
            and m["hip_height_leg"] < 0.84
        )
        return (
            (0.0 if okay else 100.0)
            + abs(m["knee_min"] - 95) / 180
            + m["thigh_vertical_fraction"] * 0.3
        )
    if posture == "kneeling":
        return abs(m["knee_min"] - 70) / 100
    if posture == "crouching":
        return abs(m["knee_min"] - 65) / 100 + abs(m["hip_height_leg"] - 0.35)
    return 0.0
