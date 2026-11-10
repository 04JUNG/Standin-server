"""Conservative correction of swapped COCO elbow and wrist labels."""

import numpy as np

_ARMS = {"left": (5, 7, 9, 11), "right": (6, 8, 10, 12)}
_ANCHORS = [5, 6, 11, 12]


def find_elbow_wrist_swap(keypoints, joint_mask, description):
    """Return a correction only when one arm clearly fits hand-on-hip contact."""
    states = description.get("arm_states") or {}
    waist_sides = {side for side in _ARMS if states.get(side) == "waist"}
    one_hand_tag = "한 손 허리" in (description.get("body_details") or "").split(" · ")
    # The default description contract has per-arm states; the compact
    # contract can also report the contact without guessing its side.
    if len(waist_sides) > 1 or (not one_hand_tag and not waist_sides):
        return None

    points = np.asarray(keypoints, dtype=float)
    mask = np.asarray(joint_mask, dtype=bool)
    if points.shape != (17, 2) or mask.shape != (17,):
        return None
    if not mask[_ANCHORS].all() or not np.isfinite(points[_ANCHORS]).all():
        return None
    torso = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
    if torso < 1:
        return None

    eligible = []
    for side, (shoulder, elbow, wrist, hip) in _ARMS.items():
        if states.get(side) not in (None, "unknown", "waist"):
            continue
        if waist_sides and side not in waist_sides:
            continue
        if not mask[[elbow, wrist]].all() or not np.isfinite(points[[elbow, wrist]]).all():
            continue
        opposite_hip = 12 if hip == 11 else 11
        hip_axis = points[hip] - points[opposite_hip]
        width = np.linalg.norm(hip_axis)
        if width < 1:
            continue
        old_gap = np.linalg.norm(points[wrist] - points[hip]) / torso
        new_gap = np.linalg.norm(points[elbow] - points[hip]) / torso
        flare = np.dot(points[wrist] - points[elbow], hip_axis / width) / torso
        upper = points[shoulder] - points[wrist]
        lower = points[elbow] - points[wrist]
        upper_len, lower_len = np.linalg.norm(upper), np.linalg.norm(lower)
        if min(upper_len, lower_len) < .25 * torso or max(upper_len, lower_len) > 1.15 * torso:
            continue
        cosine = np.dot(upper, lower) / (upper_len * lower_len)
        bend = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        if old_gap >= .55 and new_gap <= .35 and old_gap - new_gap >= .30 and flare >= .20 and 35 <= bend <= 140:
            eligible.append(dict(side=side, elbow=elbow, wrist=wrist,
                                 hip_distance_before=round(float(old_gap), 3),
                                 hip_distance_after=round(float(new_gap), 3),
                                 bend_degrees=round(float(bend), 1)))
    return eligible[0] if len(eligible) == 1 else None
