"""Probe a conservative, image-independent elbow/wrist swap on saved roughs.

The rule consumes only query keypoints and an existing Luna pose tag. Gold IDs
are used separately to measure retrieval after the decision."""

import numpy as np

ARMS = {'left': (5, 7, 9, 11), 'right': (6, 8, 10, 12)}


def hand_on_hip(description):
    tags = (description.get('body_details') or '').split(' · ')
    return '한 손 허리' in tags


def elbow_wrist_swap(points, mask, description):
    """Return a swap only when one arm clearly fits the observed contact."""
    if not hand_on_hip(description):
        return None
    points = np.asarray(points, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if points.shape != (17, 2) or mask.shape != (17,):
        return None
    anchors = [5, 6, 11, 12]
    if not mask[anchors].all() or not np.isfinite(points[anchors]).all():
        return None
    torso = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
    if torso < 1:
        return None

    states = description.get('arm_states') or {}
    eligible = []
    for side, (shoulder, elbow, wrist, hip) in ARMS.items():
        if states.get(side) not in (None, 'unknown', 'waist'):
            continue
        if not mask[[elbow, wrist]].all() or not np.isfinite(points[[elbow, wrist]]).all():
            continue
        opposite_hip = 12 if hip == 11 else 11
        hip_axis = points[hip] - points[opposite_hip]
        width = np.linalg.norm(hip_axis)
        if width < 1:
            continue
        hip_axis /= width

        old_gap = np.linalg.norm(points[wrist] - points[hip]) / torso
        new_gap = np.linalg.norm(points[elbow] - points[hip]) / torso
        flare = np.dot(points[wrist] - points[elbow], hip_axis) / torso
        upper = points[shoulder] - points[wrist]
        lower = points[elbow] - points[wrist]
        upper_len = np.linalg.norm(upper)
        lower_len = np.linalg.norm(lower)
        if min(upper_len, lower_len) < .25 * torso or max(upper_len, lower_len) > 1.15 * torso:
            continue
        bend = np.degrees(np.arccos(np.clip(np.dot(upper, lower) / (upper_len * lower_len), -1, 1)))
        if old_gap >= .55 and new_gap <= .35 and old_gap-new_gap >= .30 and flare >= .20 and 35 <= bend <= 140:
            eligible.append(dict(side=side, elbow=elbow, wrist=wrist,
                                 hip_distance_before=round(float(old_gap), 3),
                                 hip_distance_after=round(float(new_gap), 3),
                                 bend_degrees=round(float(bend), 1)))
    return eligible[0] if len(eligible) == 1 else None
