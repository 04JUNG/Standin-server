"""Fit a pose using the joints that agree across independent crop estimates."""

from math import sqrt
import numpy as np
from .crop_joints import FLIP
from .pose_recall_probe import observation

ANCHORS = np.array([5, 6, 11, 12])


OBSERVERS = ('original', 'tight_mirror', 'wide_mirror', 'rtmw', 'tight_contrast')


def consensus(models, size, lower):
    source = {name: model for name, model in models.items() if name in OBSERVERS}
    if len(source) < 2:
        return None
    reference = source.get('original') or max(source.values(), key=lambda m: sum(m['joints'][5:]))
    anchor = np.asarray(reference['keypoints'], dtype=np.float32)
    anchor_mask = np.asarray(reference['joints'], dtype=bool)
    torso = max(np.linalg.norm(anchor[[5, 6]].mean(0) - anchor[[11, 12]].mean(0)),
                np.linalg.norm(anchor[5] - anchor[6]), 10.)
    observations = []
    for model in source.values():
        coords = np.asarray(model['keypoints'], dtype=np.float32)
        visible = np.asarray(model['joints'], dtype=bool)
        scores = np.asarray(model['scores'], dtype=np.float32)
        measured = []
        for perm in (np.arange(17), FLIP):
            valid = ANCHORS[(anchor_mask & visible[perm])[ANCHORS]]
            gap = np.linalg.norm(anchor[valid] - coords[perm][valid], axis=1).mean() if len(valid) >= 2 else np.inf
            measured.append((gap, perm))
        gap, perm = min(measured, key=lambda m: m[0])
        if gap <= .43*torso:
            observations.append((coords[perm], scores[perm], visible[perm]))
    if len(observations) < 2:
        return None
    points = anchor.copy()
    confidences = np.zeros(17, dtype=np.float32)
    for j in range(17):
        candidates = [(xy[j], scores[j]) for xy, scores, valid in observations if valid[j]]
        if len(candidates) < 2:
            continue
        xy = np.stack([point for point, _ in candidates])
        scores = np.asarray([score for _, score in candidates])
        distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        agree = distance <= .22 * torso
        medoid = (agree @ scores).argmax()
        cluster = agree[medoid]
        if cluster.sum() < 2:
            continue
        points[j] = np.average(xy[cluster], axis=0, weights=scores[cluster])
        confidences[j] = scores[cluster].mean() * sqrt(cluster.mean())
    return observation(points, confidences, size, lower)
