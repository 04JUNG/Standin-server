"""Local XY camera matching with translation/positive-scale fitting.

No query rotations, mirror swaps, answer IDs, or source coordinates are used.
"""
import numpy as np


def fit_camera_distances(index, query_xy, valid_mask, scores=None):
    features, camera_valid = index
    query = np.asarray(query_xy, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if query.shape != (17, 2) or mask.shape != (17,):
        raise ValueError('expected COCO17 XY and mask')
    selected = np.arange(5, 17)[mask[5:]]
    if len(selected) < 6 or not np.isfinite(query[selected]).all():
        return np.full(features.shape[:2], np.inf)
    weights = np.ones(len(selected), dtype=np.float32)
    if scores is not None:
        confidence = np.asarray(scores, dtype=np.float32)
        if confidence.shape != (17,) or not np.isfinite(confidence[selected]).all():
            raise ValueError('expected finite selected joint scores')
        # Relative weights only; RTM scores are not calibrated probabilities.
        weights = np.clip(confidence[selected], 0., 1.)
    if weights.sum() <= 1e-8:
        return np.full(features.shape[:2], np.inf)
    weights = weights / weights.sum()
    target = query[selected].copy()
    target -= np.einsum('ki,k->i', target, weights)
    gallery = features[:, :, selected].copy()
    gallery -= np.einsum('vpki,k->vpi', gallery, weights)[:, :, None]
    variance = np.einsum('ki,ki,k->', target, target, weights)
    if variance < 1e-8:
        return np.full(features.shape[:2], np.inf)
    denominator = np.einsum('vpki,vpki,k->vp', gallery, gallery, weights)
    numerator = np.einsum('vpki,ki,k->vp', gallery, target, weights)
    scale = np.maximum(numerator / np.maximum(denominator, 1e-8), 0.)
    residual = gallery * scale[:, :, None, None] - target
    cost = np.sqrt(np.maximum(np.einsum('vpki,vpki,k->vp', residual, residual, weights), 0.) / variance)
    return np.where(camera_valid & (denominator > 1e-8), cost, np.inf)


def score_fit_camera_index(index, query_xy, valid_mask, scores=None):
    return fit_camera_distances(index, query_xy, valid_mask, scores).min(axis=0)
