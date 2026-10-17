"""Fixed-XY camera optimization with positive confidence weights and matched control."""
import numpy as np
from standin_eval.camera_refinement import project_parameters, camera_parameters, DEFAULT_STEPS
from standin_eval.dual_label_search import alternate_labels, LEFT_RIGHT
from standin_eval.confidence_cost import joint_weights


def feature_cost(features, valid, xy, mask, weights):
    weights = np.asarray(weights, float)
    if weights.shape != (17,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('finite positive COCO17 weights required')
    selected = np.flatnonzero(np.asarray(mask, bool)[5:]) + 5
    if len(selected) < 6:
        raise ValueError('at least six body joints required')
    residual = np.linalg.norm(features[..., selected, :] - np.asarray(xy)[selected], axis=-1)
    w = weights[selected]
    return np.where(valid, (residual * w).sum(axis=-1) / w.sum(), np.inf)


def objective(points, parameters, xy, mask, weights):
    features, valid = project_parameters(points, parameters)
    return feature_cost(features, valid, xy, mask, weights)


def optimize_one(points, index, cameras, xy, mask, warm, weights, *, steps=DEFAULT_STEPS):
    coarse = feature_cost(*index, xy, mask, weights)
    seeds = np.argsort(coarse, axis=0, kind='stable')[:3]
    warm = np.asarray(warm, np.float32)
    if warm.shape != (len(points), 3):
        raise ValueError('one previous camera per pose required')
    params = np.concatenate((camera_parameters(cameras)[seeds], warm[None]), axis=0)
    current = objective(points, params, xy, mask, weights)
    for scale in steps:
        for _ in range(2):
            for axis, delta in enumerate(scale):
                origin = params.copy()
                for sign in (-1., 1.):
                    proposal = origin.copy(); proposal[..., axis] += sign * delta
                    proposal[..., 0] %= 360.
                    proposal[..., 1] = np.clip(proposal[..., 1], -60, 60)
                    proposal[..., 2] = np.clip(proposal[..., 2], 0, 1.25)
                    cost = objective(points, proposal, xy, mask, weights)
                    improve = cost < current
                    params = np.where(improve[..., None], proposal, params)
                    current = np.minimum(current, cost)
    best = np.argmin(current, axis=0); cols = np.arange(len(points))
    return current[best, cols], params[best, cols]


def fit_confidence(points, index, cameras, xy, mask, warm, scores, *, weighted=True, **kwargs):
    weights = joint_weights(scores) if weighted else np.ones(17)
    alt, am = alternate_labels(xy, mask)
    direct, dp = optimize_one(points, index, cameras, xy, mask, warm, weights, **kwargs)
    swapped, sp = optimize_one(points, index, cameras, alt, am, warm, weights[LEFT_RIGHT], **kwargs)
    use = swapped < direct
    return np.minimum(direct, swapped), np.where(use[:, None], sp, dp), use
