"""Experimental weak inverse-distance prior; no runtime changes or query GT."""
import numpy as np
from standin_eval.camera_refinement import camera_parameters, parameter_cost, DEFAULT_STEPS
from standin_eval.dual_label_search import alternate_labels


def penalty(parameters, weight):
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('finite nonnegative weight required')
    return weight * np.asarray(parameters)[..., 2].astype(float)**2


def optimize_one(points, index, cameras, xy, mask, weight, *, steps=DEFAULT_STEPS):
    features, valid = index
    selected = np.flatnonzero(np.asarray(mask, dtype=bool)[5:]) + 5
    if len(selected) < 6:
        raise ValueError('at least6 observed body joints required')
    coarse_xy = np.linalg.norm(features[:, :, selected]-np.asarray(xy)[selected], axis=-1).mean(axis=-1)
    cp = camera_parameters(cameras)
    coarse = np.where(valid, coarse_xy + penalty(cp, weight)[:, None], np.inf)
    seeds = np.argsort(coarse, axis=0, kind='stable')[:3]
    params = cp[seeds]
    current = parameter_cost(points, params, xy, mask) + penalty(params, weight)
    for scale in steps:
        for _ in range(2):
            for axis, delta in enumerate(scale):
                origin = params.copy()
                for sign in (-1., 1.):
                    proposal = origin.copy()
                    proposal[..., axis] += sign*delta
                    proposal[..., 0] %= 360
                    proposal[..., 1] = np.clip(proposal[..., 1], -60, 60)
                    proposal[..., 2] = np.clip(proposal[..., 2], 0, 1.25)
                    cost = parameter_cost(points, proposal, xy, mask) + penalty(proposal, weight)
                    improve = cost < current
                    params = np.where(improve[..., None], proposal, params)
                    current = np.minimum(current, cost)
    cols = np.arange(len(points)); best = np.argmin(current, axis=0)
    result, fitted = current[best, cols], params[best, cols]
    # Keep cost and returned camera consistent, including float-rounding cases.
    coarse_params = cp[seeds[0]]
    seed_cost = parameter_cost(points, coarse_params, xy, mask) + penalty(coarse_params, weight)
    use_seed = seed_cost < result
    return np.minimum(result, seed_cost), np.where(use_seed[:, None], coarse_params, fitted)


def regularized_costs(points, index, cameras, xy, mask, weight):
    alt, altmask = alternate_labels(xy, mask)
    direct, dp = optimize_one(points, index, cameras, xy, mask, weight)
    swapped, sp = optimize_one(points, index, cameras, alt, altmask, weight)
    use_swap = swapped < direct
    return np.minimum(direct, swapped), np.where(use_swap[:, None], sp, dp)
