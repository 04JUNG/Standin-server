"""Geometry terms for the fixed-camera part target pilot."""

import numpy as np


LIFTED_BEND_MAX_CHANGE_DEG = 25.


def segment_cos(points):
    segments = np.diff(points, axis=0)
    return float(np.clip(segments[0] @ segments[1] / max(np.prod(np.linalg.norm(segments, axis=1)), 1e-10), -1., 1.))


def limb_terms(goal, screen, points, camera):
    """Residuals use screen evidence. B0 bend is an explicit separate prior."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    projected = np.linalg.norm(np.diff(screen, axis=0), axis=1)
    ratios = projected / (camera['pixel_scale'] * lengths / camera['torso_world'])
    metrics = dict(screen_bend_cos=segment_cos(screen), bend_deg=float(np.degrees(np.arccos(segment_cos(points)))),
                   foreshortening=ratios.tolist())
    terms = []
    if goal['bend_cos'] is not None:
        terms.append(.18 * (metrics['screen_bend_cos'] - goal['bend_cos']))
    terms.extend(.32 * (ratio - observed) for ratio, observed in zip(ratios, goal['foreshortening'], strict=True)
                 if observed is not None)
    prior = goal['base_bend_cos']
    if prior is not None:
        change = abs(metrics['bend_deg'] - float(np.degrees(np.arccos(np.clip(prior, -1., 1.)))))
        metrics['b0_bend_change_deg'] = change
        terms.append(.55 * np.radians(change))
    return terms, metrics
