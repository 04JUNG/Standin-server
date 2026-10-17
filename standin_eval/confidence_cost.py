"""Confidence weighting at a frozen camera; no joint removal or alignment fit."""
import numpy as np
from standin_eval.dual_label_search import LEFT_RIGHT


def joint_weights(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (17,) or not np.isfinite(scores).all():
        raise ValueError('finite COCO17 scores required')
    return .25 + .75 * np.clip((scores - .3) / .4, 0., 1.)


def frozen_costs(features, xy, mask, scores, reference):
    features, xy = np.asarray(features), np.asarray(xy)
    mask = np.asarray(mask, dtype=bool)
    weights = joint_weights(scores)
    if (features.ndim != 3 or features.shape[1:] != (17, 2) or xy.shape != (17, 2)
            or mask.shape != (17,) or not np.isfinite(features).all() or not np.isfinite(xy).all()
            or mask[5:].sum() < 6):
        raise ValueError('finite gallery/query COCO17 and at least six valid body joints required')
    uniform, weighted = [], []
    for permutation in (np.arange(17), LEFT_RIGHT):
        selected = np.flatnonzero(mask[permutation][5:]) + 5
        residual = np.linalg.norm(features[:, selected] - xy[permutation][selected], axis=-1)
        w = weights[permutation][selected]
        uniform.append(residual.mean(axis=-1))
        weighted.append((residual * w).sum(axis=-1) / w.sum())
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (len(features),) or not np.isfinite(reference).all():
        raise ValueError('stored original fitted costs required to identify original label interpretation')
    # Freeze the original winning label too. Re-minimizing both labels at the
    # single stored camera can improve even uniform scores (new cross-label start).
    choice = np.argmin(np.abs(np.stack(uniform) - reference), axis=0)
    cols = np.arange(len(features))
    control = np.stack(uniform)[choice, cols]
    if np.max(np.abs(control - reference)) >= 2e-5:
        raise ValueError('stored cost cannot be reproduced at either original label')
    return control, np.stack(weighted)[choice, cols]
