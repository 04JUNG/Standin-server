"""Geometry-only preview audit; this is not anatomical facing ground truth."""
import numpy as np

SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def centered(joints):
    p = np.asarray(joints, dtype=float)
    return p - p[[11, 12]].mean(axis=0)


def torso_front(joints):
    p = centered(joints)
    up = p[[5, 6]].mean(axis=0)
    v = np.cross(up, p[5] - p[6])
    n = np.linalg.norm(v)
    return None if n < 1e-10 else v / n


def pair_metrics(original, mirrored, tolerance=1e-4):
    p, q = centered(original), centered(mirrored)
    scale = np.linalg.norm(p[[5, 6]].mean(axis=0))
    if scale < 1e-10:
        return {'classification': 'degenerate_torso', 'hypotheses': []}
    rows = []
    front, mirror_front = torso_front(p), torso_front(q)
    for axis in (0, 2):
        sign = np.ones(3); sign[axis] = -1
        for relabel in (False, True):
            reflected = p[SWAP if relabel else np.arange(17)] * sign
            error = float(np.max(np.linalg.norm(reflected - q, axis=1)) / scale)
            dot = None if front is None or mirror_front is None else float(np.clip(np.dot(front * sign, mirror_front), -1, 1))
            rows.append({'reflection_axis': 'XZ'[axis // 2], 'anatomical_lr_swapped': relabel,
                         'max_joint_error_torso_units': error, 'matches': error <= tolerance,
                         'reflected_front_dot_preview_front': dot})
    matches = [r for r in rows if r['matches']]
    labels = {r['anatomical_lr_swapped'] for r in matches}
    classification = ('not_axis_reflection' if not matches else 'ambiguous' if len(labels) > 1
                      else 'reflection_with_lr_swap' if True in labels else 'reflection_without_lr_swap')
    return {'classification': classification, 'hypotheses': rows}
