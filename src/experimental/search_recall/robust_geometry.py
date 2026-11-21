"""Measure sketch geometry while limiting the influence of uncertain joints."""

import numpy as np
from .analyze import BONES
from .analyze import frame
from .analyze import normalized
from .evaluate import validation

def confidence_costs(camera, evidence, part):
    points = np.asarray(evidence['keypoints'], dtype=np.float32)
    mask = np.asarray(evidence['joint_mask'], dtype=bool)
    scores = np.clip(np.asarray(evidence['joint_scores'], dtype=np.float32), .3, .9)
    selected = np.arange(5, 11 if part == 'upper' else 17)
    selected = selected[mask[selected]]
    q, ok = frame(points, part, mask)
    if not ok or len(selected) < (4 if part == 'upper' else 6):
        return {}
    positions, valid = camera.frames[part]
    gallery = positions[:, :, selected]
    weights = scores[selected].copy()
    observed_lower = evidence['lower_body_observed']
    if observed_lower is False:
        weights[selected >= 13] *= .35
    # A rotated drawing can match a different camera than raw upright distance.
    # The fitted rotation is bounded and carries a cost even when points line up.
    query = q[selected]
    gx, gy = gallery[..., 0], gallery[..., 1]
    qx, qy = query[:, 0], query[:, 1]
    dot = np.einsum('cpj,j->cp', gx, qx * weights) + np.einsum('cpj,j->cp', gy, qy * weights)
    cross = np.einsum('cpj,j->cp', gx, qy * weights) - np.einsum('cpj,j->cp', gy, qx * weights)
    angle = np.clip(np.arctan2(cross, dot), -np.pi / 6, np.pi / 6)
    sin, cos = np.sin(angle)[..., None], np.cos(angle)[..., None]
    rotated_x, rotated_y = gx * cos - gy * sin, gx * sin + gy * cos
    errors = np.hypot(rotated_x - qx, rotated_y - qy)
    position = np.average(np.minimum(errors, 1.) + .25 * np.maximum(errors - 1., 0.),
                          axis=-1, weights=weights)

    start, stop = np.asarray(BONES).T
    directions, nonzero = normalized(q[stop] - q[start])
    used = mask[start] & mask[stop] & nonzero
    if part == 'upper':
        used &= (start < 11) & (stop < 11)
    left, right = start[used], stop[used]
    if len(left):
        gallery_bones, nonzero = normalized(positions[:, :, right] - positions[:, :, left])
        bx, by = gallery_bones[..., 0], gallery_bones[..., 1]
        dx, dy = directions[used].T
        alignment = (bx * cos - by * sin) * dx + (bx * sin + by * cos) * dy
        direction = np.where(nonzero, (1. - np.clip(alignment, -1., 1.)) * .5, 1.)
        bone_weights = np.minimum(scores[left], scores[right])
        if observed_lower is False:
            bone_weights[right >= 13] *= .35
        position += .45 * np.average(direction, axis=-1, weights=bone_weights)
    position += .18 * np.abs(angle)
    present = np.all(camera.scores[:, selected] >= .3, axis=-1)
    costs = np.where(camera.valid & valid & present[None, :], position, np.inf)
    best = np.argmin(costs, axis=0)
    return {pid: float(costs[view, i]) for i, (pid, view)
            in enumerate(zip(camera.pose_ids, best, strict=True)) if np.isfinite(costs[view, i])}


def soften(checked, cost, strong_geometry):
    """Let coarse shape disagreements add cost when joints support a search."""
    if not strong_geometry:
        return checked, cost
    soft_conflicts = [key for key in checked['conflicts'] if key.startswith('arm:') or key == 'posture']
    conflicts = [key for key in checked['conflicts'] if key not in soft_conflicts]
    if conflicts or checked['unknown']:
        return checked, cost
    detail_conflicts = checked.get('detail_conflicts', [])
    details = len(detail_conflicts) + len(checked.get('detail_unknown', []))
    if not soft_conflicts and not details:
        return checked, cost
    # Keep explicit contact and prop contradictions as eligibility checks.
    return dict(checked, status='passed', conflicts=conflicts,
                softened_shape_conflicts=soft_conflicts + detail_conflicts), cost + .12 * len(soft_conflicts) + .06 * details


def shortlist(engine, evidence, required, original, baseline, soften_shape):
    part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
    costs = confidence_costs(engine.camera, evidence, part)
    if not costs:
        return baseline
    pool = []
    for hit in original:
        pid = hit['pose_id']
        if pid not in costs or not 0 <= hit['distance'] <= engine.cutoffs[part]:
            continue
        robust_cost = costs[pid]
        penalty = 0.
        if required and required['active']:
            checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
            if soften_shape:
                checked, penalty = soften(checked, penalty, robust_cost < .7)
            if checked['status'] not in ('passed', 'near_match'):
                continue
            hit = dict(hit, whole_pose_validation=checked)
        hit = dict(hit, robust_geometry_cost=robust_cost)
        pool.append((robust_cost + penalty, pid, hit))
    reserved = baseline[:24]
    seen = {h['pose_id'] for h in reserved}
    return reserved + [h for _, pid, h in sorted(pool) if pid not in seen][:80 - len(reserved)]
