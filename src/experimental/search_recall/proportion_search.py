"""Compare limb shape across different drawing and model proportions."""

from dataclasses import asdict
import numpy as np
from .analyze import frame
from .evaluate import validation
from .robust_geometry import shortlist as robust_shortlist
from .robust_geometry import soften

CHAINS = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))


CAP = 80


def direction(points):
    length = np.linalg.norm(points, axis=-1)
    return points / np.maximum(length[..., None], 1e-6), length


def proportional_costs(camera, evidence, part):
    mask = np.asarray(evidence['joint_mask'], dtype=bool)
    confidence = np.clip(np.asarray(evidence['joint_scores'], dtype=np.float32), .3, .9)
    query, ok = frame(evidence['keypoints'], part, mask)
    if not ok:
        return None
    gallery, valid = camera.frames[part]
    anchors = [5, 6] if part == 'upper' else [5, 6, 11, 12]
    if part == 'upper':
        q_axis = query[6] - query[5]
        g_axis = gallery[..., 6, :] - gallery[..., 5, :]
    else:
        q_axis = query[5] + query[6] - query[11] - query[12]
        g_axis = gallery[..., 5, :] + gallery[..., 6, :] - gallery[..., 11, :] - gallery[..., 12, :]
    dot = g_axis[..., 0] * q_axis[0] + g_axis[..., 1] * q_axis[1]
    cross = g_axis[..., 0] * q_axis[1] - g_axis[..., 1] * q_axis[0]
    roll = np.clip(np.arctan2(cross, dot), -.45, .45)
    sine, cosine = np.sin(roll), np.cos(roll)

    def rotate(points):
        x, y = points[..., 0], points[..., 1]
        return np.stack((x * cosine - y * sine, x * sine + y * cosine), axis=-1)

    anchor_cost = np.mean([np.linalg.norm(rotate(gallery[..., joint, :]) - query[joint], axis=-1)
                           for joint in anchors], axis=0)
    chain_costs = []
    chain_weights = []
    used = list(anchors)
    for chain in CHAINS if part == 'full' else CHAINS[:2]:
        base, middle, end = chain
        if not mask[base]:
            continue
        parents = rotate(gallery[..., base, :])
        parent = base
        chain_sum = np.zeros_like(roll)
        total_weight = 0.
        for joint in (middle, end):
            if not mask[joint]:
                continue
            q_dir, q_length = direction(query[joint] - query[parent])
            if q_length < .02:
                continue
            segment = rotate(gallery[..., joint, :] - gallery[..., parent, :])
            g_dir, g_length = direction(segment)
            # Allow different limb proportions, while retaining some evidence
            # from projected length for foreshortening and physical contact.
            adjusted = .4 * g_length + .6 * np.clip(q_length, .5 * g_length, 2. * g_length)
            projected = parents + g_dir * adjusted[..., None]
            position = np.linalg.norm(projected - query[joint], axis=-1)
            angle_cost = (1. - np.clip(g_dir @ q_dir, -1., 1.)) * .5
            cost = np.minimum(position, 1.) + .3 * np.maximum(position - 1., 0.) + .42 * angle_cost
            weight = float(min(confidence[parent], confidence[joint]))
            if parent == base and joint == end:
                weight *= .65
            chain_sum += cost * weight
            total_weight += weight
            used.append(joint)
            parents, parent = projected, joint
        if total_weight:
            chain_costs.append(chain_sum / total_weight)
            chain_weights.append(total_weight * (.45 if base >= 11 and evidence['lower_body_observed'] is False else 1.))
    if not chain_costs:
        return None
    limbs = np.stack(chain_costs)
    average = np.average(limbs, axis=0, weights=chain_weights)
    # A single poorly read limb can dominate average joint distance. The
    # second ordering limits its influence and still compares the whole pose.
    if len(chain_costs) >= 3:
        least_reliable = int(np.argmin(chain_weights))
        retained = [i for i in range(len(chain_costs)) if i != least_reliable]
        regional = .70 * np.average(limbs[retained], axis=0,
                                    weights=np.asarray(chain_weights)[retained]) + .30 * average
    else:
        regional = average
    shared = .35 * anchor_cost + .15 * np.abs(roll)
    present = np.all(camera.scores[:, sorted(set(used))] >= .3, axis=-1)
    good = camera.valid & valid & present[None, :]
    whole = np.where(good, average + shared, np.inf)
    partial = np.where(good, regional + shared, np.inf)
    views = np.argmin(whole, axis=0)
    columns = np.arange(len(camera.pose_ids))
    return whole[views, columns], partial.min(axis=0), views


def shortlist(engine, evidence, required, original, baseline, *, fit=None):
    part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
    camera = engine.camera
    costs = fit if fit is not None else proportional_costs(camera, evidence, part)
    robust = robust_shortlist(engine, evidence, required, original, baseline, soften_shape=False)
    if costs is None:
        return robust
    whole, partial, views = costs
    rows = []
    for hit in original:
        pid = hit['pose_id']
        i = engine.validator.positions[pid]
        if not np.isfinite(whole[i]) or not 0 <= hit['distance'] <= engine.cutoffs[part]:
            continue
        penalty = 0.
        if required and required['active']:
            checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
            if checked['status'] not in ('passed', 'near_match'):
                continue
            hit = dict(hit, whole_pose_validation=checked)
        hit = dict(hit, proportion_cost=float(whole[i]), regional_cost=float(partial[i]),
                   proportion_camera=asdict(camera.cameras[int(views[i])]))
        rows.append((float(whole[i]) + penalty, float(partial[i]) + penalty, pid, hit))
    whole_rows = [row[3] for row in sorted(rows, key=lambda row: (row[0], row[2]))[:CAP]]
    regional_rows = [row[3] for row in sorted(rows, key=lambda row: (row[1], row[2]))[:CAP]]
    picked = {h['pose_id']: h for h in baseline[:12]}
    for i in range(CAP):
        for choices in (robust, whole_rows, regional_rows):
            if i < len(choices):
                hit = choices[i]
                picked.setdefault(hit['pose_id'], hit)
                if len(picked) == CAP:
                    return list(picked.values())
    return list(picked.values())


def consistent_candidates(engine, required, original, fit):
    """Find poses supported by limb direction even when proportions differ."""
    if fit is None:
        return []
    whole, regional, views = fit
    whole_top = np.argsort(whole, kind='stable')[:10]
    region_top = np.argsort(regional, kind='stable')[:10]
    best_whole, best_region = whole[whole_top[0]], regional[region_top[0]]
    if not np.isfinite(best_whole) or not np.isfinite(best_region):
        return []
    whole_rank = {i: rank for rank, i in enumerate(whole_top, 1)}
    region_rank = {i: rank for rank, i in enumerate(region_top, 1)}
    raw = {hit['pose_id']: hit for hit in original}
    choices = []
    for i in whole_rank.keys() | region_rank.keys():
        a, b = whole_rank.get(i, 11), region_rank.get(i, 11)
        same_shape = (a <= 9 and b <= 9 and a + b <= 13 and
                      whole[i] <= best_whole + .055 and regional[i] <= best_region + .055)
        partial_shape = (b <= 3 and regional[i] <= best_region + .04
                         and regional[i] + .065 < whole[i])
        if not (same_shape or partial_shape) or not np.isfinite(whole[i]) or regional[i] > .40:
            continue
        pid = engine.camera.pose_ids[i]
        if pid not in raw:
            continue
        hit = dict(raw[pid], proportion_cost=float(whole[i]), regional_cost=float(regional[i]),
                   proportion_camera=asdict(engine.camera.cameras[int(views[i])]))
        if required and required['active']:
            checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
            checked, _ = soften(checked, penalty, strong_geometry=True)
            if checked['status'] != 'passed':
                continue
            hit['whole_pose_validation'] = checked
        choices.append((min(a + b, b + 5 if partial_shape else 22), float(regional[i]), pid, hit))
    return [row[3] for row in sorted(choices)[:4]]
