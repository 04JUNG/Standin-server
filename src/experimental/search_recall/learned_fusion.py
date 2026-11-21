"""Learn continuous ranking weights on rough-grouped held-out folds."""

from collections import defaultdict
import math
import numpy as np

DETAILS = ('torso_lean_deg', 'pelvis_offset_torso', 'ankle_gap_torso', 'left_knee_deg', 'right_knee_deg')


FOLDS = 5


def fold(image_sha256):
    return int(image_sha256[:8], 16) % FOLDS


def feature_rows(branches, core, details, joint):
    geo = {h['pose_id']: (i, h) for i, h in enumerate(branches.get('geometry', {}).get('hits', []), 1)}
    sem = {h['pose_id']: (i, h) for i, h in enumerate(branches.get('semantic', {}).get('hits', []), 1)}
    confidence = float(joint.metrics.get('mean_valid_body_score') or 0.)
    visible = min(float(joint.metrics.get('valid_body_joints') or 0.) / 12, 1.)
    rows = {}

    def similarity(value, scale):
        return .5 if value is None else math.exp(-max(float(value), 0.) * scale)

    for pid in sorted(geo.keys() | sem.keys()):
        g_rank, g = geo.get(pid, (0, {}))
        s_rank, s = sem.get(pid, (0, {}))
        if s.get('matched', {}).get('visible_region_reference') and not g_rank:
            continue
        checked = (g.get('whole_pose_validation') or s.get('whole_pose_validation')
                   or s.get('matched', {}).get('whole_pose_validation') or {})
        if checked.get('status') == 'near_match':
            continue
        fit = (details or {}).get(pid) or {}
        checks = fit.get('checks') or {}
        condition = (core or {}).get(pid) or {}
        conflicts = bool(condition.get('conflicts'))
        measured_matches = sum(x.get('status') == 'match'
                               for x in (condition.get('relations') or {}).values())
        g_score = math.sqrt(6 / (5 + g_rank)) if g_rank else 0.
        s_score = math.sqrt(6 / (5 + s_rank)) if s_rank else 0.
        distance = fit.get('geometry_distance', g.get('distance'))
        features = (
            g_score, s_score, similarity(distance, 3.),
            similarity(g.get('proportion_cost'), 3.), similarity(g.get('regional_cost'), 3.),
            float(s.get('score', 0.)), similarity(fit.get('cost'), 1.2),
            *(similarity(checks.get(k, {}).get('normalized_difference'), 1.) for k in DETAILS),
            float(bool(g_rank and s_rank)), g_score * (.5 * confidence + .5 * visible),
            s_score * (1. - visible), min(1., measured_matches * .3),
        )
        rows[pid] = (int(conflicts), features)
    return rows


def priorities(rows, core, weights):
    adjusted = {}
    for pid, (conflict, features) in rows.items():
        score = float(np.dot(features, weights))
        adjusted[pid] = dict((core or {}).get(pid) or {}, priority=[conflict, -score])
    return adjusted


def residual_rows(ranked, branches, core, details, joint, projections, limbs):
    """Record physical evidence alongside the frozen baseline ordering."""
    features = feature_rows(branches, core, details, joint)
    strong = {h['pose_id']: i for i, h in enumerate(projections, 1)}
    regional = {h['pose_id']: i for i, h in enumerate(limbs, 1)}
    semantic = {h['pose_id']: h.get('matched', {}) for h in branches.get('semantic', {}).get('hits', [])}
    result = {}
    for row in ranked:
        pid = row['pose_id']
        if pid not in features or row.get('match_level') == 'near_match' or row['kind'] == 'framing_reference':
            continue
        position = len(result) + 1
        matches = semantic.get(pid, {})
        tag_rank = (matches.get('joint_geometry_and_tags') or {}).get('joint_rank')
        condition = (core or {}).get(pid) or {}
        conflicts = condition.get('conflicts') or ()
        hard_prop = any(k == 'held_object' or k == 'hand_count' or k.startswith('held_hand:') for k in conflicts)
        extra = (1. / strong[pid] if pid in strong else 0., 1. / regional[pid] if pid in regional else 0.,
                 1. / tag_rank if tag_rank else 0., float(matches.get('measured_fingertip_agreement', 0.)),
                 float(condition.get('projection_fit_support', 0.)), float(bool(conflicts)))
        result[pid] = (int(hard_prop), -.80 * math.log(position), features[pid][1] + extra)
        if len(result) == 32:
            break
    return result


def rerank_residual(ranked, features, weights):
    tiers = defaultdict(list)
    for i, row in enumerate(ranked):
        if (value := features.get(row['pose_id'])) is not None:
            tier, prior, vector = value
            tiers[tier].append((i, prior + float(np.dot(weights, vector))))
    result = list(ranked)
    for positions in tiers.values():
        sorted_positions = sorted(positions, key=lambda entry: (-entry[1], entry[0]))
        for (target, _), (source, _) in zip(positions, sorted_positions, strict=True):
            result[target] = ranked[source]
    return result
