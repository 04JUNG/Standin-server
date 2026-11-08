"""Convert saved rough detections to goals without supplying unobserved depth."""

import numpy as np

from experiments.refine_v5.part_observations import projected_evidence


CHAINS = {'right_leg': (('RightUpLeg', 'RightLeg', 'RightFoot'), (12, 14, 16)),
          'left_arm': (('LeftArm', 'LeftForeArm', 'LeftHand'), (5, 7, 9))}
MIN_SCORE = .50
ANGLE_PADDING = 12.


def bend_envelope(observation):
    """Ortho depth-sign ambiguity for the two observed projected segments."""
    cosine, ratios = observation['bend_cos'], observation['foreshortening']
    if cosine is None or any(ratio is None for ratio in ratios):
        return None
    first, second = np.clip(ratios, 0., 1.)
    xy_dot = first * second * cosine
    depth_dot = np.sqrt((1 - first**2) * (1 - second**2))
    low, high = np.degrees(np.arccos(np.clip([xy_dot + depth_dot, xy_dot - depth_dot], -1., 1.)))
    return [max(0., float(low) - ANGLE_PADDING), min(180., float(high) + ANGLE_PADDING)]


def detector_goals(case, seed, intent):
    if seed['case_id'] != case['case_id'] or seed['image_sha256'] != case['image_sha256']:
        raise ValueError('detector result does not match the frozen rough')
    if case['ownership_authority'] != 'offline_single_target_review' or case['camera']['inverse_distance_torso'] != 0:
        raise ValueError('detector goals require verified offline ownership and a fixed orthographic camera')
    if intent['right_leg'] != 'lift_forward_and_bend' or intent['left_arm'] != 'extend_and_separate_from_waist':
        raise ValueError('unsupported reviewed edit request')
    camera = case['camera']
    heads = {b['name']: np.asarray(b['matrix'])[:3, 3] for b in case['bones']}
    goals, evidence = {}, {}
    for limb, (chain, indices) in CHAINS.items():
        points = np.asarray(seed['keypoints'], float)[list(indices)]
        scores = np.asarray(seed['scores'], float)[list(indices)]
        row = dict(source='cached_rtmpose', indices=list(indices), scores=scores.tolist(), points=points.tolist())
        evidence[limb] = row
        if points.shape != (3, 2) or not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise ValueError('invalid detector output')
        if scores.min() < MIN_SCORE:
            row['status'] = 'low_detector_score'
            continue
        lengths = np.linalg.norm(np.diff([heads[name] for name in chain], axis=0), axis=1)
        if (lengths < 1e-6).any():
            raise ValueError('degenerate source chain')
        row.update(projected_evidence(points, camera['pixel_scale'], lengths, camera['torso_world']))
        interval = bend_envelope(row)
        row['plausible_bend_deg'] = interval
        if interval is None:
            row['status'] = 'unreliable_projected_lengths'
            continue
        if limb == 'right_leg':
            if interval[1] < intent['right_knee_intent_min_deg']:
                row['status'] = 'detected_knee_conflicts_with_reviewed_lift'
                continue
            # With only two observed segments, the sign of depth remains
            # unresolved. A forward-lift goal needs corroboration beyond the detector.
            row['status'] = 'unresolved_lift_depth'
            continue
        upper = min(interval[1], intent['left_elbow_intent_max_deg'])
        if interval[0] >= upper:
            row['status'] = 'detected_elbow_conflicts_with_reviewed_extension'
            continue
        uncertainty = np.clip((1. - scores[1:]) * 32., 8., 18.)
        bounds = {name: [list(point-width), list(point+width)]
                  for name, point, width in zip(chain[1:], points[1:], uncertainty, strict=True)}
        goals[limb] = dict(pixels=bounds, bend_deg=[interval[0], upper],
                           clearance_m=intent['left_waist_clearance_m'], preserve_camera_depth=True)
        row['status'] = 'applied_with_reviewed_extension_and_clearance'
    return goals, evidence
