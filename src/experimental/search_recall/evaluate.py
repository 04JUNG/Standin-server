"""Geometry and semantic candidate validation settings."""

from collections import defaultdict
from . import service_context as arm

VARIANTS = {
    'arm_control': (40, 'arm', 'original', .6),
    'no_arm_penalty': (40, 'zero_arm', 'original', .6),
    'geometry_80': (80, 'arm', 'original', .6),
    'geometry_unknown_soft': (40, 'unknown_soft', 'original', .6),
    'semantic_threshold_026': (40, 'arm', 'threshold', .6),
    'semantic_arm_soft': (40, 'arm', 'soft', .6),
    'combined': (80, 'unknown_soft', 'both', .6),
    'combined_geometry_weight': (80, 'unknown_soft', 'both', .3),
}


def validation(checked, mode):
    soft, cost = arm.soften_arms(checked)
    if mode == 'zero_arm':
        return soft, 0.
    if mode != 'unknown_soft':
        return soft, cost
    uncertain = soft['unknown']
    tolerated = [k for k in uncertain if k == 'posture' or k.startswith('hand_count:')]
    remaining = [k for k in uncertain if k not in tolerated]
    if tolerated:
        cost += .025 * len(tolerated)
        status = ('conflict' if soft['conflicts'] else 'needs_review' if remaining
                  else 'near_match' if soft['detail_conflicts'] or soft['detail_unknown'] else 'passed')
        soft = dict(soft, status=status, unknown=remaining, softened_unknown=tolerated)
    return soft, cost


class SoftSemanticValidator:
    def __init__(self, validator):
        self.validator = validator

    def check(self, required, pose_id):
        # Keep definite arm contradictions and all other requirements.
        # An unmeasurable coarse arm value is no longer a mandatory exclusion.
        checked = self.validator.check(required, pose_id)
        if checked['conflicts'] or not checked['unknown']:
            return checked
        remaining = [key for key in checked['unknown'] if not key.startswith('arm:')]
        if remaining:
            return checked
        status = ('near_match' if checked['detail_conflicts'] or checked['detail_unknown'] else 'passed')
        return dict(checked, status=status, unknown=[],
                    softened_unknown=checked['unknown'])


def geometry_variants(engine, evidence, required):
    part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
    hits = engine._clean(engine.camera.search(evidence['keypoints'], evidence['joint_mask'], part), 'geometry')
    accepted = defaultdict(list)
    for hit in hits:
        if not 0 <= hit['distance'] <= engine.cutoffs[part]:
            continue
        checked = engine.validator.check(required, hit['pose_id']) if required and required['active'] else None
        for mode in ('arm','zero_arm','unknown_soft'):
            if checked is None:
                accepted[mode].append((hit['distance'],hit))
            else:
                result, cost = validation(checked, mode)
                if result['status'] in ('passed','near_match'):
                    accepted[mode].append((hit['distance']+cost, dict(hit, whole_pose_validation=result)))
    return {name: [h for _,h in sorted(rows,key=lambda row:row[0])][:80] for name,rows in accepted.items()}


def rank(rows, gold):
    return next((i for i,r in enumerate(rows,1) if r['pose_id']==gold), None)
