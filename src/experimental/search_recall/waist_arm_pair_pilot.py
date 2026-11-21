"""Search for a waist hand paired with a lowered arm."""

from copy import copy
from types import MethodType
from .evaluate import validation
from .display import SoftSemanticValidator
from .display import prepare_engine
from src.semantic_search import SemanticQuery

GEOMETRY_CAP = 80


SEMANTIC_CAP = 40


SIDE_CONFLICT_COST = .01


VARIANT = 'combined_geometry_weight'


class WaistArmPairValidator:
    """Recognize the same two arm roles in either anatomical orientation."""

    def __init__(self, base):
        self.base = base
        self.positions = base.positions

    def check(self, required, pose_id):
        result = self.base.check(required, pose_id)
        observed = self.base.cache[pose_id]['arms']
        pair_match = {observed.get('left'), observed.get('right')} == {'waist', 'lowered'}
        arm_conflicts = [key for key in result['conflicts'] if key.startswith('arm:')]
        side_conflict = pair_match and bool(arm_conflicts)
        if side_conflict:
            conflicts = [key for key in result['conflicts'] if key not in arm_conflicts]
            status = ('conflict' if conflicts else 'needs_review' if result['unknown'] else
                      'near_match' if result['detail_conflicts'] or result['detail_unknown'] else 'passed')
            result = dict(result, conflicts=conflicts, status=status,
                          softened_arm_sides=arm_conflicts)
        return dict(result, waist_down_pair=pair_match, arm_roles=observed,
                    side_penalty=SIDE_CONFLICT_COST if side_conflict else 0.)


def use_arm_pair(engine, description):
    body_tags = set(description['body_details'].split(' · '))
    assert {'한 손 허리', '한 팔 내리기'} <= body_tags
    engine.validator = WaistArmPairValidator(engine.validator)
    engine.semantic = copy(engine.semantic)
    base_core = engine.semantic.core_evidence

    def core(query, ids):
        original = base_core(query, ids)
        checked = engine.pair_checks
        return {pid: {**original.get(pid, {}),
                      'priority': [*original.get(pid, {}).get('priority', [0, 0]),
                                   not checked[pid]['waist_down_pair']]}
                for pid in ids}

    def geometry(self, evidence, required):
        part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
        rows = []
        raw = self._clean(self.camera.search(evidence['keypoints'], evidence['joint_mask'], part), 'geometry')
        for hit in raw:
            if not 0 <= hit['distance'] <= self.cutoffs[part]:
                continue
            pid = hit['pose_id']
            checked, cost = validation(self.validator.check(required, pid), 'unknown_soft')
            if checked['status'] in ('passed', 'near_match'):
                self.pair_checks[pid] = checked
                pair_cost = 0. if checked['waist_down_pair'] else .05
                rows.append(dict(hit, whole_pose_validation=checked,
                                 penalized_distance=hit['distance'] + cost + pair_cost + checked['side_penalty']))
        rows.sort(key=lambda hit: hit['penalized_distance'])
        return dict(status='ok' if rows else 'no_result', hits=rows[:GEOMETRY_CAP],
                    backend='geometry_117', observation_scope=part)

    def semantic(self, description, required):
        adapter = copy(self.semantic)
        adapter.whole_pose_validator = SoftSemanticValidator(self.validator)
        result = adapter.search(SemanticQuery.from_description(description),
                                top_k=adapter.service.index.pose_count)
        if result.status == 'error':
            return dict(status='error', hits=[], reason=result.reason)
        hits = self._clean([hit.to_dict() for hit in result.hits], 'semantic')
        for hit in hits:
            pid = hit['pose_id']
            self.pair_checks[pid] = self.validator.check(required, pid)
        # Stable sort keeps the existing text order within each pair tier.
        hits.sort(key=lambda hit: (not self.pair_checks[hit['pose_id']]['waist_down_pair'],
                                  self.pair_checks[hit['pose_id']]['side_penalty']))
        return dict(status='ok' if hits else 'no_result', hits=hits[:SEMANTIC_CAP],
                    backend=result.backend)

    engine.pair_checks = {}
    engine.semantic.core_evidence = core
    engine.geometry_search = MethodType(geometry, engine)
    engine.semantic_search = MethodType(semantic, engine)
