"""Server contracts and soft arm validation shared by the recall stages."""
from collections import Counter
from time import perf_counter

from api.app import _library_preview_asset as _library_preview_asset
from api.pose_presentation import build_pose_presentation as build_pose_presentation
from src.config import CFG as CFG
from src.experimental.body_matching.integrated import enabled as integrated_enabled
from src.fullbody_routing import FullBodySearch
from src.search_routing import JointEvidence as JointEvidence, SemanticEvidence as SemanticEvidence, decide_route as decide_route


ARM_CONFLICT_COST = .05


ARM_UNKNOWN_COST = .025


def soften_arms(checked):
    """Retain the original checks and apply existing severity to other fields."""
    original_conflicts = checked['conflicts']
    original_unknown = checked['unknown']
    conflicts = [k for k in original_conflicts if not k.startswith('arm:')]
    unknown = [k for k in original_unknown if not k.startswith('arm:')]
    penalty = (len(original_conflicts) - len(conflicts)) * ARM_CONFLICT_COST
    penalty += (len(original_unknown) - len(unknown)) * ARM_UNKNOWN_COST
    if penalty == 0:
        return checked, penalty
    status = ('conflict' if conflicts else 'needs_review' if unknown
              else 'near_match' if checked['detail_conflicts'] or checked['detail_unknown']
              else 'passed')
    return dict(checked, status=status, conflicts=conflicts, unknown=unknown,
                original_status=checked['status'], arm_penalty=penalty), penalty


class ArmPenaltySearch(FullBodySearch):
    def geometry_search(self, evidence, required=None):
        if not (self.validator and required and required.get('active') and required.get('arms')):
            return super().geometry_search(evidence, required)
        started = perf_counter()
        part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
        hits = self._clean(self.camera.search(evidence['keypoints'], evidence['joint_mask'], part), 'geometry')
        counts, excluded, accepted = Counter(), [], []
        for hit in hits:
            if not 0 <= hit['distance'] <= self.cutoffs[part]:
                continue
            checked, cost = soften_arms(self.validator.check(required, hit['pose_id']))
            counts[checked['status']] += 1
            if checked['status'] in ('passed', 'near_match'):
                accepted.append(dict(hit, whole_pose_validation=checked,
                                     arm_penalty=cost, penalized_distance=hit['distance'] + cost))
            elif len(excluded) < 20:
                excluded.append(dict(pose_id=hit['pose_id'], **checked))
        accepted.sort(key=lambda hit: hit['penalized_distance'])
        return dict(status='ok' if accepted else 'no_result', hits=accepted[:self.pool_k],
                    backend='geometry_117', camera_count=117,
                    reason=None if accepted else 'no_original_within_geometry_cutoff',
                    observation_scope=part, retrieval_scope='whole_library_pose', cutoff=self.cutoffs[part],
                    whole_pose_validation=dict(counts=dict(counts), excluded_examples=excluded),
                    elapsed_ms=(perf_counter()-started)*1000)
