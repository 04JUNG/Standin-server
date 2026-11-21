"""Run the locally verified search strategy on observations from this request."""

import json
from types import SimpleNamespace
import numpy as np
from .hypothesis_rank_probe import ALIGNED_CLAUSES
from .hypothesis_rank_probe import BOUND_ROLES
from .hypothesis_rank_probe import CLAUSES
from .hypothesis_rank_probe import DENSE
from .learned_fusion import FOLDS
from .learned_fusion import fold
from .local_prop_detector import LocalPropDetector
from .recall_observations import RecallObservations
from pathlib import Path
from .verify_hypothesis_rank import apply_ranking

PROFILE = 'recall-90-v1'


MODEL = Path(__file__).with_name('recall-90-v1-weights.json')


OPTIONS = SimpleNamespace(
    recover_subject=True, subject_detail=True, dense=False, outline=False,
    enhanced='occluded', clauses=False, clause_slot=True, mirror_budget=True,
    bound_slot=True, aligned_slot=True, visual_slot=True, tilt_slot=True,
    prop_review_slot=True, context_slot=True, head_slot=True, detected_prop_slot=True,
)


class VerifiedRecall:
    def __init__(self, engine, assets, tags_path):
        model = json.loads(MODEL.read_text())
        if model.get('profile') != PROFILE:
            raise ValueError('Unexpected recall model profile')
        self.primary = np.asarray(model['weights'])
        self.clauses = np.asarray(model['clause_weights'])
        for weights in (self.primary, self.clauses):
            if weights.ndim != 2 or weights.shape[0] != FOLDS or not np.isfinite(weights).all():
                raise ValueError('Invalid recall weight matrix')
        self.engine = engine
        self.observations = RecallObservations(engine, assets, tags_path)
        self.assets = assets
        self.detector = None

    def observe(self, cut, path, *, extracted=None):
        hits = ()
        if len(cut['people']) == 1:
            person, = cut['people']
            report = person['dynamic_search']
            description = report['description_evidence'].get('description') or {}
            joints = report['geometry_evidence']
            if (not description.get('objects') and not joints.get('policy_block')
                    and not joints['sufficient']):
                if self.detector is None:
                    self.detector = LocalPropDetector(self.assets.prop_python, self.assets.prop_model)
                hits = self.detector.observe(path)
        return self.observations.observe(cut, path, hits, extracted=extracted)

    def rerank(self, cut, digest, evidence):
        index = fold(digest)
        result, _, _ = apply_ranking(
            self.engine, cut, evidence['hypotheses'], evidence['language'],
            self.primary[index], self.clauses[index], OPTIONS, relational=True,
            excluded_text=set((*DENSE, *CLAUSES, *BOUND_ROLES, ALIGNED_CLAUSES)),
            visual=evidence['visual'], tilt=evidence['tilt'], prop_review=evidence['prop_review'],
            contextual=evidence['contextual'], heads=evidence['heads'], detected_props=evidence['detected_props'])
        return result

    def close(self):
        if self.detector is not None:
            self.detector.close()
