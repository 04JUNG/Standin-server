"""Build recall evidence from a fresh rough and its recognized subjects."""

from copy import deepcopy
import json
import numpy as np
from rtmlib import RTMPose
from .aligned_clause_recall_probe import AlignedClauseIndex
from .aligned_clause_recall_probe import aligned_proposal
from .body_frame_recall_probe import BodyFrameIndex
from .body_frame_recall_probe import box_hypotheses
from .body_frame_recall_probe import profiles
from .bound_body_recall_probe import BoundBodyIndex
from .bound_body_recall_probe import description_roles
from .bound_body_recall_probe import geometry_support
from .clause_recall_probe import ClauseIndex
from .consensus_recall_probe import consensus
from .context_slot_probe import ContextSlots
from .continuous_camera_recall_probe import CameraRefitter
from .continuous_camera_recall_probe import model_names
from .continuous_camera_recall_probe import proposals
from .contrast_recall_probe import contrast_joints
from .crop_joints import CropJoints
from .dense_description_probe import index_text
from .description_recall_probe import DescriptionIndex
from .fitted_geometry import DepthObserver
from .fitted_geometry import FittedGeometry
from .limb_consensus_recall_probe import LimbConsensus
from .limb_consensus_recall_probe import OBSERVERS
from .limb_ink_recall_probe import LimbInkRanking
from .occluded_recall_probe import candidate_scores
from .occluded_recall_probe import expanded_camera
from .occluded_recall_probe import models_for
from .outline_ranking import OutlineRanking
from .outline_recall_probe import prepare_queries
from .outline_recall_probe import proposal_pool
from .pose_recall_probe import joint_hypotheses
from .pose_recall_probe import observation
from .pose_recall_probe import score_profiles
from .relational_projection_probe import RelationalProjection
from src.pose import _load_bgr
from src.search_routing import PoseDescription
from src.semantic_embedding import E5OnnxEncoder
from src.semantic_search import SemanticQuery
from .structured_search import StructuredIndex
from .verify_hypothesis_rank import prop_review_candidate
from .verify_hypothesis_rank import tilt_candidates
from .verify_hypothesis_rank import visual_candidates

def observation_record(query, scores):
    return dict(joints=query['joint_mask'].tolist(), scores=query['joint_scores'].tolist(),
                keypoints=query['keypoints'].tolist(), profiles=scores)


def ranked(ids, scores, limit):
    order = np.argsort(-scores, kind='stable')
    return [[str(ids[i]), float(scores[i])] for i in order[:limit] if scores[i] > 0.]


class RecallObservations:
    def __init__(self, engine, assets, tags_path):
        self.camera = engine.camera
        self.ids = engine.camera.pose_ids
        self.positions = {pid: i for i, pid in enumerate(self.ids)}
        self.structured = StructuredIndex(engine)
        self.crop = CropJoints(assets.pose_manifest).model
        self.depth = DepthObserver(assets.depth_model)
        self.fitter = FittedGeometry(self.camera)
        self.steep = FittedGeometry(expanded_camera(self.camera))
        self.bends = RelationalProjection(self.camera)
        self.body_x = RTMPose(str(assets.body_model), model_input_size=(288, 384),
                              backend='onnxruntime', device='cpu')
        self.body_frame = BodyFrameIndex(self.camera)
        self.camera_refitter = CameraRefitter(self.camera)
        self.limb_consensus = LimbConsensus(self.camera)
        records = json.loads(tags_path.read_text())['poses']
        self.lexical = DescriptionIndex(records)
        self.encoder = E5OnnxEncoder(str(assets.embedding_model), threads=3)
        self.documents = self.encoder.fit([index_text(row) for row in records])
        self.clauses = ClauseIndex(records, self.encoder)
        self.aligned = AlignedClauseIndex(records, self.encoder)
        self.bound = BoundBodyIndex(self.camera)
        self.context = ContextSlots(records)
        self.outline = OutlineRanking(self.camera)
        self.ink = LimbInkRanking(self.camera)

    def body_scores(self, description, *, lower):
        if not description:
            return None
        value = PoseDescription(**description)
        return self.structured.scores(SemanticQuery.from_description(value), value.to_dict(),
                                      legs_observed=lower, refine_arms=True)

    def add_bends(self, record, semantic, *, allow_reverse=False):
        for part in ('upper', 'full'):
            costs = self.bends.costs(record, part)
            if costs is None:
                continue
            if allow_reverse:
                xy = np.asarray(record['keypoints'])
                ratio = np.linalg.norm(xy[5] - xy[6]) / max(np.ptp(xy[5:13, 1]), 1.)
                if ratio < .48:
                    alternate = self.bends.costs(record, part, reverse=True)
                    if alternate is not None:
                        costs = np.minimum(costs, alternate + .024)
            record['profiles'][f'{part}_bend'] = profiles(self.ids, costs, semantic, ())

    def joint_models(self, person, path, image, description):
        lower = person['lower_body_observed']
        semantic = self.body_scores(description, lower=lower)
        models = {}
        for name, query, spatial in joint_hypotheses(image, person, self.crop, self.depth, path):
            measured = {kind: profiles(self.ids, costs, semantic, ())
                        for kind, costs in score_profiles(self.fitter, query, spatial)}
            models[name] = observation_record(query, measured)
        # The earlier bend pass used only the upper-body semantic cues.
        relational = self.body_scores(description, lower=False)
        for name, record in models.items():
            if name in ('original', 'tight_mirror', 'wide_mirror', 'rtmw'):
                self.add_bends(record, relational, allow_reverse=True)
        outline_models = deepcopy(models)
        strong = contrast_joints(image, person, self.crop)
        if strong is not None:
            measured = {kind: profiles(self.ids, costs, semantic, ())
                        for kind, costs in score_profiles(self.fitter, strong, None)}
            models['tight_contrast'] = observation_record(strong, measured)
            self.add_bends(models['tight_contrast'], semantic)
        joint_consensus = consensus(models, image.shape[:2], lower)
        if joint_consensus is not None:
            measured = {kind: profiles(self.ids, costs, semantic, ())
                        for kind, costs in score_profiles(self.fitter, joint_consensus, None)}
            models['consensus'] = observation_record(joint_consensus, measured)
            self.add_bends(models['consensus'], semantic)
        for name in models_for({'models': models}):
            for label, costs in candidate_scores(models[name], lower, self.fitter, self.steep):
                models[name]['profiles'][label] = profiles(self.ids, costs, semantic, ())
        return outline_models, models, semantic

    def detailed_models(self, models, person, image, path, language, semantic):
        models = deepcopy(models)
        lower = person['lower_body_observed']
        for name, detected in box_hypotheses(self.body_x, image, person.get('box'), lower):
            fits = {kind: profiles(self.ids, costs, semantic, ())
                    for kind, costs in score_profiles(self.fitter, detected, None)}
            models[name] = observation_record(detected, fits)
        if 'rtmw' in models:
            measured = self.depth.predict(path, person)
            if measured is not None:
                xy, confidence, z, _ = measured
                detected = observation(xy, confidence, image.shape[:2], lower)
                if detected is not None and np.isfinite(z).all():
                    costs = self.body_frame.costs(detected, z, lower)
                    if costs is not None:
                        models['rtmw']['profiles']['body_frame'] = profiles(self.ids, costs, semantic, ())
        candidates = proposals(models, language)
        for name in model_names(models):
            for part in (('upper', 'full') if lower else ('upper',)):
                costs = self.camera_refitter.search(models[name], candidates, part)
                if costs is not None:
                    models[name]['profiles'][f'refit_{part}'] = profiles(self.ids, costs, semantic, ())
        candidates = proposals(models, language)
        costs = self.limb_consensus.search(models, candidates, lower)
        if costs is not None:
            reference = next(name for name in OBSERVERS if name in models)
            models[reference]['profiles']['limb_consensus'] = profiles(self.ids, costs, semantic, ())
        return models

    def text_profiles(self, description):
        fields = dict(text=description.get('search_text') or '', details=description.get('body_details') or '')
        fields['both'] = ' '.join(fields.values())
        language = {}
        for field, phrase in fields.items():
            sparse = self.lexical.search(phrase)
            if sparse is not None:
                language[field] = dict(top=ranked(self.lexical.ids, sparse, 40))
        fields['both'] = ' '.join(dict.fromkeys((fields['text'], fields['details']))).strip()
        for field, phrase in fields.items():
            if phrase:
                dense = self.documents @ self.encoder.encode(phrase)[0]
                top = [[str(self.clauses.ids[i]), round(float(dense[i]), 6)]
                       for i in np.argsort(-dense, kind='stable')[:70]]
                language[f'dense_{field}'] = dict(top=top)
        return language

    def add_text_cues(self, language, description, models, lower):
        for field, scores in self.clauses.search(description):
            top = ranked(self.clauses.ids, scores, 70)
            language[field] = dict(top=[[pid, round(score, 6)] for pid, score in top])
        scores = self.aligned.search(description)
        if scores is not None:
            top = [[pid, round(score, 6)] for pid, score in ranked(self.aligned.ids, scores, 70)]
            language['aligned_clauses'] = dict(top=top, proposal=aligned_proposal(top, models))
        physical = self.bound.search(description, lower)
        if physical is not None:
            support = np.zeros(len(self.ids), dtype=np.float32)
            for pid, value in geometry_support(models).items():
                support[self.positions[pid]] = value
            role_count = sum(map(len, description_roles(description, lower)))
            for field, values in (('bound_roles', physical), ('bound_joint_roles', .62 * physical + .38 * support)):
                top = [[pid, round(score, 6)] for pid, score in ranked(self.ids, values, 60)]
                language[field] = dict(top=top, role_count=role_count)

    def visual_profiles(self, path, image, person, count, outline_models, occluded, language):
        result = {}
        queries = prepare_queries(self.outline, path, person, outline_models, count)
        if queries:
            costs = {}
            for pid in proposal_pool(outline_models, language):
                values = [value for query in queries if (value := self.outline.cost(query, pid)) is not None]
                if values:
                    costs[pid] = min(values)
            result['outline'] = dict(top=[[pid, round(float(np.exp(-4 * costs[pid])), 6)]
                                           for pid in sorted(costs, key=costs.get)])
        if count == 1:
            raster = self.ink.prepare(image, person.get('box'))
            if raster is not None:
                pool = [pid for pid in proposal_pool(occluded, language) if pid in self.positions]
                costs = {}
                for name in models_for({'models': occluded}):
                    for pid, cost in self.ink.score(raster, occluded[name], pool,
                                                    person['lower_body_observed']).items():
                        costs[pid] = min(costs.get(pid, float('inf')), cost)
                result['limb_ink'] = dict(top=[[pid, round(float(np.exp(-7 * costs[pid])), 6)]
                                                for pid in sorted(costs, key=costs.get)])
        return result

    def observe(self, cut, path, prop_hits=()):
        image = _load_bgr(str(path))
        result = {name: {} for name in ('hypotheses', 'language', 'visual', 'tilt', 'prop_review',
                                        'contextual', 'heads', 'detected_props')}
        people = cut['people']
        for person in people:
            i = person['index']
            raw = person['dynamic_search']['description_evidence'].get('description') or {}
            text = self.text_profiles(raw)
            outline, occluded, semantic = self.joint_models(person, path, image, raw)
            appearance = self.visual_profiles(path, image, person, len(people), outline, occluded, text)
            query_cues = dict(text)
            for field, scores in self.clauses.search(raw):
                query_cues[field] = dict(top=[[pid, round(score, 6)] for pid, score in ranked(self.clauses.ids, scores, 70)])
            detailed = self.detailed_models(occluded, person, image, path, query_cues, semantic)
            self.add_text_cues(text, raw, detailed, person['lower_body_observed'])
            result['hypotheses'][i] = occluded
            result['language'][i] = text
            result['visual'][i] = visual_candidates(appearance, detailed)
            result['tilt'][i] = tilt_candidates(appearance, detailed)
            result['prop_review'][i] = prop_review_candidate(person, detailed)
            result['contextual'][i] = (self.context.propose(raw, text)
                                       or self.context.fallback(raw, text, detailed))
            result['heads'][i] = self.context.head_only(person, raw, text, detailed)
            if len(people) == 1:
                result['detected_props'][i] = self.context.detected_prop(person, raw, text, detailed, prop_hits)
        return result
