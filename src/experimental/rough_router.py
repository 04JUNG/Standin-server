"""Image-only phase-1 search planner. Does not execute retrieval or grant refine."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import islice, product
import json

import numpy as np

from ..vlm.rough_slots import PersonSemantics
from .a_minimal_support import classify_minimal_support_2d
from .b1_pose_facts import extract_pose_facts_2d

POLICY_VERSION = 'rough-router-v1.2-individual'
EXCLUDED_ROUTES = ('lower_only',)
LIMBS = {'left_arm': (5, 7, 9), 'right_arm': (6, 8, 10),
         'left_leg': (11, 13, 15), 'right_leg': (12, 14, 16)}
HIDDEN = ('occluded', 'out_of_frame')
OBSERVABLE = ('visible', 'partial')


@dataclass(frozen=True)
class Observation:
    keypoints: object = None
    valid_mask: object = None
    ownership_valid: bool = True
    state: str = 'missing'


@dataclass(frozen=True)
class Channel:
    kind: str
    region: str
    queries: tuple[str, ...]
    rule_ids: tuple[str, ...]
    evidence_status: str
    candidate_budget: int = 20
    execution: str = 'not_executed_planning_only'


@dataclass(frozen=True)
class SearchPlan:
    person_index: int
    route: str
    channels: tuple[Channel, ...]
    reasons: tuple[str, ...]
    facts: tuple[str, ...] = ()
    views: tuple[str, ...] = ()
    compose: str = 'not_planned'
    policy_version: str = POLICY_VERSION
    mode: str = 'shadow'
    refine_allowed: bool = False
    candidates_changed: bool = False

    def to_dict(self):
        return asdict(self)

    @property
    def plan_id(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()


def _geometry(obs, person):
    reasons = []
    mask = np.zeros(17, dtype=bool)
    points = np.zeros((17, 2), dtype=float)
    if not obs.ownership_valid:
        return points, mask, ['observation_ownership_invalid']
    if obs.state in ('missing', 'invalid') or obs.keypoints is None or obs.valid_mask is None:
        return points, mask, ['observation_unavailable']
    try:
        raw = np.asarray(obs.keypoints, dtype=float)
        raw_mask = np.asarray(obs.valid_mask)
        if raw.shape != (17, 2) or raw_mask.shape != (17,) or raw_mask.dtype != np.bool_:
            return points, mask, ['observation_shape_or_mask_invalid']
        mask = raw_mask.copy() & np.isfinite(raw).all(axis=1)
        points = np.where(np.isfinite(raw), raw, 0.0)
    except (ValueError, TypeError, OverflowError):
        return points, mask, ['observation_invalid']
    for part, indices in LIMBS.items():
        if person.visibility_of(part) in HIDDEN:
            if mask[list(indices)].any():
                reasons.append('visibility_conflict:' + part)
            # Hips/shoulders may still be observed as torso anchors. Only mask distal joints.
            mask[list(indices[1:])] = False
    if person.visibility_of('torso') in HIDDEN:
        if mask[[5, 6, 11, 12]].any():
            reasons.append('visibility_conflict:torso')
        mask[[5, 6, 11, 12]] = False
    return points, mask, reasons


def _chain(points, mask, name, person):
    indices = LIMBS[name]
    return (person.visibility_of(name) not in HIDDEN and mask[list(indices)].all()
            and all(np.linalg.norm(points[a] - points[b]) > 1e-6
                    for a, b in zip(indices, indices[1:])))


def plan_search(person: PersonSemantics, observation: Observation) -> SearchPlan:
    """Deterministic route only; all candidate/camera/compose execution is future work."""
    reasons = list(person.issues)

    def done(route, channels=(), facts=(), views=(), compose='not_planned'):
        return SearchPlan(person.person_index, route, tuple(channels),
                          tuple(dict.fromkeys(reasons)), tuple(facts), tuple(views), compose)

    if not observation.ownership_valid:
        reasons.append('R02:person_ownership_unresolved')
        return done('unresolved_person')
    points, mask, geometry_reasons = _geometry(observation, person)
    reasons.extend(geometry_reasons)
    # Lower-only concerns the depicted crop, not simply failed upper keypoint extraction.
    upper_image = (any(person.visibility_of(p) in OBSERVABLE for p in ('torso', 'left_arm', 'right_arm'))
                   or int(mask[5:11].sum()) >= 2)
    lower_image = (any(person.visibility_of(p) in OBSERVABLE for p in ('left_leg', 'right_leg'))
                   or int(mask[13:17].sum()) >= 2)
    if lower_image and not upper_image:
        reasons.append('R04:lower_only_not_supported_phase1')
        return done('unsupported_lower_only')

    # Region anchors are measured on the same image. These are eligibility checks,
    # not a new partial feature implementation or a claim that its index is ready.
    upper_anchor = bool(mask[[5, 6]].all() and np.linalg.norm(points[5] - points[6]) > 1e-6)
    lower_anchor = bool(mask[[11, 12]].all() and np.linalg.norm(points[11] - points[12]) > 1e-6)
    torso = bool(mask[[5, 6, 11, 12]].all()
                 and np.linalg.norm((points[5] + points[6] - points[11] - points[12]) / 2) > 1e-6)
    upper = bool(upper_anchor and any(_chain(points, mask, p, person) for p in ('left_arm', 'right_arm')))
    lower = bool(lower_anchor and any(_chain(points, mask, p, person) for p in ('left_leg', 'right_leg')))
    full = upper and lower and torso
    channels = []

    def add(kind, region, queries, rule, evidence):
        channels.append(Channel(kind, region, tuple(dict.fromkeys(queries)), (rule,), evidence))

    if full:
        add('G', 'full', (), 'R04', 'observed')
    if upper or torso:
        add('G', 'upper', (), 'R04', 'observed' if upper else 'torso_only')
    if lower:
        add('G', 'lower', (), 'R04', 'observed')
    views = tuple(m.value for m in person.values('view_hypotheses'))
    upper_meanings = person.values('upper_action') + person.values('torso_orientation')
    lower_meanings = person.values('lower_configuration')
    support = sorted(person.values('support_state'), key=lambda m: (
        m.evidence_kind == 'scene_context', m.status != 'supported', m.value))[:3]
    common = list(upper_meanings) + list(lower_meanings) + list(person.values('contacts'))

    def queries_for(values):
        return [m.value for m in values]

    if upper_meanings:
        add('S', 'upper', queries_for(upper_meanings), 'R05', 'vlm_inferred')
    if lower_meanings or support:
        # Alternative support states stay separate queries, never "standing and sitting".
        bases = [m.value for m in lower_meanings] or ['']
        qs = [base for base in bases if base]
        qs.extend('; '.join(filter(None, (base, h.value))) for base in bases for h in support)
        add('S', 'lower', qs, 'R06', 'vlm_inferred')
    if common or support:
        # A field's values are alternative hypotheses. Never concatenate mutually
        # exclusive upper/lower descriptions into a single semantic request.
        choices = [person.values(name) or (None,) for name in
                   ('upper_action', 'torso_orientation', 'lower_configuration', 'contacts')]
        qs = []
        for combination in islice(product(*choices), 20):
            base = '; '.join(m.value for m in combination if m is not None)
            if base:
                qs.append(base)  # neutral pool survives support-state uncertainty
            for h in support:
                qs.append('; '.join(filter(None, (base, h.value))))
            if len(qs) >= 20:
                break
        add('S', 'full', qs[:20], 'R05', 'vlm_inferred')
    if not support and not lower_meanings:
        reasons.append('R07:lower_meaning_unspecified')
    if support and not lower:
        reasons.append('R06:lower_hypothesis_without_geometry')
    if any(m.evidence_kind == 'scene_context' for m in support):
        reasons.append('R06:context_hypothesis_does_not_filter_geometry')

    facts = extract_pose_facts_2d(points.reshape(-1), mask)
    upper_facts = tuple(f.token for f in facts if 'knee' not in f.name)
    lower_facts = tuple(f.token for f in facts if 'knee' in f.name)
    support_fact = classify_minimal_support_2d(points.reshape(-1), mask)
    fact_tokens = [f.token for f in facts]
    if upper_facts:
        add('F', 'upper', upper_facts, 'R08', 'observed')
    if lower_facts:
        add('F', 'lower', lower_facts, 'R08', 'observed')
    if support_fact.value != 'unknown':
        token = 'observed_support=' + support_fact.value
        fact_tokens.append(token)
        add('F', 'full', (token,), 'R08', 'observed')
    if not channels:
        if person.framing == 'face':
            reasons.append('R12:face_without_body_evidence')
            return done('skip_face', views=views)
        reasons.append('R12:insufficient_image_evidence')
        return done('insufficient_evidence', views=views)

    regions = {c.region for c in channels}
    if 'upper' in regions and 'lower' not in regions:
        add('D', 'lower', (), 'R07', 'unspecified')
    elif 'lower' in regions and 'upper' not in regions:
        # Allowed for a full/half image with semantic-only evidence, NOT lower-only crop.
        add('D', 'upper', (), 'R07', 'unspecified')
    material_regions = {c.region for c in channels}
    compose = ('conditional_on_valid_upper_lower_materials'
               if {'upper', 'lower'} <= material_regions else 'not_planned')
    if compose != 'not_planned':
        reasons.append('R09:compose_and_whole_candidates_compete')
    reasons.append('R10:one_camera_per_completed_candidate')
    route = 'full_observed' if full else ('upper_observed' if upper or torso else 'semantic_or_facts_only')
    return done(route, channels, fact_tokens, views, compose)
