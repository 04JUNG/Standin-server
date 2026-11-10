"""Combine projected joints, relative geometry and contact evidence."""

from copy import copy
from copy import deepcopy
from dataclasses import replace
import math
from types import MethodType
from .analyze import BoneEvidence
from . import service_context as arm
from .analyze import geometry
from .analyze import rank
from .arm_contact_probe import elbow_wrist_swap
from . import learned_fusion
from .proportion_search import consistent_candidates
from .proportion_search import proportional_costs
from .proportion_search import shortlist as proportional_shortlist
from .robust_geometry import shortlist
from src.semantic_search import SemanticQuery
from src.semantic_search_285 import SemanticSearchService
from src.semantic_shapes import cross_chest_hint
from .structured_search import rerank_semantic_geometry
from .structured_search import supports_semantic_fallback
from .verify import prepare_engine
from .visible_relations import distinct_observed_arm_roles
from .visible_relations import head_contact_unconfirmed
from .visible_relations import soften_head_contact
from .waist_arm_pair_batch import uses_pair
from .waist_arm_pair_pilot import VARIANT as PAIR_BASELINE
from .waist_arm_pair_pilot import prepare_engine as prepare_pair_engine
from .waist_arm_pair_pilot import use_arm_pair

COMPOUND_CONTACT_EXPERIMENT = 'visible_compound_contact_80'


ENSEMBLE_EXPERIMENTS = ('visible_evidence_ensemble_80', COMPOUND_CONTACT_EXPERIMENT)


FULL_HEAD_EXPERIMENTS = ('visible_full_head_80', 'visible_semantic_geometry_80',
                         'visible_waist_fingertips_80', 'visible_depth_agreement_80', *ENSEMBLE_EXPERIMENTS)


DEPTH_AGREEMENT_EXPERIMENTS = ('visible_depth_agreement_80', *ENSEMBLE_EXPERIMENTS)


SPARSE_LIMB_EXPERIMENTS = ('visible_sparse_limb_80', 'visible_observed_sparse_80',
                           'visible_head_anchor_80', 'visible_heldout_rank_80', *FULL_HEAD_EXPERIMENTS)


OBSERVED_TAG_EXPERIMENTS = ('visible_observed_tag_fallback_80', 'visible_observed_sparse_80',
                            'visible_head_anchor_80', 'visible_fingertip_contacts_80',
                            'visible_heldout_rank_80', *FULL_HEAD_EXPERIMENTS)


CONTEXT_EXPERIMENTS = ('visible_projection_context_80', 'visible_outline_80', 'visible_limb_consensus_80',
                       'visible_sparse_limb_80', *OBSERVED_TAG_EXPERIMENTS)


PAIR_PROJECTION_EXPERIMENTS = ('visible_waist_pair_projection_80', 'visible_projection_recall_80',
                               *CONTEXT_EXPERIMENTS)


JOINT_GUARD_EXPERIMENTS = ('visible_joint_partial_guard_80', *PAIR_PROJECTION_EXPERIMENTS)


JOINT_SEARCH_EXPERIMENTS = ('visible_joint_search_80', *JOINT_GUARD_EXPERIMENTS)


PARTIAL_PROJECTION_EXPERIMENTS = ('visible_partial_projection_80', *JOINT_GUARD_EXPERIMENTS)


STRONG_PROJECTION_EXPERIMENTS = ('visible_strong_projection_80', *JOINT_SEARCH_EXPERIMENTS,
                                 'visible_partial_projection_80')


GUARDED_RANK_EXPERIMENTS = ('visible_rank_balance_guarded_80', 'visible_full_tilt_80',
                           *STRONG_PROJECTION_EXPERIMENTS)


RANK_EXPERIMENTS = ('visible_rank_balance_80', *GUARDED_RANK_EXPERIMENTS)


KNEE_EXPERIMENTS = ('visible_knee_support_80', *RANK_EXPERIMENTS)


ROLL_EXPERIMENTS = ('visible_roll_consensus_80', *KNEE_EXPERIMENTS)


ROTATED_CROP_EXPERIMENTS = ('visible_crop_recovery_80', *ROLL_EXPERIMENTS)


SOFT_ARM_EXPERIMENTS = ('visible_semantic_arm_soft_80', *ROTATED_CROP_EXPERIMENTS)


ALTERNATE_XY_EXPERIMENTS = ('visible_alternate_xy_80', 'visible_soft_head_contact_80', *SOFT_ARM_EXPERIMENTS)


HAND_PLACEMENT_EXPERIMENTS = ('visible_hand_placement_80', *ALTERNATE_XY_EXPERIMENTS)


RETRYABLE = {
    'hand_screen_network_error', 'hand_screen_request_timeout',
    'hand_screen_budget_exhausted', 'hand_screen_upstream_unavailable',
    'hand_screen_work_budget_exhausted', 'hand_screen_conversion_timeout',
    'hand_screen_upstream_error',
}


CHECK_LIMIT = 'hand_screen_check_limit_exhausted'


class ScreeningUnavailable(Exception):
    pass


def supplement_semantic(engine):
    """Give body arrangement its own query and keep the existing text cap."""
    original = engine.semantic_search
    adapter = copy(engine.semantic)
    pool = adapter.pool_service
    adapter.pool_service = SemanticSearchService(pool.index, replace(pool.settings, min_score_vlm=.26))

    def retrieve(description, required=None):
        prior = original(description, required)
        query = SemanticQuery.from_description(description)
        if not query.body_details or query.visible_scope in ('head', 'shoulder_head'):
            return prior
        labels = pool.index.lexicon.labels
        actions = [labels['action', action] for action in query.actions if ('action', action) in labels]
        query.text = '. '.join([query.body_details, *actions])
        search = adapter.search(query, top_k=40)
        if search.status == 'error':
            raise RuntimeError(('field_semantic_search', search.reason))
        hits = engine._clean([h.to_dict() for h in search.hits], 'semantic')
        if not hits:
            return prior
        refs = [h for h in prior.get('hits', []) if h.get('matched', {}).get('visible_region_reference')]
        old = [h for h in prior.get('hits', []) if h not in refs]
        picked = {h['pose_id']: h for h in old[:20]}
        for hit in hits + old[20:]:
            pid = hit['pose_id']
            if pid in picked:
                continue
            if required and required['active']:
                checked = engine.validator.check(required, pid)
                if checked['status'] not in ('passed', 'near_match'):
                    continue
                hit = dict(hit, whole_pose_validation=checked)
            picked[pid] = hit
            if len(picked) == 40:
                break
        result = list(picked.values()) + refs
        return dict(prior, hits=result, status='ok' if result else prior['status'],
                    reason=None if result else prior.get('reason'))

    engine.semantic_search = retrieve


def adaptive_priority(branches, core, details, joint):
    """Balance rank evidence continuously while preserving explicit conflicts."""
    geo = branches.get('geometry', {}).get('hits', [])
    sem = branches.get('semantic', {}).get('hits', [])
    if not geo or not sem:
        return None
    metrics = joint.metrics
    visible = metrics.get('valid_body_joints') or 0
    confidence = metrics.get('mean_valid_body_score') or 0.
    weight = .38 + .25 * min(visible / 12, 1.) + .22 * min(confidence, 1.)
    if metrics.get('skeleton_source') == 'vlm_repair':
        weight -= .10
    geo_scores = {h['pose_id']: (6. / (5. + i)) ** .5 for i, h in enumerate(geo, 1)}
    sem_scores = {h['pose_id']: (6. / (5. + i)) ** .5 * (.5 + .5 * h['score'])
                  for i, h in enumerate(sem, 1)}
    adjusted = {}
    for pid in geo_scores.keys() | sem_scores.keys():
        condition = (core or {}).get(pid) or {}
        fit = (details or {}).get(pid, {}).get('cost')
        geometry_score = geo_scores.get(pid, 0.)
        if fit is not None:
            geometry_score *= 1. / (1. + .15 * fit)
        score = weight * geometry_score + (1. - weight) * sem_scores.get(pid, 0.)
        conflict = bool(condition.get('conflicts'))
        adjusted[pid] = dict(condition, priority=[int(conflict), -score])
    return adjusted


def prefer_compound_text(priorities, branches):
    hits = branches.get('semantic', {}).get('hits', [])
    matched = [hit for hit in hits if hit['score'] >= .52 and
               hit.get('matched', {}).get('specific_action_and_objects')]
    if not 1 <= len(matched) <= 4:
        return priorities
    adjusted = dict(priorities)
    for hit in matched:
        pid = hit['pose_id']
        if pid in adjusted:
            priority = adjusted[pid]['priority']
            adjusted[pid] = dict(adjusted[pid], priority=[priority[0], priority[1] - .34 * (.55 + hit['score'])])
    return adjusted


def prefer_supported_roll(core, branches, hypotheses, *, max_fit=.20):
    if not hypotheses:
        return core
    sem = branches.get('semantic', {}).get('hits', [])
    geometry = branches.get('geometry', {}).get('hits', [])
    text_rank = {hit['pose_id']: i for i, hit in enumerate(sem, 1)}
    geo_rank = {hit['pose_id']: i for i, hit in enumerate(geometry, 1)}
    support = {}
    for i, hit in enumerate(hypotheses, 1):
        pid, distance = hit['pose_id'], hit.get('fitted_cost', float('inf'))
        if pid not in geo_rank:
            continue
        if i <= 3 and distance <= max_fit:
            support[pid] = 1. - distance
        if text_rank.get(pid, 999) <= 5 and distance <= .31:
            support[pid] = max(support.get(pid, 0.), 1.22 - distance)
    if not support:
        return core
    out = {}
    for pid in text_rank.keys() | geo_rank.keys():
        info = (core or {}).get(pid) or {}
        old = info.get('priority') or [0, 0]
        fit_support = support.get(pid, 0.)
        out[pid] = dict(info, priority=[old[0], -fit_support, *old[1:]],
                        projection_fit_support=fit_support)
    return out


def is_tilted(query):
    anchors = (5, 6, 11, 12)
    if not all(query['joint_mask'][i] for i in anchors):
        return False
    a, b, c, d = (query['keypoints'][i] for i in anchors)
    x, y = (a[i] + b[i] - c[i] - d[i] for i in range(2))
    return math.hypot(x, y) >= 16 and math.atan2(abs(x), abs(y)) >= math.pi / 6


def balance_branch_ranks(core, branches, baseline, joint, *, guard_projection=False):
    """Use each branch's top matches as extra evidence after initial ranking."""
    geo = branches.get('geometry', {}).get('hits', [])
    sem = branches.get('semantic', {}).get('hits', [])
    if not geo or not sem:
        return core
    baseline_rank = {row['pose_id']: i for i, row in enumerate(baseline, 1)
                     if row.get('match_level') != 'near_match' and row['kind'] != 'framing_reference'}
    if not baseline_rank:
        return core
    confidence = float(joint.metrics.get('mean_valid_body_score') or 0.)
    visible = min(float(joint.metrics.get('valid_body_joints') or 0.) / 12, 1.)
    reliability = .5 * (confidence + visible)
    shape = {row['pose_id']: (.60 + .55 * reliability) * math.exp(-i / 5.)
             for i, row in enumerate(geo)}
    text = {row['pose_id']: (.72 + .40 * (1. - reliability)) * math.exp(-i / 5.)
            for i, row in enumerate(sem) if not row.get('matched', {}).get('visible_region_reference')}
    if not text:
        return core
    adjusted = dict(core or {})
    for pid in shape.keys() | text.keys():
        info = adjusted.get(pid) or {}
        if pid not in baseline_rank:
            adjusted[pid] = dict(info, priority=[2, 0])
            continue
        conflicts = info.get('conflicts') or ()
        explicit_prop = any(key == 'held_object' or key == 'hand_count' or key.startswith('held_hand:')
                            for key in conflicts)
        a, b = shape.get(pid, 0.), text.get(pid, 0.)
        baseline_score = math.exp(-(baseline_rank[pid] - 1) / 5.)
        score = baseline_score + .86 * max(a, b) + .22 * min(a, b)
        if guard_projection and baseline_rank[pid] <= 3:
            score += .70 * info.get('projection_fit_support', 0.)
        adjusted[pid] = dict(info, priority=[int(explicit_prop), -score])
    return adjusted


def prioritize_strong_projection(core, fitted_hits, baseline, *, preserve=(), promote_close=False):
    ranks = {hit['pose_id']: i for i, hit in enumerate(baseline, 1)}
    best = ranks.get(fitted_hits[0]['pose_id'])
    if best is not None and best <= 2:
        if not promote_close:
            return core
        first = fitted_hits[0]
        fit = first['matched']['joint_geometry_and_tags']['fitted_cost']
        fitted_hits = [hit for hit in fitted_hits[1:] if ranks.get(hit['pose_id'], 0) > 5
                       and hit['score'] >= first['score'] - .03
                       and hit['matched']['joint_geometry_and_tags']['fitted_cost'] <= fit + .025]
        if not fitted_hits:
            return core
    support = {hit['pose_id']: (.90 - .12 * i) for i, hit in enumerate(fitted_hits)}
    protected = {hit['pose_id'] for hit in preserve if ranks.get(hit['pose_id'], 999) <= 6}
    adjusted = dict(core or {})
    for pid, rank in ranks.items():
        info = adjusted.get(pid) or {}
        conflicts = info.get('conflicts') or ()
        explicit_prop = any(key == 'held_object' or key == 'hand_count' or key.startswith('held_hand:')
                            for key in conflicts)
        score = math.exp(-(rank - 1) / 5.) + max(support.get(pid, 0.), .58 if pid in protected else 0.)
        adjusted[pid] = dict(info, priority=[int(explicit_prop), -score])
    return adjusted


def keep_strong_projections(hits, proposals):
    additions = {hit['pose_id']: hit for hit in proposals}
    retained = [additions.pop(hit['pose_id'], hit) for hit in hits]
    return retained[:80 - len(additions)] + list(additions.values())


def replay(engine, bone, person, experiment, crop, path, fitted=None, depth_observer=None,
           structured=None, collect_features=None, rank_weights=None, visible_relations=None,
           person_count=1, outline=None, hand_index=None, allow_recognition_error=False):
    previous = person['dynamic_search']
    joint = arm.JointEvidence(**previous['geometry_evidence'])
    semantic = arm.SemanticEvidence(**previous['description_evidence'])
    scope = person['output_scope']
    evidence = dict(keypoints=person['keypoints'], coverage_class=person['coverage_class'],
                    joint_scores=person['raw_scores'],
                    lower_body_observed=person['lower_body_observed'],
                    joint_mask=person['quality_trace']['search_valid_joint_mask'], person_count=person_count,
                    output_scope=scope['detected'], output_scope_source=scope['source'],
                    hand_presets={side: person['hands'][side]['preset'] for side in ('left', 'right')})
    geometry_ready = previous['route'] in ('hybrid', 'geometric') and joint.sufficient and not joint.policy_block
    use_fields = structured is not None or experiment.startswith(
        ('field_', 'robust_fields_', 'proportion_fields_', 'fitted_', 'visible_'))
    observed_roles = (experiment in OBSERVED_TAG_EXPERIMENTS and geometry_ready
                      and semantic.status in ('low_certainty', 'too_generic')
                      and distinct_observed_arm_roles(evidence, semantic.description or {}))
    promoted = (structured is not None and not joint.policy_block
                and semantic.status in ('low_certainty', 'too_generic')
                and (supports_semantic_fallback(semantic.description) or observed_roles))
    use_adaptive_fusion = 'adaptive' in experiment or experiment in HAND_PLACEMENT_EXPERIMENTS
    use_robust = structured is not None or experiment.startswith('robust_')
    use_proportions = experiment.startswith(('proportion_', 'visible_'))
    soft_arms = experiment in SOFT_ARM_EXPERIMENTS
    has_pair = uses_pair(person)
    replay_pair = has_pair and experiment in PAIR_PROJECTION_EXPERIMENTS
    alternative = None
    recovered = False
    rotated_crop = experiment in ROTATED_CROP_EXPERIMENTS
    if rotated_crop and not has_pair:
        alternative = crop.observe(path, person, evidence)
        if alternative and alternative.get('recovered_geometry'):
            evidence, alternative = alternative, None
            recovered = True
            geometry_ready = True
            joint = replace(joint, sufficient=True, reasons=['validated_rotated_crop_joints'],
                            metrics=dict(joint.metrics, valid_body_joints=sum(evidence['joint_mask'][5:]),
                                         skeleton_source='local_rotated_crop'))
    if (not geometry_ready and not (use_fields and previous['route'] == 'semantic') and not promoted or
        (has_pair and not replay_pair) or (geometry_ready and cross_chest_hint(semantic, evidence))):
        return None
    use_limb_fit = (experiment == 'visible_limb_consensus_80' or
                    (experiment in SPARSE_LIMB_EXPERIMENTS and geometry_ready
                     and evidence['coverage_class'] == 'sparse' and evidence['lower_body_observed'] is False
                     and sum(evidence['joint_mask'][5:]) >= 8))
    use_head_fit = (experiment == 'visible_head_anchor_80' or
                    (experiment in FULL_HEAD_EXPERIMENTS and person_count == 1
                     and evidence['lower_body_observed'] is True))
    head_limit = .105 if experiment in FULL_HEAD_EXPERIMENTS else .18

    corrected = None
    if replay_pair:
        engine = prepare_pair_engine(engine, PAIR_BASELINE, joint=joint.sufficient)
        use_arm_pair(engine, semantic.description)
        if geometry_ready:
            corrected = elbow_wrist_swap(evidence['keypoints'], evidence['joint_mask'], semantic.description)
        if corrected:
            a, b = corrected['elbow'], corrected['wrist']
            points = list(evidence['keypoints'])
            points[a], points[b] = points[b], points[a]
            evidence['keypoints'] = points
            person['keypoints'] = points
    else:
        engine = prepare_engine(engine, previous, cap=80)
    pair_geometry = engine.geometry_search if replay_pair else None
    contacts = None
    waist_contacts = None
    compound_contact = None
    description = semantic.description or {}
    if hand_index is not None:
        if experiment in ('visible_waist_fingertips_80', *ENSEMBLE_EXPERIMENTS):
            waist_contacts = hand_index.query_waist(description)
        if waist_contacts is not None or experiment == 'visible_waist_fingertips_80':
            contacts = waist_contacts
        else:
            contacts = hand_index.query(description)
        if experiment == COMPOUND_CONTACT_EXPERIMENT and geometry_ready:
            compound_contact = hand_index.query_compound(description)
    allow_semantic_projection = (experiment == 'visible_semantic_geometry_80' or
                                 (experiment in ENSEMBLE_EXPERIMENTS and waist_contacts is None))
    if (experiment == COMPOUND_CONTACT_EXPERIMENT and evidence['lower_body_observed'] is not True
            and 'lying' in (description.get('postures') or ())):
        allow_semantic_projection = False
    semantic_distances = None
    if use_fields and not promoted and not replay_pair:
        supplement_semantic(engine)
    if structured is not None:
        prior_semantic = engine.semantic_search

        def retrieve_structured(description, required=None):
            nonlocal semantic_distances
            prior = (dict(status='no_result', hits=[]) if promoted
                     else prior_semantic(description, required))
            legs_observed = (experiment in OBSERVED_TAG_EXPERIMENTS
                             and evidence['lower_body_observed'] is True)
            refine_arms = experiment in OBSERVED_TAG_EXPERIMENTS
            if not replay_pair:
                prior = structured.semantic_candidates(engine, description, required, prior,
                    soft_arms=soft_arms, legs_observed=legs_observed, refine_arms=refine_arms, contacts=contacts)
            if experiment in JOINT_SEARCH_EXPERIMENTS and geometry_ready:
                part = 'upper' if evidence['coverage_class'] == 'upper_only' else 'full'
                matched = fitted.costs(evidence, part, None, max_roll=1.57, roll_penalty=.04)
                if matched is not None:
                    prior = structured.joint_candidates(engine, description, required, prior, matched[0],
                        preserve_matches=experiment in CONTEXT_EXPERIMENTS,
                        legs_observed=legs_observed, refine_arms=refine_arms, contacts=contacts)
                    if allow_semantic_projection and prior.get('hits'):
                        semantic_distances = fitted.semantic_fit_distances(evidence, part, matched)
            if replay_pair:
                for hit in prior.get('hits', ()):
                    pid = hit['pose_id']
                    if pid not in engine.pair_checks:
                        engine.pair_checks[pid] = engine.validator.check(required, pid)
            return prior

        engine.semantic_search = retrieve_structured
    if crop is not None and not rotated_crop:
        alternative = crop.observe(path, person, evidence)
    depth = (depth_observer.observe(path, person, evidence) if depth_observer and geometry_ready
             and (experiment not in DEPTH_AGREEMENT_EXPERIMENTS
                  or (person_count == 1 and evidence['lower_body_observed'] is True)) else None)
    relation = (visible_relations.cues(evidence, semantic.description)
                if geometry_ready and visible_relations and not replay_pair else None)
    soft_head = ((experiment == 'visible_soft_head_contact_80' or soft_arms) and geometry_ready
                 and head_contact_unconfirmed(evidence, semantic.description))
    raw_distances = {}
    roll_hypotheses = []
    strong_projections = []
    expanded_projections = []
    limb_consensus = []
    head_proposals = []
    depth_proposals = []
    outline_query = outline.prepare(path, person.get('box'), evidence) if outline and geometry_ready else None
    outline_applied = False
    semantic_fit_applied = False
    compound_applied = False
    tilted_full = False
    before_screen = []
    fuse = engine._fuse
    residual = experiment in ('visible_heldout_rank_80', *ENSEMBLE_EXPERIMENTS)

    def capture(branches, core=None, *args, **kwargs):
        nonlocal outline_applied, semantic_fit_applied, compound_applied
        if relation is not None:
            adjusted = visible_relations.priorities(branches, core, relation[0], raw_distances)
            core = soften_head_contact(core, adjusted) if soft_head else adjusted
        if not residual and (collect_features is not None or rank_weights is not None):
            rows = learned_fusion.feature_rows(branches, core, kwargs.get('detail_evidence'), joint)
            if collect_features is not None and not collect_features:
                collect_features.update(rows)
            if rank_weights is not None:
                core = learned_fusion.priorities(rows, core, rank_weights)
        if experiment == 'robust_soft_rrf_80':
            kwargs['detail_evidence'] = None
        if use_adaptive_fusion and relation is None and not replay_pair:
            adjusted = adaptive_priority(branches, core, kwargs.get('detail_evidence'), joint)
            if adjusted is not None:
                if soft_arms and evidence['coverage_class'] in ('reduced', 'sparse', 'upper_only'):
                    adjusted = prefer_compound_text(adjusted, branches)
                core = soften_head_contact(core, adjusted) if soft_head else adjusted
        if recovered and not branches.get('semantic', {}).get('hits'):
            geometry_hits = branches.get('geometry', {}).get('hits', [])
            core = {hit['pose_id']: dict((core or {}).get(hit['pose_id']) or {},
                                       priority=[0, index])
                    for index, hit in enumerate(geometry_hits)}
        if roll_hypotheses:
            core = prefer_supported_roll(core, branches, roll_hypotheses, max_fit=.32 if tilted_full else .20)
        if experiment in RANK_EXPERIMENTS and not replay_pair:
            baseline = fuse(branches, core, *args, **kwargs)
            core = balance_branch_ranks(core, branches, baseline, joint,
                                        guard_projection=experiment in GUARDED_RANK_EXPERIMENTS)
        if strong_projections:
            ranked = fuse(branches, core, *args, **kwargs)
            core = prioritize_strong_projection(core, strong_projections, ranked)
        if experiment in JOINT_SEARCH_EXPERIMENTS:
            joint_hits = [h for h in branches.get('semantic', {}).get('hits', [])
                          if h.get('matched', {}).get('joint_geometry_and_tags')]
            if joint_hits:
                joint_hits.sort(key=lambda h: h['matched']['joint_geometry_and_tags']['joint_rank'])
                best_joint = joint_hits[0]['matched']['joint_geometry_and_tags']['fitted_cost']
                stronger_geometry = (experiment in PAIR_PROJECTION_EXPERIMENTS and not replay_pair
                    and strong_projections and strong_projections[0]['fitted_cost'] + .055 < best_joint)
                if experiment in CONTEXT_EXPERIMENTS:
                    stronger_geometry = (stronger_geometry and person_count == 1
                        and not any(h['score'] >= .78 or h.get('matched', {}).get('specific_action_and_objects')
                                    or h.get('matched', {}).get('measured_fingertip_agreement', 0.) >= .78
                                    for h in joint_hits))
                if not stronger_geometry:
                    ranked = fuse(branches, core, *args, **kwargs)
                    protect = strong_projections if experiment == 'visible_joint_partial_guard_80' else ()
                    core = prioritize_strong_projection(core, joint_hits, ranked, preserve=protect,
                                                        promote_close=replay_pair)
        if contacts is not None:
            contact_hits = [h for h in branches.get('semantic', {}).get('hits', [])
                            if h['score'] >= .60 and
                            h.get('matched', {}).get('measured_fingertip_agreement', 0.) >= .92]
            if contact_hits:
                ranked = fuse(branches, core, *args, **kwargs)
                contact_hits.sort(key=lambda h: (-h['score'], -h['matched']['measured_fingertip_agreement'],
                                                h['pose_id']))
                best_contact = contact_hits[0]
                best_rank = rank(ranked, best_contact['pose_id'])
                if best_rank is not None and best_rank <= 2:
                    contact_hits = [h for h in contact_hits[1:]
                                    if h['score'] >= best_contact['score'] - .04
                                    and h['matched']['measured_fingertip_agreement'] >= .95
                                    and 5 < (rank(ranked, h['pose_id']) or 0) <= 12]
                if contact_hits:
                    core = prioritize_strong_projection(core, contact_hits[:3], ranked, preserve=strong_projections)
        if expanded_projections:
            credible = [h for h in expanded_projections if h['fitted_cost'] <= .18]
            if credible:
                ranked = fuse(branches, core, *args, **kwargs)
                best = min(credible, key=lambda h: h['fitted_cost'] + .02 * (h['expanded_projection'] == 'steep'))
                core = prioritize_strong_projection(core, [best], ranked, preserve=strong_projections)
        if limb_consensus:
            ranked = fuse(branches, core, *args, **kwargs)
            core = prioritize_strong_projection(core, limb_consensus, ranked, preserve=strong_projections)
        if head_proposals:
            ranked = fuse(branches, core, *args, **kwargs)
            core = prioritize_strong_projection(core, head_proposals, ranked,
                                                preserve=strong_projections + limb_consensus)
        hits = fuse(branches, core, *args, **kwargs)
        protected_head = None
        if experiment == COMPOUND_CONTACT_EXPERIMENT and head_proposals:
            head_id = head_proposals[0]['pose_id']
            displayed = [h['pose_id'] for h in hits if h.get('match_level') != 'near_match'
                         and h['kind'] != 'framing_reference']
            if head_id in displayed[:5]:
                protected_head = head_id
        if residual:
            rows = learned_fusion.residual_rows(hits, branches, core, kwargs.get('detail_evidence'),
                                                 joint, strong_projections, limb_consensus)
            if collect_features is not None and not collect_features:
                collect_features.update(rows)
            if rank_weights is not None:
                hits = learned_fusion.rerank_residual(hits, rows, rank_weights)
        if outline_query is not None:
            ordered = outline.rerank(hits, outline_query)
            outline_applied |= ordered != hits
            hits = ordered
        if semantic_distances is not None:
            ordered = rerank_semantic_geometry(hits, branches.get('semantic', {}).get('hits', []),
                                               semantic_distances, engine.validator.positions)
            semantic_fit_applied |= ordered != hits
            hits = ordered
        if depth_proposals:
            eligible = [i for i, h in enumerate(hits) if h.get('match_level') != 'near_match'
                        and h['kind'] != 'framing_reference']
            positions = {hits[i]['pose_id']: i for i in eligible[5:18]}
            if len(eligible) >= 6:
                candidate = next((positions[h['pose_id']] for h in depth_proposals if h['pose_id'] in positions), None)
                if candidate is not None:
                    conflicts = (hits[candidate].get('core_conditions') or {}).get('conflicts') or ()
                    if not any(k == 'held_object' or k == 'hand_count' or k.startswith('held_hand:') for k in conflicts):
                        hits.insert(eligible[4], hits.pop(candidate))
        if compound_contact is not None:
            eligible = [i for i, h in enumerate(hits) if h.get('match_level') != 'near_match'
                        and h['kind'] != 'framing_reference'][:60]
            semantic_hits = branches.get('semantic', {}).get('hits', [])
            scores = fitted.compound_contact_scores(evidence, compound_contact, semantic_hits, structured.prop_ids)
            matched = [i for i in eligible if hits[i]['pose_id'] in scores]
            if matched and len(eligible) > 5:
                best = max(matched, key=lambda i: scores[hits[i]['pose_id']])
                conflicts = (hits[best].get('core_conditions') or {}).get('conflicts') or ()
                if best > eligible[4] and not any(k in ('held_object', 'hand_count') or k.startswith('held_hand:')
                                                 for k in conflicts):
                    hits.insert(eligible[4], hits.pop(best))
                    compound_applied = True
        if protected_head is not None and not compound_applied:
            eligible = [i for i, h in enumerate(hits) if h.get('match_level') != 'near_match'
                        and h['kind'] != 'framing_reference']
            head_position = next(i for i in eligible if hits[i]['pose_id'] == protected_head)
            if head_position > eligible[4]:
                hits.insert(eligible[4], hits.pop(head_position))
        before_screen[:] = [h for h in hits if h.get('match_level') != 'near_match'
                            and h['kind'] != 'framing_reference']
        return hits

    def retrieve(self, query, required=None):
        nonlocal tilted_full
        if pair_geometry is not None:
            baseline = pair_geometry(query, required)
            part = 'upper' if query['coverage_class'] == 'upper_only' else 'full'
            if part != 'full':
                return baseline
            raw = self._clean(self.camera.search(query['keypoints'], query['joint_mask'], part), 'geometry')
            strong_projections[:] = fitted.strong_projections(self, query, required, raw, tilted=is_tilted(query))
            if use_limb_fit:
                fit = proportional_costs(self.camera, query, part)
                limb_consensus[:] = consistent_candidates(self, required, raw, fit)
            if use_head_fit:
                head_proposals[:] = fitted.strong_projections(self, query, required, raw,
                    tilted=False, head_anchor=True, head_limit=head_limit)
            if depth is not None and experiment in DEPTH_AGREEMENT_EXPERIMENTS:
                depth_proposals[:] = fitted.depth_agreement(self, query, depth, required, raw)
            proposals = strong_projections + limb_consensus + head_proposals + depth_proposals
            if not proposals:
                return baseline
            for hit in proposals:
                pid = hit['pose_id']
                self.pair_checks[pid] = hit.get('whole_pose_validation') or self.validator.check(required, pid)
            return dict(baseline, hits=keep_strong_projections(baseline['hits'], proposals), status='ok')
        if promoted:
            required = (previous.get('candidate_validation') or {}).get('requirements')
        positional = use_fields or use_adaptive_fusion or crop is not None
        options, positions, _, original = geometry(
            self, query, required, None if positional or use_robust else bone)
        if recovered:
            rows = fitted.shortlist(self, query, required, original, [])
            return dict(status='ok' if rows else 'no_result', hits=rows,
                        backend='geometry_117_local_rotated_crop', observation_scope='full')
        assert [h['pose_id'] for h in options['position_80']] == [
            h['pose_id'] for h in previous['branches']['geometry']['hits']]
        hits = (shortlist(self, query, required, original, options['position_80'],
                          soften_shape=experiment in ('robust_soft_80', 'robust_soft_rrf_80'))
                if use_robust else options['position_80' if positional else experiment])
        fit = None
        if use_proportions:
            if use_limb_fit:
                part = 'upper' if query['coverage_class'] == 'upper_only' else 'full'
                fit = proportional_costs(self.camera, query, part)
            hits = proportional_shortlist(self, query, required, original, options['position_80'], fit=fit)
        if fitted is not None and experiment.startswith('fitted_'):
            hits = fitted.shortlist(self, query, required, original, options['position_80'], depth)
        if relation is not None:
            raw_distances.update((h['pose_id'], h['distance']) for h in original)
            hits = visible_relations.ordered(positions, relation[0])[:80]
        if alternative is not None:
            alt_options, alt_positions, _, alt_raw = geometry(self, alternative, required, bone=None)
            if alternative.get('improved_crop') and relation is None:
                hits = merge_geometry(alt_options['position_80'], hits)
            elif relation is not None and alternative.get('independent_pose'):
                for h in alt_raw:
                    pid = h['pose_id']
                    raw_distances[pid] = min(raw_distances.get(pid, float('inf')), h['distance'] + .02)
                candidates = {}
                for penalty, source_hits in ((0., positions), (.02, alt_positions)):
                    for hit in source_hits:
                        pid = hit['pose_id']
                        distance = hit['distance'] + penalty
                        if pid not in candidates or distance < candidates[pid][0]:
                            candidates[pid] = distance, hit
                ordered = sorted(candidates.values(), key=lambda row: (
                    row[0] + relation[0][visible_relations.positions[row[1]['pose_id']]], row[1]['pose_id']))
                hits = [hit for _, hit in ordered[:80]]
            else:
                hits = merge_geometry(hits, alt_options['position_80'])
        tilted_full = (experiment == 'visible_full_tilt_80' and query['lower_body_observed'] is True
                       and is_tilted(query))
        if experiment in ROLL_EXPERIMENTS and (query['lower_body_observed'] is False or tilted_full):
            roll_hypotheses[:] = fitted.shortlist(self, query, required, original, [],
                                                 max_roll=1.57, roll_penalty=.04)
            merged = {hit['pose_id']: hit for hit in hits[:16]}
            for hit in roll_hypotheses + hits[16:]:
                if len(merged) >= 80:
                    break
                merged.setdefault(hit['pose_id'], hit)
            hits = list(merged.values())
        mask = query['joint_mask']
        fit_partial = (experiment in PARTIAL_PROJECTION_EXPERIMENTS and query['lower_body_observed'] is False
                       and all(mask[i] for i in (5, 6, 11, 12)) and sum(mask[7:11]) >= 3)
        if experiment in STRONG_PROJECTION_EXPERIMENTS and (query['lower_body_observed'] is True or fit_partial):
            strong_projections[:] = fitted.strong_projections(self, query, required, original,
                                                             tilted=is_tilted(query))
            if strong_projections:
                hits = keep_strong_projections(hits, strong_projections)
        if experiment == 'visible_projection_recall_80':
            expanded_projections[:] = fitted.expanded_projections(self, query, required, original, hits[:60])
            proposed = {hit['pose_id'] for hit in expanded_projections}
            tail = [hit for hit in hits[60:] if hit['pose_id'] not in proposed]
            hits = (hits[:60] + expanded_projections + tail)[:80]
        if fit is not None:
            limb_consensus[:] = consistent_candidates(self, required, original, fit)
            hits = keep_strong_projections(hits, limb_consensus)
        if use_head_fit:
            head_proposals[:] = fitted.strong_projections(self, query, required, original,
                tilted=False, head_anchor=True, head_limit=head_limit)
            hits = keep_strong_projections(hits, head_proposals)
        if depth is not None and experiment in DEPTH_AGREEMENT_EXPERIMENTS:
            depth_proposals[:] = fitted.depth_agreement(self, query, depth, required, original)
            hits = keep_strong_projections(hits, depth_proposals)
        return dict(status='ok' if hits else 'no_result', hits=hits, backend='geometry_117',
                    observation_scope='upper' if query['coverage_class'] == 'upper_only' else 'full')

    engine._fuse = capture
    engine.geometry_search = MethodType(retrieve, engine)
    decision_evidence = (replace(semantic, status='valid', reasons=['multiple_measurable_body_cues'])
                         if promoted else semantic)
    report = engine.search(arm.decide_route(joint, decision_evidence), evidence,
                           review_geometry=arm.integrated_enabled(arm.CFG))
    screen = report.get('hand_skin_screen') or {}
    reasons = {x['hand_skin_screen'].get('reason') for x in screen.get('unverified', [])}
    transient = reasons - {CHECK_LIMIT}
    if transient and transient <= RETRYABLE:
        raise ScreeningUnavailable(sorted(transient))
    recognition_failed = (allow_recognition_error and semantic.status == 'vlm_error'
                          and report['reason'] == 'search_or_description_error_no_generation')
    if transient or (not reasons and report['status'] in ('error', 'partial_error') and not recognition_failed):
        raise RuntimeError((person['index'], report['status'], report.get('reason'), sorted(map(str, reasons))))
    if report['status'] == previous['status'] == 'no_result' and previous.get('generation'):
        report['generation'] = deepcopy(previous['generation'])
    person['dynamic_search'] = report
    person['pose_presentation'] = arm.build_pose_presentation(
        report, person_index=person['index'], output_scope=scope['mesh_scope'],
        resolve_library=arm._library_preview_asset).model_dump(mode='json')
    cues = relation[1] if relation is not None else ()
    if recovered or (alternative or {}).get('improved_crop'):
        cues += ('회전과 인물 영역을 확인해 관절을 다시 읽음',)
    if outline_applied:
        cues += ('러프의 선과 후보 포즈 윤곽 비교',)
    if observed_roles:
        cues += ('관절로 확인한 서로 다른 두 팔의 역할로 검색 보강',)
    if waist_contacts is not None:
        cues += ('손끝과 허리의 접촉 거리 비교',)
    elif contacts is not None:
        cues += ('손끝에서 얼굴·반대 손목 또는 팔까지 거리 비교',)
    if head_proposals:
        cues += ('머리 위치를 관절 투영에 추가',)
    if semantic_fit_applied:
        cues += ('자연어 후보의 관절 맞춤을 확인해 순위 보강',)
    if depth_proposals:
        cues += ('2D 관절과 별도 깊이 추정이 함께 지지하는 후보',)
    if compound_applied:
        cues += ('손 접촉 위치와 관절 투영을 함께 확인해 순위 보강',)
    observed = evidence if recovered else alternative or {}
    repaired = [corrected['elbow'], corrected['wrist']] if corrected else observed.get('alternate_arm_joints', [])
    return (report, before_screen, repaired, depth is not None, promoted, cues)


def merge_geometry(hits, alternates):
    picked = {h['pose_id']: h for h in hits[:16]}
    for i in range(max(len(hits), len(alternates))):
        for hit in hits[16 + i:17 + i] + alternates[i:i + 1]:
            picked.setdefault(hit['pose_id'], hit)
        if len(picked) >= 80:
            break
    return list(picked.values())[:80]
