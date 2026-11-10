"""Combine recall proposals and screen the final candidate list."""

from copy import deepcopy
from math import atan2
from math import pi
from statistics import mean
from .hypothesis_rank_probe import ALIGNED_CLAUSES
from .hypothesis_rank_probe import BOUND_ROLES
from .hypothesis_rank_probe import DENSE
from .hypothesis_rank_probe import inputs_for_person
from .overnight_search import RETRYABLE
from .overnight_search import ScreeningUnavailable
from . import service_context as arm
import numpy as np
from src.search import pose_family_id

ENTANGLED = 'entangled_set_search_unavailable'


def reselect(engine, person, order, library_cameras, trace, *, lexical_fallback=False, crop_recovery=False,
             additional=(), geometry_review=None):
    report = person.get('dynamic_search') or {}
    joint = report.get('geometry_evidence') or {}
    trace['route'] = report.get('route')
    trace['status'] = report.get('status')
    if not crop_recovery and (joint.get('policy_block') or report.get('status') == 'blocked'):
        trace['skipped'] = joint.get('policy_block') or 'blocked'
        return False
    if report.get('status') in ('error', 'not_run'):
        trace['skipped'] = report['status']
        return False
    if report.get('route') == 'basic_model':
        trace['skipped'] = 'basic_model'
        return False
    selected = report.get('selected', [])
    valid = {row['pose_id']: row for row in selected if row.get('match_level') != 'geometry_fallback'}
    for row in additional:
        if row.get('match_level') != 'geometry_fallback':
            valid.setdefault(row['pose_id'], row)
    if not valid and report.get('status') != 'ok':
        # Only pose estimates from a person with adequate existing or recovered evidence
        # may override an undetermined route.
        if not (joint.get('sufficient') or lexical_fallback or crop_recovery):
            trace['skipped'] = 'insufficient_evidence'
            return False
    branches = report.get('branches') or {}
    for row in engine._fuse(branches, limit=200):
        if row.get('match_level') != 'near_match' and row['kind'] == 'library_original':
            valid.setdefault(row['pose_id'], row)
    if [pid for pid in order[:5]] == [row['pose_id'] for row in selected]:
        trace['skipped'] = 'same_order'
        return False
    required = (report.get('candidate_validation') or {}).get('requirements') or {}
    candidates, seen = [], set()
    rejected = trace['rejected'] = {}
    for pid in order + [row['pose_id'] for row in selected]:
        if pid not in engine.validator.positions:
            rejected[pid] = ['no_joint_positions']
            continue
        family = pose_family_id(pid, engine.meta[pid])
        if family in seen:
            if not any(row['pose_id'] == pid for row in candidates):
                rejected[pid] = ['duplicate_family']
            continue
        row = valid.get(pid)
        review = geometry_review if geometry_review and pid == geometry_review['pose_id'] else None
        checked = engine.validator.check(required, pid) if (row is None or review) and required.get('active') else None
        unresolved = checked['conflicts'] + checked['unknown'] if checked else []
        if review and unresolved:
            if any(key not in ('held_object', 'hand_count')
                   and not key.startswith(('held_hand:', 'hand_count:')) for key in unresolved):
                rejected[pid] = unresolved
                continue
            row = dict(row or {}, pose_id=pid, kind='library_original', camera=library_cameras.get(pid),
                       family=family, library_source=engine.meta[pid].get('source'), refine_allowed=False,
                       release_allowed=False, actual_mesh_validated=False, match_level='geometry_fallback',
                       review_required=True, not_exact_match=True, constraints_satisfied=False,
                       evidence={'geometry': review['evidence']},
                       core_conditions={'whole_pose_validation': checked})
        elif row is None:
            conflicts = [key for key in unresolved
                         if key in ('held_object', 'hand_count') or key.startswith('held_hand:')]
            if conflicts:
                rejected[pid] = conflicts
                continue
            row = dict(pose_id=pid, kind='library_original', camera=library_cameras.get(pid),
                       family=family, library_source=engine.meta[pid].get('source'), refine_allowed=False,
                       release_allowed=False, actual_mesh_validated=False)
        candidates.append(row)
        seen.add(family)
    trace['before_screen'] = [row['pose_id'] for row in candidates]
    if not candidates:
        return False
    deferred = getattr(engine, 'defer_library_hand_screen', False)
    if deferred:
        accepted = candidates[:5]
        trace['screen_deferred'] = True
        screening = {'rejected': [], 'unverified': []}
    else:
        hands = {side: person['hands'][side]['preset'] for side in ('left', 'right')}
        session = engine.hand_skin_gate.session(hands)
        accepted = session.filter(candidates, limit=5)
        screening = session.report()
    trace['after_screen'] = [row['pose_id'] for row in accepted]
    trace['screen_rejected'] = {row['pose_id']: row['hand_skin_screen'].get('reason')
                                for key in ('rejected', 'unverified') for row in screening[key]}
    reasons = {item['hand_skin_screen'].get('reason') for item in screening['unverified']}
    if reasons & RETRYABLE:
        raise ScreeningUnavailable(sorted(reasons))
    if not accepted:
        return False
    report.update(status='ok', reason=None, selected=accepted)
    if not deferred:
        report['hand_skin_screen'] = screening
    if crop_recovery:
        report.update(route='geometric', reason='single_subject_recovered_from_crop')
        joint['policy_block'] = None
    elif lexical_fallback and report['route'] == 'undetermined':
        report.update(route='semantic', reason='specific_visible_pose_description')
    if report.get('mirror_references'):
        previous = report['mirror_references'][0]
        original = (previous.get('mirror_comparison') or {}).get('original_pose_id')
        if original not in [r['pose_id'] for r in accepted[:4]]:
            report.pop('mirror_references', None)
            report.pop('mirror_comparison', None)
    person['pose_presentation'] = arm.build_pose_presentation(
        report, person_index=person['index'], output_scope=person['output_scope']['mesh_scope'],
        resolve_library=arm._library_preview_asset).model_dump(mode='json')
    return True


def strong_crop(models):
    for model in models.values():
        visible = [score for valid, score in zip(model['joints'][5:], model['scores'][5:], strict=True) if valid]
        if len(visible) >= 9 and mean(visible) >= .43 and 'full_xy' in model['profiles']:
            return True
    return False


def primary_subject(people, image, *, hypotheses=None):
    """Discard isolated detail detections when one body clearly dominates the cut."""
    if len(people) <= 1:
        return people

    def area(person):
        box = person.get('box')
        if not box:
            return 0.
        x1, y1, x2, y2 = box
        return max(0., min(x2, image['width']) - max(0., x1)) * max(
            0., min(y2, image['height']) - max(0., y1))

    ranked = sorted(people, key=area, reverse=True)
    main, *others = ranked
    joint = main['dynamic_search']['geometry_evidence']
    area_ratio = area(main) / (image['width'] * image['height'])
    recovered = (hypotheses is not None and joint['policy_block'] == ENTANGLED and area_ratio >= .30
                 and strong_crop(hypotheses.get(main['index'], {})))
    if (not recovered and (not joint['sufficient'] or joint['policy_block'])) or area_ratio < .18:
        return people
    for other in others:
        evidence = other['dynamic_search']['geometry_evidence']
        scope = other['output_scope']['detected']
        body_count = evidence['metrics'].get('valid_body_joints') or 0
        detail = (hypotheses is not None and scope in ('head', 'shoulder_head')
                  and body_count <= 1 and not hypotheses.get(other['index']) and area_ratio >= .40)
        max_area = .10 if recovered else .40 if detail else .28
        allowed_block = ENTANGLED if recovered else None
        if (evidence['sufficient'] or evidence['policy_block'] not in (None, allowed_block)
                or area(other) > max_area * area(main)):
            return people
        if scope not in ('head', 'shoulder_head', None) and body_count >= 4:
            return people
    return [main]


def lexical_proposal(person, language):
    """Use a specific pose phrase when the original joints are weak."""
    report = person.get('dynamic_search') or {}
    joint = report.get('geometry_evidence') or {}
    description = (report.get('description_evidence') or {}).get('description') or {}
    if (joint.get('sufficient') or not description.get('visible')
            or description.get('visible_scope') not in ('full', 'half', 'bust')):
        return None
    specific = set(description.get('postures') or ()) - {'standing', 'sitting'}
    if not specific and not description.get('body_details'):
        return None
    for field, threshold in (('details', .46), ('text', .40)):
        hits = language.get(field, {}).get('top', ())
        if len(hits) >= 3 and hits[0][1] >= threshold and hits[0][1] - hits[2][1] >= .075:
            return hits[0][0]
    return None


def supported_sentence_proposal(language, models):
    """Use a sentence winner if a full-body fit also puts it near the top."""
    candidates = language.get('dense_details', {}).get('top', ())
    if len(candidates) < 2 or candidates[0][1] - candidates[1][1] < .003:
        return None
    pose_id = candidates[0][0]
    for model in models.values():
        for part in ('full_xy', 'full_head', 'full_bend'):
            profile = model['profiles'].get(part, {})
            for mode in ('geometry', 'hybrid'):
                if any(pid == pose_id for pid, _ in profile.get(mode, {}).get('top', ())[:10]):
                    return pose_id
    return None


def clause_slot(language):
    """Reserve the last slot when two descriptions strongly support the same pose."""
    body = language.get('dense_clauses', {}).get('top', ())
    actions = language.get('dense_action_tags', {}).get('top', ())
    full = language.get('dense_details', {}).get('top', ())
    if len(body) >= 2 and len(actions) >= 2 and body[0][0] == actions[0][0]:
        if body[0][1] - body[1][1] > .026 and actions[0][1] - actions[1][1] > .018:
            return body[0][0]
    if len(full) >= 2 and full[0][1] - full[1][1] > .003:
        if full[0][0] in {pid for pid, _ in body[:3]}:
            return full[0][0]
    return None


def fused_order(first, second):
    scores = {}
    for rows in (first, second):
        for rank, row in enumerate(rows[:5], 1):
            pid = row['pose_id']
            scores[pid] = scores.get(pid, 0.) + 1. / (rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def bound_slot(language):
    physical = language.get('bound_roles') or {}
    combined = (language.get('bound_joint_roles') or {}).get('top', ())
    if physical.get('role_count', 0) < 2 or len(combined) < 2:
        return None
    pid, score = combined[0]
    if score - combined[1][1] >= .025 and pid in {item for item, _ in physical['top'][:10]}:
        return pid
    return None


def aligned_slot(language):
    return (language.get(ALIGNED_CLAUSES) or {}).get('proposal')


def visual_candidates(text, models):
    """Keep only candidates backed by both contour methods and a joint fit."""
    outline = (text.get('outline') or {}).get('top', ())
    ink = (text.get('limb_ink') or {}).get('top', ())
    outer = {pid: rank for rank, (pid, score) in enumerate(outline[:3], 1) if score >= .72}
    inner = {pid: rank for rank, (pid, score) in enumerate(ink[:3], 1) if score >= .68}
    shared = outer.keys() & inner.keys()
    if not shared:
        return []
    fit = {}
    for model in models.values():
        for methods in model['profiles'].values():
            for result in methods.values():
                for rank, (pid, _) in enumerate(result['top'][:5], 1):
                    if pid in shared:
                        fit[pid] = min(rank, fit.get(pid, 6))
    return sorted(fit, key=lambda pid: (outer[pid] + inner[pid] + fit[pid], pid))


def tilt_candidates(text, models):
    original = models.get('original') or models.get('consensus')
    torso_joints = (5, 6, 11, 12)
    if not original or not all(original['joints'][i] for i in torso_joints):
        return []
    a, b, c, d = (original['keypoints'][i] for i in torso_joints)
    x = a[0] + b[0] - c[0] - d[0]
    y = a[1] + b[1] - c[1] - d[1]
    if atan2(abs(x), abs(y)) < pi / 4:
        return []
    torso, full = {}, {}
    for name, model in models.items():
        for kind, profile in model['profiles'].items():
            if kind == 'torso':
                table = torso
            elif name.startswith('body_x') and kind.startswith(('full_', 'refit_full')):
                table = full
            else:
                continue
            data = (profile.get('geometry') or {}).get('top', ())
            for rank, (pid, _) in enumerate(data[:3], 1):
                table[pid] = min(rank, table.get(pid, 4))
    ink = (text.get('limb_ink') or {}).get('top', ())
    visible = {pid: rank for rank, (pid, _) in enumerate(ink[:10], 1)}
    shared = torso.keys() & full.keys() & visible.keys()
    return sorted(shared, key=lambda pid: (torso[pid] + full[pid] + visible[pid], pid))


def prop_review_candidate(person, models):
    report = person['dynamic_search']
    description = report['description_evidence'].get('description') or {}
    required = (report.get('candidate_validation') or {}).get('requirements') or {}
    joints = report['geometry_evidence']
    if (not required.get('hand_counts') or description.get('held_object_hands')
            or description.get('held_object_hand_counts') or not person['lower_body_observed']
            or not joints['sufficient'] or (joints['metrics'].get('mean_valid_body_score') or 0.) < .50):
        return None
    leaders, supporters = {}, {}
    for name, model in models.items():
        family = ('body_x' if name.startswith('body_x') else name
                  if name in ('original', 'rtmw', 'consensus') else 'crop')
        for method, profile in model['profiles'].items():
            top = (profile.get('geometry') or {}).get('top', ())
            for pid, _ in top[:5]:
                supporters.setdefault(pid, set()).add(family)
            if method != 'full_bend' or len(top) < 2:
                continue
            pid, cost = top[0]
            margin = top[1][1] - cost
            if cost <= .18 and margin >= .015 and margin > leaders.get(pid, (0., 0.))[0]:
                leaders[pid] = margin, cost
    eligible = [pid for pid in leaders if len(supporters[pid]) >= 3]
    if not eligible:
        return None
    pid = max(eligible, key=lambda key: (leaders[key][0], -leaders[key][1]))
    margin, cost = leaders[pid]
    return dict(pose_id=pid, evidence=dict(method='projected_limb_bend', distance=cost, margin=margin))


def apply_ranking(engine, cut, hypotheses, language, weights, clause_weights, args, *,
                  relational, excluded_text, visual, tilt, prop_review, contextual, heads,
                  detected_props):
    """Apply the display strategy using observations supplied by the caller."""
    updated = deepcopy(cut)
    changed = False
    traces = {}
    if args.recover_subject:
        updated['people'] = primary_subject(
            updated['people'], updated['image'],
            hypotheses=hypotheses if args.subject_detail else None)
        changed = len(updated['people']) != len(cut['people'])
    for person in updated['people']:
        index = person['index']
        presentation = person.get('pose_presentation') or {}
        displayed = sorted(presentation.get('library_candidates', []) +
                           presentation.get('review_candidates', []), key=lambda x: x['rank'])
        prior = [(item['pose_id'], item['kind']) for item in displayed]
        text = language.get(index, {})
        models = hypotheses.get(index, {})
        ranking_text = {k: v for k, v in text.items() if k not in excluded_text}
        ids, features = inputs_for_person(models, ranking_text, prior,
                                         relational=relational, dense=args.dense,
                                         outline=args.outline, enhanced=args.enhanced,
                                         clauses=args.clauses)
        indexes = np.argsort(-(features @ weights), kind='stable')[:10]
        order = [ids[i] for i in indexes]
        proposal = lexical_proposal(person, text) if args.recover_subject else None
        sentence = None
        if args.subject_detail and len(updated['people']) < len(cut['people']) and proposal is None:
            sentence = supported_sentence_proposal(text, models)
            if sentence is not None:
                order = list(dict.fromkeys(order[:4] + [sentence] + order[4:]))
        if proposal is not None:
            if (person.get('dynamic_search') or {}).get('route') == 'undetermined':
                order = [proposal]
            else:
                order = list(dict.fromkeys(order[:4] + [proposal] + order[4:]))
        cameras = {item['pose_id']: item.get('camera') for item in displayed}
        crop_recovery = (args.subject_detail and len(updated['people']) < len(cut['people'])
                         and person['dynamic_search']['geometry_evidence']['policy_block'] == ENTANGLED)
        trace = traces[index] = dict(shortlist=order)
        clause_person = deepcopy(person) if clause_weights is not None else None
        changed |= reselect(engine, person, order, cameras, trace,
                            lexical_fallback=proposal is not None, crop_recovery=crop_recovery)
        if clause_person is not None:
            clause_text = {k: v for k, v in text.items() if k not in (*DENSE, *BOUND_ROLES, ALIGNED_CLAUSES)}
            clause_ids, clause_features = inputs_for_person(
                models, clause_text, prior, relational=relational,
                enhanced=args.enhanced, clauses=True)
            indexes = np.argsort(-(clause_features @ clause_weights), kind='stable')[:10]
            alternate_order = [clause_ids[i] for i in indexes]
            if proposal and clause_person['dynamic_search']['route'] == 'undetermined':
                alternate_order = [proposal]
            elif proposal or sentence:
                alternate_order = list(dict.fromkeys(
                    alternate_order[:4] + [proposal or sentence] + alternate_order[4:]))
            alternate_trace = trace['clauses'] = dict(shortlist=alternate_order)
            reselect(engine, clause_person, alternate_order, cameras, alternate_trace,
                     lexical_fallback=proposal is not None, crop_recovery=crop_recovery)
            first = person['dynamic_search'].get('selected', [])
            second = clause_person['dynamic_search'].get('selected', [])
            if second:
                merged = fused_order(first, second)
                if (args.mirror_budget and len(merged) >= 5
                        and person['dynamic_search'].get('mirror_references')
                        and merged[4] in {row['pose_id'] for row in second[:3]}
                        and merged[4] in order):
                    # Presentation can show only four originals when it has
                    # a mirror reference. Prefer the corroborated candidate.
                    merged[3], merged[4] = merged[4], merged[3]
                merged_trace = trace['fused'] = dict(shortlist=merged)
                changed |= reselect(engine, person, merged, cameras, merged_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery, additional=second)
        if args.clause_slot:
            candidate = clause_slot(text)
            chosen = [row['pose_id'] for row in person['dynamic_search'].get('selected', [])]
            if candidate and chosen and candidate not in chosen:
                updated_order = chosen[:4] + [candidate] + chosen[4:]
                second_trace = trace['clause_slot'] = dict(shortlist=updated_order, proposal=candidate)
                changed |= reselect(engine, person, updated_order, cameras, second_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery)
        if args.bound_slot and not trace.get('clause_slot', {}).get('after_screen'):
            candidate = bound_slot(text)
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            if candidate and chosen and candidate not in chosen:
                position = 3 if report.get('mirror_references') else 4
                updated_order = chosen[:position] + [candidate] + chosen[position:]
                bound_trace = trace['bound_slot'] = dict(shortlist=updated_order, proposal=candidate)
                changed |= reselect(engine, person, updated_order, cameras, bound_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery)
        if args.aligned_slot and not any(trace.get(stage, {}).get('after_screen')
                                        for stage in ('bound_slot', 'clause_slot')):
            candidate = aligned_slot(text)
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            if candidate and chosen and candidate not in chosen:
                position = 3 if report.get('mirror_references') else 4
                updated_order = chosen[:position] + [candidate] + chosen[position:]
                aligned_trace = trace['aligned_slot'] = dict(shortlist=updated_order, proposal=candidate)
                changed |= reselect(engine, person, updated_order, cameras, aligned_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery)
        if args.visual_slot and not any(trace.get(stage, {}).get('after_screen')
                                        for stage in ('aligned_slot', 'bound_slot', 'clause_slot')):
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            position = 3 if report.get('mirror_references') else 4
            shown = set(chosen[:position+1])
            shown.update(row['pose_id'] for row in report.get('mirror_references', []))
            candidate = next((pid for pid in visual.get(index, ()) if pid not in shown), None)
            if candidate and chosen:
                updated_order = chosen[:position] + [candidate] + chosen[position:]
                visual_trace = trace['visual_slot'] = dict(shortlist=updated_order, proposal=candidate)
                changed |= reselect(engine, person, updated_order, cameras, visual_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery)
        evidence_slots = (trace.get(stage, {}) for stage in
                          ('visual_slot', 'aligned_slot', 'bound_slot'))
        confirmed_slot = any(step.get('proposal') in step.get('after_screen', ())
                             for step in evidence_slots)
        if args.tilt_slot and not confirmed_slot:
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            position = 3 if report.get('mirror_references') else 4
            shown = set(chosen[:position+1])
            shown.update(row['pose_id'] for row in report.get('mirror_references', []))
            candidate = next((pid for pid in tilt.get(index, ()) if pid not in shown), None)
            if candidate and chosen:
                updated_order = chosen[:position] + [candidate] + chosen[position:]
                tilt_trace = trace['tilt_slot'] = dict(shortlist=updated_order, proposal=candidate)
                changed |= reselect(engine, person, updated_order, cameras, tilt_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery)
                confirmed_slot |= candidate in tilt_trace.get('after_screen', ())
        if args.prop_review_slot and not confirmed_slot:
            review = prop_review.get(index)
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            if review and chosen and review['pose_id'] not in chosen:
                position = 3 if report.get('mirror_references') else 4
                displaced = chosen[position:position+1]
                required = (report.get('candidate_validation') or {}).get('requirements') or {}
                if displaced and required.get('active'):
                    checked = engine.validator.check(required, displaced[0])
                    if checked['status'] == 'passed':
                        continue
                updated_order = chosen[:position] + [review['pose_id']] + chosen[position:]
                review_trace = trace['prop_review_slot'] = dict(
                    shortlist=updated_order, proposal=review['pose_id'])
                changed |= reselect(engine, person, updated_order, cameras, review_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery, geometry_review=review)
                confirmed_slot |= review['pose_id'] in review_trace.get('after_screen', ())
        if args.context_slot and not confirmed_slot:
            contextual_pick = contextual.get(index)
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            if contextual_pick and chosen and contextual_pick['pose_id'] not in chosen:
                candidate = contextual_pick['pose_id']
                position = 3 if report.get('mirror_references') else 4
                updated_order = chosen[:position] + [candidate] + chosen[position:]
                context_trace = trace['context_slot'] = dict(
                    shortlist=updated_order, proposal=candidate, basis=contextual_pick['basis'])
                changed |= reselect(engine, person, updated_order, cameras, context_trace,
                                    lexical_fallback=proposal is not None,
                                    crop_recovery=crop_recovery)
        if args.head_slot and (head := heads.get(index)):
            report = person['dynamic_search']
            if report['route'] == 'undetermined' and not report.get('selected'):
                head_order = [head['pose_id']]
                head_trace = trace['head_slot'] = dict(shortlist=head_order, basis=head['basis'])
                changed |= reselect(engine, person, head_order, cameras, head_trace,
                                    lexical_fallback=True)
        if args.detected_prop_slot and (prop := detected_props.get(index)):
            report = person['dynamic_search']
            chosen = [row['pose_id'] for row in report.get('selected', [])]
            candidate = prop['pose_id']
            if candidate not in chosen:
                position = 3 if report.get('mirror_references') else 4
                prop_order = chosen[:position] + [candidate] + chosen[position:]
                prop_trace = trace['detected_prop_slot'] = dict(
                    shortlist=prop_order, proposal=candidate, basis=prop['basis'])
                changed |= reselect(engine, person, prop_order, cameras, prop_trace,
                                    lexical_fallback=True)
    return updated, changed, traces
