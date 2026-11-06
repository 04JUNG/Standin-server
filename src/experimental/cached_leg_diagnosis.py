"""Offline leg intent from original cached VLM text, independent of arm adapters.

Named anatomical sides and explicit bilateral legs are linguistic evidence;
neither proves a visible chain, ground contact, weight support or cross order.
Development thresholds compare shapes, not evaluation scores or image accuracy.
"""
import math
import re

MENTION = re.compile(r'(?P<side>오른|왼)(?:쪽)?\s*(?P<part>무릎|다리|발|팔|손)(?!가락|바닥|등)')
BOTH = re.compile(r'(?:두\s*다리|양\s*다리|양쪽\s*다리|양\s*무릎|두\s*무릎)')
UNCERTAIN = re.compile(r'않|아니|불확|불명|모호|가려|추정|듯|보이지|안\s*(?:뻗|펴|굽|꼬)')
SCREEN = re.compile(r'화면|사진|그림|관찰자|보는\s*사람')
LEG_CROSS = re.compile(r'(?:다리|발목|종아리)[^,，。.!?;\n]{0,40}(?:꼬|꼰|교차|엇갈)|(?:꼬인|꼰|교차한|엇갈린)\s*(?:다리|발목|종아리)')
LEG_ACTIONS = {'straight_leg', 'bent_leg', 'raised_knee', 'kneeling_ground'}


def _claim(span, part):
    if UNCERTAIN.search(span):
        return None
    # A hand placed on a knee is not a raised leg observation.
    if part == '무릎' and re.search(r'바닥|지면', span) and re.search(r'꿇|닿|붙|대고', span):
        action = 'kneeling_ground'
    elif part == '무릎' and re.search(r'세운|세우|세워|높게|높이|무릎이\s*올라', span):
        action = 'raised_knee'
    elif re.search(r'펴|뻗', span) or (re.search(r'곧게', span) and not re.search(r'곧게\s*(?:서|선)', span)):
        action = 'straight_leg'
    elif re.search(r'굽|구부', span):
        action = 'bent_leg'
    else:
        return None
    return {'action': action, 'elevated': bool(re.search(r'책상\s*위|탁자\s*위', span)),
            'support_surface': 'table' if re.search(r'책상\s*위|탁자\s*위', span) else
                               'ground' if action == 'kneeling_ground' else 'unknown'}


def targets_from_cached(description):
    rows = {s: {'side': s, 'action': 'unknown', 'confidence': 'low', 'evidence': [],
                'laterality_source': 'unknown', 'chain_visible': False, 'visible': False,
                'elevated': False, 'support_surface': 'unknown', 'reasons': []}
            for s in ('left', 'right')}
    result = {'schema': 'standin.cached-leg-target.v1', 'legs': rows, 'relations': [],
              'source': 'original_cached_vlm', 'new_api_calls': 0, 'new_api_tokens': 0,
              'automatic_image_accuracy_measured': False}
    if description.get('visible') is not True or description.get('visible_scope') != 'full':
        for row in rows.values():
            row['reasons'] = ['lower_body_not_observed_in_cached_description']
        return result
    review = description.get('laterality_review') or {}
    arm_actions = {'touch_face', 'touch_head', 'arm_raised', 'phone_hold', 'umbrella_grip',
                   'chin_support', 'knee_rest', 'folded_arms'}
    # A reviewed arm abstention is not a reviewed leg abstention. Unknown/global
    # changes remain conservative; explicit arm actions do not erase leg text.
    leg_changes = [c for c in review.get('changes', []) if c.get('action') not in arm_actions]
    abstain = review.get('status') == 'abstain' or any(c.get('status') == 'abstain' for c in leg_changes)
    conflicts = bool(leg_changes)
    claims = {s: [] for s in rows}
    for field in ('body_details', 'context_evidence'):
        value = description.get(field) or ''
        mentions = list(MENTION.finditer(value))
        for index, match in enumerate(mentions):
            if match['part'] not in ('다리', '무릎', '발'):
                continue
            prefix = re.split(r'[,，。.!?;\n]', value[:match.start()])[-1]
            if SCREEN.search(prefix):
                continue
            # Don't read a named knee inside an earlier arm/hand clause as its
            # own action. Explicit kneeling/raised verbs are still required.
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(value)
            span = re.split(r'[,，。.!?;\n]', value[match.start():end], 1)[0].strip()
            claim = _claim(span, match['part'])
            if claim:
                side = 'left' if match['side'] == '왼' else 'right'
                claims[side].append({**claim, 'text': span, 'source_field': field,
                                     'laterality_source': 'explicit_natural_language'})
        for match in BOTH.finditer(value):
            prefix = re.split(r'[,，。.!?;\n]', value[:match.start()])[-1]
            if SCREEN.search(prefix):
                continue
            end = next((m.start() for m in mentions if m.start() > match.start()), len(value))
            span = re.split(r'[,，。.!?;\n]', value[match.start():end], 1)[0].strip()
            claim = _claim(span, '다리')
            if claim:
                for side in rows:
                    claims[side].append({**claim, 'text': span, 'source_field': field,
                                         'laterality_source': 'explicit_bilateral_natural_language'})
        # Crossing is an atomic relationship even if the family/order is not
        # available. It must never become two independent straight-leg edits.
        if LEG_CROSS.search(value) and not UNCERTAIN.search(value):
            result['relations'].append({'kind': 'crossed_legs', 'top_side': 'unknown',
                                        'source_field': field, 'evidence': value,
                                        'requires_atomic_relation_checker': True})
    for side, row in rows.items():
        evidence = claims[side]
        row['evidence'] = evidence
        if not evidence:
            row['reasons'] = ['no_explicit_leg_action']
            continue
        row['visible'] = True  # textual observation, not visible chain proof
        actions = {c['action'] for c in evidence}
        if len(actions) != 1:
            row['reasons'] = ['contradictory_explicit_leg_actions']
            continue
        row.update(action=next(iter(actions)), confidence=description.get('certainty', 'low'),
                   laterality_source=evidence[0]['laterality_source'],
                   elevated=any(c['elevated'] for c in evidence))
        surfaces = {c['support_surface'] for c in evidence} - {'unknown'}
        row['support_surface'] = next(iter(surfaces)) if len(surfaces) == 1 else 'unknown'
        if abstain or conflicts:
            row.update(confidence='low', laterality_source='explicitly_abstained' if abstain else 'conflicting_review')
            row['reasons'] = ['explicit_laterality_review_abstained' if abstain else 'laterality_review_changes_require_resolution']
        # No limb chain evidence is inferred from an absent review or from arm
        # review text. Only an actual explicit leg-chain record can enable it.
        for relation in description.get('pose_relations') or []:
            if (relation.get('kind') == 'leg_chain' and relation.get('side') in (side, 'both')
                    and relation.get('chain_visible') is True and relation.get('side_confidence') == 'high'):
                row['chain_visible'] = not abstain and not conflicts
    return result


def diagnose(target, state):
    if target.get('schema') != 'standin.cached-leg-target.v1':
        raise ValueError('cached leg target required')
    rows = []
    for side in ('left', 'right'):
        wanted = target['legs'][side]
        measured = state.get('legs', {}).get(side)
        row = {'side': side, 'target': wanted, 'measured': measured, 'status': 'held',
               'checks': [], 'reasons': [], 'candidate_allowed': False, 'auto_apply_allowed': False}
        rows.append(row)
        if target['relations']:
            row['reasons'] = ['coupled_crossing_requires_family_and_order_checker']
            continue
        if wanted['confidence'] != 'high' or wanted['action'] not in LEG_ACTIONS:
            row['reasons'] = wanted['reasons'] or ['uncertain_or_missing_target']
            continue
        if not measured or measured.get('valid') is not True:
            row['reasons'] = ['native_measurements_unavailable']
            continue
        def check(key, predicate):
            value = measured.get(key)
            known = type(value) in (int, float) and math.isfinite(value)
            row['checks'].append({'metric': key, 'value': value,
                                  'status': ('match' if predicate(value) else 'mismatch') if known else 'held'})
        action = wanted['action']
        if action == 'straight_leg':
            check('flexion_deg', lambda x: x <= 25.)
        elif action == 'bent_leg':
            check('flexion_deg', lambda x: x >= 35.)
        elif action == 'raised_knee':
            check('knee_above_other_torso', lambda x: x >= .25)
            check('flexion_deg', lambda x: x >= 35.)
        elif action == 'kneeling_ground':
            check('knee_above_other_torso', lambda x: x <= -.25)
            check('flexion_deg', lambda x: x >= 70.)
        statuses = {c['status'] for c in row['checks']}
        row['shape_status'] = 'mismatch' if 'mismatch' in statuses else 'held' if 'held' in statuses else 'match'
        row['status'] = row['shape_status']
        surface = wanted['support_surface']
        if surface != 'unknown':
            row['support_status'] = 'held'
            row['reasons'].append(surface + '_surface_geometry_not_provided')
            if row['status'] == 'match':
                row['status'] = 'shape_matched_contact_held'
        row['candidate_allowed'] = action == 'straight_leg' and row['shape_status'] == 'mismatch'
        if row['candidate_allowed']:
            row['reasons'].append('local_shape_candidate_only_requires_surface_review')
    return {'schema': 'standin.cached-leg-diagnosis.v1', 'legs': rows,
            'candidate_sides': [r['side'] for r in rows if r['candidate_allowed']],
            'auto_apply_allowed': False, 'score': None,
            'threshold_source': 'development_shape_tolerances_not_evaluation_answers'}
