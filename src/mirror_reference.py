"""One real opposite-hand library reference, never a match or a new pose."""
from copy import deepcopy


def _opposite(side):
    return {'left': 'right', 'right': 'left'}.get(side, side)


def one_observed_side(description, required=None):
    if not description:
        return None
    known = {s for s, state in description.arm_states.items()
             if s in ('left', 'right') and state != 'unknown'}
    relations = {r.get('side') for r in description.pose_relations
                 if r.get('chain_visible') is True and r.get('side_confidence') == 'high'}
    held = set(description.held_object_hands.values())
    known |= (relations | held) & {'left', 'right'}
    required = required or {}
    known |= {s for s, state in required.get('arms', {}).items()
              if s in ('left', 'right') and state != 'unknown'}
    required_hands = set(required.get('object_hands', {}).values())
    known |= required_hands & {'left', 'right'}
    if ('both' in relations | held | required_hands
            or 'two' in required.get('hand_counts', {}).values() or len(known) != 1):
        return None
    return next(iter(known))


def mirror_reference(engine, selected, description, required, skin_session, *, head_only=False):
    """Recheck a bound family member against explicitly swapped side conditions.

    All other conditions are kept. The original observation is not rewritten,
    hidden limbs are not filled, and the returned row has no retrieval score.
    """
    side = one_observed_side(description, required)
    def result(reason, row=None):
        return row, dict(status='ready' if row else 'unavailable', reason=reason,
                         observed_side=side, reference_side=_opposite(side))
    if not side:
        return None, dict(status='not_requested', reason='one_known_arm_required')
    budget = min(engine.top_k, 5)
    if budget < 2:
        return result('display_budget_too_small')
    if not engine.validator or (skin_session is None and head_only is not True) or required is None:
        return result('validation_unavailable')
    from .pose_query import prepare_pose_query
    from .semantic_search import SemanticQuery
    from .semantic_adapter_285 import excluded_pose_ids
    records = getattr(engine.semantic, 'core_records', {})
    body = getattr(engine.semantic, 'body', {})
    hashes = getattr(engine.camera, 'hashes', {})
    def bound(pid):
        meta = records.get(pid, {}).get('meta') or {}
        digest = hashes.get(pid)
        return (meta if digest and meta.get('bvh_sha256') == digest
                and body.get(pid, {}).get('source_sha256') == digest
                and isinstance(meta.get('is_mirror'), bool)
                and isinstance(meta.get('pose_family_id'), str) and meta['pose_family_id'] else None)
    # Same arm/action conditions on the opposite anatomical side, not an
    # unconditioned search or permission to ignore a failed condition.
    swapped = deepcopy(required)
    swapped['arms'] = {_opposite(s): state for s, state in required['arms'].items()}
    swapped['object_hands'] = {obj: _opposite(s) for obj, s in required.get('object_hands', {}).items()}
    if not hasattr(engine.semantic, 'conditions'):
        return result('condition_validation_unavailable')
    prepared = prepare_pose_query(SemanticQuery.from_description(description), engine.semantic.service.index.lexicon)
    mirrored = deepcopy(prepared)
    mirrored['held_object_hands'] = {obj: _opposite(s) for obj, s in prepared.get('held_object_hands', {}).items()}
    for relation in mirrored.get('action_relations', []):
        relation['side'] = _opposite(relation.get('side'))
    excluded = excluded_pose_ids()
    selected_ids = {row['pose_id'] for row in selected}
    reason = 'bound_mirror_not_available'
    for original in selected[:budget - 1]:
        pid = original['pose_id']
        meta = bound(pid)
        if (not meta or pid not in engine.allowed or pid in excluded
                or original.get('review_required') or original.get('not_exact_match')
                or original.get('constraints_satisfied') is False):
            continue
        original_check = engine.validator.check(required, pid)
        if original_check['status'] not in ('passed', 'not_required'):
            continue
        family = meta['pose_family_id']
        for other in sorted(engine.allowed - excluded - selected_ids):
            other_meta = bound(other)
            if (not other_meta or other_meta['pose_family_id'] != family
                    or other_meta['is_mirror'] == meta['is_mirror']
                    or (not engine.include_stored_compositions and engine.meta[other].get('source') == 'generated_composition')):
                continue
            checked = engine.validator.check(swapped, other)
            if checked['status'] not in ('passed', 'not_required'):
                reason = 'mirror_condition_check_failed'
                continue
            core = engine.semantic.conditions(mirrored, records[other], other)
            if ((mirrored.get('enabled') and not core) or core and
                        (core['conflicts'] or any(r['status'] != 'match' for r in core['relations'].values())
                         or (mirrored.get('objects') and core['object_state'] != 'match'))):
                reason = 'mirror_condition_check_failed'
                continue
            row = dict(pose_id=other, source_bvh_sha256=hashes[other], family=family, kind='mirror_reference',
                library_source=engine.meta[other].get('source'), camera=deepcopy(original.get('camera')),
                semantic_preview_camera=deepcopy(original.get('semantic_preview_camera')),
                rank_score=None, score=None, score_kind=None, evidence={}, review_required=True, not_exact_match=True,
                constraints_satisfied=False, match_level='opposite_hand_reference', lookup_allowed=True,
                refine_allowed=False, release_allowed=False, actual_mesh_validated=False,
                mirror_comparison=dict(family_id=family, original_pose_id=pid, original_bvh_sha256=hashes[pid],
                    is_mirror=other_meta['is_mirror'], observed_side=side, reference_side=_opposite(side),
                    reason='opposite_hand_reference', matches_observed_side=False),
                core_conditions=dict(whole_pose_validation=engine.validator.check(required, other),
                                     reference_conditions=checked, reference_relations=core))
            if head_only is True:
                from .hand_scope import head_hand_omission
                return result('opposite_hand_reference', {**row, 'hand_skin_screen': head_hand_omission()})
            accepted = skin_session.filter([row], limit=1)
            if accepted:
                return result('opposite_hand_reference', accepted[0])
            reason = 'mirror_hand_skin_not_clear'
    return result(reason)
