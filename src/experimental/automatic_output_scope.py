"""Local per-person adapter to the dynamic-routing anatomical framing contract."""
from src.body_scope import BodyScope, ScopeDetection, observed_legs, promote_for_observed_legs
from .partial_body_heading import owns_observations


def choose_output_scope(person, observation, image_size, *, contact=False):
    raw = observation.get('body_scope')
    if raw not in BodyScope._value2member_map_:
        raise ValueError('missing fresh per-person framing evidence')
    head = observation.get('head_center')
    if head is not None and not owns_observations(person['bbox'], {'head_center': head}, image_size):
        raise ValueError('framing evidence belongs to a different person')
    scope = ScopeDetection(BodyScope(raw), 'vlm_person')
    regions = set(observation.get('visible_regions', []))
    # Only corroborated leg observations can widen a crop. On extreme face
    # closeups, a skeleton detector may put a complete body on the face outline.
    legs = observed_legs(person['keypoints'], person['valid_mask'], (), image_size) if regions & {'knees','feet'} else ()
    scope = promote_for_observed_legs(scope, legs)
    output = scope.detected.value
    reasons = []
    if output in ('bust', 'head') and (contact or regions & {'forearms', 'hands'}):
        output = 'half'; reasons.append('preserve_observed_arms_and_hands')
    elif output == 'head':
        # User-requested output family is full / upper body / shoulders-head.
        # Preserve a shoulder/neck base rather than exporting a detached face.
        output = 'bust'; reasons.append('minimum_output_is_shoulders_to_head')
    return dict(output_scope=output, raw_detected_scope=raw, dynamic_scope=scope.detected.value,
        dynamic_scope_source=scope.source, evidence=observation, observed_legs=list(legs),
        policy_adjustments=reasons, head_ownership_checked=head is not None,
        source='fresh_local_vlm+dynamic_routing_body_scope', manual_scope=False)
