"""Provider response shape and content-free diagnostics for pose descriptions."""
import json
import re


class DescriptionResponseError(ValueError):
    def __init__(self, code, field):
        self.code, self.field = code, field
        super().__init__(f"pose_description:{code}:{field}")


def description_payload(text):
    if not isinstance(text, str) or not text.strip():
        raise DescriptionResponseError('missing_text', 'response')
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise DescriptionResponseError('missing_json', 'response')
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        raise DescriptionResponseError('invalid_json', 'response') from None


def description_schema(count):
    """Fix slot count without requiring imagined limbs or a successful pose match."""
    from ..semantic_vocab import VOCABULARY

    def choice(values):
        return {'type': 'string', 'enum': list(values)}

    def slots(values):
        return {'type': 'object', 'properties': {
            side: choice(values) for side in ('left', 'right')}}

    props = {
        'visible': {'type': 'boolean'},
        'body_details': {'type': 'string'}, 'search_text': {'type': 'string'},
        'context_evidence': {'type': 'string'},
        'certainty': choice(('high', 'medium', 'low')),
        'visible_scope': choice(('full', 'half', 'bust', 'shoulder_head', 'head')),
        'head_view': choice(('front', 'profile_left', 'profile_right', 'back',
                             'three_quarter_left', 'three_quarter_right', 'unknown')),
        'arm_states': slots(('lowered', 'waist', 'forward', 'raised', 'spread', 'unknown')),
        'body_observations': {'type': 'object', 'properties': {
            key: choice(values) for key, values in {
                'knees': ('bent', 'straight', 'mixed', 'unknown'),
                'legs': ('apart', 'together', 'staggered', 'unknown'),
                'arms': ('lowered', 'raised', 'spread', 'mixed', 'unknown'),
                'torso': ('forward', 'backward', 'upright', 'unknown'),
            }.items()}},
        'held_object_hands': {'type': 'object', 'additionalProperties': choice(('left', 'right', 'both', 'unknown'))},
        'held_object_hand_counts': {'type': 'object', 'additionalProperties': choice(('one', 'two', 'unknown'))},
        'pose_relations': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'kind': choice(('touch_head', 'touch_face', 'check_wrist', 'hand_chest', 'arm_raised')),
            'side': choice(('left', 'right', 'both', 'unknown')),
            'chain_visible': {'type': 'boolean'},
            'side_confidence': choice(('high', 'unknown')), 'evidence': {'type': 'string'},
        }}},
    }
    groups = {'postures': 'posture', 'actions': 'action', 'domains': 'domain',
              'activities': 'activity', 'objects': 'object', 'intents': 'intent', 'emotions': 'emotion'}
    for name, group in groups.items():
        props[name] = {'type': 'array', 'items': choice(VOCABULARY[group])}
    return {'type': 'object', 'required': ['people'], 'properties': {
        'people': {'type': 'array', 'minItems': count, 'maxItems': count,
                   'items': {'type': 'object', 'required': ['visible', 'body_details', 'search_text', 'certainty'],
                             'properties': props}},
    }}
