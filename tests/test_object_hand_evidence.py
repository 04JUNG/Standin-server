"""Hand shape is not held-object identity or physical contact evidence."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src.library import _synthetic_pose
from src.natural_pose_schema import validate_proposal
from src.natural_pose_worker import compile_spec
from src.pose_requirements import WholePoseValidator, audit_generated, requirements
from src.semantic_search import SemanticQuery


TAXONOMY = {'vocabulary': {'posture': {'standing': '서기'}, 'action': {'hold': '잡기'},
    'domain': {}, 'activity': {}, 'object': {'staff': '지팡이', 'phone': '휴대폰'},
    'emotion': {}, 'intent': {}}}


def proposal(bindings=None):
    parts = dict(lower_posture='서기', upper_action='왼손 지팡이, 오른손 내밀기',
                 active_hand='both', hands={'L': 'grip', 'R': 'open'}, head_direction='정면')
    if bindings is not None:
        parts['held_object_hands'] = bindings
    return dict(base='st_neutral', spec={'hands': {'L': 'grip', 'R': 'open'}},
                tags={'posture': ['standing'], 'action': ['hold'], 'object': ['staff']},
                body_parts=parts)


def check(parts, *, count='one', wanted_side=None, obj='staff', bound=True):
    points = _synthetic_pose('standing')
    camera = SimpleNamespace(pose_ids=['a'], points=[points], scores=[np.ones(17)], hashes={'a': 'sha'})
    record = {'meta': {'bvh_sha256': 'sha' if bound else 'stale', 'body_parts': parts}}
    validator = WholePoseValidator(camera, {'a': record}, {})
    required = dict(active=True, posture=None, arms={}, body_states={}, hand_counts={obj: count},
                    object_hands={obj: wanted_side} if wanted_side else {})
    return validator.check(required, 'a')


def test_shape_and_explicit_object_intent_are_separate_and_survive_compile():
    value = validate_proposal(proposal({'staff': 'left'}), TAXONOMY)
    assert value.body_parts.hands.L == 'grip'
    assert value.body_parts.held_object_hands == {'staff': 'left'}
    assert compile_spec(value)['hands'] == {'L': 'grip', 'R': 'open'}


def test_old_stored_proposals_remain_readable_without_staff_inference():
    value = validate_proposal(proposal(), TAXONOMY)
    assert value.body_parts.held_object_hands is None
    result = check({'hands': {'left': 'grip', 'right': 'open'}, 'active_hand': 'left'})
    assert result['status'] == 'needs_review'
    assert result['checks']['hand_count:staff']['candidate'] is None


@pytest.mark.parametrize('bindings', [[], {'staff': 'L'}, {'staff': 'screen_left'},
                                      {'staff': True}, {'staff': ['left']}, {'': 'left'}])
def test_malformed_object_bindings_are_rejected(bindings):
    with pytest.raises(ValidationError):
        validate_proposal(proposal(bindings), TAXONOMY)


def test_binding_requires_explicit_known_object_tag():
    with pytest.raises(ValueError, match='held_object_not_in_object_tags'):
        validate_proposal(proposal({'phone': 'left'}), TAXONOMY)
    raw = proposal({'invented': 'left'})
    raw['tags']['object'] = ['invented']
    with pytest.raises(ValueError, match='unknown search-v2 tag'):
        validate_proposal(raw, TAXONOMY)


@pytest.mark.parametrize('shape', ['relaxed', 'open', 'point'])
def test_explicit_empty_or_pointing_hand_cannot_claim_a_grasp(shape):
    raw = proposal({'staff': 'left'})
    raw['body_parts']['hands']['L'] = shape
    raw['spec']['hands']['L'] = shape
    with pytest.raises(ValueError, match='held_object_hand_shape_incompatible'):
        validate_proposal(raw, TAXONOMY)


def test_unknown_binding_does_not_require_or_infer_a_grasp():
    raw = proposal({'staff': 'unknown'})
    raw['body_parts']['hands']['L'] = 'open'
    value = validate_proposal(raw, TAXONOMY)
    result = check({'hands': {'left': 'grip'}, 'held_object_hands': value.body_parts.held_object_hands})
    assert result['status'] == 'needs_review'


def test_explicit_staff_intent_passes_count_but_is_not_mesh_contact_certificate():
    result = check({'hands': {'left': 'grip', 'right': 'open'},
                    'held_object_hands': {'staff': 'left'}}, wanted_side='left')
    assert result['status'] == 'passed'
    assert result['checks']['hand_count:staff']['candidate'] == 'one'
    assert result['checks']['held_hand:staff']['candidate'] == 'left'
    assert result['actual_mesh_contact_validated'] is False


def test_explicit_other_object_or_active_hand_cannot_satisfy_staff():
    for parts in ({'hands': {'left': 'grip'}, 'active_hand': 'L'},
                  {'hands': {'left': 'grip'}, 'held_object_hands': {'phone': 'left'}},
                  {'hands': {'left': 'grip'}, 'held_object_hands': {}}):
        assert check(parts)['checks']['hand_count:staff']['status'] == 'unknown'


def test_wrong_hand_is_a_conflict_even_when_count_matches():
    result = check({'held_object_hands': {'staff': 'right'}}, wanted_side='left')
    assert result['checks']['hand_count:staff']['status'] == 'match'
    assert result['checks']['held_hand:staff']['status'] == 'conflict'
    assert result['status'] == 'conflict'


def test_one_hand_is_not_both_and_unbound_labels_never_pass():
    parts = {'held_object_hands': {'staff': 'left'}}
    assert check(parts, count='two')['status'] == 'conflict'
    result = check(parts, wanted_side='left', bound=False)
    assert result['status'] == 'needs_review'
    assert all(c['status'] == 'unknown' for c in result['checks'].values())


@pytest.mark.parametrize('bad', [{}, {'phone': 'unknown'}, {'phone': 'left'},
                                {'phone': []}, ['phone']])
def test_explicit_binding_is_authoritative_over_legacy_phone_labels(bad):
    result = check({'hands': {'left': 'phone', 'right': 'phone'},
                    'held_object_hands': bad}, obj='phone', count='two')
    assert result['status'] != 'passed'


def test_legacy_bound_object_labels_still_work_without_new_field():
    result = check({'hands': {'left': 'phone', 'right': 'phone'}}, obj='phone', count='two')
    # The existing spatial shared-grip check remains active; inspect the count only.
    assert result['checks']['hand_count:phone']['status'] == 'match'
    assert not result['actual_mesh_contact_validated']


def test_query_preserves_explicit_side_separately_from_count():
    query = SemanticQuery(objects=['staff'], held_object_hands={'staff': 'left'})
    result = requirements(query, {})
    assert result['hand_counts'] == {'staff': 'one'}
    assert result['object_hands'] == {'staff': 'left'}
    query.held_object_hands['staff'] = 'unknown'
    assert requirements(query, {})['object_hands'] == {}


def test_generated_audit_keeps_explicit_binding_and_fails_on_missing_or_bad_hash(monkeypatch):
    import src.bvh
    points = _synthetic_pose('standing')
    monkeypatch.setattr(src.bvh, 'load_coco17', lambda path: (points, np.ones(17)))
    text = 'unit fixture; actual BVH/IK is tested in the opt-in local replay'
    candidate = {'bvh': text, 'bvh_sha256': hashlib.sha256(text.encode()).hexdigest(),
                 'meta_json': {'body_parts': {'hands': {'L': 'grip', 'R': 'open'},
                                             'held_object_hands': {'staff': 'left'}}}}
    required = dict(active=True, posture=None, arms={}, body_states={},
                    hand_counts={'staff': 'one'}, object_hands={'staff': 'left'})
    assert audit_generated(required, candidate)['status'] == 'passed'
    missing = deepcopy(candidate)
    missing['meta_json']['body_parts']['held_object_hands'] = {}
    assert audit_generated(required, missing)['status'] == 'needs_review'
    candidate['bvh_sha256'] = 'changed'
    with pytest.raises(ValueError, match='generated_bvh_hash_mismatch'):
        audit_generated(required, candidate)
