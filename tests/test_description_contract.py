"""Malformed close-up descriptions remain failures; schema never invents evidence."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.vlm.client import GeminiVLMClient, parse_pose_descriptions
from src.vlm.description_contract import DescriptionResponseError, description_payload, description_schema


@pytest.mark.parametrize('count', [1, 2, 5])
def test_schema_keeps_exact_slots_and_allows_abstention(count):
    schema = description_schema(count)
    people = schema['properties']['people']
    assert people['minItems'] == people['maxItems'] == count
    fields = people['items']['properties']
    assert fields['certainty']['enum'] == ['high', 'medium', 'low']
    assert 'shoulder_head' in fields['visible_scope']['enum']
    assert 'body_observations' not in people['items']['required']
    assert 'coordinates' not in fields
    desc = parse_pose_descriptions({'people': [{'visible': False, 'certainty': 'low'}] * count}, count)
    assert all(not p.visible and not p.arm_states and not p.postures for p in desc.values())


@pytest.mark.parametrize('text,code', [(None, 'missing_text'), ('private response', 'missing_json'),
    ('{"people":oops}', 'invalid_json'), ('{"people":[]}', 'people_count_mismatch'),
    ('{"people":[null]}', 'invalid_person')])
def test_content_free_validation_codes(text, code):
    with pytest.raises(DescriptionResponseError) as caught:
        parse_pose_descriptions(description_payload(text), 1)
    assert caught.value.code == code
    assert 'private response' not in str(caught.value)


def test_gemini_uses_schema_and_invalid_response_is_not_retried_or_success(monkeypatch):
    calls, logs = [], []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({'people': []}))
    monkeypatch.setattr('src.vlm.client.log_warn', lambda *args, **kw: logs.append(kw))
    client = GeminiVLMClient(client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    result = client.describe_people(Image.new('RGB', (32, 32)), [(0, 0, 1, 1)])
    assert len(calls) == 1
    assert calls[0]['config'].response_json_schema['properties']['people']['minItems'] == 1
    assert result.status == 'error' and result.error_reason == 'invalid_response'
    assert not result.descriptions and result.retryable is False
    assert logs[0]['validationCode'] == 'people_count_mismatch'
    assert logs[0]['validationField'] == 'people'
    assert all('text' not in key.lower() for key in logs[0])


def test_valid_head_description_preserves_unknown_hidden_limbs():
    d = parse_pose_descriptions({'people': [{'visible': True, 'visible_scope': 'head',
        'head_view': 'profile_left', 'body_details': '머리가 왼쪽을 향함',
        'search_text': '왼쪽으로 향한 머리', 'certainty': 'medium'}]}, 1)[0]
    assert d.visible_scope == 'head' and d.head_view == 'profile_left'
    assert d.arm_states == {} and d.body_observations == {} and d.held_object_hands == {}
