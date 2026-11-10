"""One Luna low request for the evaluated six-field search description."""
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from ..config import CFG
from ..logging_setup import log_info, log_warn
from ..search_routing import DescribeOutcome
from .description_contract import DescriptionResponseError

VERSION = 'six-fields-single-low-v16'


@lru_cache(maxsize=1)
def profile():
    resource = Path(__file__).with_name('luna_search_v16.json')
    payload = json.loads(resource.read_text(encoding='utf-8'))
    if payload['version'] != VERSION:
        raise ValueError('Luna search description profile mismatch')
    reverse = {field: {label: key for key, label in labels.items()}
               for field, labels in payload['catalog'].items()}
    if any(len(reverse[field]) != len(labels) for field, labels in payload['catalog'].items()):
        raise ValueError('Luna search labels must be unambiguous')
    return payload, reverse


def validate(value, schema, path='person'):
    kind = schema['type']
    expected = {'object': dict, 'array': list, 'string': str, 'boolean': bool}
    if not isinstance(value, expected[kind]):
        raise DescriptionResponseError('invalid_six_field_type', path)
    if 'enum' in schema and value not in schema['enum']:
        raise DescriptionResponseError('invalid_six_field_value', path)
    if kind == 'string' and len(value) > schema.get('maxLength', 1000000):
        raise DescriptionResponseError('invalid_six_field_length', path)
    if kind == 'object':
        properties = schema['properties']
        if set(schema.get('required', ())) - value.keys() or value.keys() - properties.keys():
            raise DescriptionResponseError('invalid_six_field_keys', path)
        for key, item in value.items():
            validate(item, properties[key], f'{path}.{key}')
    elif kind == 'array':
        for i, item in enumerate(value):
            validate(item, schema['items'], f'{path}.{i}')


def project(row, settings, reverse):
    support = row['support']
    mapped = {}
    for group, fields in settings['group_fields'].items():
        for field, key in fields.items():
            labels = row[group] if key is None else row[group][key]
            mapped[key or group] = [reverse[field][label] for label in labels]
    held = {item['object']: item for item in support.get('held_objects', ())
            if item['object'] in mapped['objects']}
    configurations = mapped['configurations']
    labels = settings['catalog']['body_configuration']
    return dict(mapped, visible=support['visible'], certainty=support['certainty'],
                visible_scope=support['visible_scope'], head_view=support['head_view'],
                arm_states=support['arm_states'],
                body_observations={k: v for k, v in support.get('body', {}).items() if v != 'unknown'},
                body_details=' · '.join(labels[key] for key in configurations), search_text=row['description'],
                context_evidence=support.get('context_evidence', ''),
                pose_relations=support.get('pose_relations', []),
                held_object_hands={key: value['hand'] for key, value in held.items()},
                held_object_hand_counts={key: value['count'] for key, value in held.items()})


def describe(client, image, boxes):
    from .client import describe_failure_reason, parse_pose_descriptions

    if not boxes:
        return DescribeOutcome('ok', descriptions={}, attempts=0, provider=client.provider)
    settings, reverse = profile()
    schema = settings['schema']
    started = client._clock()
    single = len(boxes) == 1
    if single:
        box = boxes[0] if boxes[0] is not None else (0., 0., 1., 1.)
        prompt = settings['prompt'].format(box=json.dumps(box))
        contract = schema
    else:
        instruction = ('지정 인물별로 독립적으로 분석한다. 입력 박스 순서대로 각각 동일한 6필드 JSON 객체를 '
                       'people 배열에 담는다. 다른 인물의 손과 섞지 않는다. 박스 목록: ')
        prompt = instruction + json.dumps(boxes) + '\n\n' + settings['prompt'].split('\n\n', 1)[1]
        contract = {'type': 'object', 'properties': {'people': {'type': 'array', 'items': schema}},
                    'required': ['people'], 'additionalProperties': False}
    try:
        response = client._client.models.generate_content(
            model=client._model, contents=[prompt, client._to_part(image)],
            config=SimpleNamespace(system_instruction=settings['system'], response_json_schema=contract,
                                   reasoning_effort='low',
                                   http_options=SimpleNamespace(timeout=CFG.routing_describe_timeout_ms)))
        payload = json.loads(response.text)
        validate(payload, contract)
        people = [payload] if single else payload['people']
        rows = [project(row, settings, reverse) for row in people]
        descriptions = parse_pose_descriptions({'people': rows}, len(boxes), vocabulary=settings['catalog'])
    except Exception as error:
        reason, retryable, delay = describe_failure_reason(error)
        elapsed = (client._clock() - started) * 1000
        log_warn('openai_pose_describe', model=client._model, promptVersion=VERSION, status='failed',
                 failureReason=reason, errorType=type(error).__name__, elapsedMs=round(elapsed))
        return DescribeOutcome('error', error_reason=reason, retryable=retryable,
                               retry_after_seconds=delay, elapsed_ms=elapsed, provider=client.provider)
    elapsed = (client._clock() - started) * 1000
    log_info('openai_pose_describe', model=client._model, promptVersion=VERSION, status='ok',
             people=len(boxes), elapsedMs=round(elapsed))
    return DescribeOutcome('ok', descriptions=descriptions, elapsed_ms=elapsed, provider=client.provider)
