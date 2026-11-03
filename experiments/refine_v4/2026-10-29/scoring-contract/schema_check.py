"""Offline JSON Schema checker for ALL keywords used by the shipped draft-07 schema.

Not a general-purpose JSON Schema implementation. Fails on unsupported keywords;
the schema is independently compiled and checked with Ajv as well.
"""
import re

SUPPORTED = {'$schema','title','type','additionalProperties','required','properties',
             'minLength','const','enum','items','minimum','maximum','pattern','allOf','if','then','else'}

def check_supported(schema):
    assert set(schema) <= SUPPORTED, ('unsupported schema keyword', set(schema) - SUPPORTED)
    for key, value in schema.items():
        if key == 'properties':
            for child in value.values(): check_supported(child)
        elif key in ('items','if','then','else'):
            check_supported(value)
        elif key == 'allOf':
            for child in value: check_supported(child)

def validate(value, schema, path='$'):
    types = {'object': lambda x: isinstance(x, dict), 'array': lambda x: isinstance(x, list),
             'integer': lambda x: type(x) is int, 'string': lambda x: isinstance(x, str), 'null': lambda x: x is None}
    if 'type' in schema:
        wanted = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        assert any(types[t](value) for t in wanted), (path, 'type')
    if 'const' in schema: assert value == schema['const'], (path, 'const')
    if 'enum' in schema: assert value in schema['enum'], (path, 'enum')
    if isinstance(value, dict):
        assert set(schema.get('required', [])) <= value.keys(), (path, 'required')
        if schema.get('additionalProperties') is False:
            assert value.keys() <= schema.get('properties', {}).keys(), (path, 'additionalProperties')
        for k, v in value.items():
            if k in schema.get('properties', {}): validate(v, schema['properties'][k], path + '.' + k)
    if isinstance(value, list) and 'items' in schema:
        for i, v in enumerate(value): validate(v, schema['items'], path + f'[{i}]')
    if isinstance(value, str):
        assert len(value) >= schema.get('minLength', 0), (path, 'minLength')
        if 'pattern' in schema: assert re.search(schema['pattern'], value), (path, 'pattern')
    if type(value) is int:
        assert schema.get('minimum', value) <= value <= schema.get('maximum', value), (path, 'range')
    for child in schema.get('allOf', []): validate(value, child, path)
    if 'if' in schema:
        try: validate(value, schema['if'], path)
        except AssertionError: branch = 'else'
        else: branch = 'then'
        if branch in schema: validate(value, schema[branch], path)
