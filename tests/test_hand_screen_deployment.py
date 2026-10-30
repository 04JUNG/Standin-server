"""Rollout overrides cannot silently change another container or resource."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('hand_screen_deploy',
    Path(__file__).resolve().parents[1] / 'scripts/prepare_hand_screen_deployment.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def task():
    return {'cpu': '2048', 'memory': '4096', 'taskRoleArn': 'unchanged-role',
            'containerDefinitions': [
                {'name': 'inference', 'image': 'inference@sha256:fixed',
                 'environment': [{'name': 'HAND_SKIN_SCREEN_BUDGET_SECONDS', 'value': '45'}],
                 'secrets': [{'name': 'GEMINI_API_KEY', 'valueFrom': 'unchanged-reference'}]},
                {'name': 'converter', 'image': 'converter@sha256:fixed',
                 'environment': [{'name': 'CONVERTER_MAX_CONCURRENT_PROCESSES', 'value': '1'}]},
                {'name': 'body-assets-init', 'image': 'initializer@sha256:fixed'}]}


@pytest.mark.parametrize('container', ['inference', 'converter'])
@pytest.mark.parametrize('value', [None, ''])
def test_empty_override_is_exact_noop(container, value):
    original = task()
    assert deploy.prepare(original, container, value) == original


@pytest.mark.parametrize('container', ['inference', 'converter'])
@pytest.mark.parametrize('value', ['1', '2', '3', '4', '5'])
def test_only_selected_capacity_changes(container, value):
    original = task()
    before = deepcopy(original)
    after = deploy.prepare(original, container, value)
    expected = deepcopy(original)
    target = next(c for c in expected['containerDefinitions'] if c['name'] == container)
    name = deploy.SETTINGS[container]
    target['environment'] = [r for r in target['environment'] if r['name'] != name]
    target['environment'].append({'name': name, 'value': value})
    assert original == before and after == expected


@pytest.mark.parametrize('value', ['0', '6', '-1', '02', '2.0', ' 2', 2, True])
def test_bad_override_fails_before_deployment(value):
    with pytest.raises(ValueError):
        deploy.prepare(task(), 'inference', value)


@pytest.mark.parametrize('problem', ['duplicate_container', 'missing_container', 'duplicate_env', 'secret'])
def test_ambiguous_target_is_rejected(problem):
    value = task()
    target = value['containerDefinitions'][0]
    if problem == 'duplicate_container': value['containerDefinitions'].append(deepcopy(target))
    if problem == 'missing_container': target['name'] = 'other'
    if problem == 'duplicate_env': target['environment'] *= 2
    if problem == 'secret': target['secrets'].append({'name': 'HAND_SKIN_SCREEN_CONCURRENCY'})
    with pytest.raises(ValueError): deploy.prepare(value, 'inference', '2')
