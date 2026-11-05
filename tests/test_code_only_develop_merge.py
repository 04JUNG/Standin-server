"""Code-only develop merges must not start an automatic deployment."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('enabled,event,message,expected', [
    ('true', 'push', 'feat: normal develop change', True),
    ('true', 'push', 'feat: body defaults [skip deploy]', False),
    ('true', 'push', 'feat: body defaults [SKIP DEPLOY]', False),
    ('false', 'push', 'feat: normal develop change', False),
    ('false', 'push', 'feat: body defaults [skip deploy]', False),
    ('true', 'workflow_dispatch', '', True),
    ('false', 'workflow_dispatch', '', False),
])
def test_actual_deployment_guard(enabled, event, message, expected):
    workflow = yaml.safe_load((ROOT / '.github/workflows/deploy.yml').read_text())
    condition = workflow['jobs']['deploy']['if']
    expression = condition.replace('vars.', 'variables.').replace('&&', ' and ').replace('||', ' or ').replace('!contains', 'not contains')
    github = SimpleNamespace(event_name=event, event=SimpleNamespace(head_commit=SimpleNamespace(message=message)))
    actual = eval(expression, {'__builtins__': {}}, {
        'github': github, 'variables': SimpleNamespace(AUTO_DEPLOY_ENABLED=enabled),
        'contains': lambda value, part: part.lower() in value.lower(),
    })
    assert actual is expected


def test_code_only_marker_does_not_suppress_develop_ci():
    ci = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    triggers = ci.get('on', ci.get(True))
    assert 'develop' in triggers['push']['branches']
    assert '[skip deploy]' not in (ROOT / '.github/workflows/ci.yml').read_text()
