"""Apply one explicit capacity override; empty values preserve the ECS task."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hand_skin_capacity import validate_concurrency

SETTINGS = {
    'inference': 'HAND_SKIN_SCREEN_CONCURRENCY',
    'converter': 'CONVERTER_MAX_CONCURRENT_PROCESSES',
}


def prepare(task, container_name, value):
    if container_name not in SETTINGS:
        raise ValueError('unsupported hand screen container')
    result = copy.deepcopy(task)
    containers = [c for c in result['containerDefinitions'] if c['name'] == container_name]
    if len(containers) != 1:
        raise ValueError('exactly one target container required')
    container = containers[0]
    rows = container.get('environment', [])
    env = {r['name']: r['value'] for r in rows}
    if len(env) != len(rows):
        raise ValueError('duplicate environment names in task definition')
    if value in (None, ''):
        return result
    if not isinstance(value, str) or value not in {'1', '2', '3', '4', '5'}:
        raise ValueError('capacity override must be an integer in [1,5]')
    validate_concurrency(int(value))
    name = SETTINGS[container_name]
    if name in {s['name'] for s in container.get('secrets', [])}:
        raise ValueError('capacity setting conflicts with an existing secret')
    env[name] = value
    container['environment'] = [{'name': k, 'value': v} for k, v in env.items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--container', choices=SETTINGS, required=True)
    args = parser.parse_args()
    value = os.getenv(SETTINGS[args.container], '')
    result = prepare(json.loads(args.task.read_text()), args.container, value)
    args.task.write_text(json.dumps(result), encoding='utf-8')
    print('hand screen capacity prepared; unspecified settings preserved')


if __name__ == '__main__':
    main()
