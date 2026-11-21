"""Render the staging-only recall defaults into the existing inference task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PIN = Path(__file__).resolve().parents[1] / 'config/staging-recall-90-v1.json'
STAGING_FAMILY = 'StandinStagingAppInferenceTask76CF0AC4'


def configure(task, enabled, pin):
    if enabled not in ('0', '1'):
        raise ValueError('ROUTING_RECALL_ENABLED must be 0 or 1')
    if task.get('family') != STAGING_FAMILY:
        raise ValueError('recall defaults apply to the staging inference task only')
    [container] = [c for c in task['containerDefinitions'] if c['name'] == 'inference']
    rows = container.get('environment', [])
    env = {row['name']: row['value'] for row in rows}
    if len(env) != len(rows):
        raise ValueError('duplicate environment names in task definition')
    for name in ('ROUTING_RECALL_ASSETS_DIR', 'ROUTING_RECALL_ASSETS_URI',
                 'ROUTING_RECALL_ASSETS_SHA256', 'ROUTING_RECALL_PROP_PYTHON'):
        env.pop(name, None)
    if enabled == '1':
        if (env.get('ROUTING_FULLBODY_ENABLED') != '1'
                or env.get('VLM_PROVIDER') != 'openai'
                or env.get('OPENAI_MODEL') != 'gpt-6-luna'
                or env.get('OPENAI_REASONING_EFFORT') != 'low'):
            raise ValueError('staging recall requires full-body search and gpt-6-luna low')
        prefix = 's3://standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk/pose-library/runtime-assets/recall-90-v1/'
        digest = pin['sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or pin['uri'] != (
                f'{prefix}recall-models-{digest[:16]}.tar'):
            raise ValueError('invalid staging recall model pin')
        env.update(ROUTING_RECALL_ASSETS_DIR='/app/data/recall-90-v1',
                   ROUTING_RECALL_ASSETS_URI=pin['uri'], ROUTING_RECALL_ASSETS_SHA256=digest,
                   ROUTING_RECALL_PROP_PYTHON='/opt/recall-prop/bin/python')
    env['ROUTING_LUNA_SIX_FIELD_ENABLED'] = enabled
    container['environment'] = [{'name': name, 'value': value} for name, value in env.items()]
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--enabled', choices=('0', '1'), required=True)
    args = parser.parse_args()
    pin = json.loads(PIN.read_text()) if args.enabled == '1' else {}
    result = configure(json.loads(args.task.read_text()), args.enabled, pin)
    args.task.write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
