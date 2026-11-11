"""Render the Recall profile into the inference task for its environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


PIN = Path(__file__).resolve().parents[1] / 'config/staging-recall-90-v1.json'
STAGING_FAMILY = 'StandinStagingAppInferenceTask76CF0AC4'


def configure(task, enabled, pin, *, production=False):
    if enabled not in ('0', '1'):
        raise ValueError('ROUTING_RECALL_ENABLED must be 0 or 1')
    if not production and task.get('family') != STAGING_FAMILY:
        raise ValueError('recall defaults apply to the staging inference task only')
    if production and task.get('family') == STAGING_FAMILY:
        raise ValueError('production recall cannot use the staging inference task')
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
            raise ValueError('recall requires full-body search and gpt-6-luna low')
        prefix = 's3://standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk/pose-library/runtime-assets/recall-90-v1/'
        if production:
            library = re.fullmatch(r's3://([a-z0-9.-]+)/pose-library/.+', env.get('POSE_LIBRARY_URI', ''))
            if library is None or library[1].startswith('standinstagingapp-'):
                raise ValueError('production recall requires the production pose library bucket')
            prefix = f's3://{library[1]}/pose-library/runtime-assets/recall-90-v1/'
        digest = pin['sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or pin['uri'] != (
                f'{prefix}recall-models-{digest[:16]}.tar'):
            raise ValueError('invalid recall model pin for this environment')
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
    parser.add_argument('--production', action='store_true')
    args = parser.parse_args()
    pin = {}
    if args.enabled == '1':
        if args.production:
            task = json.loads(args.task.read_text())
            env = {e['name']:e['value'] for c in task['containerDefinitions'] if c['name']=='inference' for e in c.get('environment',[])}
            bucket = re.fullmatch(r's3://([a-z0-9.-]+)/pose-library/.+', env.get('POSE_LIBRARY_URI',''))
            if bucket is None or bucket[1].startswith('standinstagingapp-'):
                raise ValueError('production Recall requires a production library bucket')
            profile=json.loads((PIN.parents[0]/'production-runtime-parity.json').read_text())
            pin = {'uri': os.getenv('ROUTING_RECALL_ASSETS_URI') or f"s3://{bucket[1]}/{profile['recallKey']}",
                   'sha256': os.getenv('ROUTING_RECALL_ASSETS_SHA256') or profile['recallSha256']}
        else:
            pin = json.loads(PIN.read_text())
    result = configure(json.loads(args.task.read_text()), args.enabled, pin,
                       production=args.production)
    args.task.write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
