"""Render an explicit staging rollout contract; no AWS writes/secret retrieval.

Input is the current task definition fetched by the existing GitHub deploy role.
Existing secrets, role ARNs and non-routing fields are preserved verbatim.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def prepare(task, *, enabled, variant='', canary=''):
    if enabled not in ('0', '1'):
        raise ValueError('ROUTING_FULLBODY_ENABLED must be 0 or 1')
    result = copy.deepcopy(task)
    containers = [c for c in result['containerDefinitions'] if c['name'] == 'inference']
    if len(containers) != 1:
        raise ValueError('exactly one inference container required')
    container = containers[0]
    rows = container.get('environment', [])
    env = {r['name']: r['value'] for r in rows}
    if len(env) != len(rows):
        raise ValueError('duplicate environment names in task definition')
    secret_names = {s['name'] for s in container.get('secrets', [])}
    env['ROUTING_FULLBODY_ENABLED'] = enabled
    if enabled == '1':
        if variant and variant != 'humanart-m':
            raise ValueError('fullbody rollout conflicts with POSE_MODEL_VARIANT; use humanart-m')
        if canary and canary != 'canary-100':
            raise ValueError('direct fullbody rollout requires canary-100')
        env.update(POSE_BACKEND='rtmlib', POSE_MODEL_VARIANT='humanart-m',
                   POSE_CANARY_STAGE='canary-100', POSE_STRICT='1',
                   ROUTING_PREPARE_ASSETS='1', POSE_CALIBRATION_FROM_MANIFEST='1')
        if container.get('healthCheck'):
            container['healthCheck']['startPeriod'] = 300  # ECS maximum; keep command/retries intact
        if not env.get('POSE_MODEL_URI'):
            raise ValueError('fullbody deployment requires the infra-owned POSE_MODEL_URI')
        if not env.get('POSE_LIBRARY_URI'):
            raise ValueError('fullbody deployment requires the infra-owned POSE_LIBRARY_URI')
        # Do not change the provider silently, or retrieve any secret here.
        provider = env.get('VLM_PROVIDER')
        key = {'gemini': 'GEMINI_API_KEY', 'openai': 'OPENAI_API_KEY'}.get(provider)
        if not key or not (env.get(key) or key in secret_names):
            raise ValueError('fullbody deployment requires a real VLM provider and its configured secret')
        # Library/model downloads + first derived-cache construction happen here.
        # This is readiness grace, not relaxed ALB or application health checks.
    else:
        env['POSE_MODEL_VARIANT'] = variant or env.get('POSE_MODEL_VARIANT', 'cascade')
        env['POSE_CANARY_STAGE'] = canary or env.get('POSE_CANARY_STAGE', 'canary-100')
    container['environment'] = [{'name': k, 'value': v} for k, v in env.items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--enabled', required=True)
    parser.add_argument('--variant', default='')
    parser.add_argument('--canary', default='')
    args = parser.parse_args()
    result = prepare(json.loads(args.task.read_text()), enabled=args.enabled,
                     variant=args.variant, canary=args.canary)
    args.task.write_text(json.dumps(result), encoding='utf-8')
    if args.enabled == '1':
        print('fullbody rollout prepared; required asset/VLM references present')
    else:
        print('explicit legacy rollout prepared')


if __name__ == '__main__':
    main()
