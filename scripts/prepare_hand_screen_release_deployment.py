"""Connect a pinned, read-only hand-screen release to staging inference startup.

An explicit enabled=0 with a release prepares the file without using it. An
explicit enabled=0 without a release removes this owned connection (rollback).
Empty settings preserve the existing task exactly. This helper makes no ECS call.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.install_hand_screen_release import KEY, manifest

BUCKET = 'standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
INIT = 'hand-screen-release-init'
VOLUME = 'hand-screen-release'
ROOT = '/app/hand-screen-release'
FLAGS = {'HAND_SKIN_PREVALIDATED_PATH', 'HAND_SKIN_PREVALIDATED_SHA256'}


def prepare(task, *, key='', manifest_sha256='', manifest_raw=None, image='', enabled=''):
    if not any((key, manifest_sha256, enabled)):
        return copy.deepcopy(task)
    if (not str(task.get('family', '')).startswith('StandinStagingAppInferenceTask')
            or enabled not in ('0', '1')):
        raise ValueError('explicit staging hand screen rollout required')
    result = copy.deepcopy(task)
    targets = [c for c in result['containerDefinitions'] if c['name'] == 'inference']
    if len(targets) != 1 or any(c['name'] == 'converter' for c in result['containerDefinitions']):
        raise ValueError('exactly one isolated inference container required')
    target = targets[0]
    rows = target.get('environment', [])
    env = {r['name']: r['value'] for r in rows}
    if len(env) != len(rows) or FLAGS & {s['name'] for s in target.get('secrets', [])}:
        raise ValueError('ambiguous hand screen environment')
    owned = [c for c in result['containerDefinitions'] if c['name'] == INIT]
    if len(owned) > 1 or (owned and owned[0].get('command') != [
            'python', '/app/scripts/install_hand_screen_release.py']):
        raise ValueError('unknown hand screen initializer')
    result['containerDefinitions'] = [c for c in result['containerDefinitions'] if c['name'] != INIT]
    target['dependsOn'] = [d for d in target.get('dependsOn', []) if d['containerName'] != INIT]
    target['mountPoints'] = [m for m in target.get('mountPoints', []) if m['sourceVolume'] != VOLUME]
    env = {k: v for k, v in env.items() if k not in FLAGS}
    volumes = result.get('volumes', [])
    reserved = [v for v in volumes if v['name'] == VOLUME]
    if len(reserved) > 1 or (reserved and reserved[0] != {'name': VOLUME, 'host': {}}):
        raise ValueError('unknown hand screen volume')
    result['volumes'] = [v for v in volumes if v['name'] != VOLUME]
    if not key and not manifest_sha256 and enabled == '0':
        target['environment'] = [{'name': k, 'value': v} for k, v in env.items()]
        return result
    if (not KEY.fullmatch(key) or not manifest_sha256 or manifest_raw is None
            or not re.fullmatch(r'[0-9]{12}\.dkr\.ecr\.ap-northeast-2\.amazonaws\.com/standin/inference:[a-f0-9]{40}', image)):
        raise ValueError('immutable staging release and inference image required')
    release = manifest(manifest_raw, manifest_sha256)
    if enabled == '1':
        env.update(HAND_SKIN_PREVALIDATED_PATH=ROOT+'/catalog.json',
                   HAND_SKIN_PREVALIDATED_SHA256=release['catalog']['sha256'])
    target['environment'] = [{'name': k, 'value': v} for k, v in env.items()]
    target['dependsOn'].append({'containerName': INIT, 'condition': 'SUCCESS'})
    target['mountPoints'].append(dict(sourceVolume=VOLUME, containerPath=ROOT, readOnly=True))
    if any(m['containerPath'] == ROOT and m['sourceVolume'] != VOLUME for m in target['mountPoints']):
        raise ValueError('hand screen mount path collision')
    result['volumes'].append({'name': VOLUME, 'host': {}})
    initializer = dict(name=INIT, image=image, user='0', essential=False,
        command=['python', '/app/scripts/install_hand_screen_release.py'],
        memoryReservation=512,
        environment=[{'name': k, 'value': v} for k, v in dict(
            HAND_SCREEN_RELEASE_BUCKET=BUCKET,
            HAND_SCREEN_RELEASE_MANIFEST_KEY=key,
            HAND_SCREEN_RELEASE_MANIFEST_SHA256=manifest_sha256,
            HAND_SCREEN_RELEASE_ROOT=ROOT).items()],
        mountPoints=[dict(sourceVolume=VOLUME, containerPath=ROOT, readOnly=False)])
    if 'logConfiguration' in target:
        initializer['logConfiguration'] = copy.deepcopy(target['logConfiguration'])
        initializer['logConfiguration']['options']['awslogs-stream-prefix'] = INIT
    result['containerDefinitions'].append(initializer)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--environment', choices=['staging'], required=True)
    args = parser.parse_args()
    key = os.getenv('HAND_SCREEN_RELEASE_MANIFEST_KEY', '')
    pin = os.getenv('HAND_SCREEN_RELEASE_MANIFEST_SHA256', '')
    enabled = os.getenv('HAND_SCREEN_PREVALIDATION_ENABLED', '')
    raw = None
    if key or pin:
        if not KEY.fullmatch(key) or not re.fullmatch(r'[a-f0-9]{64}', pin):
            raise ValueError('invalid pinned hand screen manifest')
        import boto3
        response = boto3.client('s3').get_object(Bucket=BUCKET, Key=key)
        with response['Body'] as body:
            raw = body.read(65537)
    result = prepare(json.loads(args.task.read_text()), key=key, manifest_sha256=pin,
                     manifest_raw=raw, image=args.image, enabled=enabled)
    args.task.write_text(json.dumps(result, indent=2)+'\n')
    print('staging hand screen release prepared; no ECS update performed')


if __name__ == '__main__':
    main()
