"""Explicit staging-only private verified-FBX cache and immutable measurement reuse."""
import argparse
import copy
import json
import os
from pathlib import Path
import re

CONFIG = Path(__file__).resolve().parents[1] / 'config/staging-final-cache.json'


def prepare(task, config, enabled='1', reuse_rest='1'):
    if not task.get('family', '').startswith('StandinStagingApp'):
        raise ValueError('final cache rollout is staging-only')
    if enabled not in ('0', '1') or reuse_rest not in ('0', '1'):
        raise ValueError('rollout flags must be 0 or 1')
    sha = config.get('librarySha256', '')
    if (config.get('version') != 'staging-verified-final-cache-v1'
            or not re.fullmatch('[a-f0-9]{64}', sha)
            or config.get('bucket') != 'standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
            or config.get('dataPrefix') != 'verified-final-data/v1/'
            or config.get('libraryKey') != 'verified-final-control/library-' + sha + '.json'
            or config.get('revocationsKey') != 'verified-final-control/revoked.json'):
        raise ValueError('invalid operator-pinned staging cache configuration')
    result = copy.deepcopy(task)
    containers = [c for c in result.get('containerDefinitions', []) if c['name'] == 'converter']
    if len(containers) != 1:
        raise ValueError('exactly one converter container required')
    main = containers[0]
    rows = main.get('environment', [])
    env = {e['name']: e['value'] for e in rows}
    if len(env) != len(rows):
        raise ValueError('duplicate environment names')
    uri = 's3://' + config['bucket'] + '/'
    values = {
        'CONVERTER_FINAL_CACHE_ROOT': '/tmp/standin-verified-final' if enabled == '1' else '',
        'CONVERTER_FINAL_CACHE_S3_URI': uri + config['dataPrefix'] if enabled == '1' else '',
        'CONVERTER_FINAL_LIBRARY_URI': uri + config['libraryKey'] if enabled == '1' else '',
        'CONVERTER_FINAL_LIBRARY_SHA256': sha if enabled == '1' else '',
        'CONVERTER_FINAL_REVOCATIONS_URI': uri + config['revocationsKey'] if enabled == '1' else '',
        'CONVERTER_FINAL_PREWARM_ENABLED': '0',
        'CONVERTER_REUSE_REST_METRICS': reuse_rest,
    }
    if values.keys() & {s['name'] for s in main.get('secrets', [])}:
        raise ValueError('cache setting collides with a secret')
    env.update(values)
    main['environment'] = [{'name': k, 'value': v} for k, v in env.items()]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    args = parser.parse_args()
    result = prepare(json.loads(args.task.read_text()), json.loads(CONFIG.read_text()),
        os.getenv('STAGING_FINAL_CACHE_ENABLED', '1'), os.getenv('STAGING_REUSE_REST_METRICS', '1'))
    args.task.write_text(json.dumps(result), encoding='utf-8')
    print('staging private final cache policy prepared; images and unrelated settings preserved')
