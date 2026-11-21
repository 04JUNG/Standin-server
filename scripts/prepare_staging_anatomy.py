"""Restore the checked-in staging preset release during every converter deployment."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / 'config/staging-anatomy-release.json').read_text(encoding='utf-8'))
INITIALIZER = (ROOT / 'scripts/staging_body_assets_init.py').read_text(encoding='utf-8')

def prepare(task):
    if not task.get('family', '').startswith('StandinStagingApp'):
        raise ValueError('Anatomy preparation requires staging')
    result = copy.deepcopy(task)
    containers = {c['name']: c for c in result['containerDefinitions']}
    main, init = containers['converter'], containers['body-assets-init']
    if not any(d.get('containerName') == init['name'] and d.get('condition') == 'SUCCESS' for d in main.get('dependsOn', [])):
        raise ValueError('Missing reviewed body initializer dependency')
    def setenv(container, values):
        env = {e['name']: e['value'] for e in container.get('environment', [])}
        env.update(values)
        container['environment'] = [{'name': k, 'value': v} for k, v in env.items()]
    setenv(init, {'BODY_ASSETS_ROLE': 'converter', 'BODY_ASSETS_ROOT': '/app/body-staging',
                 'BODY_ASSETS_MANIFEST_KEY': RELEASE['prefix'] + '/manifest.json',
                 'BODY_ASSETS_MANIFEST_SHA256': RELEASE['manifestSha256']})
    setenv(main, {'CONVERTER_SKIN_ENABLED': 'true', 'CONVERTER_OUTLINE_ENABLED': 'true',
                 'CONVERTER_SKIN_ASSET_ROOT': '/app/body-staging/skin',
                 'CONVERTER_CHARACTER_REGISTRY': '/app/body-staging/characters.json',
                 'CONVERTER_SHAPE_REGISTRY': '/app/body-staging/shape-bakes.json',
                 **{'BODY_' + cid.upper().replace('-', '_') + '_URI': '/app/body-staging/fbx/' + cid + '.fbx' for cid in RELEASE['characterIds']}})
    init['command'] = ['python', '-c', INITIALIZER]
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task_definition', type=Path); args = ap.parse_args()
    task = prepare(json.loads(args.task_definition.read_text(encoding='utf-8-sig')))
    args.task_definition.write_text(json.dumps(task, indent=2) + '\n', encoding='utf-8')
    print('Pinned staging preset release prepared')
