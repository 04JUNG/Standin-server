"""Headless local API/Blender probe. Never starts the installed Standin client.

Run with a real registered rigged-rest character and a BVH (including generated
BVH). Optional shape baking uses an existing approved local shape registry.
"""
import argparse
import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bvh', type=Path, required=True)
    parser.add_argument('--character', type=Path, required=True)
    parser.add_argument('--character-id', required=True)
    parser.add_argument('--shape-registry', type=Path)
    parser.add_argument('--shape-control', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from fastapi.testclient import TestClient
    from converter_api.app import create_app
    from converter_api.registry import CharacterRegistry
    from converter_api.runner import BlenderRunner, RunnerSettings
    from converter_api.body_preview import preview_contract
    from converter.shape_spec import digest_shape, load_registry
    from PIL import Image, ImageStat

    args.output.mkdir(parents=True, exist_ok=True)
    character = args.character.resolve()
    character_sha = hashlib.sha256(character.read_bytes()).hexdigest()
    bvh = args.bvh.read_bytes()
    source_sha = hashlib.sha256(bvh).hexdigest()
    registry_path = args.output / 'characters.json'
    registry_path.write_text(json.dumps({'schema_version': 1, 'characters': {args.character_id: {
        'display_name': 'Readiness verification', 'artifact_uri_env': 'STANDIN_READY_QA_CHARACTER',
        'sha256': character_sha, 'rig_profile': 'mixamo', 'revision': 'local-qa',
    }}}))
    os.environ['STANDIN_READY_QA_CHARACTER'] = str(character)
    if args.shape_registry:
        os.environ['CONVERTER_SHAPE_REGISTRY'] = str(args.shape_registry.resolve())
    runner = BlenderRunner(replace(RunnerSettings.from_env(ROOT), timeout_seconds=180,
        camera_timeout_seconds=180, retarget_cache_bytes=64 * 1024 * 1024))
    registry = CharacterRegistry(registry_path, default_character_id=args.character_id,
                                artifact_cache_dir=args.output / 'character-cache')
    cases = [('first', 'half', 'fist'), ('hand-change', 'half', 'relaxed'), ('head', 'head', None)]
    if args.shape_registry:
        cases.append(('shape', 'half', 'fist'))
    rows = []
    with TestClient(create_app(registry=registry, runner=runner)) as client:
        for label, scope, left in cases:
            fields = {'character_id': args.character_id, 'output_scope': scope, 'preview_format': 'none',
                'expected_bvh_sha256': source_sha, 'expected_character_sha256': character_sha,
                'expected_preview_revision': preview_contract()['preview_revision'],
                'camera_rotation': json.dumps([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}
            if left:
                fields.update(hand_left=left, hand_right='relaxed')
            if label == 'shape':
                config = load_registry()
                entry = config['characters'][args.character_id]
                controls = {c['key']: c['default'] for c in entry['controls']}
                for pair in args.shape_control:
                    key, value = pair.split('=', 1)
                    controls[key] = float(value)
                source = config['sources'][entry['rules']]
                fields['shape'] = json.dumps({'schemaVersion': 'body-shape.v1', 'controls': controls,
                    'shapeSha256': digest_shape(args.character_id, character_sha, controls)})
                fields['expected_shape_source'] = json.dumps({'rules': entry['rules'],
                    'blendSha256': source['blend_sha256'], 'controllerSha256': source['controller_sha256']})
            start = time.monotonic()
            response = client.post('/convert-framed-export', data=fields,
                                  files={'bvh': ('final.bvh', bvh, 'application/octet-stream')})
            assert response.status_code == 200, (label, response.text)
            value = response.json()
            fbx = base64.b64decode(value['fbx_base64'], validate=True)
            digest = hashlib.sha256(fbx).hexdigest()
            assert value['artifact_contract'] == 'verified-fbx-v1' and value['fbx_sha256'] == digest
            assert 'preview_base64' not in value and value['source_bvh_sha256'] == source_sha
            (args.output / f'{label}.fbx').write_bytes(fbx)
            ready_s = time.monotonic() - start
            start = time.monotonic()
            image = client.post('/render-final-preview', data={'artifact_sha256': digest, 'aligned': 'true'},
                                files={'fbx': ('final.fbx', fbx, 'application/octet-stream')})
            assert image.status_code == 200, (label, image.text)
            assert image.headers['X-Standin-Artifact-SHA256'] == digest
            assert hashlib.sha256(image.content).hexdigest() == image.headers['X-Standin-Preview-Artifact-SHA256']
            destination = args.output / f'{label}.png'
            destination.write_bytes(image.content)
            assert max(ImageStat.Stat(Image.open(destination).convert('RGB')).stddev) > 1
            assert hashlib.sha256((args.output / f'{label}.fbx').read_bytes()).hexdigest() == digest
            row = {'case': label, 'scope': scope, 'ready_s': round(ready_s, 3),
                   'preview_s': round(time.monotonic() - start, 3), 'sha256': digest, 'bytes': len(fbx)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {'character_sha256': character_sha, 'source_bvh_sha256': source_sha,
              'blender': vars(runner.inspect_blender()), 'cases': rows}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
