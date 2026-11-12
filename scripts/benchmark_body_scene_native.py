"""Real native Converter ASGI prepare/solve/export, fresh store per OFF/ON.

Fixture selection only; actual registry/Blender/canonical state/fresh verifier.
No network listener, BFF/app UX, server configuration change or deployment.
"""
import argparse
import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('assets', 'blender', 'out'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--cases')
    args = parser.parse_args()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from converter_api.registry import CharacterRegistry
    from converter_api.runner import BlenderRunner, RunnerSettings
    from converter_api.interactive import register_interactive
    from converter.body_scene_transfer import FLAG
    from converter.solved_state import V1_ENCODING
    from converter.solved_state_v2 import ENCODING as V2_ENCODING
    from converter.shape_spec import resolve_shape, digest_shape
    assets, output = Path(args.assets).resolve(), Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    character_config = json.loads((assets/'characters.json').read_text())
    rows = []
    cases = json.loads(Path(args.cases).read_text()) if args.cases else [
        dict(character='male-base-v0-8-0', encoding=V2_ENCODING),
        dict(character='female-base-v0-4-0', encoding=V1_ENCODING)]
    for index, case in enumerate(cases):
        cid, encoding = case['character'], case.get('encoding', V2_ENCODING)
        name = f'{index:02d}-'+cid
        raw = Path(case.get('bvh', assets/'ob_015.bvh')).read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        row = dict(character=cid, case=case, encoding=encoding, source_sha256=source_sha, runs={})
        for variant in (('off', 'on') if index % 2 == 0 else ('on', 'off')):
            directory = output/(name+'-'+variant)
            directory.mkdir()
            probe = directory/'probe'
            probe.mkdir()
            token = secrets.token_hex(32)
            os.environ.update({FLAG: '1' if variant == 'on' else '0',
                'CONVERTER_INTERACTIVE_ENABLED': 'true', 'CONVERTER_INTERACTIVE_TOKEN': token,
                'CONVERTER_INTERACTIVE_ROOT': str(directory/'private'),
                'CONVERTER_INTERACTIVE_BODY_FORMAT': 'v2',
                'CONVERTER_INTERACTIVE_BINARY_STATE': '1',
                'CONVERTER_PREPARED_PRESETS_ENABLED': '0', 'CONVERTER_FINAL_CACHE_ENABLED': '0',
                'CONVERTER_SHAPE_REGISTRY': str(assets/'shape-bakes.json'),
                'STANDIN_SCENE_PROBE_DIR': str(probe), 'STANDIN_LAB_FIRST_ENTRY_PROFILE': '1',
                'STANDIN_LAB_PREVIEW_TRACE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            cfg = character_config['characters'][cid]
            row['character_sha256'] = cfg['sha256']
            shape = None
            if case.get('controls'):
                recipe = resolve_shape(cid, cfg['sha256'])
                values = {c['key']: c['default'] for c in recipe['controls']}
                values.update(case['controls'])
                shape = dict(schemaVersion='body-shape.v1', controls=values,
                             shapeSha256=digest_shape(cid, cfg['sha256'], values))
            registry = CharacterRegistry(assets/'characters.json', default_character_id=cid,
                environ={cfg['artifact_uri_env']: str(assets/'fbx'/(cid+'.fbx'))})
            settings = replace(RunnerSettings.from_env(ROOT), blender_binary=args.blender,
                               temp_root=directory, shape_timeout_seconds=600, fast_scene_reset=False)
            app = FastAPI()
            register_interactive(app, registry, BlenderRunner(settings))
            owner = 'a'*64
            payload = dict(selection=dict(schema='interactive-selection-v1', ownerBinding=owner,
                candidateId='fixture', body=dict(characterId=cid, assetSha256=cfg['sha256']), shape=shape,
                shapeSource=None, source=dict(variant='base', originalBvhSha256=source_sha, outputBvhSha256=source_sha),
                camera=dict(version='candidate-camera-v2', reference='search-fit', source_bvh_sha256=source_sha,
                    projection='orthographic', rotation=[[1,0,0],[0,1,0],[0,0,1]], distance_torso=None),
                hands=None, scope='full', mirror=False), operation='experiment', revision=1,
                person='0', bvhBase64=base64.b64encode(raw).decode(),
                expiresAt=datetime.fromtimestamp(time.time()+1800, timezone.utc).isoformat())
            headers = {'Authorization': 'Bearer '+token, 'X-Standin-Owner': owner,
                       'X-Standin-State-Encoding': encoding}
            with TestClient(app) as client:
                load_before = os.getloadavg()
                start = time.monotonic()
                response = client.post('/interactive/v1/solve', json=payload, headers=headers)
                if response.status_code != 200:
                    raise ValueError('native solve failed: '+str(response.status_code)+' '+response.text[:500])
                value = response.json()
                assert response.headers['x-standin-state-encoding'] == encoding
                snapshot = value['snapshotId']
                model = client.get('/interactive/v1/snapshots/'+snapshot+'/model', headers=headers)
                assert model.status_code == 200
                model_seconds = time.monotonic()-start
                response = client.post('/interactive/v1/snapshots/'+snapshot+'/export', headers=headers)
                if response.status_code != 200:
                    raise ValueError('native export failed: '+str(response.status_code)+' '+response.text[:500])
                receipt = response.json()
                artifact = client.get('/interactive/v1/snapshots/'+snapshot+'/file', headers=headers)
                assert artifact.status_code == 200
                elapsed = time.monotonic()-start
                assert hashlib.sha256(artifact.content).hexdigest() == receipt['artifactSha256']
                (output/(name+'-'+variant+'.fbx')).write_bytes(artifact.content)
                (directory/'receipt.json').write_text(json.dumps(receipt, indent=2))
                probe_rows = [json.loads(p.read_text()) for p in probe.glob('process-*.json')]
                body = next(p for p in probe_rows if 'prepare_body' in p['stages'])
                worker = json.loads((probe/'worker-report.json').read_text())
                expected = resolve_shape(cid, cfg['sha256'], shape)
                assert worker['shape_bake']['controls'] == expected['values']
                assert worker['shape_bake']['neutral_preparation']['status'] == 'disabled'
                io = body['io']
                row['runs'][variant] = dict(file_ready_seconds=elapsed, model_ready_seconds=model_seconds,
                    body_prepare_seconds=body['stages']['prepare_body']['seconds'],
                    body_run_job_seconds=body['stages']['run_job']['seconds'],
                    peak_rss_bytes=max(p['peak_rss_bytes'] for p in probe_rows),
                    scene_transfer=worker.get('scene_transfer'), probes=probe_rows,
                    correspondence=receipt['correspondence'], canonical_encoding=encoding,
                    shape_controls=worker['shape_bake']['controls'], controls_match=True,
                    neutral_preparation=worker['shape_bake']['neutral_preparation'],
                    host_load_before=load_before, host_load_after=os.getloadavg(),
                    fbx_io={op:dict(disk_calls=sum(e['operation']==op and not e['memory'] for e in io),
                        memory_calls=sum(e['operation']==op and e['memory'] for e in io),
                        disk_bytes=sum((e['bytes'] or 0) for e in io if e['operation']==op and not e['memory']))
                        for op in ('parse', 'write')})
            app.state.interactive._volume_lock.close()
        job = dict(off=str(output/(name+'-off.fbx')), on=str(output/(name+'-on.fbx')),
                   result=str(output/(name+'-comparison.json')))
        path = output/(name+'-compare-job.json')
        path.write_text(json.dumps(job))
        with (output/(name+'-compare.log')).open('w') as log:
            completed = subprocess.run([args.blender, '--background', '--factory-startup', '--disable-autoexec',
                '--python-exit-code', '1', '--python', str(ROOT/'scripts/benchmark_body_scene_transfer.py'),
                '--', '--compare', str(path)], stdout=log, stderr=subprocess.STDOUT, timeout=120)
        row['comparison'] = json.loads(Path(job['result']).read_text())
        row['ok'] = completed.returncode == 0
        rows.append(row)
        (output/'results.json').write_text(json.dumps(rows, indent=2))
        print(cid, row['ok'], {k:round(v['file_ready_seconds'],3) for k,v in row['runs'].items()}, flush=True)
    assert all(r['ok'] for r in rows)


if __name__ == '__main__':
    main()
