"""Uncached localhost HTTP file-ready comparisons; no BFF, PNG or deployment.

Run inside the pinned Linux/amd64 image, sequentially after offline QA. Each
variant has a fresh API process and private cache directories. The response's
neutral-preparation time is distinct from POST -> verified FBX saved on disk.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', required=True)
    ap.add_argument('--bvh', required=True)
    ap.add_argument('--character', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=19971)
    args = ap.parse_args()
    if not 1024 <= args.port <= 65535:
        ap.error('choose an unprivileged local port')
    assets, out = Path(args.assets), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    bvh = Path(args.bvh).read_bytes()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    characters = json.loads((assets/'characters.json').read_text())['characters']
    manifest = json.loads((assets/'.body-assets-ready.json').read_text())
    registry = json.loads((assets/'shape-bakes.json').read_text())
    if (len(args.character) != len(set(args.character))
            or any(cid not in registry['characters'] or cid not in characters for cid in args.character)):
        raise ValueError('choose distinct registered shape characters')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'CONVERTER_CHARACTER_REGISTRY': str(assets/'characters.json'),
        'CONVERTER_DEFAULT_CHARACTER_ID': args.character[0], 'CONVERTER_SHAPE_REGISTRY': str(assets/'shape-bakes.json'),
        'CONVERTER_FINAL_CACHE_ENABLED': '0', 'CONVERTER_RETARGET_CACHE_BYTES': '0',
        'CONVERTER_MAX_CONCURRENT_PROCESSES': '1', 'CONVERTER_SHAPE_TIMEOUT_SECONDS': '600'}
    # Resolve every character's approved bytes by digest, including old aliases.
    by_sha = {sha(p.read_bytes()): p for p in sorted((assets/'fbx').glob('*.fbx'))}
    for character in characters.values():
        # The registry also lists generic characters outside this body release.
        # They are never requested by this preset-only benchmark.
        if character['sha256'] in by_sha:
            env[character['artifact_uri_env']] = str(by_sha[character['sha256']])
    if any(characters[cid]['sha256'] not in by_sha for cid in args.character):
        raise ValueError('selected approved character bytes are missing')
    rows = []
    url = f'http://127.0.0.1:{args.port}'
    for i, cid in enumerate(args.character):
        row = dict(character=cid, bvh_sha256=sha(bvh), runs={}, ok=False)
        for variant in (('baseline', 'prepared') if i % 2 == 0 else ('prepared', 'baseline')):
            directory = out/(cid+'-'+variant); directory.mkdir()
            child_env = {**env, 'CONVERTER_PREPARED_PRESETS_ENABLED': '1' if variant=='prepared' else '0'}
            for key, name in [('CONVERTER_TEMP_ROOT','temp'), ('CONVERTER_CHARACTER_CACHE_DIR','characters'),
                              ('CONVERTER_INTERACTIVE_ROOT','private'), ('CONVERTER_FINAL_CACHE_ROOT','final')]:
                child_env[key] = str(directory/name)
                (directory/name).mkdir()
            with (directory/'api.log').open('w') as log:
                server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'converter_api.app:app',
                    '--host', '127.0.0.1', '--port', str(args.port)], cwd=ROOT, env=child_env, stdout=log, stderr=log)
                try:
                    for _ in range(180):
                        if server.poll() is not None:
                            raise RuntimeError('local benchmark API failed to start; see private log')
                        try:
                            with urllib.request.urlopen(url+'/preview-contract', timeout=5) as response:
                                contract = json.load(response)
                            break
                        except (urllib.error.URLError, TimeoutError):
                            time.sleep(.1)
                    else:
                        raise RuntimeError('local benchmark API startup deadline')
                    boundary = 'preparedbench'+uuid.uuid4().hex
                    fields = dict(character_id=cid, output_scope='full', preview_format='none',
                        expected_bvh_sha256=sha(bvh), expected_character_sha256=characters[cid]['sha256'],
                        expected_preview_revision=contract['preview_revision'])
                    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                             for k,v in fields.items()]
                    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="bvh"; filename="pose.bvh"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),
                              bvh, f'\r\n--{boundary}--\r\n'.encode()]
                    request = urllib.request.Request(url+'/convert-framed', data=b''.join(parts), headers={
                        'Content-Type': 'multipart/form-data; boundary='+boundary, 'X-Standin-Work-Budget-Ms': '300000'})
                    started = time.monotonic()
                    with urllib.request.urlopen(request, timeout=650) as response:
                        raw = response.read()
                    response_seconds = time.monotonic()-started
                    result = json.loads(raw)
                    artifact = base64.b64decode(result['fbx_base64'], validate=True)
                    if (sha(artifact) != result['fbx_sha256'] or not result['artifact_validation']['skin_verified']
                            or result['character_sha256'] != registry['characters'][cid]['asset_sha256']
                            or result['source_bvh_sha256'] != sha(bvh)):
                        raise ValueError('HTTP artifact identity or validation failed')
                    target = out/(cid+'-'+variant+'.fbx')
                    with target.open('xb') as handle:
                        handle.write(artifact); handle.flush(); os.fsync(handle.fileno())
                    if sha(target.read_bytes()) != result['fbx_sha256']:
                        raise ValueError('saved HTTP artifact checksum mismatch')
                    preparation = result['shape_bake']['neutral_preparation']
                    if preparation['status'] != ('prepared' if variant=='prepared' else 'disabled'):
                        raise ValueError('HTTP neutral path differs from requested condition')
                    row['runs'][variant] = dict(response_seconds=response_seconds,
                        file_ready_seconds=time.monotonic()-started, neutral_preparation=preparation,
                        artifact_sha256=sha(artifact), artifact_bytes=len(artifact),
                        validation=result['artifact_validation'], final_cache=result.get('final_cache'))
                finally:
                    server.terminate()
                    try: server.wait(timeout=10)
                    except subprocess.TimeoutExpired: server.kill(); server.wait()
        comparison = out/(cid+'-comparison.json')
        job = out/(cid+'-compare-job.json')
        job.write_text(json.dumps(dict(baseline=str(out/(cid+'-baseline.fbx')),
            prepared=str(out/(cid+'-prepared.fbx')), result=str(comparison))))
        with (out/(cid+'-compare.log')).open('w') as log:
            subprocess.run([env['BLENDER_BINARY'], '--background', '--factory-startup', '--python-exit-code', '1',
                '--python', str(ROOT/'scripts/benchmark_prepared_bodies.py'), '--', '--compare', str(job)],
                check=True, stdout=log, stderr=subprocess.STDOUT)
        row['comparison'] = json.loads(comparison.read_text()); row['ok'] = row['comparison']['ok']
        rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(cases=rows, manifest=manifest,
            scope='localhost HTTP POST through decoded, hash-verified FBX saved; no BFF/client/PNG',
            os_file_cache='uncontrolled; one sample per condition; sequential Blender only'), indent=2)+'\n')
        print('HTTP_PREPARED_COMPARISON', cid, row['ok'], flush=True)


if __name__ == '__main__':
    main()
