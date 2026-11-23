"""Offline replay through real Converter HTTP routes, Blender and FBX verifier.

Use the pinned Linux image with --network none, 2 CPUs and 4 GiB. Mount approved
assets read-only at /app/body-staging. No BFF, production DB, VLM or deployment.
The camera is a fixed front descriptor: camera fitting is outside body_prepare.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OfflineClient:
    """Use the real local HTTP server without adding test packages to its image."""
    def __init__(self, out):
        self.out = out
        self.url = 'http://127.0.0.1:18973'

    def __enter__(self):
        self.log = (self.out/'api.log').open('wb')
        self.server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'converter_api.app:app',
            '--host', '127.0.0.1', '--port', '18973'], cwd=ROOT, stdout=self.log, stderr=self.log)
        try:
            for _ in range(100):
                if self.server.poll() is not None:
                    raise RuntimeError('local API startup failed; inspect api.log')
                try:
                    if self.get('/preview-contract').status_code == 200:
                        return self
                except urllib.error.URLError:
                    pass
                time.sleep(.1)
            raise RuntimeError('local API startup deadline')
        except BaseException:
            self.__exit__()
            raise

    def request(self, path, method, data=None, headers=None):
        request = urllib.request.Request(self.url+path, data=data, method=method, headers=headers or {})
        try:
            response = urllib.request.urlopen(request, timeout=650)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            code, raw = response.code, response.read()
        class Result:
            status_code, content = code, raw
            def json(self):
                return json.loads(self.content)
        return Result()

    def get(self, path, headers=None):
        return self.request(path, 'GET', headers=headers)

    def post(self, path, json=None, headers=None):
        return self.request(path, 'POST', data=globals()['json'].dumps(json).encode() if json is not None else b'',
            headers={**(headers or {}), 'Content-Type': 'application/json'})

    def __exit__(self, *exc):
        self.server.terminate()
        try:
            self.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cid = 'male-base-v0-8-0'
    registry = json.loads((args.assets/'characters.json').read_text())
    character = registry['characters'][cid]
    token = 'local-body-trace-offline-token-20261123'
    os.environ.update(CONVERTER_CHARACTER_REGISTRY=str(args.assets/'characters.json'),
        CONVERTER_SHAPE_REGISTRY=str(args.assets/'shape-bakes.json'),
        CONVERTER_DEFAULT_CHARACTER_ID=cid, CONVERTER_INTERACTIVE_ENABLED='true',
        CONVERTER_INTERACTIVE_TOKEN=token, CONVERTER_INTERACTIVE_ROOT=str(args.out/'private'),
        CONVERTER_CHARACTER_CACHE_DIR=str(args.out/'characters'),
        CONVERTER_TEMP_ROOT=str(args.out/'temp'), CONVERTER_SHAPE_TIMEOUT_SECONDS='600',
        CONVERTER_FINAL_CACHE_ENABLED='0', CONVERTER_MAX_CONCURRENT_PROCESSES='2',
        CONVERTER_INTERACTIVE_BINARY_STATE='1')
    os.environ[character['artifact_uri_env']] = str(args.assets/'fbx'/(cid+'.fbx'))
    from converter.hand_library import load_library
    from converter.solved_state_v2 import ENCODING
    from converter.solved_state import digest, canonical
    from converter.body_trace import validated_event

    rows, events = [], []

    log_offset = 0
    def refresh_events():
        nonlocal log_offset
        with (args.out/'api.log').open('rb') as stream:
            stream.seek(log_offset)
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
            log_offset = stream.tell()
        for line in lines:
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                continue
            if value.get('event') in {'converter_body_stage', 'converter_body_reuse', 'converter_body_solver'}:
                value = validated_event(value)
            elif value.get('event') in {'interactive_stage', 'interactive_blender', 'interactive_body_cache'}:
                value = {k: v for k, v in value.items() if k in {'event', 'stage', 'duration_ms',
                    'succeeded', 'queue_wait_ms', 'execution_ms', 'return_code', 'hit'}}
            else:
                continue
            if value is not None:
                events.append(value)

    headers = {'Authorization': 'Bearer '+token, 'X-Standin-State-Encoding': ENCODING,
        'X-Standin-Work-Budget-Ms': '300000'}
    poses = ['g1__M_Move9__n05_f0810', 'sd_cmu_124_01_f00424']
    with OfflineClient(args.out) as client:
        for round_index in range(args.rounds):
            for pose in (poses if round_index % 2 == 0 else poses[::-1]):
                raw = (args.assets/'bvh'/(pose+'.bvh')).read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                owner = digest(canonical(['offline-trace', pose, round_index]))
                camera = dict(version='candidate-camera-v2', reference='search-fit',
                    rotation=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    projection='orthographic', distance_torso=None, source_bvh_sha256=sha)
                selection = dict(schema='interactive-selection-v1', ownerBinding=owner,
                    candidateId=pose, source=dict(originalBvhSha256=sha, outputBvhSha256=sha,
                        variant='base', refine=None),
                    body=dict(characterId=cid, assetSha256=character['sha256']),
                    shape=None, shapeSource=None, hands=dict(left=None, right=None,
                        library_sha256=load_library().sha256), scope='full', mirror=False, camera=camera)
                payload = dict(selection=selection, operation=f'trace-{round_index}-{pose}', person='0',
                    revision=1, bvhBase64=base64.b64encode(raw).decode(),
                    expiresAt=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=55)).isoformat())
                offset = len(events)
                started = time.monotonic()
                response = client.post('/interactive/v1/solve', json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f'{pose}: solve status {response.status_code}: {response.json()}')
                solve_seconds = time.monotonic()-started
                snapshot = response.json()['snapshotId']
                packet = response.json().get('packet')
                read_headers = {**headers, 'X-Standin-Owner': owner}
                model = client.get(f'/interactive/v1/snapshots/{snapshot}/model', headers=read_headers)
                if model.status_code != 200:
                    raise RuntimeError('model download failed')
                result = client.post(f'/interactive/v1/snapshots/{snapshot}/export', headers=read_headers)
                if result.status_code != 200 or result.json().get('verified') is not True:
                    raise RuntimeError('FBX export/reload verification failed')
                file = client.get(f'/interactive/v1/snapshots/{snapshot}/file', headers=read_headers)
                if file.status_code != 200 or hashlib.sha256(file.content).hexdigest() != result.json()['artifactSha256']:
                    raise RuntimeError('FBX download integrity failed')
                target = args.out/f'{round_index}-{pose}.fbx'
                target.write_bytes(file.content)
                refresh_events()
                cold = dict(round=round_index, pose=pose, sourceBvhSha256=sha,
                    solve_seconds=solve_seconds, file_ready_seconds=time.monotonic()-started,
                    artifact_sha256=result.json()['artifactSha256'], bytes=len(file.content),
                    receipt=result.json(), packet=packet, events=events[offset:])
                # New operation, same owner/source/body/shape: test actual body/core reuse.
                payload['operation'] += '-repeat'
                payload['revision'] += 1
                offset = len(events)
                warm_started = time.monotonic()
                repeated = client.post('/interactive/v1/solve', json=payload, headers=headers)
                if repeated.status_code != 200:
                    raise RuntimeError('repeat solve failed')
                sid = repeated.json()['snapshotId']
                reexport = client.post(f'/interactive/v1/snapshots/{sid}/export', headers=read_headers)
                refile = client.get(f'/interactive/v1/snapshots/{sid}/file', headers=read_headers)
                if reexport.status_code != 200 or refile.status_code != 200:
                    raise RuntimeError('repeat export failed')
                refresh_events()
                cold['repeat_seconds'] = time.monotonic()-warm_started
                cold['repeat_events'] = events[offset:]
                rows.append(cold)
                (args.out/'results.json').write_text(json.dumps(dict(scope=__doc__, cases=rows), indent=2))
                print('OFFLINE_BODY_REPLAY', pose, round(solve_seconds, 3), round(cold['file_ready_seconds'], 3),
                    'repeat', round(cold['repeat_seconds'], 3), flush=True)


if __name__ == '__main__':
    main()
