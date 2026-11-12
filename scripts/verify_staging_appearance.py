"""Fresh-process skin and outline verification for pinned current staging bodies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--blender', default='/Applications/Blender.app/Contents/MacOS/Blender')
    p.add_argument('--character', action='append')
    p.add_argument('--cases', type=Path, help='Additional shape/pose cases with source, characterId, statureM')
    p.add_argument('--cases-only', action='store_true')
    p.add_argument('--skin', action='append', choices=('flesh','flesh-guide','gray-guide','muscle'))
    p.add_argument('--outline-width', type=float)
    a = p.parse_args()
    release = json.loads((ROOT / 'config/staging-anatomy-release.json').read_bytes())
    manifest = json.loads((ROOT / 'converter/skin_assets.v1.json').read_bytes())
    a.out.mkdir(parents=True, exist_ok=True)
    cases = [dict(name=cid, characterId=cid, source=str(a.models / 'fbx' / (cid + '.fbx')),
                  statureM=1.78 if cid.startswith('male') else 1.75)
             for cid in ([] if a.cases_only else (a.character or release['characterIds']))]
    if a.cases:
        cases.extend(json.loads(a.cases.read_bytes()))
    if a.skin:
        cases = [dict(case, name=case['name']+'--'+skin, skinId=skin) for case in cases for skin in a.skin]
    rows = []
    for case in cases:
        cid = case['characterId']
        common = dict(character_id=cid, character_sha256=manifest['models'][cid]['character_sha256'],
                      binding_sha256=hashlib.sha256(case['name'].encode()).hexdigest(), output_scope='full')
        source = Path(case['source'])
        row = dict(case, sourceSha256=sha(source))
        for stage in ('skin', 'outline'):
            directory = a.out / case['name'] / stage
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, directory / 'source.fbx')
            job = dict(common, source_artifact_sha256=sha(source))
            if stage == 'skin':
                job['skin'] = dict(version='fbx-skin-v1', id=case.get('skinId','flesh'))
            else:
                job.update(appearance=dict(version='fbx-outline-v1', width=a.outline_width if a.outline_width is not None else case.get('width', 2)),
                           stature_m=case['statureM'], culling='CullingOff', surface='outward-transparent')
            (directory / 'job.json').write_text(json.dumps(job))
            started = time.monotonic()
            for action in ('export', 'verify'):
                with (directory / (action + '.log')).open('w') as log:
                    subprocess.run([a.blender, '--background', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
                                    '--python', str(ROOT / 'converter' / (stage + '_worker.py')), '--', action, str(directory)],
                                   cwd=ROOT, env={**os.environ, 'CONVERTER_SKIN_ASSET_ROOT': str(a.assets)},
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360)
            receipt = json.loads((directory / 'verified.json').read_bytes())
            assert receipt['file_verified'] and receipt['base']['typed_fbx_values_preserved']
            assert all(v == 0 for k, v in receipt['base'].items() if k.startswith('max')), receipt['base']
            source = directory / 'artifact.fbx'
            assert sha(source) == receipt['fbx_sha256']
            row[stage] = dict(sha256=sha(source), bytes=source.stat().st_size, path=str(source),
                              seconds=time.monotonic()-started, receipt=receipt)
        rows.append(row)
        (a.out / 'report.json').write_text(json.dumps(dict(cases=rows, cspVisualVerified=False), indent=2) + '\n')
        print('VERIFIED', case['name'], flush=True)


if __name__ == '__main__':
    main()
