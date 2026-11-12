"""Retain existing independent skin/outline verifiers for experiment outputs."""
import argparse
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
    for key in ('matrix', 'assets', 'out', 'blender'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    from converter_api.runner import RunnerSettings, BlenderRunner
    from converter_api.skin import _prepare_uncached as skin
    from converter_api.outline import _prepare_uncached as outline
    from converter.outline_spec import DEFAULT_SURFACE
    source, output = Path(args.matrix), Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CONVERTER_SKIN_ASSET_ROOT'] = str(Path(args.assets)/'skin-assets')
    settings = replace(RunnerSettings.from_env(ROOT), blender_binary=args.blender,
        temp_root=output, shape_timeout_seconds=600,
        process_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    runner = BlenderRunner(settings)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    matrix = json.loads((source/'results.json').read_text())
    rows = []
    for index in (0, 3):
        row = matrix[index]
        for variant in ('off', 'on'):
            cid = row['case']['character']
            data = (source/f'{index:02d}-{variant}.fbx').read_bytes()
            job = dict(source_artifact_sha256=sha(data), skin=dict(version='fbx-skin-v1', id='flesh-guide'),
                binding_sha256='a'*64, character_id=cid, character_sha256=row['character_sha256'], output_scope='full')
            start = time.monotonic()
            skinned, sr = skin(runner, data, job, checkpoint=lambda: None)
            (output/(cid+'-'+variant+'-skin.fbx')).write_bytes(skinned)
            oj = dict(source_artifact_sha256=sha(skinned), appearance=dict(version='fbx-outline-v1', width=2),
                binding_sha256='a'*64, character_id=cid, character_sha256=row['character_sha256'], output_scope='full',
                stature_m=1.7, culling='CullingOff', surface=DEFAULT_SURFACE)
            outlined, receipt = outline(runner, skinned, oj, checkpoint=lambda: None)
            (output/(cid+'-'+variant+'-composed.fbx')).write_bytes(outlined)
            rows.append(dict(character=cid, flag=variant, seconds=time.monotonic()-start,
                             skin=sr, outline=receipt))
            (output/'results.json').write_text(json.dumps(rows, indent=2))
            print(cid, variant, sr['file_verified'], receipt['file_verified'], flush=True)


if __name__ == '__main__':
    main()
