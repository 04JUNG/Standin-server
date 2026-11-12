"""Compare uncached first conversions with/without prepared neutral preset bodies.

No completed-FBX cache, PNG, VLM, upload or deployment is used. Final FBXs are
independently reloaded in Blender to compare geometry, topology, rig and weights.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare(job):
    import numpy as np
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    def read(path):
        rt.reset_scene()
        arm, meshes = import_full_character(path)
        if len(meshes) != 1:
            raise ValueError('benchmark expects one body mesh')
        mesh = meshes[0]
        points = np.empty(len(mesh.data.vertices)*3, dtype=np.float32)
        mesh.data.vertices.foreach_get('co', points)
        bones = {b.name: np.asarray(arm.matrix_world@b.matrix_local) for b in arm.data.bones}
        names = {g.index: g.name for g in mesh.vertex_groups}
        weights = {(v.index, names[g.group]): g.weight for v in mesh.data.vertices for g in v.groups}
        topology = [tuple(face.vertices) for face in mesh.data.polygons]
        return points, bones, weights, topology, np.asarray(mesh.matrix_world)
    a, b = read(job['baseline']), read(job['prepared'])
    if a[0].shape != b[0].shape or a[1].keys() != b[1].keys() or a[2].keys() != b[2].keys() or a[3] != b[3]:
        raise ValueError('prepared result changed mesh/rig/skin topology')
    result = {
        'vertices': int(a[0].size//3), 'bones': len(a[1]),
        'vertex_max_delta': float(np.max(np.abs(a[0]-b[0]))),
        'rig_max_delta': max(float(np.max(np.abs(a[1][n]-b[1][n]))) for n in a[1]),
        'mesh_matrix_max_delta': float(np.max(np.abs(a[4]-b[4]))),
        'weight_max_delta': max(abs(v-b[2][k]) for k,v in a[2].items()),
    }
    result['ok'] = all(result[k] <= limit for k,limit in [
        ('vertex_max_delta', 1e-6), ('rig_max_delta', 1e-6),
        ('mesh_matrix_max_delta', 1e-6), ('weight_max_delta', 1e-7)])
    Path(job['result']).write_text(json.dumps(result, indent=2))
    if not result['ok']:
        raise ValueError(result)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry')
    ap.add_argument('--characters-root')
    ap.add_argument('--character', action='append', default=[])
    ap.add_argument('--bvh', action='append', default=[])
    ap.add_argument('--out')
    ap.add_argument('--blender', default=os.getenv('BLENDER_BINARY', 'blender'))
    ap.add_argument('--compare')
    args = ap.parse_args(argv)
    if args.compare:
        compare(json.loads(Path(args.compare).read_text()))
        return
    if not all([args.registry, args.characters_root, args.character, args.bvh, args.out]):
        ap.error('registry, characters-root, character, bvh and out are required')
    os.environ['CONVERTER_SHAPE_REGISTRY'] = str(Path(args.registry).resolve())
    os.environ['CONVERTER_FINAL_CACHE_ENABLED'] = '0'
    from converter.shape_spec import load_registry, resolve_shape
    from converter_api.runner import BlenderRunner, RunnerSettings
    registry = load_registry()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for cid in args.character:
        for source in args.bvh:
            recipe = resolve_shape(cid, registry['characters'][cid]['asset_sha256'])
            bvh = Path(source).read_bytes()
            name = str(len(rows)).zfill(2) + '-' + cid
            row = {'character': cid, 'bvh_sha256': hashlib.sha256(bvh).hexdigest(),
                   'bvh_name': Path(source).name, 'runs': {}, 'ok': False}
            # Alternate order to reduce first-run/file-system-cache bias.
            order = ['baseline', 'prepared'] if len(rows)%2 == 0 else ['prepared', 'baseline']
            try:
                for variant in order:
                    settings = replace(RunnerSettings.from_env(ROOT), blender_binary=args.blender,
                        retarget_cache_bytes=0, fast_scene_reset=False, shape_timeout_seconds=600,
                        process_env={'CONVERTER_FINAL_CACHE_ENABLED': '0',
                            'CONVERTER_PREPARED_PRESETS_ENABLED': '1' if variant == 'prepared' else '0'})
                    started = time.monotonic()
                    result = BlenderRunner(settings).convert(bvh_bytes=bvh,
                        character_path=Path(args.characters_root)/(cid+'.fbx'), character_id=cid,
                        character_sha256=recipe['asset_sha256'], conversion_id=name+'-'+variant,
                        shape_recipe=recipe)
                    row['runs'][variant] = {'seconds': time.monotonic()-started,
                        'preparation': result.report['shape_bake']['neutral_preparation'],
                        'validation': result.report.get('artifact_validation'),
                        'shape_controls': result.report['shape_bake']['controls']}
                    (output/(name+'-'+variant+'.fbx')).write_bytes(result.artifact)
                    if variant == 'prepared' and row['runs'][variant]['preparation']['status'] != 'prepared':
                        raise ValueError('prepared path was not executed')
                comparison = output/(name+'-comparison.json')
                job = output/(name+'-compare-job.json')
                job.write_text(json.dumps({'baseline': str(output/(name+'-baseline.fbx')),
                    'prepared': str(output/(name+'-prepared.fbx')), 'result': str(comparison)}))
                with (output/(name+'-compare.log')).open('w') as log:
                    subprocess.run([args.blender,'--background','--factory-startup','--python-exit-code','1',
                        '--python',str(Path(__file__).resolve()),'--','--compare',str(job)],
                        check=True, stdout=log, stderr=subprocess.STDOUT)
                row['comparison'] = json.loads(comparison.read_text())
                row['ok'] = row['comparison']['ok'] and row['runs']['baseline']['shape_controls'] == row['runs']['prepared']['shape_controls']
            except Exception as error:
                row['error'] = str(error)
            rows.append(row)
            (output/'results.json').write_text(json.dumps(rows, indent=2))
            print('PREPARED_BENCHMARK', json.dumps(row), flush=True)
    if not all(row['ok'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
