"""Sequential uncached converter OFF/ON runs and fresh-process FBX comparison.

No server, bundle write, deploy, release or cross-job result cache. The measured
file-ready boundary is BlenderRunner.convert, not BFF/app end-to-end latency.
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


def compare_files(job):
    import numpy as np
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.final_artifact import validate
    rt.import_character = import_full_character

    def read(path):
        qa = validate(Path(path), job.get('scope', 'full'))
        arm = next(o for o in __import__('bpy').context.scene.objects if o.type == 'ARMATURE')
        meshes = sorted((o for o in __import__('bpy').context.scene.objects if o.type == 'MESH'), key=lambda o: o.name)
        frames = {b.name: (b.parent.name if b.parent else None,
                           np.asarray(arm.matrix_world @ b.matrix_local).copy()) for b in arm.data.bones}
        data = []
        for obj in meshes:
            mesh = obj.evaluated_get(__import__('bpy').context.evaluated_depsgraph_get()).data
            normals = obj.matrix_world.to_3x3().inverted().transposed()
            groups = {g.index: g.name for g in obj.vertex_groups}
            data.append(dict(points=np.asarray([list(obj.matrix_world @ v.co) for v in mesh.vertices]),
                normals=np.asarray([list((normals @ n.vector).normalized()) for n in mesh.corner_normals]),
                weights={(v.index, groups[g.group]): g.weight for v in mesh.vertices for g in v.groups},
                topology=[list(p.vertices) for p in mesh.polygons], matrix=np.asarray(obj.matrix_world).copy()))
        return qa, frames, data

    qa, a, am = read(job['off'])
    qb, b, bm = read(job['on'])
    assert a.keys() == b.keys() and len(am) == len(bm), 'rig/mesh count changed'
    errors = dict(position_m=0., matrix=0., weight=0., normal=0.)
    for name in a:
        assert a[name][0] == b[name][0], 'bone parent changed'
        errors['matrix'] = max(errors['matrix'], float(np.max(np.abs(a[name][1]-b[name][1]))))
    for x, y in zip(am, bm):
        assert x['topology'] == y['topology'], 'topology changed'
        assert x['weights'].keys() == y['weights'].keys(), 'weighted joint identity changed'
        assert x['points'].shape == y['points'].shape and x['normals'].shape == y['normals'].shape
        errors['position_m'] = max(errors['position_m'], float(np.max(np.linalg.norm(x['points']-y['points'], axis=1))))
        errors['normal'] = max(errors['normal'], float(np.max(np.linalg.norm(x['normals']-y['normals'], axis=1))))
        errors['matrix'] = max(errors['matrix'], float(np.max(np.abs(x['matrix']-y['matrix']))))
        errors['weight'] = max(errors['weight'], max(abs(w-y['weights'][k]) for k, w in x['weights'].items()))
    limits = dict(position_m=1e-4, matrix=1e-5, weight=1e-6, normal=1e-4)
    result = dict(ok=all(np.isfinite(v) and v <= limits[k] for k, v in errors.items()),
        errors=errors, limits=limits, topology_equal=True, joint_identity_equal=True,
        off_validation=qa, on_validation=qb)
    Path(job['result']).write_text(json.dumps(result, indent=2))
    if not result['ok']:
        raise ValueError(result)


def run(args):
    from converter.shape_spec import resolve_shape, digest_shape
    from converter_api.runner import BlenderRunner, RunnerSettings
    from converter.body_scene_transfer import FLAG
    os.environ['CONVERTER_SHAPE_REGISTRY'] = str(Path(args.registry).resolve())
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = json.loads(Path(args.cases).read_text())
    for case in manifest:
        cid = case['character']
        character = Path(args.characters_root)/(cid+'.fbx')
        asset_sha = hashlib.sha256(character.read_bytes()).hexdigest()
        raw = None
        if case.get('controls'):
            initial = resolve_shape(cid, asset_sha)
            values = {c['key']: c['default'] for c in initial['controls']}
            values.update(case['controls'])
            raw = dict(schemaVersion='body-shape.v1', controls=values, shapeSha256=digest_shape(cid, asset_sha, values))
        recipe = resolve_shape(cid, asset_sha, raw)
        source = Path(case['bvh']).read_bytes()
        row = dict(case=case, character_sha256=asset_sha, source_sha256=hashlib.sha256(source).hexdigest(), runs={})
        name = f'{len(rows):02d}'
        # First-run bias alternates; each variant is still a fresh child/job.
        for variant in (('off', 'on') if len(rows) % 2 == 0 else ('on', 'off')):
            env = {FLAG: '1' if variant == 'on' else '0', 'CONVERTER_PREPARED_PRESETS_ENABLED': '0',
                   'CONVERTER_FINAL_CACHE_ENABLED': '0', 'STANDIN_LAB_FIRST_ENTRY_PROFILE': '1',
                   'STANDIN_LAB_PREVIEW_TRACE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
            probe = output/f'{name}-{variant}-probe'
            probe.mkdir()
            env['STANDIN_SCENE_PROBE_DIR'] = str(probe)
            if case.get('budget') is not None:
                env['CONVERTER_EXPERIMENT_BODY_SCENE_SECONDS'] = str(case['budget'])
            settings = replace(RunnerSettings.from_env(ROOT), blender_binary=args.blender, temp_root=output,
                               retarget_cache_bytes=0, fast_scene_reset=False, shape_timeout_seconds=600, process_env=env)
            started = time.monotonic()
            try:
                result = BlenderRunner(settings).convert(bvh_bytes=source, character_path=character, character_id=cid,
                    character_sha256=asset_sha, conversion_id=f'scene-{name}-{variant}', shape_recipe=recipe,
                    output_scope=case.get('scope', 'full'), hands=case.get('hands'))
                (output/f'{name}-{variant}.fbx').write_bytes(result.artifact)
                (output/f'{name}-{variant}-report.json').write_text(json.dumps(result.report, indent=2))
                profile = result.report['lab_call_profile']
                row['runs'][variant] = dict(seconds=time.monotonic()-started,
                    scene_transfer=result.report.get('scene_transfer'),
                    stage_timings=result.report.get('lab_stage_timings'),
                    all_fbx_parses=sum(r['function'] == 'fbx.parse' for r in profile),
                    import_loader_parses=sum(r['function'] == 'fbx.parse' and r['caller'] == 'import_fbx.py:load' for r in profile),
                    memory_imports=sum(r['function'] == 'fbx.parse' and (r.get('file') or '').endswith('.memory-rig.fbx') for r in profile),
                    retarget_exports=sum(r['function'] == 'retarget.export_fbx' for r in profile),
                    memory_exports=sum(r['function'] == 'retarget.export_fbx' and (r.get('file') or '').endswith('.memory-rig.fbx') for r in profile),
                    shape_controls=result.report['shape_bake']['controls'])
                row['runs'][variant]['probes'] = [json.loads(p.read_text()) for p in probe.glob('*.json')]
            except Exception as exc:
                row['runs'][variant] = dict(seconds=time.monotonic()-started, error=type(exc).__name__)
                row['ok'] = False
                break
        if all('error' not in row['runs'].get(v, {'error': 'missing'}) for v in ('off', 'on')):
            job = dict(off=str(output/f'{name}-off.fbx'), on=str(output/f'{name}-on.fbx'),
                       scope=case.get('scope', 'full'), result=str(output/f'{name}-comparison.json'))
            path = output/f'{name}-compare-job.json'
            path.write_text(json.dumps(job))
            with (output/f'{name}-compare.log').open('w') as log:
                completed = subprocess.run([args.blender, '--background', '--factory-startup', '--disable-autoexec',
                    '--python-exit-code', '1', '--python', str(Path(__file__).resolve()), '--', '--compare', str(path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=120)
            if Path(job['result']).exists():
                row['comparison'] = json.loads(Path(job['result']).read_text())
            row['ok'] = completed.returncode == 0
        rows.append(row)
        (output/'results.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    if not all(row['ok'] for row in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('registry', 'characters-root', 'cases', 'out', 'blender', 'compare'):
        parser.add_argument('--'+option)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    if args.compare:
        compare_files(json.loads(Path(args.compare).read_text()))
    elif all((args.registry, args.characters_root, args.cases, args.out, args.blender)):
        run(args)
    else:
        parser.error('registry, characters-root, cases, out and blender are required')


if __name__ == '__main__':
    main()
