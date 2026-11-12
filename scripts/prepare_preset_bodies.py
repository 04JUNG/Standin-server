"""Offline preparation only: immutable neutral FBXs/rest frames for registered presets.

python scripts/prepare_preset_bodies.py --registry shape-bakes.local.json \
    --out /new/prepared-release --rules <approved-body-rules>

No uploads, service changes, or runtime custom-body writes. The generated
shape-bakes.local.json activates these prototypes in a local Converter run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.prepared_body import identity, body_key, preset_values, read_prepared, MAX_RECEIPT_BYTES, _read  # noqa: E402
from converter.shape_spec import load_registry  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def require_runtime():
    import bpy
    from converter.protocol import EXPECTED_BLENDER_VERSION, EXPECTED_BLENDER_BUILD_HASH
    if (bpy.app.version_string.split()[0] != EXPECTED_BLENDER_VERSION
            or bpy.app.build_hash.decode() != EXPECTED_BLENDER_BUILD_HASH):
        raise ValueError('prepared bodies require the pinned Blender runtime')
    return bpy


def reuse_existing(recipe, entry, output):
    """Preserve an already approved prototype byte-for-byte, only on this runtime."""
    started = time.monotonic()
    prototype = read_prepared(recipe, entry)  # exact code/build/platform + every hash
    raw = _read(entry['path'], MAX_RECEIPT_BYTES)
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('prepared receipt changed during reuse')
    key = prototype['key']; output = Path(output)
    receipt = output/'qa'/f'prepared-{key}.json'
    if receipt.exists():
        raise ValueError('duplicate prepared identity; output must be fresh')
    for path, data in [(output/'fbx'/f'prepared-{key}.fbx', prototype['data']), (receipt, raw)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return dict(path=str(receipt), sha256=entry['sha256'], key=key,
                fbx_bytes=len(prototype['data']), generation_seconds=0.,
                reused_existing=True, reuse_seconds=time.monotonic()-started)


def bake(recipe, output):
    bpy = require_runtime()
    from converter.shape_bake import _load, import_full_character
    from converter.final_artifact import validate
    from converter import retarget as rt
    expected = identity(recipe)
    if expected is None:
        raise ValueError('only registered preset defaults may be prepared')
    key = body_key(expected)
    output = Path(output)
    # Identical aliases can share a prototype; never recompute customer settings.
    receipt_path = output / 'qa' / ('prepared-' + key + '.json')
    if receipt_path.exists():
        raise ValueError('duplicate prepared identity; output must be fresh')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output) as tmp:
        module = _load(recipe)
        bundle = module.export_fixed(str(Path(tmp) / 'neutral'))
        original = json.loads((bundle / 'manifest.json').read_text())
        source = bundle / 'model.fbx'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != original['fbx_sha256']:
            raise ValueError('neutral export receipt changed')
        # Same importer/policy as the production shape path, including real leaf bones.
        rt.import_character = import_full_character
        validation = validate(source, 'full')
        arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
        rest = {b.name: [list(row) for row in arm.matrix_world @ b.matrix_local] for b in arm.data.bones}
        receipt = {'identity': expected, 'key': key, 'fbx_sha256': digest,
                   'fbx_bytes': source.stat().st_size, 'rest_frames': rest,
                   'neutral_validation': validation}
        target = output / 'fbx' / ('prepared-' + key + '.fbx')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        write(receipt_path, receipt)
        if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
            raise ValueError('prepared receipt exceeded budget')
    entry = {'path': str(receipt_path), 'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    read_prepared(recipe, entry)  # use the exact runtime reader before publishing an index
    return {**entry, 'generation_seconds': time.monotonic() - started, 'key': key,
            'fbx_bytes': receipt['fbx_bytes']}


def run_worker(job):
    require_runtime()  # reusing old files must not bypass the actual Blender pin
    output = Path(job['out'])
    registry = job['registry']
    entries, results, completed = {}, [], {}
    for character_id in job['characters']:
        raw = registry['characters'][character_id]
        recipe = {**raw, **registry['sources'][raw['rules']],
                  'values': preset_values(raw), 'shape_sha256': None}
        try:
            key = body_key(identity(recipe))
            if key in completed:
                result = {**completed[key], 'generation_seconds': 0., 'alias_reused': True}
            else:
                previous = registry.get('prepared_neutrals', {}).get('entries', {}).get(character_id)
                result = None
                if previous is not None:
                    try:
                        result = reuse_existing(recipe, previous, output)
                    except (OSError, ValueError, TypeError, KeyError):
                        pass  # stale/corrupt optimization input uses the approved generator
                if result is None:
                    result = bake(recipe, output)
                completed[key] = result
            entries[character_id] = {k: result[k] for k in ('path', 'sha256')}
            results.append({'character': character_id, 'ok': True, **result})
            print('PREPARED_PRESET', character_id, round(result['generation_seconds'], 3), flush=True)
        except Exception as error:
            results.append({'character': character_id, 'ok': False, 'error': str(error)})
            print('PREPARED_PRESET_FAILED', character_id, type(error).__name__, flush=True)
    write(output / 'results.json', results)
    if not all(r['ok'] for r in results):
        raise RuntimeError('not all presets prepared; no registry published')
    registry['prepared_neutrals'] = {'schema': 'prepared-neutral-index.v1', 'entries': entries}
    write(output / 'shape-bakes.local.json', registry)
    print('PREPARED_PRESETS_COMPLETE', len(entries), flush=True)


def stage_prepared(bundle, registry, package, deploy_root):
    """Add verified prototypes to an immutable body release; never uploads."""
    bundle, package = Path(bundle).resolve(), Path(package)
    prepared = json.loads((bundle / 'shape-bakes.local.json').read_text())['prepared_neutrals']
    if prepared.get('schema') != 'prepared-neutral-index.v1':
        raise ValueError('invalid prepared index')
    entries = prepared['entries']
    if not entries or set(entries) - set(registry['characters']):
        raise ValueError('prepared preset is not in this release')
    copied = {}
    for cid, entry in entries.items():
        recorded = Path(entry['path'])
        if recorded.parent.name != 'qa' or not re.fullmatch(r'prepared-[a-f0-9]{64}\.json', recorded.name):
            raise ValueError('prepared receipt escaped bundle')
        # Bundles built inside Linux containers can be packaged on another host.
        # Anchor to this bundle, never read an absolute path from the build host.
        metadata = bundle / 'qa' / recorded.name
        local_entry = {**entry, 'path': str(metadata)}
        raw = registry['characters'][cid]
        recipe = {**raw, **registry['sources'][raw['rules']], 'values': preset_values(raw)}
        data = read_prepared(recipe, local_entry, check_runtime=False)
        name = 'prepared-' + data['key']
        for folder, extension, content in [('fbx', '.fbx', data['data']), ('qa', '.json', metadata.read_bytes())]:
            target = package / folder / (name + extension)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        copied[cid] = {'path': deploy_root.rstrip('/') + '/qa/' + name + '.json', 'sha256': entry['sha256']}
    registry['prepared_neutrals'] = {'schema': prepared['schema'], 'entries': copied}
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry')
    ap.add_argument('--out')
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument('--character', action='append', default=[])
    selection.add_argument('--rules', action='append', default=[], help='prepare only these approved body versions')
    selection.add_argument('--catalog', help='approved body-models.json; select exposed default/preset entries')
    ap.add_argument('--catalog-sha256', help='required approval pin with --catalog')
    ap.add_argument('--blender', default=os.getenv('BLENDER_BINARY', 'blender'))
    ap.add_argument('--worker')
    args = ap.parse_args(argv)
    if args.worker:
        run_worker(json.loads(Path(args.worker).read_text()))
        return
    if not args.registry or not args.out:
        ap.error('--registry and --out are required')
    registry = load_registry(Path(args.registry).resolve())
    if not args.character and not args.rules and not args.catalog:
        ap.error('explicit --character or --rules selection is required; historical bodies are not prepared automatically')
    characters = args.character or [cid for cid, recipe in registry['characters'].items() if recipe['rules'] in args.rules]
    if args.catalog:
        from scripts.prepared_preset_scope import pinned_catalog, select
        characters = select(pinned_catalog(args.catalog, args.catalog_sha256), registry,
                            {'male': 9, 'female': 11})
    if not characters or len(characters) != len(set(characters)) or set(characters) - set(registry['characters']):
        ap.error('choose distinct registered characters')
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    job = output / '.preparation-job.json'
    try:
        write(job, {'out': str(output), 'registry': registry, 'characters': characters})
        subprocess.run([args.blender, '--background', '--factory-startup', '--python-exit-code', '1',
                        '--python', str(Path(__file__).resolve()), '--', '--worker', str(job)], check=True)
    finally:
        job.unlink(missing_ok=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
