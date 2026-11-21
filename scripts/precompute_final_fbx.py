"""Plan/resume local final FBXs from an active library and approved body catalog.

No upload/deploy capability. Run --plan first, then --pose/--limit for a pilot.
Runtime and batch must use the same converter image, registries and policy env.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter_api.final_cache import FinalStore, canonical, digest, identity, policy_sha256
from converter_api.registry import CharacterRegistry
from converter_api.runner import BlenderRunner, RunnerSettings
from converter.shape_spec import resolve_shape, digest_shape
from converter_api.final_inputs import load_inputs, install_inputs, approved_default


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--pose', action='append')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--scope', default='full')
    ap.add_argument('--camera', help='Exact baked camera rotation JSON, never inferred')
    ap.add_argument('--explicit-defaults', action='store_true', help='Generate explicit default controls separately from shape=null')
    ap.add_argument('--seconds-per-asset', type=float, default=61.9)
    ap.add_argument('--bytes-per-asset', type=int, default=10 * 1024 * 1024)
    args = ap.parse_args()
    library = args.library.resolve(strict=True)
    defaults, sources, registered = load_inputs(library, args.catalog)
    store = FinalStore(args.out)
    pin = install_inputs(store, registered)
    os.environ['CONVERTER_FINAL_CACHE_ROOT'] = str(store.root)
    os.environ['CONVERTER_FINAL_LIBRARY_SHA256'] = pin
    base_plan = dict(version='final-fbx-batch-v1', policy_sha256=policy_sha256(), library_db_sha256=digest((library / 'poses.db').read_bytes()),
        catalog_sha256=json.loads(registered)['catalog_sha256'], library_manifest_sha256=digest(registered), sources=sources,
        defaults=[{k: a[k] for k in ('characterId', 'assetSha256', 'shape', 'shapeSource')} for a in defaults],
        options=dict(scope=args.scope, camera_rotation=json.loads(args.camera) if args.camera else None,
                     explicit_defaults=args.explicit_defaults),
        total=len(sources) * len(defaults),
        estimate=dict(seconds=len(sources) * 2 * args.seconds_per_asset,
                      bytes=len(sources) * 2 * args.bytes_per_asset,
                      basis='planning estimate; replace with measured pilot, serial one Blender worker'))
    store.atomic('plan.json', canonical(base_plan))
    plan_id = digest(canonical(base_plan))
    store.atomic('plan-' + plan_id + '.json', canonical(base_plan))
    print(json.dumps({k: v for k, v in base_plan.items() if k not in ('sources', 'defaults')}, indent=2), flush=True)
    if args.plan:
        return 0
    os.environ['CONVERTER_FINAL_CACHE_ROOT'] = str(store.root)
    registry = CharacterRegistry.from_env()
    settings = replace(RunnerSettings.from_env(), timeout_seconds=90)
    runner = BlenderRunner(settings)
    selected = [s for s in sources if not args.pose or s['pose_id'] in args.pose]
    if args.pose and set(args.pose) - {s['pose_id'] for s in selected}:
        raise ValueError('requested pose not registered')
    if args.limit is not None:
        selected = selected[:args.limit]
    records = []
    # Append-only history survives a new invocation; the per-invocation progress
    # never claims that unvisited combinations were checked in this run.
    for source in selected:
        for body in defaults:
            started = time.monotonic()
            cid = body['characterId']
            row = dict(pose_id=source['pose_id'], character_id=cid, status='failed')
            try:
                character, default_recipe = approved_default(body, registry)
                raw = None
                if args.explicit_defaults:
                    controls = {c['key']: c['default'] for c in body['shape']['controls']}
                    raw = dict(schemaVersion='body-shape.v1', controls=controls,
                               shapeSha256=digest_shape(cid, body['assetSha256'], controls))
                recipe = default_recipe if raw is None else resolve_shape(cid, body['assetSha256'], raw, body['shapeSource'])
                if recipe is None:
                    raise ValueError('approved default source missing')
                options = dict(bvh_bytes=Path(source['path']).read_bytes(), character_path=character.path,
                    character_id=cid, character_sha256=body['assetSha256'], conversion_id=str(uuid.uuid4()),
                    shape_recipe=recipe, output_scope=args.scope,
                    camera_rotation=base_plan['options']['camera_rotation'])
                from converter.hand_contact import selected_hand_screen_enabled
                options['selected_hand_screen'] = selected_hand_screen_enabled(args.scope)
                binding = identity(settings, options)
                row['key'] = digest(canonical(binding))
                result = runner.convert(**options)
                store.get(binding)  # Never count unregistered fallback as a prepared asset.
                row.update(status='ready', artifact_sha256=result.artifact_sha256, bytes=len(result.artifact),
                           cache=result.report.get('final_cache'),
                           source_fingers=result.report.get('source_fingers'),
                           artifact_validation=result.report.get('artifact_validation'))
            except Exception as exc:
                row['error'] = type(exc).__name__ + ':' + str(exc)[:300]
            row['seconds'] = round(time.monotonic() - started, 3)
            records.append(row)
            with (store.root / 'history.jsonl').open('ab') as history:
                history.write(canonical(dict(plan_id=plan_id, **row)) + b'\n')
            store.atomic('progress.json', canonical(dict(total=base_plan['total'],
                attempted=len(records), ready=sum(r['status'] == 'ready' for r in records),
                failed=sum(r['status'] == 'failed' for r in records),
                not_attempted_this_run=base_plan['total'] - len(records), results=records)))
            print(json.dumps(row), flush=True)
    return int(any(r['status'] == 'failed' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
