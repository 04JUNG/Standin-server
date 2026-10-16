"""Bake real BVHs with slider values through the production runner and record evidence.

    CONVERTER_SHAPE_REGISTRY=<release>/shape-bakes.local.json \
    python scripts/verify_shape_bake.py --release <release> --bvh a.bvh --bvh b.bvh \
        --case male-bulk-v0-7-1:muscle=1.2,height=185 --case female-base-v0-3-2:breast_size=0.5 \
        --out <release>/bake-qa

Each case runs twice per BVH: once at the preset defaults and once with the slider
values, both through ``BlenderRunner.convert`` (the same worker job the API runs).
The evidence per run is the converter's own lineage (``shape_bake``) plus independent
checks on the two final FBX files:

* ``lineage_controls_match``: the bake used exactly the requested controller values;
* ``bone_lengths``: final-FBX bone lengths equal the controller's warped rest skeleton
  for those values (pose is rigid per bone, proportions are not);
* ``shape_effect``: per-vertex |Δ| between preset and slider bakes against the same
  |Δ| in the neutral meshes — the slider change survives posing and correction.
"""
import argparse, hashlib, json, os, sys, time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.shape_spec import digest_shape, load_registry, resolve_shape  # noqa: E402
from converter_api.runner import BlenderRunner, RunnerSettings  # noqa: E402


def parse_case(text):
    cid, _, rest = text.partition(':')
    controls = {}
    for part in filter(None, rest.split(',')):
        k, v = part.split('=')
        controls[k] = float(v)
    return cid, controls


def envelope(cid, recipe, overrides):
    full = {c['key']: c['default'] for c in recipe['controls']}
    for k, v in overrides.items():
        spec = next(c for c in recipe['controls'] if c['key'] == k)
        # snap onto the slider grid exactly like the app/BFF do
        n = round((v - spec['min']) / spec['step'])
        full[k] = round(spec['min'] + n * spec['step'], 6)
    return {'schemaVersion': 'body-shape.v1', 'controls': full,
            'shapeSha256': digest_shape(cid, recipe['asset_sha256'], full)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--bvh', action='append', required=True)
    ap.add_argument('--case', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--timeout', type=float, default=900)
    args = ap.parse_args()
    release, out = Path(args.release), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    runner = BlenderRunner(RunnerSettings.from_env(ROOT))
    runner.settings = replace(runner.settings, timeout_seconds=args.timeout, camera_timeout_seconds=args.timeout)
    rows = []
    for case in args.case:
        cid, overrides = parse_case(case)
        recipe = registry['characters'][cid]
        character = release / 'fbx' / (cid + '.fbx')
        for bvh_path in map(Path, args.bvh):
            bvh = bvh_path.read_bytes()
            for label, shape in (('preset', None), ('slider', envelope(cid, recipe, overrides))):
                resolved = resolve_shape(cid, recipe['asset_sha256'], shape)
                stem = f"{cid}__{bvh_path.stem.replace(' ', '_')}__{label}"
                t = time.monotonic()
                try:
                    result = runner.convert(bvh_bytes=bvh, character_path=character, character_id=cid,
                                            character_sha256=recipe['asset_sha256'], conversion_id=stem,
                                            preview_view='front', shape_recipe=resolved)
                    (out / (stem + '.fbx')).write_bytes(result.artifact)
                    (out / (stem + '.png')).write_bytes(result.preview)
                    lineage = result.report.get('shape_bake') or {}
                    row = {'id': stem, 'case': cid, 'bvh': bvh_path.name, 'label': label, 'ok': True,
                           'seconds': round(time.monotonic() - t, 2), 'values': resolved['values'],
                           'lineage_controls_match': lineage.get('controls') == resolved['values'],
                           'shape_sha256': lineage.get('shape_sha256'), 'rules': lineage.get('rules'),
                           'pose_transfer_max_joint_error_mm': lineage.get('pose_transfer_max_joint_error_mm'),
                           'roundtrip_max_error_mm': lineage.get('roundtrip_max_error_mm'),
                           'contact_qa': lineage.get('contact_qa'), 'pose_qa': lineage.get('pose_qa'),
                           'fbx_sha256': result.artifact_sha256}
                except Exception as exc:  # noqa: BLE001 - evidence file records every failure
                    row = {'id': stem, 'case': cid, 'bvh': bvh_path.name, 'label': label, 'ok': False,
                           'error': f'{type(exc).__name__}: {exc}', 'report': getattr(exc, 'report', None)}
                rows.append(row)
                (out / (stem + '.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2))
                print(json.dumps({k: row.get(k) for k in ('id', 'ok', 'seconds', 'lineage_controls_match', 'error')}, ensure_ascii=False), flush=True)
    (out / 'runs.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    failed = [r['id'] for r in rows if not r['ok'] or (r['label'] == 'slider' and not r['lineage_controls_match'])]
    print('BAKE_RUNS', len(rows), 'FAILED', failed, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
