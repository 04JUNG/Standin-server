"""Derive a new immutable Converter body release that adds prepared neutral bodies.

    python scripts/add_prepared_bodies_to_release.py \
        --release <dir with the pinned release's manifest.json and shape-bakes.json> \
        --prepared <prepare_preset_bodies.py output built in the pinned Linux Converter image> \
        --version <new manifest version> --out <new empty dir>

Every existing file keeps its exact path, size, SHA-256 and roles. Only
shape-bakes.json changes (adds ``prepared_neutrals``) and fbx/qa prepared files
are added. Nothing is uploaded: the output holds the new/changed bytes, the new
manifest and ``upload-plan.json`` (server-side copies for unchanged objects,
conditional puts for new ones). The body initializer's path, size and role
rules are checked here so a release it would refuse is never produced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_preset_bodies import stage_prepared  # noqa: E402
from scripts.staging_body_assets_init import check_budget, validate_files, PREPARED_PROFILE, RELEASE_CAP  # noqa: E402
from scripts.prepared_preset_scope import select, pinned_catalog  # noqa: E402
from converter.prepared_body import MAX_RECEIPT_BYTES  # noqa: E402

DEPLOY = '/app/body-staging/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def initializer_budget(files, profile, *, legacy_errors=False):
    try:
        validate_files(files)
        return check_budget(files, profile)
    except ValueError as error:
        # Preserve the original small-release helper's CLI error contract.
        if legacy_errors:
            raise SystemExit(f'body initializer would refuse release: {error}') from None
        raise


def build(release, prepared, version, out, *, catalog=None, budget_profile=None, release_sha256=None):
    release, prepared, out = Path(release), Path(prepared), Path(out)
    manifest_raw = (release/'manifest.json').read_bytes()
    if release_sha256 is not None and sha(manifest_raw) != release_sha256:
        raise ValueError('source release manifest checksum mismatch')
    manifest = json.loads(manifest_raw)
    if manifest.get('schema') != 'body-assets.v1':
        raise SystemExit('not a body-assets.v1 manifest')
    initializer_budget(manifest['files'], manifest.get('budgetProfile'), legacy_errors=budget_profile is None)
    files = {f['path']: f for f in manifest['files']}
    registry_raw = (release/'shape-bakes.json').read_bytes()
    if sha(registry_raw) != files['shape-bakes.json']['sha256']:
        raise SystemExit('local shape-bakes.json is not the pinned release copy')
    registry = json.loads(registry_raw)
    if budget_profile is None and 'prepared_neutrals' in registry:
        raise SystemExit('release already has prepared neutrals; use the explicit extension profile')
    original_registry = copy.deepcopy(registry)
    prepared_index = json.loads((prepared/'shape-bakes.local.json').read_text())['prepared_neutrals']['entries']
    if budget_profile == PREPARED_PROFILE:
        if catalog is None or set(select(catalog, registry, {'male': 9, 'female': 11})) != set(prepared_index):
            raise ValueError('prepared bundle must cover the exact approved 20 presets')
    # Production packaging must never label a macOS prototype as Linux usable.
    for entry in prepared_index.values():
        with (prepared/'qa'/Path(entry['path']).name).open('rb') as stream:
            raw = stream.read(MAX_RECEIPT_BYTES + 1)
        if len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError('prepared receipt exceeded budget')
        receipt = json.loads(raw)
        if receipt['identity']['runtime'][2:4] != ['linux', 'x86_64']:
            raise ValueError('release requires Linux/amd64 prepared bodies')
    out.mkdir(parents=True, exist_ok=False)
    # Verifies every prepared receipt/FBX against this exact registry recipe.
    entries = stage_prepared(prepared, registry, out, DEPLOY)
    previous = original_registry.get('prepared_neutrals', {}).get('entries', {})
    registry['prepared_neutrals']['entries'] = {**previous, **entries}
    unchanged_recipe = copy.deepcopy(registry)
    unchanged_recipe.pop('prepared_neutrals', None)
    original_registry.pop('prepared_neutrals', None)
    if unchanged_recipe != original_registry:
        raise ValueError('prepared packaging changed source recipes')
    new_registry = dump(registry)
    (out/'shape-bakes.json').write_bytes(new_registry)
    added, changed = [], ['shape-bakes.json']
    files['shape-bakes.json'] = {**files['shape-bakes.json'], 'size': len(new_registry), 'sha256': sha(new_registry)}
    for folder in ('fbx', 'qa'):
        for path in sorted((out/folder).glob('prepared-*')):
            name = f'{folder}/{path.name}'
            if name in files:
                if files[name] != {'path': name, 'roles': ['converter'], 'size': path.stat().st_size, 'sha256': sha(path.read_bytes())}:
                    raise ValueError('existing prepared path identity changed')
                path.unlink()  # unchanged object stays in the server-side copy plan
                continue
            data = path.read_bytes()
            files[name] = {'path': name, 'roles': ['converter'], 'size': len(data), 'sha256': sha(data)}
            added.append(name)
    total = sum(f['size'] for f in files.values())
    cap = initializer_budget(list(files.values()), budget_profile, legacy_errors=budget_profile is None)
    new_manifest = {'schema': 'body-assets.v1', 'version': version,
                    'files': [files[name] for name in sorted(files)]}
    if budget_profile:
        new_manifest['budgetProfile'] = budget_profile
    raw = dump(new_manifest)
    if len(raw) > 65536:
        raise ValueError('body manifest exceeds initializer limit')
    (out/'manifest.json').write_bytes(raw)
    digest = sha(raw)
    plan = {'manifestSha256': digest, 'totalBytes': total, 'headroomBytes': cap - total,
            'budgetProfile': budget_profile, 'capBytes': cap,
            'copyUnchanged': sorted(n for n in files if n not in added and n not in changed),
            'putNew': sorted(added + changed), 'putManifestLast': 'manifest.json',
            'preparedEntries': entries}
    (out/'upload-plan.json').write_bytes(dump(plan))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True)
    ap.add_argument('--release-sha256', help='required approval pin with --budget-profile')
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--catalog')
    ap.add_argument('--catalog-sha256')
    ap.add_argument('--budget-profile', choices=[PREPARED_PROFILE])
    args = ap.parse_args()
    if args.budget_profile and not args.release_sha256:
        ap.error('--release-sha256 is required with --budget-profile')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*', args.version):
        raise SystemExit('invalid version')
    catalog = pinned_catalog(args.catalog, args.catalog_sha256) if args.catalog else None
    plan = build(args.release, args.prepared, args.version, args.out,
                 catalog=catalog, budget_profile=args.budget_profile, release_sha256=args.release_sha256)
    print(json.dumps({k: v for k, v in plan.items() if k != 'copyUnchanged'}, indent=2))
    print('unchanged objects to copy:', len(plan['copyUnchanged']))


if __name__ == '__main__':
    main()
