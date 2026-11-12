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
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_preset_bodies import stage_prepared  # noqa: E402
from scripts.staging_body_assets_init import ALLOWED_PATH, LIMIT  # noqa: E402

DEPLOY = '/app/body-staging/'
RELEASE_CAP = 768 * 1024 * 1024  # staging_body_assets_init total limit


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def build(release, prepared, version, out):
    release, prepared, out = Path(release), Path(prepared), Path(out)
    manifest = json.loads((release/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'body-assets.v1':
        raise SystemExit('not a body-assets.v1 manifest')
    files = {f['path']: f for f in manifest['files']}
    registry_raw = (release/'shape-bakes.json').read_bytes()
    if sha(registry_raw) != files['shape-bakes.json']['sha256']:
        raise SystemExit('local shape-bakes.json is not the pinned release copy')
    registry = json.loads(registry_raw)
    if 'prepared_neutrals' in registry:
        raise SystemExit('release already has prepared neutrals; derive from its source release instead')
    out.mkdir(parents=True, exist_ok=False)
    # Verifies every prepared receipt/FBX against this exact registry recipe.
    entries = stage_prepared(prepared, registry, out, DEPLOY)
    new_registry = dump(registry)
    (out/'shape-bakes.json').write_bytes(new_registry)
    added, changed = [], ['shape-bakes.json']
    files['shape-bakes.json'] = {**files['shape-bakes.json'], 'size': len(new_registry), 'sha256': sha(new_registry)}
    for folder in ('fbx', 'qa'):
        for path in sorted((out/folder).glob('prepared-*')):
            name = f'{folder}/{path.name}'
            if name in files:
                raise SystemExit(f'{name} already exists in the release')
            data = path.read_bytes()
            files[name] = {'path': name, 'roles': ['converter'], 'size': len(data), 'sha256': sha(data)}
            added.append(name)
    for name, entry in files.items():
        if not ALLOWED_PATH.fullmatch(name) or not 0 < entry['size'] <= LIMIT:
            raise SystemExit(f'body initializer would refuse {name}')
    total = sum(f['size'] for f in files.values())
    if total > RELEASE_CAP:
        raise SystemExit(f'release would be {total} bytes, above the initializer cap {RELEASE_CAP}')
    new_manifest = {'schema': 'body-assets.v1', 'version': version,
                    'files': [files[name] for name in sorted(files)]}
    raw = dump(new_manifest)
    (out/'manifest.json').write_bytes(raw)
    digest = sha(raw)
    plan = {'manifestSha256': digest, 'totalBytes': total, 'headroomBytes': RELEASE_CAP - total,
            'copyUnchanged': sorted(n for n in files if n not in added and n not in changed),
            'putNew': sorted(added + changed), 'putManifestLast': 'manifest.json',
            'preparedEntries': entries}
    (out/'upload-plan.json').write_bytes(dump(plan))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True)
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*', args.version):
        raise SystemExit('invalid version')
    plan = build(args.release, args.prepared, args.version, args.out)
    print(json.dumps({k: v for k, v in plan.items() if k != 'copyUnchanged'}, indent=2))
    print('unchanged objects to copy:', len(plan['copyUnchanged']))


if __name__ == '__main__':
    main()
