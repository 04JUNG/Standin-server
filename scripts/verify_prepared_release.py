"""Install a local release with the real initializer and audit all 20 prototypes.

Run in the pinned Linux/amd64 image, with a fresh writable /app/body-staging
volume. No network, upload, or deployment. --base supplies unchanged objects
from add_prepared_bodies_to_release.py's copy plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.prepared_body import preset_values, restore
from converter.shape_spec import load_registry
from scripts.prepared_preset_scope import pinned_catalog, select
from scripts.staging_body_assets_init import install


def verify(base, release, catalog, root, result):
    base, release, root = map(Path, (base, release, root))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('verification requires a fresh install root')
    raw = (release/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    prefix = 'body-models/releases/local-prepared-verification'
    def fetch(key, limit):
        if key == prefix+'/manifest.json':
            return raw
        name = key.removeprefix(prefix+'/')
        path = release/name
        if not path.is_file():
            path = base/name
        with path.open('rb') as stream:
            return stream.read(limit+1)
    ready = install(fetch, dict(BODY_ASSETS_ROLE='converter', BODY_ASSETS_MANIFEST_KEY=prefix+'/manifest.json',
        BODY_ASSETS_MANIFEST_SHA256=hashlib.sha256(raw).hexdigest(), BODY_ASSETS_ROOT=str(root)))
    registry = load_registry(root/'shape-bakes.json')
    ids = select(catalog, registry, {'male': 9, 'female': 11})
    rows = []
    with tempfile.TemporaryDirectory(prefix='prepared-release-verify-') as scratch:
        for cid in ids:
            recipe = registry['characters'][cid]
            recipe = {**recipe, **registry['sources'][recipe['rules']], 'values': preset_values(recipe),
                      'prepared_neutral': registry['prepared_neutrals']['entries'][cid]}
            entry = recipe['prepared_neutral']
            receipt_raw = Path(entry['path']).read_bytes()
            receipt = json.loads(receipt_raw)
            prototype, status = restore(recipe, Path(scratch)/cid)
            if status != 'prepared' or receipt['identity']['runtime'][2:4] != ['linux', 'x86_64']:
                raise ValueError('installed prototype was not adopted on Linux/amd64')
            rows.append(dict(character=cid, status=status, key=prototype['key'], asset_sha256=recipe['asset_sha256'],
                blend_sha256=recipe['blend_sha256'], controller_sha256=recipe['controller_sha256'],
                runtime=receipt['identity']['runtime'], fbx_bytes=receipt['fbx_bytes'],
                fbx_sha256=receipt['fbx_sha256'], receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                neutral_validation=receipt['neutral_validation']))
    evidence = dict(ready=ready, presets=rows, totalBytes=sum(f['size'] for f in manifest['files']),
        installedBytes=sum(f['size'] for f in manifest['files'] if 'converter' in f['roles']),
        manifestSha256=hashlib.sha256(raw).hexdigest(), budgetProfile=manifest.get('budgetProfile'))
    Path(result).write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'release', 'catalog', 'catalog-sha256', 'result'):
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--root', default='/app/body-staging')
    args = ap.parse_args()
    report = verify(args.base, args.release, pinned_catalog(args.catalog, args.catalog_sha256), args.root, args.result)
    print(json.dumps(dict(presets=len(report['presets']), totalBytes=report['totalBytes'])))
