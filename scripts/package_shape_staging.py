"""Package verified shape-bake sources for a new, immutable S3 staging release.

    python scripts/package_shape_staging.py --release <build_shape_release out> \
        --runtime standin-body-0.7.1=<male .ssr> --runtime standin-female-body-0.3.2=<female .ssr> \
        --bake-qa <verify_shape_bake out> --version shape-20261016-v1 [--upload]

Upload is opt-in. This command never updates ECS, production, or existing keys.

Checks before anything is written:
* every bake run in --bake-qa passed (and complete pose-matched inspect_shape_bake.py evidence required);
* each runtime is SSR1/SSR2; an SSR2 header must name the same rules and source SHA-256
  as the registry (the app refuses a mismatch anyway — this catches it before release);
* every asset's FBX hash equals its registry asset_sha256.
"""
import argparse, hashlib, json, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.shape_spec import load_registry  # noqa: E402
from scripts.shape_inspection_metrics import require_inspection

BUCKET = 'standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
DEPLOY = '/app/body-staging/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def runtime_header(path):
    raw = Path(path).read_bytes()
    magic = raw[:4]
    if magic == b'SSR2':
        size = struct.unpack('<I', raw[4:8])[0]
        return 'standin-shape-runtime.v2', json.loads(raw[8:8 + size])
    if magic == b'SSR1':
        size = struct.unpack('<I', raw[4:8])[0]
        return 'standin-shape-runtime.v1', json.loads(raw[8:8 + size])
    raise SystemExit(f'{path}: not a shape runtime')


def publish(package, files, digest, prefix):
    for f in files:
        subprocess.run(['aws', 's3api', 'put-object', '--bucket', BUCKET, '--key', prefix + '/' + f['path'], '--body', str(package / f['path']), '--if-none-match', '*'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['aws', 's3api', 'put-object', '--bucket', BUCKET, '--key', prefix + '/manifest.json', '--body', str(package / 'manifest.json'), '--if-none-match', '*', '--content-type', 'application/json'], check=True, stdout=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory(prefix='shape-s3-verify-') as tmp:
        for f in files + [{'path': 'manifest.json', 'size': (package / 'manifest.json').stat().st_size, 'sha256': digest}]:
            p = Path(tmp) / 'object'
            subprocess.run(['aws', 's3api', 'get-object', '--bucket', BUCKET, '--key', prefix + '/' + f['path'], str(p)], check=True, stdout=subprocess.DEVNULL)
            assert p.stat().st_size == f['size'] and sha(p.read_bytes()) == f['sha256'], f['path']
    return {'bucket': BUCKET, 'prefix': prefix, 'objects_verified': len(files) + 1, 'all_sizes_and_sha256_match': True, 'manifest_sha256': digest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--runtime', action='append', default=[], help='<rules>=<path to .ssr>')
    ap.add_argument('--bake-qa', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--upload', action='store_true')
    args = ap.parse_args()
    release = Path(args.release)
    package = release / 'package'
    if package.exists():
        raise SystemExit('use a fresh package directory; never overwrite a published release')
    registry = load_registry(release / 'shape-bakes.local.json')
    bodies = json.loads((release / 'body-models.local.json').read_text())
    characters = json.loads((release / 'characters.local.json').read_text())
    runs = json.loads((Path(args.bake_qa) / 'runs.json').read_text())
    if not runs or not all(r.get('ok') for r in runs) or not all(r.get('lineage_controls_match') for r in runs if r['label'] == 'slider'):
        raise SystemExit('every bake run must pass before packaging')
    inspection = Path(args.bake_qa) / 'inspection.json'
    if not inspection.is_file():
        raise SystemExit('pose-matched independent inspection is required')
    require_inspection(runs, json.loads(inspection.read_text()))
    runtimes = dict(item.split('=', 1) for item in args.runtime)
    package.mkdir()

    def copy(src, rel):
        dest = package / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest
    runtime_meta = {}
    for rules, src in registry['sources'].items():
        for key, ext in (('blend', '.blend'), ('controller', '.py')):
            rel = f'sources/{rules}{ext}'
            assert sha(copy(src[key], rel).read_bytes()) == src[key + '_sha256']
            src[key] = DEPLOY + rel
        if rules in runtimes:
            fmt, header = runtime_header(runtimes[rules])
            if fmt.endswith('v2') and (header['rules'] != rules or header['source'] != {'blendSha256': src['blend_sha256'], 'controllerSha256': src['controller_sha256']}):
                raise SystemExit(f'{runtimes[rules]}: runtime was built from another source than {rules}')
            rel = f'runtime/{rules}.ssr'
            data = copy(runtimes[rules], rel).read_bytes()
            runtime_meta[rules] = {'path': DEPLOY + rel, 'sha256': sha(data), 'bytes': len(data), 'format': fmt, 'rules': rules, 'presets': set(header['presets'])}
    env = {}
    for body in bodies['assets']:
        cid = body['characterId']
        recipe = registry['characters'][cid]
        assert sha(copy(release / 'fbx' / (cid + '.fbx'), f'fbx/{cid}.fbx').read_bytes()) == recipe['asset_sha256'] == body['assetSha256']
        png = release / 'png' / (cid + '.png')
        if png.exists() and 'neutralPreview' in body:
            copy(png, f'png/{cid}.png')
            body['neutralPreview']['path'] = DEPLOY + f'png/{cid}.png'
        meta = runtime_meta.get(recipe['rules'])
        if meta and recipe['preset'] in meta['presets']:
            body['shapeRuntime'] = {k: meta[k] for k in ('path', 'sha256', 'bytes', 'format', 'rules')} | {'preset': recipe['preset']}
        # supportedPoseIds stay as built/curated: BVH file names are not library pose ids.
        env[characters['characters'][cid]['artifact_uri_env']] = DEPLOY + f'fbx/{cid}.fbx'

    def write(name, value):
        p = package / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    write('shape-bakes.json', registry)
    write('body-models.json', bodies)
    write('characters.json', characters)
    write('qa/bake-runs.json', runs)
    if inspection.exists():
        write('qa/bake-inspection.json', json.loads(inspection.read_text()))
    files = []
    for p in sorted(package.rglob('*')):
        if not p.is_file():
            continue
        n = p.relative_to(package).as_posix()
        roles = (['converter'] if n.startswith('sources/') or n in ('characters.json', 'shape-bakes.json')
                 else ['bff'] if n.startswith(('png/', 'runtime/')) or n == 'body-models.json'
                 else ['inference', 'converter'] if n.startswith('fbx/') else ['inference'])
        files.append({'path': n, 'roles': roles, 'size': p.stat().st_size, 'sha256': sha(p.read_bytes())})
    write('manifest.json', {'schema': 'body-assets.v1', 'version': args.version, 'files': files})
    digest = sha((package / 'manifest.json').read_bytes())
    prefix = f'body-models/releases/{args.version}-{digest[:12]}'
    summary = {'version': args.version, 'bucket': BUCKET, 'manifestKey': prefix + '/manifest.json', 'manifestSha256': digest,
               'bodyEnvironment': env, 'sources': sorted(registry['sources']), 'fileCount': len(files),
               'bytes': sum(f['size'] for f in files), 'activation': 'not activated',
               'flagsAfterDeployment': {'BODY_SHAPE_EDIT_ENABLED': 'true', 'BODY_SHAPE_BAKE_ENABLED': 'true'}}
    if args.upload:
        summary['upload'] = publish(package, files, digest, prefix)
    (release / 'release.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
