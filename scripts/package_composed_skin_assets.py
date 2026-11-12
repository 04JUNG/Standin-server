"""Extend the reviewed current body assets with flesh + authored muscle ink.
Run compose_flesh_muscle.py first. Existing UVs, models and texture entries stay exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--artwork', type=Path, required=True)
    args = parser.parse_args()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    path = ROOT / 'converter/skin_assets.v1.json'
    manifest = json.loads(path.read_bytes())
    current = json.loads((ROOT / 'config/staging-anatomy-release.json').read_bytes())['characterIds']
    manifest['revision'] = 'staging-composed-skins-20261112-v3'
    manifest['authoring_sha256']['compose_flesh_muscle.py'] = sha((ROOT / 'scripts/skin_fbx/compose_flesh_muscle.py').read_bytes())
    for cid in current:
        gender = 'female' if cid.startswith('female') else 'male'
        for kind, suffix in [('textures', ''), ('preview_textures', '-preview')]:
            source = args.artwork / f'{gender}-flesh-muscle{suffix}.png'
            name = f'anatomy-{gender}-flesh-muscle{suffix}.png'
            data = source.read_bytes()
            shutil.copyfile(source, args.assets / name)
            manifest['models'][cid][kind]['flesh-muscle'] = dict(path=name, bytes=len(data), sha256=sha(data))
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    path.write_bytes(raw)
    (args.assets / 'manifest.json').write_bytes(raw)
    print('COMPOSED_SKIN_MANIFEST', sha(raw))


if __name__ == '__main__':
    main()
