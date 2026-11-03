"""Verify every generated JPEG and package only the immutable thumbnail overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from src.thumbnail_overlay import _record, _raster, _root, manifest_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    manifest = manifest_for(args.directory)
    if manifest is None:
        raise ValueError('completed manifest required')
    root = _root(args.directory)
    paths = [root / 'manifest.json']
    for number, source in enumerate(sorted(manifest['pose_records']), 1):
        record = _record(root, manifest, source)
        paths.append(root / source / 'pose.json')
        for name, info in sorted(record['images'].items()):
            path = root / source / name
            _raster(path, info)
            paths.append(path)
        if number % 500 == 0:
            print(f'verified {number}/{manifest["pose_count"]} poses', flush=True)
    if len(paths) != 193447:
        raise ValueError('incomplete image package')
    with tarfile.open(args.archive, 'w:gz', compresslevel=1) as archive:
        for path in paths:
            info = archive.gettarinfo(str(path), str(path.relative_to(args.directory)).replace('\\', '/'))
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mode = 0o644
            with path.open('rb') as data:
                archive.addfile(info, data)
    with args.archive.open('rb') as data:
        digest = hashlib.file_digest(data, 'sha256').hexdigest()
    receipt = dict(asset_version=manifest['asset_version'], archive=str(args.archive.resolve()),
                   sha256=digest, bytes=args.archive.stat().st_size, files=len(paths),
                   poses=manifest['pose_count'], images=manifest['image_count'])
    args.archive.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
