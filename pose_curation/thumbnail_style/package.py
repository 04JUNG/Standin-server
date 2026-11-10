"""Verify every generated JPEG and package only the immutable thumbnail overlay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
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
    def checked_paths(source):
        record = _record(root, manifest, source)
        selected = [root / source / 'pose.json']
        for name, info in sorted(record['images'].items()):
            path = root / source / name
            _raster(path, info)
            selected.append(path)
        return selected

    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, selected in enumerate(pool.map(checked_paths, sorted(manifest['pose_records'])), 1):
            paths.extend(selected)
            if number % 500 == 0:
                print(f'verified {number}/{manifest["pose_count"]} poses', flush=True)
    if len(paths) != len(manifest['pose_records']) * 33 + 1:
        raise ValueError('incomplete image package')
    # These are generated regular files, not a filesystem backup. Avoid expensive
    # per-file platform ownership lookups and prefetch a bounded batch of bytes.
    with ThreadPoolExecutor(max_workers=8) as pool, tarfile.open(args.archive, 'w:gz', compresslevel=1) as archive:
        for start in range(0, len(paths), 256):
            batch = paths[start:start + 256]
            for path, data in zip(batch, pool.map(Path.read_bytes, batch)):
                info = tarfile.TarInfo(path.relative_to(args.directory).as_posix())
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
            if start % 16384 == 0:
                print(f'packaged {min(start + 256, len(paths))}/{len(paths)} files', flush=True)
    with args.archive.open('rb') as data:
        digest = hashlib.file_digest(data, 'sha256').hexdigest()
    receipt = dict(asset_version=manifest['asset_version'], archive=str(args.archive.resolve()),
                   sha256=digest, bytes=args.archive.stat().st_size, files=len(paths),
                   poses=manifest['pose_count'], sources=len(manifest['pose_records']), images=manifest['image_count'])
    args.archive.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
