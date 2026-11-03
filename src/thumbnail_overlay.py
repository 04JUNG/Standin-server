"""Pinned, appearance-only thumbnail release layered over an unchanged library.

The operator supplies both an immutable archive URI and SHA256. Startup checks
the complete inventory and every asset before publishing it. BVH, database,
search metadata and review decisions cannot be present in this archive.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

from PIL import Image

STYLE = 'outline-half-strong-v1'
SCOPES = ('full', 'half', 'bust', 'shoulder_head')
VIEWS = tuple(f'yaw_{a:03d}' for a in range(0, 360, 45))
LEGACY_VIEWS = dict(front='yaw_000', three_quarter='yaw_045', side='yaw_090', back='yaw_180')
HASH = re.compile(r'[a-f0-9]{64}')


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path, maximum):
    if path.is_symlink() or not 0 < path.stat().st_size <= maximum:
        raise ValueError('invalid styled thumbnail metadata')
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError('invalid styled thumbnail metadata')
    return value


def _root(data_dir):
    base = Path(data_dir).resolve()
    root = base / 'thumbs/styled'
    if (base / 'thumbs').is_symlink() or root.is_symlink() or not root.resolve().is_relative_to(base):
        raise ValueError('invalid thumbnail overlay root')
    return root


@lru_cache(maxsize=4)
def _manifest(path_string, mtime, size):
    manifest = _json(Path(path_string), 4_000_000)
    expected = dict(schema_version=1, style=STYLE, status='complete', pose_count=5862,
                    image_count=187584, scopes=list(SCOPES), angles=list(range(0, 360, 45)),
                    width=512, height=512, pitch_degrees=0, projection='orthographic',
                    outline_width=.65, contrast=2.6, lighten=.22, background='white')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('incomplete or unsupported thumbnail overlay')
    poses, records = manifest.get('poses', {}), manifest.get('pose_records', {})
    if (len(poses) != 5862 or len(records) != 5862 or set(poses.values()) != set(records)
            or not all(HASH.fullmatch(h) and HASH.fullmatch(d) for h, d in records.items())
            or not HASH.fullmatch(manifest.get('renderer_revision', ''))
            or not re.fullmatch(STYLE + r'-[a-f0-9]{12}', manifest.get('asset_version', ''))):
        raise ValueError('invalid thumbnail overlay inventory')
    return manifest


def manifest_for(data_dir):
    path = _root(data_dir) / 'manifest.json'
    if not path.exists():
        return None
    stat = path.stat()
    return _manifest(str(path), stat.st_mtime_ns, stat.st_size)


def _record(root, manifest, source):
    if not HASH.fullmatch(source) or source not in manifest['pose_records']:
        raise ValueError('unregistered styled thumbnail source')
    folder = root / source
    if folder.is_symlink():
        raise ValueError('invalid styled thumbnail source directory')
    path = folder / 'pose.json'
    if _sha(path) != manifest['pose_records'][source]:
        raise ValueError('styled thumbnail record changed')
    record = _json(path, 65536)
    names = {f'{s}__{v}.jpg' for s in SCOPES for v in VIEWS}
    if (record.get('style') != STYLE or record.get('source_bvh_sha256') != source
            or record.get('renderer_revision') != manifest['renderer_revision']
            or manifest['poses'].get(record.get('pose_id')) != source
            or set(record.get('images', {})) != names):
        raise ValueError('styled thumbnail lineage changed')
    return record


def _raster(path, info):
    if path.is_symlink() or not 3 <= path.stat().st_size <= 2_000_000:
        raise ValueError('invalid styled thumbnail raster')
    data = path.read_bytes()
    if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
        raise ValueError('styled thumbnail raster changed')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or image.size != (512, 512):
            raise ValueError('invalid styled thumbnail dimensions')
        image.verify()
    return data


def read_styled_poster(data_dir, source, scope, view):
    manifest = manifest_for(data_dir)
    if manifest is None or scope not in SCOPES:
        return None
    if view not in VIEWS:
        raise ValueError('invalid styled thumbnail view')
    root = _root(data_dir)
    name = f'{scope}__{view}.jpg'
    try:
        record = _record(root, manifest, source)
        data = _raster(root / source / name, record['images'][name])
    except FileNotFoundError as error:
        # rough_posters allows a fallback only for genuinely old/missing bundles.
        # A declared complete style release must never mix angles or styles.
        raise ValueError('advertised styled thumbnail is missing') from error
    from .directional_posters import CHARACTER_SHA
    from .pose_posters import headers
    meta = dict(kind='default-body', render_method='frozen-retarget-v1',
                source_bvh_sha256=source, character_id='standin-master-v2', character_sha256=CHARACTER_SHA,
                renderer_revision=manifest['renderer_revision'], image_sha256=record['images'][name]['sha256'], scope=scope)
    result = headers(meta)
    result.update({'X-Standin-Poster-Asset-Version': manifest['asset_version'],
                   'X-Standin-Thumbnail-Style': STYLE,
                   'X-Standin-Native-Render-Method': 'reviewed-surface-outline-v1',
                   'X-Standin-Preview-View': view,
                   'X-Standin-Asset-Library-Version': manifest['source_library_version']})
    return data, result


def legacy_thumbnail_path(data_dir, pose_id, view):
    manifest = manifest_for(data_dir)
    if manifest is None:
        return None
    source = manifest['poses'].get(pose_id)
    if source is None or view not in LEGACY_VIEWS:
        raise ValueError('unregistered styled legacy thumbnail')
    root = _root(data_dir)
    record = _record(root, manifest, source)
    name = f'full__{LEGACY_VIEWS[view]}.jpg'
    path = root / source / name
    _raster(path, record['images'][name])
    return path


def _extract(archive, destination):
    seen, total = set(), 0
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            name = member.name
            parts = PurePosixPath(name).parts
            if (not member.isfile() or name in seen or '\\' in name or ':' in name
                    or '..' in parts or parts[:2] != ('thumbs', 'styled')
                    or PurePosixPath(name).is_absolute() or len(parts) not in (3, 4)):
                raise ValueError('thumbnail archive contains a forbidden entry')
            valid = (parts == ('thumbs', 'styled', 'manifest.json') or
                     (len(parts) == 4 and HASH.fullmatch(parts[2]) and
                      (parts[3] == 'pose.json' or parts[3] in {f'{s}__{v}.jpg' for s in SCOPES for v in VIEWS})))
            total += member.size
            if not valid or not 0 < member.size <= 4_000_000 or total > 8_000_000_000 or len(seen) >= 193447:
                raise ValueError('invalid thumbnail archive inventory')
            seen.add(name)
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out)
    if len(seen) != 193447:
        raise ValueError('incomplete thumbnail archive')


def ensure_thumbnail_overlay(data_dir, db_path, uri=None, archive_sha=None):
    uri = uri if uri is not None else os.getenv('POSE_THUMBNAIL_OVERLAY_URI', '')
    archive_sha = archive_sha if archive_sha is not None else os.getenv('POSE_THUMBNAIL_OVERLAY_SHA256', '')
    if not uri and not archive_sha:
        return False
    if not uri.startswith('s3://') or not HASH.fullmatch(archive_sha):
        raise ValueError('thumbnail overlay needs an S3 URI and pinned SHA256')
    base = Path(data_dir).resolve()
    root = _root(base)
    marker = root / 'archive.sha256'
    if marker.is_file() and marker.read_text() == archive_sha:
        manifest = manifest_for(base)
        if manifest and manifest['library_db_sha256'] == _sha(Path(db_path)):
            return False
        raise ValueError('installed thumbnail overlay database changed')
    from .library_source import _download_s3
    with tempfile.TemporaryDirectory(prefix='thumbnail-overlay-', dir=base) as temporary:
        work = Path(temporary)
        archive = work / 'overlay.tar.gz'
        _download_s3(uri, archive)
        if _sha(archive) != archive_sha:
            raise ValueError('thumbnail archive SHA256 mismatch')
        unpacked = work / 'unpacked'
        _extract(archive, unpacked)
        manifest = manifest_for(unpacked)
        library = _json(base / 'library_manifest.json', 65536)
        if (manifest['library_db_sha256'] != _sha(Path(db_path))
                or manifest['source_library_version'] != library['library_version']):
            raise ValueError('thumbnail overlay belongs to a different library')
        incoming = _root(unpacked)
        for source in manifest['pose_records']:
            record = _record(incoming, manifest, source)
            for name, info in record['images'].items():
                _raster(incoming / source / name, info)
        (incoming / 'archive.sha256').write_text(archive_sha)
        # Startup only: no requests can observe a partially installed release.
        if root.exists():
            raise ValueError('remove the previous overlay only through a fresh task deployment')
        root.parent.mkdir(parents=True, exist_ok=True)
        incoming.rename(root)
    _manifest.cache_clear()
    return True
