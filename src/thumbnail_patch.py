"""Opt-in immutable raster corrections, verified against the installed release."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

from PIL import Image

HASH = re.compile(r'[a-f0-9]{64}')
NAMES = re.compile(r'full__yaw_(000|045|090|135|180|225|270|315)\.jpg')
STAGING_BUCKET = 'standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
STAGING_LIBRARY_URI = f's3://{STAGING_BUCKET}/pose-library/releases/lib-20261105-c7817702-32e4eb027242.tar.gz'


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read(root):
    path = root / 'manifest.json'
    if path.is_symlink() or not 0 < path.stat().st_size < 65536:
        raise ValueError('invalid thumbnail patch manifest')
    return json.loads(path.read_bytes())


def _verify(root, base, *, check_images=True):
    from .thumbnail_overlay import manifest_for, _record, _root
    from .directional_posters import CHARACTER_SHA
    patch = _read(root)
    original = manifest_for(base)
    if (patch.get('schema_version') != 1 or patch.get('environment') != 'staging'
            or patch.get('base_library_uri') != STAGING_LIBRARY_URI
            or original is None or patch.get('source_library_version') != original['source_library_version']
            or patch.get('base_asset_version') != original['asset_version']
            or patch.get('character_sha256') != CHARACTER_SHA
            or not HASH.fullmatch(patch.get('source_bvh_sha256', ''))
            or not HASH.fullmatch(patch.get('renderer_revision', ''))
            or not HASH.fullmatch(patch.get('recipe_sha256', ''))
            or patch.get('renderer_revision') != patch.get('recipe_sha256')
            or not HASH.fullmatch(patch.get('surface_sha256', ''))
            or not re.fullmatch(r'staging-raster-patch-[a-f0-9]{12}', patch.get('asset_version', ''))
            or not isinstance(patch.get('recipe'), dict)
            or not patch['recipe'] or not all(isinstance(value, str) and HASH.fullmatch(value)
                                            for value in patch['recipe'].values())
            or _digest(json.dumps(patch['recipe'], sort_keys=True).encode()) != patch['recipe_sha256']
            or patch.get('finger_joints_applied') != 30
            or not isinstance(patch.get('images'), dict) or not 1 <= len(patch['images']) <= 8):
        raise ValueError('thumbnail patch release lineage mismatch')
    source = patch['source_bvh_sha256']
    record = _record(_root(base), original, source)
    if (original['poses'].get(patch.get('pose_id')) != source
            or patch.get('base_renderer_revision') != record['renderer_revision']):
        raise ValueError('thumbnail patch source lineage mismatch')
    for name, info in patch['images'].items():
        if (not NAMES.fullmatch(name) or not isinstance(info, dict)
                or info.get('old_sha256') != record['images'][name]['sha256']
                or not HASH.fullmatch(info.get('sha256', ''))):
            raise ValueError('thumbnail patch key or original digest mismatch')
        if check_images:
            _image(root / name, info)
    return patch


def _image(path, info):
    if path.is_symlink() or not 0 < path.stat().st_size <= 2_000_000:
        raise ValueError('invalid thumbnail patch image')
    data = path.read_bytes()
    if len(data) != info.get('bytes') or _digest(data) != info['sha256']:
        raise ValueError('thumbnail patch image digest mismatch')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or image.size != (512, 512):
            raise ValueError('invalid thumbnail patch image dimensions')
        image.verify()
    return data


def read_patch(base, source, name, original_bytes):
    if not os.getenv('POSE_THUMBNAIL_PATCH_URI'):
        return None
    root = Path(base) / 'thumbs/raster-patch'
    if not root.exists():
        return None
    if root.is_symlink():
        raise ValueError('invalid thumbnail patch root')
    patch = _verify(root, base, check_images=False)
    if source != patch['source_bvh_sha256'] or name not in patch['images']:
        return None
    info = patch['images'][name]
    if _digest(original_bytes) != info['old_sha256']:
        raise ValueError('thumbnail patch original image changed')
    return _image(root / name, info), patch


def ensure_thumbnail_patch(base, uri=None, archive_sha=None):
    uri = os.getenv('POSE_THUMBNAIL_PATCH_URI', '') if uri is None else uri
    archive_sha = os.getenv('POSE_THUMBNAIL_PATCH_SHA256', '') if archive_sha is None else archive_sha
    if not uri and not archive_sha:
        return False
    if not uri.startswith('s3://') or not HASH.fullmatch(archive_sha):
        raise ValueError('thumbnail patch requires an immutable S3 URI and SHA256')
    if (os.getenv('POSE_LIBRARY_URI') != STAGING_LIBRARY_URI
            or not uri.startswith(f's3://{STAGING_BUCKET}/pose-library/thumbnail-patches/')):
        raise ValueError('thumbnail patch is restricted to the pinned staging library and bucket')
    base = Path(base).resolve()
    root = base / 'thumbs/raster-patch'
    if root.exists():
        raise ValueError('thumbnail patches require a fresh task deployment')
    from .library_source import _download_s3
    with tempfile.TemporaryDirectory(prefix='thumbnail-patch-', dir=base) as temporary:
        work = Path(temporary)
        archive = work / 'patch.tar.gz'
        _download_s3(uri, archive)
        if _digest(archive.read_bytes()) != archive_sha:
            raise ValueError('thumbnail patch archive digest mismatch')
        out = work / 'unpacked'; out.mkdir()
        seen = set()
        with tarfile.open(archive, 'r:gz') as stream:
            for item in stream:
                if (not item.isfile() or item.name in seen
                        or (item.name != 'manifest.json' and not NAMES.fullmatch(item.name))
                        or not 0 < item.size <= 2_000_000 or len(seen) >= 9):
                    raise ValueError('forbidden thumbnail patch archive entry')
                seen.add(item.name)
                (out / item.name).write_bytes(stream.extractfile(item).read())
        patch = _verify(out, base)
        if seen != {'manifest.json', *patch['images']}:
            raise ValueError('incomplete thumbnail patch archive')
        root.parent.mkdir(parents=True, exist_ok=True)
        out.rename(root)
    return True
