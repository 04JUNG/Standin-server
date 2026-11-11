"""Operator-pinned final poses for the exact reviewed default character.

Only catalogued BVH digests can select an asset. Refined/user BVHs, other
characters, mirroring and the exact-solver kill switch retain normal retargeting.
The public BVH is reconstructed from the reviewed skeleton, not its donor pose.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

VERSION = 'reviewed-library-fbx-v1'
HEX = re.compile(r'[a-f0-9]{64}')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=4)
def _catalog(root, expected):
    if not HEX.fullmatch(expected):
        raise ValueError('reviewed library requires a pinned manifest')
    base = Path(root).resolve(strict=True)
    path = base / 'manifest.json'
    if path.is_symlink() or path.stat().st_size > 1_000_000 or sha(path) != expected:
        raise ValueError('reviewed library manifest changed')
    value = json.loads(path.read_bytes())
    if value.get('version') != VERSION or not isinstance(value.get('poses'), dict) or not 1 <= len(value['poses']) <= 10000:
        raise ValueError('invalid reviewed library catalog')
    if not HEX.fullmatch(value.get('character_sha256', '')):
        raise ValueError('invalid reviewed character')
    for source, row in value['poses'].items():
        if not HEX.fullmatch(source) or not HEX.fullmatch(row.get('fbx_sha256', '')) or row.get('path') != source + '.fbx':
            raise ValueError('invalid reviewed asset binding')
    return base, value


def restore(job):
    root, expected = os.getenv('REVIEWED_LIBRARY_ROOT', ''), os.getenv('REVIEWED_LIBRARY_SHA256', '')
    if not root and not expected:
        return None
    if not root or not expected:
        raise ValueError('incomplete reviewed library configuration')
    base, catalog = _catalog(root, expected)
    source = sha(job['bvh_path'])
    row = catalog['poses'].get(source)
    if row is None or job.get('shape_recipe') or job.get('mirror') or job.get('force_exact_v324'):
        return None
    if job['character_sha256'] != catalog['character_sha256'] or sha(job['character_fbx']) != catalog['character_sha256']:
        return None
    asset = base / row['path']
    if asset.is_symlink() or not asset.is_file() or not 0 < asset.stat().st_size < 32 * 1024**2 or sha(asset) != row['fbx_sha256']:
        raise ValueError('reviewed FBX integrity changed')
    shutil.copyfile(asset, job['output_path'])
    return dict(ok=True, output_mode='rigged_rest', frame=0, mirrored=False,
                src_profile='mixamo', dst_profile='mixamo', mapped_bones=row['bones'],
                missing_required=[], warnings=[],
                reviewed_library=dict(version=VERSION, pose_id=row['pose_id'],
                    source_bvh_sha256=source, reviewed_fbx_sha256=row['fbx_sha256'],
                    manifest_sha256=expected, character_sha256=catalog['character_sha256']))


def transfer_registered_fingers(job):
    """Keep source articulation for an ordinary retarget onto another body."""
    root, expected = os.getenv('REVIEWED_LIBRARY_ROOT', ''), os.getenv('REVIEWED_LIBRARY_SHA256', '')
    if not root or not expected or job.get('force_exact_v324'):
        return None
    _, catalog = _catalog(root, expected)
    if sha(job['bvh_path']) not in catalog['poses']:
        return None
    from converter.reviewed_source_fingers import bake_source_fingers
    return bake_source_fingers(job['output_path'], job['bvh_path'],
                               mirror=job['mirror'], hands=job.get('hands'))
