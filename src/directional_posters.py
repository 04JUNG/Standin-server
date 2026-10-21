"""Read the approved, source-bound 4-scope/8-direction library JPEGs.

This is a frozen default-body reference, not the chosen character, search camera,
or an export grant. No converter/Blender work occurs on a request path.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from PIL import Image

VARIANT = 'scope8-v1'
METHOD = 'frozen-native-scope8-v1'
SCOPES = ('full', 'half', 'bust', 'shoulder_head')
VIEWS = tuple(f'yaw_{a:03d}' for a in range(0, 360, 45))
CHARACTER_SHA = '7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d'
SOLVER_SHA = '52eae0b0eb6a7023540fdf023ad4321f24c11d439f3ecf9f390b5b72358b24e3'
REVISIONS = frozenset((
    '14d60ac3973bb86a9cb9b01c3bdfc4e89f233013373b5e6715ef9db40582518f',
    'a7ebf9dafc931a65ad171571c9033d45890693a5b24d976edd5ae3867d2a0f53',
))
MAX_IMAGE_BYTES = 2 * 1024 * 1024


def _json(path: Path, maximum: int) -> dict:
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError('invalid directional metadata')
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError('invalid directional metadata')
    return result


def read_directional_poster(data_dir: Path, source_sha: str, scope='full', view='yaw_000'):
    if (len(source_sha) != 64 or any(c not in '0123456789abcdef' for c in source_sha)
            or scope not in SCOPES or view not in VIEWS):
        raise ValueError('invalid directional identity')
    base = data_dir.resolve()
    root = base / 'thumbs' / 'scope8'
    if root.is_symlink() or not root.resolve().is_relative_to(base):
        raise ValueError('invalid directional asset directory')
    stem = source_sha + '__' + scope + '__' + view
    meta = _json(root / (stem + '.json'), 32768)
    angle = int(view[4:])
    expected = dict(schema_version=2, render_method=METHOD, source_bvh_sha256=source_sha,
        character_id='standin-master-v2', character_sha256=CHARACTER_SHA,
        scope=scope, view=view, azimuth_degrees=angle, native_camera_yaw=(180-angle) % 360,
        pitch_degrees=0, projection='orthographic', projection_matches_search=False,
        selected_body=False, hands_applied=False, source_frame=0, preview_only=True,
        anatomy_approval=False, width=512, height=512, format='jpeg', quality=90,
        style='approved-anatomy-detail-dark-5scope-8view-v1', solver_manifest_sha256=SOLVER_SHA)
    if (any(meta.get(k) != v or type(meta.get(k)) is not type(v) for k, v in expected.items())
            or meta.get('renderer_revision') not in REVISIONS):
        raise ValueError('directional poster lineage changed')
    framing = meta.get('framing', {})
    if (not isinstance(framing, dict) or framing.get('scope') != scope
            or framing.get('version') != 'skin-regions-v2-shoulder-head'
            or not 0 < framing.get('vertices_after', 0) <= framing.get('vertices_before', 0)
            or framing.get('bones_preserved', 0) <= 0 or meta.get('border_pixels', 0) < 12):
        raise ValueError('directional crop changed')
    if scope == 'full':
        if framing['vertices_before'] != framing['vertices_after'] or framing.get('cap_faces') != 0:
            raise ValueError('full-body crop changed')
    elif framing['vertices_after'] >= framing['vertices_before']:
        raise ValueError('anatomical crop missing')
    image = root / (stem + '.jpg')
    if image.is_symlink() or not 3 <= image.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError('invalid directional image')
    data = image.read_bytes()
    if (meta.get('bytes') != len(data) or hashlib.sha256(data).hexdigest() != meta.get('image_sha256')
            or not data.startswith(b'\xff\xd8\xff')):
        raise ValueError('directional image changed')
    with Image.open(io.BytesIO(data)) as decoded:
        if decoded.format != 'JPEG' or decoded.size != (512, 512):
            raise ValueError('invalid directional raster')
        decoded.load()
    manifest = _json(base / 'library_manifest.json', 65536)
    declared = manifest.get('directional_thumbnails', {})
    if (not isinstance(declared, dict) or declared.get('render_method') != METHOD or declared.get('status') != 'complete'
            or meta['renderer_revision'] not in declared.get('renderer_revisions', [])
            or scope not in declared.get('output_scopes', []) or view not in declared.get('views', [])):
        raise ValueError('unpublished directional asset')
    from src.pose_posters import headers
    result = headers({**meta, 'kind': 'default-body'})
    result.update({'X-Standin-Poster-Asset-Version': VARIANT,
                   'X-Standin-Preview-View': view,
                   'X-Standin-Asset-Library-Version': manifest['library_version']})
    return data, result
