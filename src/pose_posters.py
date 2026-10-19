"""Read-only default-body posters. Missing assets use a bounded pose-only drawing.

No request path imports the converter or starts Blender. Posters are canonical
front/full references, never evidence of selected shape, hands or search camera.
"""
from __future__ import annotations

from converter.framing import OUTPUT_SCOPES, validate_scope
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import tempfile

VERSION = "default-body-poster-v1"
CHARACTER = "standin-master-v2"
MAX_IMAGE_BYTES = 2 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def character_sha256() -> str:
    return json.loads((ROOT / 'config/characters.example.json').read_text())["characters"][CHARACTER]["sha256"]


def revision() -> str:
    return digest(b''.join((ROOT / name).read_bytes().replace(b'\r\n', b'\n') for name in (
        'src/pose_posters.py', 'converter/preview_style.py', 'converter/preview_studio.sl',
        'converter/framing.py', 'converter/protocol.py', 'converter/worker.py',
        'converter/camera.py', 'converter/convert.py', 'converter/retarget.py',
        'converter/thumbnail_render.py', 'converter/framed_model.py')))


def identity(source_sha256: str, scope="full") -> dict:
    validate_scope(scope)
    return dict(version=VERSION, source_bvh_sha256=source_sha256,
                character_id=CHARACTER, character_sha256=character_sha256(),
                renderer_revision=revision(), view='front', scope=scope,
                projection='orthographic', projection_matches_search=False,
                selected_body=False, hands_applied=False)


def read_poster(directory: Path, source: str, scope="full") -> tuple[bytes, dict]:
    if len(source) != 64 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('invalid poster source')
    validate_scope(scope)
    root = directory.resolve()
    stem = source + '__' + scope
    manifest = root / (stem + '.json')
    if manifest.is_symlink() or manifest.stat().st_size > 8192:
        raise ValueError('invalid poster manifest')
    meta = json.loads(manifest.read_text())
    if not isinstance(meta, dict):
        raise ValueError('invalid poster manifest')
    from src.human_reference import METHOD, reference_identity
    if meta.get('render_method') not in (None, METHOD):
        raise ValueError('unknown human preview method')
    expected = reference_identity(source, scope) if meta.get('render_method') == METHOD else identity(source, scope)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('poster lineage changed')
    image = root / (stem + '.png')
    if image.is_symlink() or not 8 <= image.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError('invalid poster image')
    data = image.read_bytes()
    if (not data.startswith(b'\x89PNG\r\n\x1a\n') or meta.get('image_sha256') != digest(data)
            or meta.get('bytes') != len(data)):
        raise ValueError('poster image changed')
    return data, {**meta, 'kind': 'default-body'}


@lru_cache(maxsize=32)
def pose_only(source: bytes, scope="full") -> bytes:
    # Render the exact verified bytes (including an actual mirrored BVH), not a
    # path that could change after hashing. Never reflect an image with CSS.
    from src.thumbnail_renderer import render_thumbnail
    from src.bvh import load_coco17
    with tempfile.TemporaryDirectory(prefix='standin-pose-poster-') as temp:
        path = Path(temp) / 'source.bvh'
        path.write_bytes(source)
        keypoints, scores = load_coco17(str(path), 0)
        # Pose-only fallback: anatomical joint visibility, not a mesh-cut claim.
        # A raised leg cannot leak into a half/body-free reference by screen Y.
        if scope != 'full':
            keep = {'half': set(range(13)), 'bust': set(range(7)) | {11, 12},
                    'shoulder_head': set(range(7)) | {11, 12}, 'head': set(range(5))}[scope]
            if scope in ('bust', 'shoulder_head'):
                # Approximate torso band, along the body's own torso axis.
                # Shoulder/head keeps only the top 10%; bust retains upper torso.
                # These are reference cut points, never an exact skin-region cut.
                keypoints = keypoints.copy()
                shoulder = (keypoints[5] + keypoints[6]) * .5
                hip = (keypoints[11] + keypoints[12]) * .5
                offset = (shoulder - hip) * (.90 if scope == 'shoulder_head' else .55)
                keypoints[11] += offset
                keypoints[12] += offset
            scores = scores.copy()
            for index in range(17):
                if index not in keep:
                    scores[index] = 0
        image = render_thumbnail(keypoints, scores, 0.0, 256)
    result = io.BytesIO()
    image.save(result, format='PNG')
    return result.getvalue()


def poster(directory: Path, source: bytes, scope="full") -> tuple[bytes, dict]:
    if not source or len(source) > 2_000_000:
        raise ValueError('invalid BVH size')
    validate_scope(scope)
    source_sha = digest(source)
    try:
        return read_poster(directory, source_sha, scope)
    except (OSError, ValueError, KeyError, TypeError):
        data = pose_only(source, scope)
        return data, {**identity(source_sha, scope), 'kind': 'pose-only',
                      'character_id': 'none', 'character_sha256': 'none',
                      'image_sha256': digest(data), 'bytes': len(data)}


def headers(meta: dict) -> dict[str, str]:
    return {
        'Cache-Control': 'private, no-store',
        'X-Standin-Poster-Kind': meta['kind'],
        'X-Standin-Poster-Version': VERSION,
        'X-Standin-Preview-Method': meta.get('render_method', 'frozen-retarget-v1') if meta['kind'] == 'default-body' else 'pose-only',
        'X-Standin-Template-SHA256': meta.get('template_sha256', 'none'),
        'X-Standin-Source-BVH-SHA256': meta['source_bvh_sha256'],
        'X-Standin-Character-Id': meta['character_id'],
        'X-Standin-Character-SHA256': meta['character_sha256'],
        'X-Standin-Preview-Revision': meta['renderer_revision'],
        'X-Standin-Artifact-SHA256': meta['image_sha256'],
        'X-Standin-Output-Scope': meta['scope'],
        'X-Standin-Preview-Projection': 'orthographic',
        'X-Standin-Preview-Only': 'true',
    }
