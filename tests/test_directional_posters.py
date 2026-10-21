import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest
from src import directional_posters as native


def asset(root, scope='full', view='yaw_000', source=b'public source BVH'):
    source_sha = hashlib.sha256(source).hexdigest()
    directory = root / 'thumbs/scope8'
    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new('RGB', (512, 512), 'gray').save(buffer, format='JPEG', quality=90)
    data = buffer.getvalue()
    angle = int(view[4:])
    meta = dict(schema_version=2, render_method=native.METHOD, source_bvh_sha256=source_sha,
        character_id='standin-master-v2', character_sha256=native.CHARACTER_SHA,
        renderer_revision=next(iter(native.REVISIONS)), scope=scope, view=view,
        azimuth_degrees=angle, native_camera_yaw=(180-angle) % 360, pitch_degrees=0,
        projection='orthographic', projection_matches_search=False, selected_body=False,
        hands_applied=False, source_frame=0, preview_only=True, anatomy_approval=False,
        width=512, height=512, format='jpeg', quality=90, image_sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data), style='approved-anatomy-detail-dark-5scope-8view-v1',
        solver_manifest_sha256=native.SOLVER_SHA, border_pixels=12,
        framing=dict(version='skin-regions-v2-shoulder-head', scope=scope, vertices_before=100,
                     vertices_after=100 if scope == 'full' else 50, bones_preserved=52, cap_faces=0))
    stem = directory / (source_sha + '__' + scope + '__' + view)
    image = stem.with_suffix('.jpg')
    image.write_bytes(data)
    metadata = stem.with_suffix('.json')
    metadata.write_text(json.dumps(meta))
    (root / 'library_manifest.json').write_text(json.dumps(dict(library_version='lib-20261021-21b1b3d9',
        directional_thumbnails=dict(status='complete', render_method=native.METHOD,
            renderer_revisions=list(native.REVISIONS), output_scopes=native.SCOPES, views=native.VIEWS))))
    return source_sha, data, meta, image, metadata


@pytest.mark.parametrize('scope', native.SCOPES)
@pytest.mark.parametrize('view', native.VIEWS)
def test_exact_scope_view_source_and_original_jpeg_bytes(tmp_path, scope, view):
    source, data, meta, _, _ = asset(tmp_path, scope, view)
    result, headers = native.read_directional_poster(tmp_path, source, scope, view)
    assert result == data
    assert headers['X-Standin-Preview-Method'] == native.METHOD
    assert headers['X-Standin-Preview-View'] == view
    assert headers['X-Standin-Output-Scope'] == scope
    assert headers['X-Standin-Artifact-SHA256'] == meta['image_sha256']
    assert headers['X-Standin-Preview-Only'] == 'true'


@pytest.mark.parametrize('key', ['source_bvh_sha256', 'character_sha256', 'renderer_revision',
    'scope', 'view', 'solver_manifest_sha256', 'format', 'source_frame', 'native_camera_yaw',
    'style', 'width', 'image_sha256', 'bytes', 'projection_matches_search', 'selected_body', 'hands_applied'])
def test_changed_lineage_or_content_fails_closed(tmp_path, key):
    source, _, meta, _, path = asset(tmp_path)
    path.write_text(json.dumps({**meta, key: 'changed'}))
    with pytest.raises((ValueError, TypeError)):
        native.read_directional_poster(tmp_path, source)


def test_wrong_pixels_symlinks_and_path_traversal_are_rejected(tmp_path):
    source, _, meta, image, path = asset(tmp_path)
    image.write_bytes(b'\xff\xd8\xffnot a JPEG')
    meta.update(image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), bytes=image.stat().st_size)
    path.write_text(json.dumps(meta))
    with pytest.raises((ValueError, OSError)):
        native.read_directional_poster(tmp_path, source)
    image.unlink()
    outside = tmp_path / 'elsewhere.jpg'
    outside.write_bytes(b'private')
    image.symlink_to(outside)
    with pytest.raises(ValueError):
        native.read_directional_poster(tmp_path, source)
    for wrong in ('../' + source, 'A' * 64, 'a' * 63):
        with pytest.raises(ValueError):
            native.read_directional_poster(tmp_path, wrong)


def test_head_is_not_mislabeled_as_a_new_scope(tmp_path):
    source, *_ = asset(tmp_path)
    with pytest.raises(ValueError):
        native.read_directional_poster(tmp_path, source, 'head')
