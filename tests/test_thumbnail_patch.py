import hashlib
import io
import json
import shutil
import tarfile

from PIL import Image
import pytest

from tests.test_thumbnail_overlay import assets
from src import thumbnail_overlay as overlay
from src import thumbnail_patch as patch
from src.directional_posters import CHARACTER_SHA


@pytest.fixture
def corrected(assets, monkeypatch):
    monkeypatch.setenv('POSE_THUMBNAIL_PATCH_URI', f's3://{patch.STAGING_BUCKET}/pose-library/thumbnail-patches/test.tar.gz')
    base, source, folder, _ = assets
    manifest = overlay.manifest_for(base)
    # The baseline fixture contains one complete 32-image reviewed source.
    record = json.loads((folder / 'pose.json').read_text())
    root = base / 'thumbs/raster-patch'; root.mkdir()
    name = 'full__yaw_045.jpg'
    Image.new('RGB', (512, 512), '#888888').save(root / name, 'JPEG')
    data = (root / name).read_bytes()
    recipe = {'renderer': 'a' * 64}
    revision = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    doc = dict(schema_version=1, environment='staging', pose_id=record['pose_id'],
        base_library_uri=patch.STAGING_LIBRARY_URI,
        source_bvh_sha256=source, character_sha256=CHARACTER_SHA,
        source_library_version=manifest['source_library_version'],
        base_asset_version=manifest['asset_version'], base_renderer_revision=record['renderer_revision'],
        renderer_revision=revision, recipe_sha256=revision, recipe=recipe,
        asset_version='staging-raster-patch-' + 'a' * 12, finger_joints_applied=30, surface_sha256='e' * 64,
        images={name: dict(old_sha256=record['images'][name]['sha256'],
                           sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))})
    (root / 'manifest.json').write_text(json.dumps(doc))
    return base, source, root, doc, data


def test_patch_replaces_only_bound_key_and_preserves_original_overlay(corrected):
    base, source, root, doc, data = corrected
    baseline = (base / 'thumbs/styled/manifest.json').read_bytes()
    result, headers = overlay.read_styled_poster(base, source, 'full', 'yaw_045')
    assert result == data
    assert headers['X-Standin-Preview-Revision'] == doc['renderer_revision']
    assert headers['X-Standin-Poster-Asset-Version'] == doc['asset_version']
    assert overlay.legacy_thumbnail_path(base, doc['pose_id'], 'three_quarter') == root / 'full__yaw_045.jpg'
    assert overlay.read_styled_poster(base, source, 'full', 'yaw_000')[0] != data
    assert overlay.read_styled_poster(base, source, 'half', 'yaw_045')[0] != data
    assert (base / 'thumbs/styled/manifest.json').read_bytes() == baseline


def test_disabling_config_ignores_even_a_previous_installed_patch(corrected, monkeypatch):
    base, source, _, _, data = corrected
    monkeypatch.delenv('POSE_THUMBNAIL_PATCH_URI')
    assert overlay.read_styled_poster(base, source, 'full', 'yaw_045')[0] != data


@pytest.mark.parametrize('field,value', [
    ('source_library_version', 'another-library'), ('pose_id', 'other-pose'),
    ('character_sha256', 'f' * 64), ('source_bvh_sha256', 'f' * 64),
    ('base_renderer_revision', 'f' * 64), ('finger_joints_applied', 0),
    ('environment', 'production'), ('recipe_sha256', 'f' * 64)])
def test_wrong_release_or_lineage_fails_closed(corrected, field, value):
    base, source, root, doc, _ = corrected
    doc[field] = value; (root / 'manifest.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        overlay.read_styled_poster(base, source, 'full', 'yaw_045')


def test_wrong_old_image_and_corrupted_new_image_fail_closed(corrected):
    base, source, root, doc, _ = corrected
    doc['images']['full__yaw_045.jpg']['old_sha256'] = 'f' * 64
    (root / 'manifest.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError): patch._verify(root, base)
    doc['images']['full__yaw_045.jpg']['old_sha256'] = json.loads((base / 'thumbs/styled' / source / 'pose.json').read_text())['images']['full__yaw_045.jpg']['sha256']
    (root / 'manifest.json').write_text(json.dumps(doc))
    (root / 'full__yaw_045.jpg').write_bytes(b'corrupt')
    with pytest.raises(ValueError): patch._verify(root, base)


def test_immutable_archive_install_and_no_config_noop(corrected, monkeypatch):
    base, source, root, doc, data = corrected
    archive = base / 'release.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for path in sorted(root.iterdir()): stream.add(path, arcname=path.name)
    shutil.rmtree(root)
    from src import library_source
    monkeypatch.setattr(library_source, '_download_s3', lambda uri, target: shutil.copyfile(archive, target))
    assert patch.ensure_thumbnail_patch(base, '', '') is False
    monkeypatch.setenv('POSE_LIBRARY_URI', patch.STAGING_LIBRARY_URI)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert patch.ensure_thumbnail_patch(base, f's3://{patch.STAGING_BUCKET}/pose-library/thumbnail-patches/patch.tar.gz', digest)
    assert overlay.read_styled_poster(base, source, 'full', 'yaw_045')[0] == data
    with pytest.raises(ValueError): patch.ensure_thumbnail_patch(base, f's3://{patch.STAGING_BUCKET}/pose-library/thumbnail-patches/patch.tar.gz', digest)


@pytest.mark.parametrize('library,uri', [
    ('s3://production-bucket/library.tar.gz', f's3://{patch.STAGING_BUCKET}/pose-library/thumbnail-patches/x.tar.gz'),
    (patch.STAGING_LIBRARY_URI, 's3://production-bucket/x.tar.gz'),
    (patch.STAGING_LIBRARY_URI, f's3://{patch.STAGING_BUCKET}/pose-library/releases/x.tar.gz')])
def test_production_or_wrong_prefix_rejected_before_download(tmp_path, monkeypatch, library, uri):
    monkeypatch.setenv('POSE_LIBRARY_URI', library)
    from src import library_source
    monkeypatch.setattr(library_source, '_download_s3', lambda *args: pytest.fail('must not download'))
    with pytest.raises(ValueError, match='restricted'):
        patch.ensure_thumbnail_patch(tmp_path, uri, 'a' * 64)


def test_archive_traversal_rejected_before_install(corrected, monkeypatch):
    base, _, root, _, _ = corrected
    archive = base / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        item = tarfile.TarInfo('../poses.db'); item.size = 1
        stream.addfile(item, io.BytesIO(b'x'))
    shutil.rmtree(root)
    from src import library_source
    monkeypatch.setattr(library_source, '_download_s3', lambda uri, target: shutil.copyfile(archive, target))
    monkeypatch.setenv('POSE_LIBRARY_URI', patch.STAGING_LIBRARY_URI)
    with pytest.raises(ValueError, match='forbidden'):
        patch.ensure_thumbnail_patch(base,
            f's3://{patch.STAGING_BUCKET}/pose-library/thumbnail-patches/bad.tar.gz',
            hashlib.sha256(archive.read_bytes()).hexdigest())
    assert not root.exists()
