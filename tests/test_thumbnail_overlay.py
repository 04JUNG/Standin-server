import hashlib
import io
import json
import tarfile

from PIL import Image
import pytest

from src import thumbnail_overlay as overlay


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / 'thumbs/styled'
    source = hashlib.sha256(b'pose').hexdigest()
    folder = root / source
    folder.mkdir(parents=True)
    stream = io.BytesIO()
    Image.new('RGB', (512, 512), 'white').save(stream, format='JPEG')
    raster = stream.getvalue()
    info = dict(bytes=len(raster), sha256=hashlib.sha256(raster).hexdigest())
    images = {}
    for scope in overlay.SCOPES:
        for view in overlay.VIEWS:
            name = f'{scope}__{view}.jpg'
            (folder / name).write_bytes(raster)
            images[name] = info
    record = dict(style=overlay.STYLE, source_bvh_sha256=source, renderer_revision='a'*64,
                  pose_id='sample', images=images)
    (folder / 'pose.json').write_text(json.dumps(record))
    poses = {f'pose-{i}': hashlib.sha256(str(i).encode()).hexdigest() for i in range(5861)}
    poses['sample'] = source
    records = {s: 'b'*64 for s in poses.values()}
    records[source] = overlay._sha(folder / 'pose.json')
    manifest = dict(schema_version=1, style=overlay.STYLE, status='complete', pose_count=5862,
        image_count=187584, scopes=list(overlay.SCOPES), angles=list(range(0,360,45)),
        width=512, height=512, pitch_degrees=0, projection='orthographic',
        outline_width=.65, contrast=2.6, lighten=.22, background='white',
        poses=poses, pose_records=records, renderer_revision='a'*64,
        asset_version=overlay.STYLE+'-'+'c'*12, source_library_version='lib-source')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, source, folder, raster


def test_directional_native_and_legacy_routes_share_approved_asset(assets):
    from src.directional_posters import read_directional_poster
    from src.rough_posters import rough_poster
    from src.native_png_posters import read_native_png
    from src.thumbnails import find_thumbnail
    root, source, folder, raster = assets
    image, headers = read_directional_poster(root, source, 'half', 'yaw_315')
    assert image.startswith(b'\x89PNG')
    assert headers['X-Standin-Thumbnail-Style'] == overlay.STYLE
    assert headers['X-Standin-Output-Scope'] == 'half'
    image, headers = rough_poster(root, source, 'half', 'yaw_045')
    assert headers['X-Standin-Preview-View'] == 'yaw_315'
    assert headers['X-Standin-Poster-Fallback'] == 'none'
    png, headers = read_native_png(root, source, 'bust')
    assert png.startswith(b'\x89PNG')
    assert headers['X-Standin-Artifact-SHA256'] == hashlib.sha256(png).hexdigest()
    assert find_thumbnail(str(root), 'sample', 'side') == folder / 'full__yaw_090.jpg'


@pytest.mark.parametrize('damage', ['pixels', 'missing', 'record', 'source'])
def test_advertised_overlay_never_falls_back_to_old_assets(assets, damage):
    root, source, folder, _ = assets
    if damage == 'pixels':
        (folder / 'full__yaw_000.jpg').write_bytes(b'not the approved JPEG')
    elif damage == 'missing':
        (folder / 'full__yaw_000.jpg').unlink()
    elif damage == 'record':
        (folder / 'pose.json').write_text('{}')
    else:
        source = '0'*64
    with pytest.raises((ValueError, OSError)):
        overlay.read_styled_poster(root, source, 'full', 'yaw_000')


def test_missing_requested_angle_does_not_silently_use_front(assets):
    from src.rough_posters import rough_poster
    root, source, folder, _ = assets
    (folder / 'full__yaw_315.jpg').unlink()
    with pytest.raises(ValueError, match='missing'):
        rough_poster(root, source, 'full', 'yaw_045')


def test_explicit_compatible_revision_preserves_actual_producer(assets):
    root, source, folder, _ = assets
    path = root / 'thumbs/styled/manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(renderer_revision='c'*64, renderer_revisions=['a'*64, 'c'*64])
    path.write_text(json.dumps(manifest))
    _, headers = overlay.read_styled_poster(root, source, 'full', 'yaw_000')
    assert headers['X-Standin-Preview-Revision'] == 'a'*64
    record = json.loads((folder / 'pose.json').read_text())
    record['renderer_revision'] = 'd'*64
    (folder / 'pose.json').write_text(json.dumps(record))
    manifest['pose_records'][source] = overlay._sha(folder / 'pose.json')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='lineage'):
        overlay.read_styled_poster(root, source, 'full', 'yaw_000')


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'poses.db', 'bvh/a.bvh',
                                  'thumbs/styled/../../poses.db', 'thumbs\\styled\\manifest.json'])
def test_archive_cannot_change_library_or_escape(tmp_path, name):
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        item = tarfile.TarInfo(name); item.size = 1
        tar.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        overlay._extract(archive, tmp_path / 'out')


def test_archive_rejects_link_even_inside_allowed_namespace(tmp_path):
    archive = tmp_path / 'link.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        item = tarfile.TarInfo('thumbs/styled/manifest.json')
        item.type = tarfile.SYMTYPE; item.linkname = '/etc/passwd'
        tar.addfile(item)
    with pytest.raises(ValueError):
        overlay._extract(archive, tmp_path / 'out')


def test_disabled_overlay_preserves_existing_deployments(tmp_path, monkeypatch):
    monkeypatch.delenv('POSE_THUMBNAIL_OVERLAY_URI', raising=False)
    monkeypatch.delenv('POSE_THUMBNAIL_OVERLAY_SHA256', raising=False)
    assert overlay.ensure_thumbnail_overlay(tmp_path, tmp_path/'poses.db') is False
    assert overlay.read_styled_poster(tmp_path, 'a'*64, 'full', 'yaw_000') is None


def test_partial_or_unpinned_configuration_fails_before_download(tmp_path):
    with pytest.raises(ValueError):
        overlay.ensure_thumbnail_overlay(tmp_path, tmp_path/'poses.db', 's3://bucket/file', '')


def test_expanded_library_keeps_existing_style_and_lineage(assets):
    root, source, _, _ = assets
    path = root / 'thumbs/styled/manifest.json'
    manifest = json.loads(path.read_text())
    for i in range(58):
        digest = hashlib.sha256(f'new-{i}'.encode()).hexdigest()
        manifest['poses'][f'new-{i}'] = digest
        manifest['pose_records'][digest] = 'b' * 64
    manifest.update(pose_count=5920, image_count=5920 * 32)
    path.write_text(json.dumps(manifest))
    assert overlay.manifest_for(root)['pose_count'] == 5920
    assert overlay.read_styled_poster(root, source, 'full', 'yaw_000')[1]['X-Standin-Thumbnail-Style'] == overlay.STYLE


@pytest.mark.parametrize('count,images', [(True, 32), (0, 0), (10001, 320032), (5920, 187584), (5863, 187616)])
def test_manifest_cannot_advertise_incomplete_or_unbounded_library(assets, count, images):
    root, _, _, _ = assets
    path = root / 'thumbs/styled/manifest.json'
    manifest = json.loads(path.read_text());manifest.update(pose_count=count, image_count=images)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        overlay.manifest_for(root)


def test_byte_identical_library_ids_share_one_raster_set(assets):
    import sqlite3

    root, source, folder, _ = assets
    path = root / 'thumbs/styled/manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(pose_count=2, source_count=1, image_count=overlay.IMAGES_PER_POSE,
                    poses={'sample': source, 'sample-alias': source},
                    pose_records={source: overlay._sha(folder / 'pose.json')})
    path.write_text(json.dumps(manifest))
    db = root / 'poses.db'
    bvh = root / 'bvh'
    bvh.mkdir()
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE poses (pose_id TEXT PRIMARY KEY, bvh_path TEXT)')
        for pose in manifest['poses']:
            conn.execute('INSERT INTO poses VALUES (?,?)', (pose, f'data/bvh/{pose}.bvh'))
            (bvh / f'{pose}.bvh').write_bytes(b'pose')

    parsed = overlay.manifest_for(root)
    overlay._verify_library_pose_map(root, db, parsed)
    assert overlay.legacy_thumbnail_path(root, 'sample-alias', 'front') == folder / 'full__yaw_000.jpg'

    archive = root / 'shared.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for item in [path, *folder.iterdir()]:
            tar.add(item, arcname=item.relative_to(root).as_posix())
    overlay._extract(archive, root / 'extracted')

    # Matching ID counts alone cannot authorize an alias to different bytes.
    (bvh / 'sample-alias.bvh').write_bytes(b'different pose')
    with pytest.raises(ValueError, match='BVH does not match'):
        overlay._verify_library_pose_map(root, db, parsed)

    # Legacy release manifests still require one independent source per ID.
    manifest.pop('source_count')
    manifest['image_count'] = 2 * overlay.IMAGES_PER_POSE
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inventory'):
        overlay.manifest_for(root)
