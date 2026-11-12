"""Current anatomy targets, shape-compatible topology and staging-only activation."""
import hashlib
import json
from pathlib import Path

from converter import skin_spec, outline_spec
from scripts.prepare_staging_anatomy import RELEASE
from scripts.staging_body_assets_init import install


def test_current_targets_are_exact_and_do_not_inherit_old_guide_or_crop_acceptance():
    models = skin_spec.manifest()['models']
    for cid in RELEASE['characterIds']:
        row = models[cid]
        assert row['bones'] == 54 and row['vertices'] > 100000
        assert set(row['textures']) == {'flesh'}
        assert skin_spec.certified(row['character_sha256'], 'full', dict(version='fbx-skin-v1', id='flesh'))
        assert not skin_spec.certified(row['character_sha256'], 'full', dict(version='fbx-skin-v1', id='muscle'))
        assert not skin_spec.certified(row['character_sha256'], 'head', dict(version='fbx-skin-v1', id='flesh'))
        for width in (.5, 1.2, 2):
            assert outline_spec.certified(row['character_sha256'], 'full', width)
        assert not outline_spec.certified(row['character_sha256'], 'head', 2)
        assert not outline_spec.certified('f'*64, 'full', 2)


def test_pinned_skin_assets_install_to_private_volume_with_checked_bytes(tmp_path):
    blobs = {'characters.json': b'{"characters":{}}', 'skin/flesh.png': b'bounded-test-image', 'skin/body-uv.json.gz': b'bounded-test-uv'}
    manifest = json.dumps(dict(schema='body-assets.v1', version='test', files=[
        dict(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest(), roles=['converter'])
        for path, data in blobs.items()])).encode()
    key = RELEASE['prefix'] + '/manifest.json'
    env = dict(BODY_ASSETS_ROLE='converter', BODY_ASSETS_ROOT=str(tmp_path),
               BODY_ASSETS_MANIFEST_KEY=key, BODY_ASSETS_MANIFEST_SHA256=hashlib.sha256(manifest).hexdigest())
    install(lambda request, limit: manifest if request == key else blobs[request[len(RELEASE['prefix'])+1:]], env)
    assert all((tmp_path / path).read_bytes() == data for path, data in blobs.items())
