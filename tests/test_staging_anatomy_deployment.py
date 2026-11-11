import copy
import hashlib
import json
import pytest
from scripts.verify_staging_anatomy import verify
from scripts.prepare_staging_anatomy import prepare, RELEASE
from scripts.staging_body_assets_init import install


def fixture():
    def env(values):
        return [{"name": k, "value": v} for k, v in values.items()]
    return prepare({"family": "StandinStagingApp-ConverterTask", "containerDefinitions": [
        {"name": "converter", "dependsOn": [{"containerName": "body-assets-init", "condition": "SUCCESS"}], "environment": env({
            "CONVERTER_CHARACTER_REGISTRY": "/app/body-staging/characters.json",
            "CONVERTER_SHAPE_REGISTRY": "/app/body-staging/shape-bakes.json",
            "BODY_MALE_BASE_V0_8_0_URI": "/app/body-staging/fbx/male-base-v0-8-0.fbx",
            "BODY_FEMALE_BASE_V0_4_0_URI": "/app/body-staging/fbx/female-base-v0-4-0.fbx",
        })},
        {"name": "body-assets-init", "environment": env({
            "BODY_ASSETS_ROLE": "converter", "BODY_ASSETS_ROOT": "/app/body-staging",
            "BODY_ASSETS_MANIFEST_KEY": "body-models/releases/anatomy-20261111-bf9439491bef-converter/manifest.json",
            "BODY_ASSETS_MANIFEST_SHA256": "bf9439491befc529b1a7b1072bfa46b2fb91f2d3eef93ed36b52f847a0534c3c",
        })},
    ]})


def test_approved_release_is_not_mutated():
    task = fixture()
    before = copy.deepcopy(task)
    assert verify(task) is task
    assert task == before


def test_restore_all_twenty_presets_preserves_other_task_settings():
    task = fixture(); task['containerDefinitions'][0]['image'] = 'reviewed-image'
    task['containerDefinitions'][0]['environment'] = []
    before = copy.deepcopy(task)
    result = prepare(task)
    assert result['containerDefinitions'][0]['image'] == 'reviewed-image'
    assert task == before and len(RELEASE['characterIds']) == 20
    assert verify(result) is result
    task['family'] = 'StandinApp-ConverterTask'
    with pytest.raises(ValueError): prepare(task)


@pytest.mark.parametrize("case", ["production", "old_manifest", "missing_female", "dependency"])
def test_reject_release_regression(case):
    task = fixture()
    if case == "production":
        task["family"] = "StandinApp-ConverterTask"
    elif case == "old_manifest":
        task["containerDefinitions"][1]["environment"][2]["value"] = "body-models/old/manifest.json"
    elif case == "missing_female":
        task["containerDefinitions"][0]["environment"].pop()
    else:
        task["containerDefinitions"][0]["dependsOn"] = []
    with pytest.raises(ValueError):
        verify(task)


def test_initializer_rejects_corruption_without_replacing_existing_catalog(tmp_path):
    target = tmp_path / 'characters.json'; target.write_bytes(b'previous')
    good = b'{"characters":{}}'
    manifest = json.dumps({'schema': 'body-assets.v1', 'version': 'test', 'files': [
        {'path': 'characters.json', 'roles': ['converter'], 'size': len(good), 'sha256': hashlib.sha256(good).hexdigest()},
    ]}).encode()
    env = {'BODY_ASSETS_ROLE': 'converter', 'BODY_ASSETS_ROOT': str(tmp_path),
           'BODY_ASSETS_MANIFEST_KEY': RELEASE['prefix'] + '/manifest.json',
           'BODY_ASSETS_MANIFEST_SHA256': hashlib.sha256(manifest).hexdigest()}
    def fetch(key, limit): return manifest if key.endswith('/manifest.json') else b'x' * len(good)
    with pytest.raises(ValueError, match='checksum'): install(fetch, env)
    assert target.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.body-assets-*'))


def test_initializer_keeps_individual_asset_limit_and_rejects_path_escape(tmp_path):
    for path, size in [('fbx/../characters.json', 1), ('fbx/huge.fbx', 300 * 1024 * 1024 + 1)]:
        manifest = json.dumps({'schema': 'body-assets.v1', 'version': 'test', 'files': [
            {'path': path, 'roles': ['converter'], 'size': size, 'sha256': 'a' * 64},
        ]}).encode()
        env = {'BODY_ASSETS_ROLE': 'converter', 'BODY_ASSETS_ROOT': str(tmp_path), 'BODY_ASSETS_MANIFEST_KEY': RELEASE['prefix'] + '/manifest.json', 'BODY_ASSETS_MANIFEST_SHA256': hashlib.sha256(manifest).hexdigest()}
        with pytest.raises(ValueError): install(lambda key, limit: manifest, env)
