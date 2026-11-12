"""Prepared data never changes body identity or bypasses final pose validation."""
import hashlib
import json
from pathlib import Path

import pytest
from converter import prepared_body as p
from converter.shape_spec import resolve_shape
from tests.converter.test_shape_bake_contract import write_registry, CHARACTER, envelope


def bundle(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    recipe = resolve_shape(CHARACTER, 'a' * 64)
    ident = p.identity(recipe)
    key = p.body_key(ident)
    root = tmp_path / 'bundle'
    fbx = root / 'fbx' / ('prepared-' + key + '.fbx')
    fbx.parent.mkdir(parents=True)
    data = b'Kaydara FBX Binary  \x00\x1a\x00fixture'
    fbx.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    receipt = {'identity': ident, 'key': key, 'fbx_sha256': digest, 'fbx_bytes': len(data),
               'rest_frames': {'Hips': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
               'neutral_validation': {'version': 'final-fbx-v1', 'artifact_sha256': digest,
                 'scope': 'full', 'bones': 1, 'meshes': 1, 'vertices': 3, 'polygons': 1, 'skin_verified': True}}
    meta = root / 'qa' / ('prepared-' + key + '.json')
    meta.parent.mkdir()
    def publish():
        meta.write_text(json.dumps(receipt))
        recipe['prepared_neutral'] = {'path': str(meta), 'sha256': hashlib.sha256(meta.read_bytes()).hexdigest()}
    publish()
    return recipe, fbx, meta, receipt, publish


def test_exact_preset_gets_private_copy_and_verified_rest_frames(tmp_path, monkeypatch):
    recipe, fbx, _, receipt, _ = bundle(tmp_path, monkeypatch)
    result, status = p.restore(recipe, tmp_path / 'job')
    assert status == 'prepared'
    assert result['rest_frames'] == receipt['rest_frames']
    assert result['path'].read_bytes() == fbx.read_bytes()
    result['path'].write_bytes(b'private solver change')
    assert fbx.read_bytes().startswith(b'Kaydara')


def test_custom_shapes_and_disabled_flag_never_read_or_store_prototypes(tmp_path, monkeypatch):
    recipe, _, _, _, _ = bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(p, '_read', lambda *_: pytest.fail('custom settings must not read shared data'))
    recipe['values']['muscle'] = 1
    assert p.restore(recipe, tmp_path / 'job') == (None, 'custom_shape')
    assert not (tmp_path / 'job').exists()
    monkeypatch.setenv('CONVERTER_PREPARED_PRESETS_ENABLED', '0')
    assert p.restore(recipe, tmp_path / 'job') == (None, 'disabled')


def test_explicit_defaults_and_implicit_defaults_share_prototype_identity(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    a = resolve_shape(CHARACTER, 'a' * 64)
    b = resolve_shape(CHARACTER, 'a' * 64, envelope({}))
    assert a['shape_sha256'] is None and b['shape_sha256'] is not None
    assert p.identity(a) == p.identity(b)


def test_registry_is_the_only_source_of_prepared_paths(tmp_path, monkeypatch):
    recipe, _, meta, _, _ = bundle(tmp_path, monkeypatch)
    registry_path = tmp_path / 'shape-bakes.json'
    registry = json.loads(registry_path.read_text())
    registry['prepared_neutrals'] = {'schema': 'prepared-neutral-index.v1',
                                    'entries': {CHARACTER: recipe['prepared_neutral']}}
    registry_path.write_text(json.dumps(registry))
    assert resolve_shape(CHARACTER, 'a' * 64)['prepared_neutral']['path'] == str(meta)
    with pytest.raises(ValueError):
        resolve_shape(CHARACTER, 'a' * 64, {**envelope({}), 'prepared_neutral': recipe['prepared_neutral']})


@pytest.mark.parametrize('change', ['receipt', 'fbx', 'absent', 'symlink', 'oversized', 'source'])
def test_bad_data_falls_back_without_granting_a_substitute_body(tmp_path, monkeypatch, change):
    recipe, fbx, meta, _, _ = bundle(tmp_path, monkeypatch)
    if change == 'receipt': meta.write_text('{}')
    if change == 'fbx': fbx.write_bytes(b'changed')
    if change == 'absent': meta.unlink()
    if change == 'symlink':
        other = tmp_path / 'other.fbx'; fbx.rename(other); fbx.symlink_to(other)
    if change == 'oversized': monkeypatch.setattr(p, 'MAX_FBX_BYTES', 4)
    if change == 'source': Path(recipe['controller']).write_text('changed authoring rules')
    assert p.restore(recipe, tmp_path / 'job') == (None, 'invalid_prepared')
    assert not (tmp_path / 'job').exists()


@pytest.mark.parametrize('field', ['rules', 'preset', 'asset_sha256', 'blend_sha256', 'controller_sha256', 'runtime', 'values'])
def test_re_signed_but_wrong_identity_is_not_used(tmp_path, monkeypatch, field):
    recipe, _, _, receipt, publish = bundle(tmp_path, monkeypatch)
    receipt['identity'][field] = 'different'
    receipt['key'] = p.body_key(receipt['identity'])
    publish()
    assert p.restore(recipe, tmp_path / 'job') == (None, 'invalid_prepared')


@pytest.mark.parametrize('invalid', [[], {'Hips': []}, {'Hips': [[0]*4]*4},
    {'Hips': [[float('nan'),0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}])
def test_invalid_rest_frames_do_not_reach_the_pose_solver(tmp_path, monkeypatch, invalid):
    recipe, _, _, receipt, publish = bundle(tmp_path, monkeypatch)
    receipt['rest_frames'] = invalid; publish()
    assert p.restore(recipe, tmp_path / 'job') == (None, 'invalid_prepared')


def test_old_registry_without_preparation_keeps_original_behavior(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    recipe = resolve_shape(CHARACTER, 'a' * 64)
    assert 'prepared_neutral' not in recipe
    assert p.restore(recipe, tmp_path / 'job') == (None, 'not_prepared')


def test_packaging_preserves_verified_bytes_and_rewrites_only_operator_paths(tmp_path, monkeypatch):
    from scripts.prepare_preset_bodies import stage_prepared
    recipe, fbx, meta, _, _ = bundle(tmp_path, monkeypatch)
    registry = json.loads((tmp_path / 'shape-bakes.json').read_text())
    prepared_registry = {**registry, 'prepared_neutrals': {'schema': 'prepared-neutral-index.v1',
                          'entries': {CHARACTER: recipe['prepared_neutral']}}}
    (meta.parent.parent / 'shape-bakes.local.json').write_text(json.dumps(prepared_registry))
    output = tmp_path / 'package'
    entries = stage_prepared(meta.parent.parent, registry, output, '/app/body-staging')
    assert entries[CHARACTER]['path'] == '/app/body-staging/qa/' + meta.name
    assert entries[CHARACTER]['sha256'] == recipe['prepared_neutral']['sha256']
    assert (output / 'fbx' / fbx.name).read_bytes() == fbx.read_bytes()
    assert (output / 'qa' / meta.name).read_bytes() == meta.read_bytes()
    assert registry['characters'][CHARACTER]['asset_sha256'] == recipe['asset_sha256']


def test_packaging_refuses_a_different_model_release(tmp_path, monkeypatch):
    from scripts.prepare_preset_bodies import stage_prepared
    recipe, _, meta, _, _ = bundle(tmp_path, monkeypatch)
    registry = json.loads((tmp_path / 'shape-bakes.json').read_text())
    prepared_registry = {**registry, 'prepared_neutrals': {'schema': 'prepared-neutral-index.v1',
                          'entries': {CHARACTER: recipe['prepared_neutral']}}}
    (meta.parent.parent / 'shape-bakes.local.json').write_text(json.dumps(prepared_registry))
    registry['characters'][CHARACTER]['asset_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='identity'):
        stage_prepared(meta.parent.parent, registry, tmp_path / 'package', '/app/body-staging')


def test_runtime_refuses_a_bundle_built_for_another_platform(tmp_path, monkeypatch):
    recipe, _, _, receipt, publish = bundle(tmp_path, monkeypatch)
    receipt['identity']['runtime'][2] = 'different-platform'
    receipt['key'] = p.body_key(receipt['identity']); publish()
    assert p.restore(recipe, tmp_path / 'job') == (None, 'invalid_prepared')


def test_packaging_can_relocate_a_container_built_bundle(tmp_path, monkeypatch):
    from scripts.prepare_preset_bodies import stage_prepared
    recipe, fbx, meta, _, _ = bundle(tmp_path, monkeypatch)
    registry = json.loads((tmp_path / 'shape-bakes.json').read_text())
    recorded = {**recipe['prepared_neutral'], 'path': '/build-container/qa/' + meta.name}
    (meta.parent.parent / 'shape-bakes.local.json').write_text(json.dumps({**registry,
        'prepared_neutrals': {'schema': 'prepared-neutral-index.v1', 'entries': {CHARACTER: recorded}}}))
    output = tmp_path / 'package'
    assert stage_prepared(meta.parent.parent, registry, output, '/app/body-staging')
    assert (output / 'fbx' / fbx.name).read_bytes() == fbx.read_bytes()


def test_prepared_paths_install_with_the_existing_body_asset_allowlist(tmp_path, monkeypatch):
    from scripts.prepare_preset_bodies import stage_prepared
    from scripts.staging_body_assets_init import install
    recipe, _, meta, _, _ = bundle(tmp_path, monkeypatch)
    registry = json.loads((tmp_path / 'shape-bakes.json').read_text())
    (meta.parent.parent / 'shape-bakes.local.json').write_text(json.dumps({**registry,
        'prepared_neutrals': {'schema': 'prepared-neutral-index.v1',
                             'entries': {CHARACTER: recipe['prepared_neutral']}}}))
    package = tmp_path / 'package'; target = tmp_path / 'installed'
    stage_prepared(meta.parent.parent, registry, package, str(target))
    (package / 'shape-bakes.json').write_text(json.dumps(registry))
    (package / 'characters.json').write_text('{"schema_version":1,"characters":{}}')
    entries = []
    for file in package.rglob('*'):
        if file.is_file(): entries.append({'path': file.relative_to(package).as_posix(),
            'size': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'roles': ['converter']})
    manifest = json.dumps({'schema': 'body-assets.v1', 'version': 'prepared-fixture', 'files': entries}).encode()
    prefix = 'body-models/releases/prepared-fixture'
    def fetch(key, _limit):
        return manifest if key == prefix + '/manifest.json' else (package / key.removeprefix(prefix + '/')).read_bytes()
    report = install(fetch, {'BODY_ASSETS_ROLE':'converter', 'BODY_ASSETS_MANIFEST_KEY':prefix+'/manifest.json',
        'BODY_ASSETS_MANIFEST_SHA256':hashlib.sha256(manifest).hexdigest(), 'BODY_ASSETS_ROOT':str(target)})
    assert report['files'] == 4
    assert json.loads((target / 'shape-bakes.json').read_text())['prepared_neutrals']['entries'][CHARACTER]['path'].startswith(str(target))


@pytest.mark.parametrize('options', [[], ['--rules','unregistered-rules'], ['--character','unknown'], ['--character',CHARACTER,'--character',CHARACTER]])
def test_builder_refuses_implicit_or_invalid_preparation_scope(tmp_path, monkeypatch, options):
    from scripts.prepare_preset_bodies import main
    write_registry(tmp_path, monkeypatch)
    output=tmp_path/'offline-output'
    with pytest.raises(SystemExit):
        main(['--registry',str(tmp_path/'shape-bakes.json'),'--out',str(output),*options])
    assert not output.exists()


def test_cross_platform_packaging_retains_the_build_platform_binding(tmp_path, monkeypatch):
    from scripts.prepare_preset_bodies import stage_prepared
    recipe, fbx, meta, receipt, _ = bundle(tmp_path, monkeypatch)
    receipt['identity']['runtime'][2:4] = ['linux','x86_64']
    receipt['key'] = p.body_key(receipt['identity'])
    new_meta=meta.parent/('prepared-'+receipt['key']+'.json')
    new_fbx=fbx.parent/('prepared-'+receipt['key']+'.fbx')
    new_fbx.write_bytes(fbx.read_bytes());new_meta.write_text(json.dumps(receipt))
    entry={'path':str(new_meta),'sha256':hashlib.sha256(new_meta.read_bytes()).hexdigest()}
    registry=json.loads((tmp_path/'shape-bakes.json').read_text())
    (meta.parent.parent/'shape-bakes.local.json').write_text(json.dumps({**registry,
        'prepared_neutrals':{'schema':'prepared-neutral-index.v1','entries':{CHARACTER:entry}}}))
    output=tmp_path/'package'
    stage_prepared(meta.parent.parent,registry,output,'/app/body-staging')
    packaged=json.loads((output/'qa'/new_meta.name).read_text())
    assert packaged['identity']['runtime'][2:4] == ['linux','x86_64']
