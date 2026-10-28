"""shape-bake.v2 registry boundary. Real Blender baking is verified by scripts/verify_shape_bake.py."""
import hashlib
import json

import pytest

from converter.shape_spec import ShapeError, digest_shape, load_registry, resolve_shape

CHARACTER = 'standin-master-v2'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_registry(tmp_path, monkeypatch, *, rules='standin-body-9.9.9', version=None, controls=None,
                   defaults=None, asset_sha='a' * 64, schema='shape-bake.v2', character=CHARACTER, tamper=None):
    blend = tmp_path / 'source.blend'
    blend.write_bytes(b'BLENDER-fixture')
    controller = tmp_path / 'standin_body.py'
    controller.write_text(f"bl_info={{}}\nVERSION='{version or rules}'\nRIG_VERSION='x'\n", encoding='utf-8')
    source = {'blend': str(blend), 'controller': str(controller),
              'blend_sha256': _sha(blend), 'controller_sha256': _sha(controller)}
    recipe = {'rules': rules, 'preset': 'BASE', 'asset': 'asset-1', 'asset_sha256': asset_sha,
              'defaults': defaults or {'muscle': 0.0, 'head_scale': 0.927, 'height_cm': 178.0},
              'controls': controls or [
                  {'key': 'muscle', 'min': 0, 'max': 2, 'step': .01, 'default': 0},
                  {'key': 'head_scale', 'min': .907, 'max': 1, 'step': .01, 'default': .927},
                  {'key': 'height', 'min': 160, 'max': 195, 'step': 1, 'default': 178},
              ]}
    registry = {'schema': schema, 'sources': {rules: source}, 'characters': {character: recipe}}
    if tamper:
        tamper(registry)
    path = tmp_path / 'shape-bakes.json'
    path.write_text(json.dumps(registry), encoding='utf-8')
    monkeypatch.setenv('CONVERTER_SHAPE_REGISTRY', str(path))
    return registry


DEFAULTS = {'muscle': 0, 'head_scale': .927, 'height': 178}


def envelope(controls, character=CHARACTER, asset_sha='a' * 64):
    # The BFF sends every control (missing keys = default) and digests that full set.
    full = {**DEFAULTS, **controls}
    return {'schemaVersion': 'body-shape.v1', 'controls': full,
            'shapeSha256': digest_shape(character, asset_sha, full)}


def test_digest_matches_javascript():
    assert digest_shape('body', 'a' * 64, {'muscle': 1.0}) == hashlib.sha256(
        ('[' + '"body-shape.v1","body","' + 'a' * 64 + '",[["muscle",1]]]').encode()).hexdigest()


def test_any_approved_version_is_data_not_code(tmp_path, monkeypatch):
    # A version this code has never heard of works once the registry approves it.
    write_registry(tmp_path, monkeypatch, rules='standin-body-0.7.1')
    recipe = resolve_shape(CHARACTER, 'a' * 64, envelope({'muscle': 1.5}))
    assert recipe['rules'] == 'standin-body-0.7.1'
    assert recipe['values'] == {'muscle': 1.5, 'head_scale': .927, 'height_cm': 178}
    assert recipe['blend'].endswith('source.blend')
    assert recipe['shape_sha256'] == envelope({'muscle': 1.5})['shapeSha256']


def test_breast_extension_uses_registered_range_without_changing_legacy(tmp_path, monkeypatch):
    controls = [{'key': 'breast_size', 'min': -1, 'max': 2, 'step': .01, 'default': 0}]
    write_registry(tmp_path, monkeypatch, rules='standin-female-body-0.3.4.breast6',
                   controls=controls, defaults={'breast_size': 0})
    def request(value):
        values = {'breast_size': value}
        return {'schemaVersion': 'body-shape.v1', 'controls': values,
                'shapeSha256': digest_shape(CHARACTER, 'a' * 64, values)}
    for value in [-1, 0, 1, 1.5, 2]:
        assert resolve_shape(CHARACTER, 'a' * 64, request(value))['values']['breast_size'] == value
    for value in [-1.01, 2.01, 1.005]:
        with pytest.raises(ShapeError):
            resolve_shape(CHARACTER, 'a' * 64, request(value))
    write_registry(tmp_path, monkeypatch, rules='standin-female-body-0.3.4',
                   controls=[{**controls[0], 'max': 1}], defaults={'breast_size': 0})
    with pytest.raises(ShapeError, match='out of range'):
        resolve_shape(CHARACTER, 'a' * 64, request(2))


def test_default_shape_without_request_has_no_digest(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    recipe = resolve_shape(CHARACTER, 'a' * 64)
    assert recipe['shape_sha256'] is None and recipe['values']['muscle'] == 0


def test_controller_version_must_match_rules(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch, rules='standin-body-0.7.1', version='standin-body-0.7.0')
    with pytest.raises(ShapeError, match='VERSION'):
        load_registry()


def test_changed_source_bytes_close_the_registry(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    (tmp_path / 'source.blend').write_bytes(b'edited')
    with pytest.raises(ShapeError, match='checksum'):
        load_registry()


def test_old_v1_registry_is_rejected(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch, schema='shape-bake.v1')
    with pytest.raises(ShapeError):
        load_registry()


def test_recipe_must_reference_an_approved_source(tmp_path, monkeypatch):
    def other(r):
        r['characters'][CHARACTER]['rules'] = 'standin-body-0.0.1'
    write_registry(tmp_path, monkeypatch, tamper=other)
    with pytest.raises(ShapeError, match='unapproved'):
        load_registry()


def test_preset_default_off_its_grid_closes_registry(tmp_path, monkeypatch):
    # head_scale .927 on a grid anchored at .90 with step .01 would reject every request.
    write_registry(tmp_path, monkeypatch, controls=[{'key': 'head_scale', 'min': .9, 'max': 1, 'step': .01, 'default': .927}])
    with pytest.raises(ShapeError, match='off step'):
        load_registry()


def test_expected_source_must_match(tmp_path, monkeypatch):
    registry = write_registry(tmp_path, monkeypatch, rules='standin-body-0.7.1')
    source = registry['sources']['standin-body-0.7.1']
    ok = {'rules': 'standin-body-0.7.1', 'blendSha256': source['blend_sha256'], 'controllerSha256': source['controller_sha256']}
    assert resolve_shape(CHARACTER, 'a' * 64, envelope({'muscle': 1}), ok)['rules'] == 'standin-body-0.7.1'
    for change in ({'rules': 'standin-body-0.7.0'}, {'blendSha256': 'b' * 64}, {'controllerSha256': 'c' * 64}):
        with pytest.raises(ShapeError, match='shape source mismatch'):
            resolve_shape(CHARACTER, 'a' * 64, envelope({'muscle': 1}), {**ok, **change})


@pytest.mark.parametrize('value', [True, '1', float('nan'), float('inf'), -.1, 2.1, .005])
def test_invalid_values_fail_closed(tmp_path, monkeypatch, value):
    write_registry(tmp_path, monkeypatch)
    with pytest.raises(ShapeError):
        resolve_shape(CHARACTER, 'a' * 64, {'schemaVersion': 'body-shape.v1', 'controls': {'muscle': value}, 'shapeSha256': 'b' * 64})


def test_values_on_a_shifted_grid_are_accepted(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    assert resolve_shape(CHARACTER, 'a' * 64, envelope({'head_scale': .957}))['values']['head_scale'] == .957
    with pytest.raises(ShapeError, match='off step'):
        resolve_shape(CHARACTER, 'a' * 64, envelope({'head_scale': .95}))


@pytest.mark.parametrize('change', [{'controls': {'unknown': 0}}, {'shapeSha256': 'b' * 64}, {'schemaVersion': 'body-shape.v0'}])
def test_stale_or_unknown_shape_rejected(tmp_path, monkeypatch, change):
    write_registry(tmp_path, monkeypatch)
    with pytest.raises(ShapeError):
        resolve_shape(CHARACTER, 'a' * 64, {**envelope({'muscle': 1}), **change})


def test_asset_or_capability_mismatch_rejected(tmp_path, monkeypatch):
    write_registry(tmp_path, monkeypatch)
    with pytest.raises(ShapeError):
        resolve_shape(CHARACTER, 'b' * 64)
    with pytest.raises(ShapeError):
        resolve_shape('unregistered', 'a' * 64, envelope({'muscle': 1}, 'unregistered'))
    assert resolve_shape('unregistered', 'a' * 64) is None


def test_no_registry_means_no_baking(monkeypatch):
    monkeypatch.delenv('CONVERTER_SHAPE_REGISTRY', raising=False)
    assert resolve_shape(CHARACTER, 'a' * 64) is None
    with pytest.raises(ShapeError, match='unavailable'):
        resolve_shape(CHARACTER, 'a' * 64, envelope({'muscle': 1}))


def test_missing_shape_source_closes_health(tmp_path, monkeypatch):
    from tests.converter.test_api_contract import _client
    monkeypatch.setenv('CONVERTER_SHAPE_REGISTRY', str(tmp_path / 'missing.json'))
    response = _client(tmp_path).get('/healthz')
    assert response.status_code == 503
    assert response.json()['checks']['shape_sources']['ok'] is False


def test_health_lists_approved_versions(tmp_path, monkeypatch):
    from tests.converter.test_api_contract import _client
    client = _client(tmp_path)
    write_registry(tmp_path, monkeypatch, rules='standin-female-body-0.3.2')
    check = client.get('/healthz').json()['checks']['shape_sources']
    assert check == {'ok': True, 'schema': 'shape-bake.v2', 'models': 1, 'rules': ['standin-female-body-0.3.2']}


def _framed(client, **fields):
    from tests.converter.test_api_contract import VALID_BVH
    return client.post('/convert-framed', files={'bvh': ('final.bvh', VALID_BVH, 'application/octet-stream')},
                       data={'output_scope': 'full', 'preview_view': 'front',
                             'expected_bvh_sha256': hashlib.sha256(VALID_BVH).hexdigest(), **fields})


def test_api_refuses_another_source_before_blender(tmp_path, monkeypatch):
    from tests.converter.test_api_contract import FakeRunner, _client
    runner = FakeRunner()
    client = _client(tmp_path, runner=runner)
    asset_sha = hashlib.sha256(b'character').hexdigest()
    registry = write_registry(tmp_path, monkeypatch, rules='standin-body-0.7.1', asset_sha=asset_sha)
    source = registry['sources']['standin-body-0.7.1']
    expected = {'rules': 'standin-body-0.7.0', 'blendSha256': source['blend_sha256'], 'controllerSha256': source['controller_sha256']}
    response = _framed(client, shape=json.dumps(envelope({'muscle': 1}, asset_sha=asset_sha)),
                       expected_shape_source=json.dumps(expected))
    assert response.status_code == 409, response.text
    assert response.json()['error']['code'] == 'SHAPE_SOURCE_MISMATCH'
    assert runner.calls == []


def test_api_expected_source_needs_a_shape(tmp_path, monkeypatch):
    from tests.converter.test_api_contract import FakeRunner, _client
    runner = FakeRunner()
    write_registry(tmp_path, monkeypatch)
    response = _framed(_client(tmp_path, runner=runner), expected_shape_source='{}')
    assert response.status_code == 400 and runner.calls == []


def test_shape_does_not_claim_exact_parent_mode():
    from dataclasses import replace
    from pathlib import Path
    from converter_api.runner import BlenderRunner, ConversionRejectedError, RunnerSettings
    runner = BlenderRunner(replace(RunnerSettings.from_env(), force_exact_v324=True))
    with pytest.raises(ConversionRejectedError, match='exact V3.2.4'):
        runner.convert(bvh_bytes=b'x', character_path=Path('/unused'), character_id='x', character_sha256='a' * 64,
                       conversion_id='test', shape_recipe={})


def test_shaped_jobs_get_their_own_timeout(monkeypatch):
    from converter_api.runner import RunnerSettings
    monkeypatch.setenv('CONVERTER_TIMEOUT_SECONDS', '30')
    assert RunnerSettings.from_env().shape_timeout_seconds == 300
    monkeypatch.setenv('CONVERTER_SHAPE_TIMEOUT_SECONDS', '420')
    assert RunnerSettings.from_env().shape_timeout_seconds == 420
