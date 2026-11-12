import copy
import hashlib
import json

import pytest

from scripts.prepared_preset_scope import select, pinned_catalog
from scripts.staging_body_assets_init import (check_budget, validate_files, install,
    PREPARED_PROFILE, RELEASE_CAP, PREPARED_CAP)
from scripts.add_prepared_bodies_to_release import build
from tests.converter.test_prepared_body import bundle, CHARACTER


def file(path, size=1, roles=None):
    return dict(path=path, size=size, roles=roles or ['converter'], sha256='a'*64)


def pair(key='a'*64, size=64*1024*1024):
    return [file(f'fbx/prepared-{key}.fbx', size), file(f'qa/prepared-{key}.json')]


def test_budget_extends_only_prepared_converter_files():
    base = [file(f'fbx/base-{n}.fbx', RELEASE_CAP//3) for n in range(3)]
    files = base + pair()
    validate_files(files)
    assert check_budget(files, PREPARED_PROFILE) == RELEASE_CAP + PREPARED_CAP
    with pytest.raises(ValueError): check_budget(files)
    for role in ('bff', 'inference'):
        with pytest.raises(ValueError): check_budget(files, PREPARED_PROFILE, role)
    with pytest.raises(ValueError): check_budget(files, 'unknown')
    with pytest.raises(ValueError): check_budget(files + [file('fbx/extra.fbx')], PREPARED_PROFILE)


@pytest.mark.parametrize('change', ['unpaired', 'too-many', 'prepared-cap', 'roles', 'extension', 'receipt-size'])
def test_budget_rejects_invalid_supplement(change):
    files = pair()
    if change == 'unpaired': files.pop()
    if change == 'too-many': files = sum((pair(f'{n:064x}', 1) for n in range(23)), [])
    if change == 'prepared-cap': files = sum((pair(f'{n:064x}') for n in range(3)), [])
    if change == 'roles': files[0]['roles'] = ['inference', 'converter']
    if change == 'extension': files[0]['path'] = files[0]['path'].replace('fbx/', 'qa/')
    if change == 'receipt-size': files[1]['size'] = 256*1024+1
    with pytest.raises(ValueError): check_budget(files, PREPARED_PROFILE)


def scope_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr('converter.prepared_body.platform.machine', lambda: 'x86_64')
    monkeypatch.setattr('converter.prepared_body.sys.platform', 'linux')
    recipe, _, meta, _, _ = bundle(tmp_path, monkeypatch)
    registry = json.loads((tmp_path/'shape-bakes.json').read_text())
    source = registry['sources'][recipe['rules']]
    catalog = dict(assets=[])
    entries = {}
    for i in range(20):
        cid = f'preset-{i}'
        registry['characters'][cid] = copy.deepcopy(registry['characters'][CHARACTER])
        catalog['assets'].append(dict(characterId=cid, role='default' if i in (0, 9) else 'preset',
            gender='male' if i < 9 else 'female', assetSha256=recipe['asset_sha256'],
            shape=dict(controls=copy.deepcopy(recipe['controls'])),
            shapeSource=dict(rules=recipe['rules'], blendSha256=source['blend_sha256'],
                             controllerSha256=source['controller_sha256'])))
        entries[cid] = recipe['prepared_neutral']
    (meta.parent.parent/'shape-bakes.local.json').write_text(json.dumps(dict(
        prepared_neutrals=dict(schema='prepared-neutral-index.v1', entries=entries))))
    release = tmp_path/'release'; release.mkdir()
    raw = json.dumps(registry).encode(); (release/'shape-bakes.json').write_bytes(raw)
    (release/'manifest.json').write_text(json.dumps(dict(schema='body-assets.v1', version='source', files=[
        dict(file('shape-bakes.json', len(raw)), sha256=hashlib.sha256(raw).hexdigest()),
        file('characters.json', 2)])))
    return registry, catalog, release, meta.parent.parent


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'source', 'asset', 'count', 'defaults'])
def test_scope_refuses_catalog_registry_disagreement(tmp_path, monkeypatch, change):
    registry, catalog, _, _ = scope_fixture(tmp_path, monkeypatch)
    if change == 'duplicate': catalog['assets'].append(catalog['assets'][0])
    if change == 'missing': registry['characters'].pop('preset-0')
    if change == 'source': catalog['assets'][0]['shapeSource']['blendSha256'] = 'b'*64
    if change == 'asset': catalog['assets'][0]['assetSha256'] = 'b'*64
    if change == 'count': catalog['assets'].pop()
    if change == 'defaults': catalog['assets'][0]['shape']['controls'][0]['default'] += 1
    with pytest.raises(ValueError): select(catalog, registry, {'male': 9, 'female': 11})


def test_catalog_pin_is_required(tmp_path):
    path = tmp_path/'catalog.json'; path.write_bytes(b'{}')
    with pytest.raises(ValueError): pinned_catalog(path, 'a'*64)


def test_release_reuses_framework_and_preserves_existing_files_and_entries(tmp_path, monkeypatch):
    registry, catalog, release, prepared = scope_fixture(tmp_path, monkeypatch)
    original = json.loads((release/'manifest.json').read_text())
    plan = build(release, prepared, 'test', tmp_path/'out', catalog=catalog, budget_profile=PREPARED_PROFILE)
    result = json.loads((tmp_path/'out/manifest.json').read_text())
    assert result['budgetProfile'] == PREPARED_PROFILE
    assert len(plan['preparedEntries']) == 20
    assert next(f for f in result['files'] if f['path']=='characters.json') == original['files'][1]
    assert json.loads((tmp_path/'out/shape-bakes.json').read_text())['characters'] == registry['characters']
    assert len([f for f in result['files'] if f['path'].startswith('fbx/prepared-')]) == 1  # identical aliases
    # Re-derive from a prepared release: preserve hashes and avoid duplicate paths.
    second = build(tmp_path/'out', prepared, 'test2', tmp_path/'out2', catalog=catalog, budget_profile=PREPARED_PROFILE)
    assert second['putNew'] == ['shape-bakes.json']
    assert len(second['preparedEntries']) == 20


def test_release_rejects_duplicate_source_paths_before_collapsing_them(tmp_path, monkeypatch):
    _, catalog, release, prepared = scope_fixture(tmp_path, monkeypatch)
    manifest = json.loads((release/'manifest.json').read_text()); manifest['files'].append(manifest['files'][0])
    (release/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='duplicate'):
        build(release, prepared, 'bad', tmp_path/'out', catalog=catalog, budget_profile=PREPARED_PROFILE)


def test_installer_corruption_preserves_previous_release_and_rollback(tmp_path):
    key = 'body-models/releases/test/presets-aaaaaaaaaaaa/manifest.json'
    data = {'characters.json': b'{}', 'shape-bakes.json': b'old'}
    env = dict(BODY_ASSETS_ROLE='converter', BODY_ASSETS_MANIFEST_KEY=key, BODY_ASSETS_ROOT=str(tmp_path/'installed'))
    def run(payload, corrupt=False):
        raw = json.dumps(dict(schema='body-assets.v1', version='test', files=[
            dict(file(p, len(v)), sha256=hashlib.sha256(v).hexdigest()) for p,v in payload.items()])).encode()
        env['BODY_ASSETS_MANIFEST_SHA256'] = hashlib.sha256(raw).hexdigest()
        return install(lambda k, _: raw if k==key else (b'corrupt' if corrupt else payload[k.split('/presets-aaaaaaaaaaaa/')[1]]), env)
    run(data)
    before = {p.relative_to(tmp_path/'installed'): p.read_bytes() for p in (tmp_path/'installed').rglob('*') if p.is_file()}
    with pytest.raises(ValueError): run({'characters.json': b'new'}, corrupt=True)
    assert before == {p.relative_to(tmp_path/'installed'): p.read_bytes() for p in (tmp_path/'installed').rglob('*') if p.is_file()}
    run({**data, 'shape-bakes.json': b'new'})
    run(data)
    assert (tmp_path/'installed/shape-bakes.json').read_bytes() == b'old'
