"""Deriving a body release with prepared neutrals keeps every existing object exact."""
import hashlib
import json
from pathlib import Path

import pytest

import scripts.add_prepared_bodies_to_release as tool


def sha(data):
    return hashlib.sha256(data).hexdigest()


def release(tmp_path, *, registry=None, extra_size=None):
    root = tmp_path/'release'; root.mkdir()
    registry = registry or {'schema': 'shape-bake.v2', 'sources': {}, 'characters': {'male-base-v0-8-0': {}}}
    raw = (json.dumps(registry, indent=2) + '\n').encode()
    (root/'shape-bakes.json').write_bytes(raw)
    files = [{'path': 'characters.json', 'roles': ['converter'], 'size': 10, 'sha256': 'a'*64},
             {'path': 'fbx/male-base-v0-8-0.fbx', 'roles': ['converter'], 'size': extra_size or 100, 'sha256': 'b'*64},
             {'path': 'shape-bakes.json', 'roles': ['converter'], 'size': len(raw), 'sha256': sha(raw)}]
    (root/'manifest.json').write_text(json.dumps({'schema': 'body-assets.v1', 'version': 'old', 'files': files}))
    return root


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Stand-in for stage_prepared: real one needs Blender-built receipts and source files."""
    key = 'c'*64
    receipt = json.dumps({'identity': {'runtime': ['5.2.0', 'fbe6228777e7', 'linux', 'x86_64', 'd'*64]}}).encode()
    prepared = tmp_path/'prepared'; (prepared/'qa').mkdir(parents=True)
    path = prepared/'qa'/f'prepared-{key}.json'; path.write_bytes(receipt)
    (prepared/'shape-bakes.local.json').write_text(json.dumps({'prepared_neutrals': {
        'schema': 'prepared-neutral-index.v1', 'entries': {
            'male-base-v0-8-0': {'path': str(path), 'sha256': sha(receipt)}}}}))
    def stage(bundle, registry, package, deploy_root):
        for folder, ext, data in [('fbx', '.fbx', b'Kaydara FBX'), ('qa', '.json', receipt)]:
            (Path(package)/folder).mkdir(parents=True, exist_ok=True)
            (Path(package)/folder/f'prepared-{key}{ext}').write_bytes(data)
        entries = {'male-base-v0-8-0': {'path': deploy_root + f'qa/prepared-{key}.json', 'sha256': sha(receipt)}}
        registry['prepared_neutrals'] = {'schema': 'prepared-neutral-index.v1', 'entries': entries}
        return entries
    monkeypatch.setattr(tool, 'stage_prepared', stage)


def test_only_registry_changes_and_prepared_files_are_added(tmp_path, staged):
    source = release(tmp_path)
    old = {f['path']: f for f in json.loads((source/'manifest.json').read_text())['files']}
    plan = tool.build(source, tmp_path/'prepared', 'new-v1', tmp_path/'out')
    new = {f['path']: f for f in json.loads((tmp_path/'out/manifest.json').read_text())['files']}
    assert set(new) - set(old) == {'fbx/prepared-' + 'c'*64 + '.fbx', 'qa/prepared-' + 'c'*64 + '.json'}
    assert all(new[p] == old[p] for p in old if p != 'shape-bakes.json')
    registry = json.loads((tmp_path/'out/shape-bakes.json').read_text())
    assert registry['prepared_neutrals']['entries']['male-base-v0-8-0']['path'].startswith('/app/body-staging/qa/')
    assert new['shape-bakes.json']['sha256'] == sha((tmp_path/'out/shape-bakes.json').read_bytes())
    assert plan['copyUnchanged'] == ['characters.json', 'fbx/male-base-v0-8-0.fbx']
    assert plan['manifestSha256'] == sha((tmp_path/'out/manifest.json').read_bytes())
    assert [f['path'] for f in json.loads((tmp_path/'out/manifest.json').read_text())['files']] == sorted(new)


def test_release_above_initializer_cap_is_refused(tmp_path, staged):
    source = release(tmp_path, extra_size=200*1024*1024)
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['files'].extend({'path': f'fbx/extra-{i}.fbx', 'size': 200*1024*1024,
                              'sha256': 'd'*64, 'roles': ['converter']} for i in range(3))
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match='initializer'):
        tool.build(source, tmp_path/'prepared', 'new-v1', tmp_path/'out')


def test_already_prepared_or_tampered_registry_is_refused(tmp_path, staged):
    registry = {'schema': 'shape-bake.v2', 'sources': {}, 'characters': {}, 'prepared_neutrals': {}}
    with pytest.raises(SystemExit, match='already has prepared'):
        tool.build(release(tmp_path, registry=registry), tmp_path/'prepared', 'v', tmp_path/'out')
    other = tmp_path/'other'; other.mkdir()
    source = release(other)
    (source/'shape-bakes.json').write_text('{}')
    with pytest.raises(SystemExit, match='pinned release copy'):
        tool.build(source, tmp_path/'prepared', 'v', tmp_path/'out2')


def test_extended_profile_requires_an_explicit_manifest_approval_pin(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['add_prepared_bodies_to_release.py', '--release', str(tmp_path/'source'),
        '--prepared', str(tmp_path/'prepared'), '--version', 'v', '--out', str(tmp_path/'out'),
        '--budget-profile', tool.PREPARED_PROFILE])
    with pytest.raises(SystemExit) as error:
        tool.main()
    assert error.value.code == 2 and not (tmp_path/'out').exists()
