import hashlib
import io
import json
import shutil
import tarfile

import pytest

from scripts.fullbody_preflight_environment import export
from scripts.prepare_staging_recall_deployment import PIN, STAGING_FAMILY, configure
from src import search_recall_assets


def staging_task():
    env = {
        'ROUTING_FULLBODY_ENABLED': '1',
        'VLM_PROVIDER': 'openai',
        'OPENAI_MODEL': 'gpt-6-luna',
        'OPENAI_REASONING_EFFORT': 'low',
        'REFINE_THUMBNAIL_RENDERER': 'converter',
    }
    return {
        'family': STAGING_FAMILY,
        'containerDefinitions': [{
            'name': 'inference',
            'environment': [{'name': key, 'value': value} for key, value in env.items()],
        }],
    }


def environment(task):
    return {row['name']: row['value'] for row in task['containerDefinitions'][0]['environment']}


def test_enable_then_disable_staging_recall():
    task = staging_task()
    pin = json.loads(PIN.read_text())
    enabled = configure(task, '1', pin)
    env = environment(enabled)
    assert env['ROUTING_RECALL_ASSETS_URI'] == pin['uri']
    assert env['ROUTING_RECALL_ASSETS_SHA256'] == pin['sha256']
    assert env['ROUTING_LUNA_SIX_FIELD_ENABLED'] == '1'
    assert 'ROUTING_RECALL_PROP_PYTHON=/opt/recall-prop/bin/python\n' in export(enabled)
    assert env['REFINE_THUMBNAIL_RENDERER'] == 'converter'
    assert env['OPENAI_REASONING_EFFORT'] == 'low'

    disabled = environment(configure(enabled, '0', {}))
    assert not any(key.startswith('ROUTING_RECALL_') for key in disabled)
    assert disabled['ROUTING_LUNA_SIX_FIELD_ENABLED'] == '0'
    assert disabled['REFINE_THUMBNAIL_RENDERER'] == 'converter'


def test_cannot_enable_on_another_task():
    task = staging_task()
    task['family'] = 'production-inference'
    with pytest.raises(ValueError, match='staging inference task only'):
        configure(task, '1', json.loads(PIN.read_text()))


@pytest.fixture
def pinned_archive(tmp_path, monkeypatch):
    bundle = tmp_path / 'fixture.tar'
    with tarfile.open(bundle, 'w') as archive:
        for name in sorted(search_recall_assets.REQUIRED_FILES):
            member = tarfile.TarInfo(name)
            member.size = 3
            archive.addfile(member, io.BytesIO(b'abc'))
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    destination = tmp_path / 'models'
    monkeypatch.setenv('ROUTING_RECALL_ASSETS_DIR', str(destination))
    monkeypatch.setenv('ROUTING_RECALL_ASSETS_URI', 's3://private-staging/pose-library/runtime-assets/recall-90-v1/fixture.tar')
    monkeypatch.setenv('ROUTING_RECALL_ASSETS_SHA256', sha)
    calls = []

    def download(uri, target):
        calls.append(uri)
        shutil.copyfile(bundle, target)

    monkeypatch.setattr(search_recall_assets, '_download_s3', download)
    return destination, calls


def test_provisions_archive_and_reuses_it(pinned_archive):
    destination, calls = pinned_archive
    assert search_recall_assets.ensure_recall_assets() == destination
    assert (destination / 'body.onnx').read_bytes() == b'abc'
    assert search_recall_assets.ensure_recall_assets() == destination
    assert len(calls) == 1


def test_checksum_failure_leaves_no_installed_directory(pinned_archive, monkeypatch):
    destination, _ = pinned_archive
    monkeypatch.setenv('ROUTING_RECALL_ASSETS_SHA256', 'a' * 64)
    with pytest.raises(ValueError, match='checksum mismatch'):
        search_recall_assets.ensure_recall_assets()
    assert not destination.exists()


def test_archive_cannot_escape_model_directory(tmp_path):
    bundle = tmp_path / 'fixture.tar'
    with tarfile.open(bundle, 'w') as archive:
        archive.addfile(tarfile.TarInfo('../unexpected'))
    with pytest.raises(ValueError, match='unexpected recall model archive entry'):
        search_recall_assets.extract_archive(bundle, tmp_path / 'models')
    assert not (tmp_path / 'unexpected').exists()
