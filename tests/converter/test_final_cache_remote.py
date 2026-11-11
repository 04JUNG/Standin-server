"""Private persistence survives task-local eviction without weakening lineage."""
import io
import json
from pathlib import Path

import pytest

from converter_api.final_cache import canonical, digest
from converter_api.final_cache_remote import RemoteStore, location
from tests.converter.test_final_cache import Runner, request_options


class Missing(Exception):
    response = {'Error': {'Code': 'NoSuchKey'}}


class FakeS3:
    def __init__(self):
        self.objects, self.writes = {}, []
        self.fail = False
        self.fail_receipt = False

    def get_object(self, *, Bucket, Key):
        if self.fail: raise OSError('sensitive internal endpoint must not leak')
        if (Bucket, Key) not in self.objects: raise Missing()
        data = self.objects[Bucket, Key]
        return {'Body': io.BytesIO(data), 'ContentLength': len(data)}

    def put_object(self, *, Bucket, Key, Body, **options):
        assert options['ServerSideEncryption'] == 'AES256'
        if self.fail_receipt and Key.endswith('.json'): raise OSError('write unavailable')
        self.writes.append(Key)
        self.objects[Bucket, Key] = Body


@pytest.fixture
def remote(request_options, monkeypatch):
    fake = FakeS3()
    root = Path(__import__('os').environ['CONVERTER_FINAL_CACHE_ROOT'])
    fake.objects['test-bucket', 'control/library.json'] = (root / 'library.json').read_bytes()
    fake.objects['test-bucket', 'control/revoked.json'] = b'[]'
    remote = RemoteStore('s3://test-bucket/cache/data/', 's3://test-bucket/control/library.json',
                         's3://test-bucket/control/revoked.json', fake)
    monkeypatch.setattr(RemoteStore, 'from_env', classmethod(lambda cls: remote))
    return fake


def test_new_task_and_job_reuse_verified_bytes_with_fresh_receipt(request_options, remote, tmp_path, monkeypatch):
    first = Runner().convert(**request_options)
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(tmp_path / 'new-task'))
    second = Runner().convert(**{**request_options, 'conversion_id': 'new-owned-job'})
    assert first.artifact == second.artifact and Runner.calls == 1
    assert second.report['final_cache']['status'] == 'hit'
    assert second.report['final_cache']['blender_runs'] == 0
    assert second.conversion_id == second.report['conversion_id'] == 'new-owned-job'
    assert all(k.startswith('cache/data/') for k in remote.writes)
    assert remote.writes[0].endswith('.fbx') and remote.writes[1].endswith('.json')


def test_shared_revocation_wins_over_local_hot_entry(request_options, remote):
    first = Runner().convert(**request_options)
    remote.objects['test-bucket', 'control/revoked.json'] = canonical([first.artifact_sha256])
    second = Runner().convert(**request_options)
    assert Runner.calls == 2 and second.report['final_cache']['status'] == 'miss'


def test_corrupt_remote_artifact_never_becomes_a_hit(request_options, remote, tmp_path, monkeypatch):
    Runner().convert(**request_options)
    blob = next(k for k in remote.objects if k[1].endswith('.fbx'))
    remote.objects[blob] = b'corrupt'
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(tmp_path / 'new-task'))
    second = Runner().convert(**request_options)
    assert Runner.calls == 2 and second.report['final_cache']['status'] == 'miss'


def test_remote_outage_falls_back_once_without_leaking_endpoint(request_options, remote):
    remote.fail = True
    result = Runner().convert(**request_options)
    assert Runner.calls == 1 and result.report['ok']
    assert result.report['final_cache']['status'] == 'miss'
    assert 'sensitive' not in json.dumps(result.report)


def test_missing_required_revocations_disables_cache_hits(request_options, remote):
    Runner().convert(**request_options)
    del remote.objects['test-bucket', 'control/revoked.json']
    second = Runner().convert(**request_options)
    assert Runner.calls == 2 and second.report['final_cache']['status'] == 'miss'


def test_partial_publication_is_not_a_ready_entry(request_options, remote, tmp_path, monkeypatch):
    remote.fail_receipt = True
    first = Runner().convert(**request_options)
    assert first.report['ok'] and first.report['final_cache']['status'] == 'miss'
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(tmp_path / 'new-task'))
    remote.fail_receipt = False
    Runner().convert(**request_options)
    assert Runner.calls == 2


def test_private_or_refined_source_is_never_published(request_options, remote):
    result = Runner().convert(**{**request_options, 'bvh_bytes': b'private refined input'})
    assert result.report['final_cache']['reason'] == 'unregistered_source_or_options'
    assert remote.writes == []


def test_control_paths_and_artifact_names_cannot_enter_write_prefix():
    for uri in ['https://example.com/x', 's3://test-bucket/a/../b', 's3://test-bucket//x', 's3://test-bucket/x?secret=bad']:
        with pytest.raises(ValueError): location(uri)
    with pytest.raises(ValueError): RemoteStore('s3://test-bucket/data', 's3://test-bucket/data/control', 's3://test-bucket/control/revoked', FakeS3())
    remote = RemoteStore('s3://test-bucket/data', 's3://test-bucket/control/library', 's3://test-bucket/control/revoked', FakeS3())
    for name in ['../secret', 'library.json', 'job_private.json']:
        with pytest.raises(ValueError): remote.read(name, 100)


def test_stale_frozen_solver_receipt_is_not_authorized(request_options, remote, tmp_path, monkeypatch):
    Runner().convert(**request_options)
    key = next(k for k in remote.objects if k[1].startswith('cache/data/') and k[1].endswith('.json'))
    metadata = json.loads(remote.objects[key])
    metadata['report']['solver_manifest_sha256'] = '0' * 64
    metadata['report_sha256'] = digest(canonical(metadata['report']))
    remote.objects[key] = canonical(metadata)
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(tmp_path / 'new-task'))
    result = Runner().convert(**request_options)
    assert Runner.calls == 2 and result.report['final_cache']['status'] == 'miss'
    assert not list((tmp_path / 'new-task').glob('*.fbx'))


@pytest.mark.parametrize('corruption', ['metadata_list', 'report_list', 'hands_list', 'fingers_list', 'roundtrip_list'])
def test_malformed_remote_receipts_recompute_instead_of_500(request_options, remote, tmp_path, monkeypatch, corruption):
    Runner().convert(**request_options)
    key = next(k for k in remote.objects if k[1].startswith('cache/data/') and k[1].endswith('.json'))
    metadata = json.loads(remote.objects[key])
    if corruption == 'metadata_list': metadata = []
    else:
        if corruption == 'report_list': metadata['report'] = []
        if corruption == 'hands_list': metadata['report']['hands'] = ['invalid']
        if corruption == 'fingers_list': metadata['report']['source_fingers'] = ['invalid']
        if corruption == 'roundtrip_list': metadata['report']['source_fingers']['saved_fbx_roundtrip'] = ['invalid']
        metadata['report_sha256'] = digest(canonical(metadata['report']))
    remote.objects[key] = canonical(metadata)
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(tmp_path / 'new-task'))
    result = Runner().convert(**request_options)
    assert Runner.calls == 2 and result.report['ok'] and result.report['final_cache']['status'] == 'miss'
