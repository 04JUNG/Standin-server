import hashlib
import os
from pathlib import Path
import pytest
from converter_api.verified_files import VerifiedDigests
from converter_api.body_metadata import BodyMetadataCache
from converter.interactive_adapter import runtime_revision
from converter.solved_state import canonical, digest


def test_digest_rejects_stale_bytes_even_with_restored_mtime(tmp_path):
    path = tmp_path/'asset'; path.write_bytes(b'first')
    cache = VerifiedDigests(); old = path.stat()
    assert cache.digest(path) == hashlib.sha256(b'first').hexdigest()
    path.write_bytes(b'other'); os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert cache.digest(path) == hashlib.sha256(b'other').hexdigest()
    path.unlink()
    with pytest.raises(OSError): cache.digest(path)


@pytest.mark.skipif(os.name != 'posix', reason='Only POSIX ctime supports the digest fast path')
def test_digest_singleflight_and_ttl(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path/'asset'; path.write_bytes(b'asset')
    now = [0]; cache = VerifiedDigests(ttl=10, clock=lambda: now[0])
    opened = []; original = Path.open
    def read(self, *args, **kwargs):
        opened.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', read)
    with ThreadPoolExecutor(8) as executor:
        results = list(executor.map(cache.digest, [path]*8))
    assert len(set(results)) == 1 and len(opened) == 1
    now[0] = 11; cache.digest(path)
    assert len(opened) == 2


def test_body_reuse_keeps_exact_hash_and_binding_gates(tmp_path, monkeypatch):
    import converter_api.body_metadata as module
    (tmp_path/'body.blend').write_bytes(b'snapshot')
    binding = {'owner': 'one'}
    metadata = dict(schema='interactive-body-base-v1', binding=binding,
        runtimeRevision=runtime_revision(), snapshotSha256=digest(b'snapshot'),
        model={'bones': [{'name': 'root'}], 'meshes': ['large geometry']})
    (tmp_path/'base.json').write_bytes(canonical(metadata))
    cache = BodyMetadataCache(); calls = []; original = module.decode
    monkeypatch.setattr(module, 'decode', lambda raw: calls.append(1) or original(raw))
    first = cache.read(tmp_path, binding)
    assert first['model'] == {'bones': [{'name': 'root'}]}
    first['model']['bones'].clear()
    assert cache.read(tmp_path, binding)['model']['bones'] == [{'name': 'root'}]
    assert len(calls) == 1
    assert cache.read(tmp_path, {'owner': 'other'}) is None
    assert cache.read(tmp_path, binding) is not None
    (tmp_path/'body.blend').write_bytes(b'tampered')
    assert cache.read(tmp_path, binding) is None
    (tmp_path/'body.blend').write_bytes(b'snapshot')
    monkeypatch.setattr(module, 'runtime_revision', lambda: 'changed')
    assert cache.read(tmp_path, binding) is None


def test_health_shape_verifies_each_source_without_relaxing_checks(tmp_path, monkeypatch):
    import converter_api.health_assets as module
    blend, controller = tmp_path/'base.blend', tmp_path/'controller.py'
    blend.write_bytes(b'blend'); controller.write_text("VERSION = 'body.v1'\n")
    source = dict(blend=str(blend), controller=str(controller),
        blend_sha256=digest(blend.read_bytes()), controller_sha256=digest(controller.read_bytes()))
    value = {'sources': {'body.v1': source}}
    monkeypatch.setattr(module, 'load_registry', lambda **kw: value)
    assert module.health_shape_registry() == value
    blend.write_bytes(b'wrong')
    with pytest.raises(ValueError): module.health_shape_registry()
