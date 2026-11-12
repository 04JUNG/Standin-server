"""Persistence, exact lineage, fail-closed registration and coalescing."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from converter_api.final_cache import FinalStore, cached_final, canonical, digest, identity
from converter_api.runner import ConversionResult, RunnerSettings
from converter.scene_contract import FBX_MAGIC
from converter.protocol import SOLVER_VERSION, SOLVER_MANIFEST_SHA256, RETARGET_SHA256, ANKLE_POLICY_SHA256
from converter.hand_library import load_library, normalize_request


class Runner:
    calls = 0
    gate = threading.Lock()

    def __init__(self):
        self.settings = RunnerSettings('unused', Path('converter/worker.py'))

    @cached_final
    def convert(self, *, bvh_bytes, character_path, character_id, character_sha256, conversion_id,
                mirror=False, output_scope='full', camera_rotation=None, hands=None,
                shape_recipe=None, preview_view=None, preview_model=False, thumbnail=None,
                hand_contact_screen=False, selected_hand_screen=False):
        with self.gate:
            type(self).calls += 1
        time.sleep(.01)
        artifact = FBX_MAGIC + bvh_bytes + canonical([hands, mirror, camera_rotation, output_scope])
        sha = digest(artifact)
        report = dict(ok=True, solver_version=SOLVER_VERSION, solver_manifest_sha256=SOLVER_MANIFEST_SHA256,
            retarget_sha256=RETARGET_SHA256, ankle_policy_sha256=ANKLE_POLICY_SHA256,
            artifact_sha256=sha, artifact_size=len(artifact),
            source_bvh_sha256=digest(bvh_bytes), character_id=character_id,
            character_sha256=character_sha256, camera_rotation=camera_rotation, mirrored=mirror,
            source_fingers=dict(status='transferred', body_bones_preserved=True,
                saved_fbx_roundtrip=dict(verified=True)),
            artifact_validation=dict(version='final-fbx-v1', artifact_sha256=sha, scope=output_scope,
                skin_verified=True, bones=54, meshes=1, vertices=100, polygons=100))
        if hands:
            report['hands'] = {**normalize_request(hands), 'library_sha256':load_library().sha256,
                               'body_bones_preserved':True}
        if shape_recipe:
            report['shape_bake'] = {k:shape_recipe[k] for k in
                                   ('shape_sha256','rules','blend_sha256','controller_sha256')}
            report['shape_bake']['controls'] = shape_recipe['values']
        return ConversionResult(conversion_id, artifact, sha, digest(bvh_bytes), report)


@pytest.fixture
def request_options(tmp_path, monkeypatch):
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(tmp_path / 'store'))
    Runner.calls = 0
    root = tmp_path / 'store'
    root.mkdir()
    registered = canonical(dict(version='final-library-inputs-v1',
        sources=[digest(b'registered source')], characters={'male-base-v0-8-0':digest(FBX_MAGIC+b'approved')}))
    (root/'library.json').write_bytes(registered)
    monkeypatch.setenv('CONVERTER_FINAL_LIBRARY_SHA256', digest(registered))
    character = tmp_path / 'body.fbx'
    character.write_bytes(FBX_MAGIC + b'approved')
    return dict(bvh_bytes=b'registered source', character_path=character,
                character_id='male-base-v0-8-0', character_sha256=digest(character.read_bytes()),
                conversion_id='job1')


def test_restart_and_new_job_share_only_bytes(request_options):
    a = Runner().convert(**request_options)
    b = Runner().convert(**{**request_options, 'conversion_id': 'job2'})
    assert Runner.calls == 1
    assert a.artifact == b.artifact
    assert b.conversion_id == b.report['conversion_id'] == 'job2'
    assert b.report['final_cache']['blender_runs'] == 0
    assert b.execution_ms == 0


def test_concurrent_requests_coalesce_across_runner_instances(request_options):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: Runner().convert(**request_options), range(4)))
    assert Runner.calls == 1
    assert len({r.artifact_sha256 for r in results}) == 1


@pytest.mark.parametrize('change', [dict(bvh_bytes=b'refined-result'), dict(mirror=True),
    dict(output_scope='head'), dict(camera_rotation=[[1,0,0],[0,1,0],[0,0,1]]),
    dict(hands={'left':'relaxed','right':None})])
def test_exact_options_and_return_to_default(request_options, change):
    default = Runner().convert(**request_options)
    Runner().convert(**{**request_options, **change})
    restored = Runner().convert(**request_options)
    assert Runner.calls == 2
    assert default.artifact == restored.artifact
    assert restored.report['final_cache']['status'] == 'hit'


def test_corrupt_and_policy_changed_safe_fallback(request_options, monkeypatch):
    Runner().convert(**request_options)
    root = Path(__import__('os').environ['CONVERTER_FINAL_CACHE_ROOT'])
    next(root.glob('*.fbx')).write_bytes(b'broken')
    assert Runner().convert(**request_options).report['final_cache']['status'] == 'miss'
    assert Runner.calls == 2
    monkeypatch.setattr('converter_api.final_cache.policy_sha256', lambda: 'b' * 64)
    assert Runner().convert(**request_options).report['final_cache']['status'] == 'miss'
    assert Runner.calls == 3


def test_revocation_is_checked_on_every_hit(request_options):
    result = Runner().convert(**request_options)
    root = Path(__import__('os').environ['CONVERTER_FINAL_CACHE_ROOT'])
    (root / 'revoked.json').write_text(json.dumps([result.artifact_sha256]))
    result = Runner().convert(**request_options)
    assert result.report['final_cache']['status'] == 'miss'
    assert 'registration_failed' in result.report['final_cache']['reason']
    assert Runner.calls == 2


def test_unverified_entries_never_register(request_options, monkeypatch):
    monkeypatch.setattr('converter_api.final_cache.validate', lambda *args: (_ for _ in ()).throw(ValueError('unverified')))
    for _ in range(2):
        assert Runner().convert(**request_options).report['final_cache']['status'] == 'miss'
    assert Runner.calls == 2
    root = Path(__import__('os').environ['CONVERTER_FINAL_CACHE_ROOT'])
    assert not list(root.glob('*.fbx'))


def test_appearance_and_render_paths_remain_uncached(request_options):
    for _ in range(2):
        Runner().convert(**{**request_options, 'preview_model': True})
    assert Runner.calls == 2


def test_cache_unavailable_keeps_safe_path(request_options, monkeypatch, tmp_path):
    path = tmp_path / 'file'
    path.write_text('not a directory')
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT', str(path))
    assert Runner().convert(**request_options).report['final_cache']['reason'] == 'storage_unavailable'
    assert Runner.calls == 1


def test_processes_coalesce_and_crashed_owner_releases_lock(request_options):
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    pipes, processes = [], []
    def run(connection):
        try:
            connection.send(Runner().convert(**request_options).report['final_cache']['status'])
        except Exception as exc:
            connection.send(type(exc).__name__)
        finally:
            connection.close()
    for _ in range(3):
        parent, child = ctx.Pipe()
        process = ctx.Process(target=run, args=(child,))
        process.start()
        pipes.append(parent)
        processes.append(process)
    statuses = [p.recv() for p in pipes]
    for p in processes:
        p.join(10)
        assert p.exitcode == 0
    assert statuses.count('miss') == 1
    assert statuses.count('hit') == 2


def test_shape_controls_sources_and_explicit_defaults_have_distinct_identity(request_options, tmp_path):
    recipe = dict(rules='test-approved-v1', values={'height_cm':168.684899, 'heads':7.943109},
                  defaults={'height_cm':168.684899, 'heads':7.943109}, shape_sha256=None)
    for k in ('blend', 'controller'):
        p = tmp_path / k
        p.write_text(k)
        recipe[k] = str(p)
        recipe[k + '_sha256'] = digest(p.read_bytes())
    Runner().convert(**{**request_options, 'shape_recipe':recipe})
    Runner().convert(**{**request_options, 'shape_recipe':{**recipe, 'values':{**recipe['values'], 'heads':8.3}}})
    restored = Runner().convert(**{**request_options, 'shape_recipe':recipe})
    assert restored.report['final_cache']['status'] == 'hit'
    assert Runner.calls == 2
    explicit = Runner().convert(**{**request_options, 'shape_recipe':{**recipe, 'shape_sha256':'e'*64}})
    assert explicit.report['final_cache']['status'] == 'miss'
    assert Runner.calls == 3


def test_cancelled_work_cannot_use_hit_or_publish_late_output(request_options):
    from converter_api.work_queue import WorkControl, current_work, WorkStopped
    Runner().convert(**request_options)
    control = WorkControl()
    control.cancelled.set()
    token = current_work.set(control)
    try:
        with pytest.raises(WorkStopped):
            Runner().convert(**request_options)
    finally:
        current_work.reset(token)
    assert Runner.calls == 1


def test_refined_private_sources_never_enter_shared_library_cache(request_options):
    request_options['bvh_bytes'] = b'user refined input'
    for _ in range(2):
        result = Runner().convert(**request_options)
        assert result.report['final_cache']['reason'] == 'unregistered_source_or_options'
    assert Runner.calls == 2
    root = Path(__import__('os').environ['CONVERTER_FINAL_CACHE_ROOT'])
    assert not list(root.glob('*.fbx'))


def test_unpinned_or_malformed_input_registry_falls_back(request_options):
    root = Path(__import__('os').environ['CONVERTER_FINAL_CACHE_ROOT'])
    (root / 'library.json').write_text('broken')
    assert Runner().convert(**request_options).report['final_cache']['status'] == 'miss'
    assert Runner.calls == 1


def test_policy_fingerprint_matches_docker_runtime_copy_allowlist(tmp_path, monkeypatch):
    import shutil
    import converter_api.final_cache as cache
    original = cache.policy_sha256()
    for line in (cache.ROOT/'Dockerfile.converter').read_text().splitlines():
        if line.startswith('COPY '):
            for name in line.split()[1:-1]:
                source = cache.ROOT/name
                if source.is_file():
                    target = tmp_path/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source,target)
    monkeypatch.setattr(cache,'ROOT',tmp_path)
    assert cache.policy_sha256() == original


def test_uncached_worker_oserror_is_not_executed_twice(request_options):
    class Broken:
        settings = Runner().settings
        calls = 0
        @cached_final
        def convert(self, *, bvh_bytes, character_path, character_id, character_sha256, conversion_id):
            self.calls += 1
            raise OSError('worker disk failure')
    runner = Broken()
    with pytest.raises(OSError):
        runner.convert(**{**request_options, 'bvh_bytes': b'private unregistered input'})
    assert runner.calls == 1


def test_crashed_lock_owner_does_not_strand_resumed_work(request_options):
    import multiprocessing
    import os
    ctx=multiprocessing.get_context('fork')
    store=FinalStore(os.environ['CONVERTER_FINAL_CACHE_ROOT'])
    def crash():
        with store.lock('c'*64,lambda:None):
            os._exit(7)
    process=ctx.Process(target=crash)
    process.start();process.join(5)
    assert process.exitcode==7
    with store.lock('c'*64,lambda:None):
        pass


def test_interrupted_publication_never_serves_orphan_bytes(request_options, monkeypatch):
    import os
    runner=Runner()
    binding=identity(runner.settings,request_options)
    result=Runner.convert.__wrapped__(runner,**request_options)
    store=FinalStore(os.environ['CONVERTER_FINAL_CACHE_ROOT'])
    atomic=store.atomic
    def interrupted(name,data):
        if name.endswith('.json'):
            raise OSError('interrupted before metadata commit')
        return atomic(name,data)
    monkeypatch.setattr(store,'atomic',interrupted)
    with pytest.raises(OSError):store.put(binding,result)
    with pytest.raises(FileNotFoundError):store.get(binding)
    monkeypatch.setattr(store,'atomic',atomic)
    store.put(binding,result)
    assert store.get(binding)[0]==result.artifact
