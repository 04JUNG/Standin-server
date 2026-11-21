"""Private latency metrics preserve failures and exclude payload contents."""
import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from converter_api.interactive import SnapshotStore, measured
from converter_api.work_queue import current_work
from converter.interactive_adapter import runtime_revision
from converter.solved_state import digest
from fastapi import HTTPException


def test_stage_correlates_without_payload_or_exception_text(caplog):
    token = current_work.set(SimpleNamespace(request_id='req_test'))
    try:
        with caplog.at_level(logging.INFO, logger='standin.converter'):
            assert measured('validate', lambda value: value, 'private geometry') == 'private geometry'
            with pytest.raises(ValueError, match='private failure'):
                measured('validate', lambda: (_ for _ in ()).throw(ValueError('private failure')))
        records = [json.loads(record.message) for record in caplog.records]
        assert [row['succeeded'] for row in records] == [True, False]
        assert all(row['request_id'] == 'req_test' and row['duration_ms'] >= 0 for row in records)
        assert 'private geometry' not in caplog.text
        assert 'private failure' not in caplog.text
    finally:
        current_work.reset(token)


def test_blender_queue_execution_are_separate_and_accumulate(monkeypatch, caplog, tmp_path):
    control = SimpleNamespace(request_id='req_test', execution_ms=7, check=lambda phase: None)
    token = current_work.set(control)
    store = SnapshotStore.__new__(SnapshotStore)
    store.runner = SimpleNamespace(_process_slots=nullcontext(), settings=SimpleNamespace(
        blender_binary='test-blender', shape_timeout_seconds=300))
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: SimpleNamespace(returncode=0, poll=lambda: 0))
    ticks = iter([1., 1.25, 2.])
    monkeypatch.setattr('converter_api.interactive.time.monotonic', lambda: next(ticks))
    try:
        with caplog.at_level(logging.INFO, logger='standin.converter'):
            store._child('solve', tmp_path/'job', tmp_path/'identity', tmp_path/'base', tmp_path/'solved', lambda: None)
        row = json.loads(caplog.records[-1].message)
        assert row['queue_wait_ms'] == 250
        assert row['execution_ms'] == 750
        assert control.execution_ms == 757
        assert row['request_id'] == 'req_test'
    finally:
        current_work.reset(token)


def test_published_model_remains_readable_during_export_but_never_bypasses_hash(tmp_path):
    solved = tmp_path/'solved'
    solved.mkdir()
    raw = b'{"immutable":"published model"}'
    (solved/'packet.json').write_bytes(raw)
    entry = dict(busy=True, stateHash='a'*64, runtimeRevision=runtime_revision(),
                 directory=tmp_path, packetSha256=digest(raw), checkpoint=lambda: None)
    store = SnapshotStore.__new__(SnapshotStore)
    store._get = lambda snapshot, owner: entry
    assert store.read('snapshot', 'owner', 'model') == raw
    with pytest.raises(HTTPException) as pending_file:
        store.read('snapshot', 'owner', 'file')
    assert pending_file.value.status_code == 409
    (solved/'packet.json').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='packet changed'):
        store.read('snapshot', 'owner', 'model')
    del entry['stateHash']
    with pytest.raises(HTTPException) as pending_model:
        store.read('snapshot', 'owner', 'model')
    assert pending_model.value.status_code == 409


def test_health_timing_separates_handler_wait_and_failure_without_changing_response(monkeypatch, caplog):
    import asyncio
    from converter_api.health_timing import HealthTimingMiddleware, current_health, health_handler_started, measured_health
    now = [0.]
    monkeypatch.setattr('converter_api.health_timing.time.perf_counter', lambda: now[0])
    messages = []

    async def handler(scope, receive, send):
        now[0] = 2.
        health_handler_started()
        def failing_probe():
            now[0] = 8.
            raise ValueError('private path or geometry')
        with pytest.raises(ValueError, match='private path'):
            measured_health('blender', failing_probe)
        await send({'type': 'http.response.start', 'status': 503, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'unchanged'})

    async def send(message):
        messages.append(message)

    async def receive():
        return {'type': 'http.request', 'body': b''}

    scope = {'type': 'http', 'path': '/healthz', 'headers': [
        (b'x-request-id', b'req_12345678-1234-1234-1234-123456789abc')]}
    with caplog.at_level(logging.INFO, logger='standin.converter'):
        asyncio.run(HealthTimingMiddleware(handler)(scope, receive, send))
    row = json.loads(caplog.records[-1].message)
    assert row['duration_ms'] == 8000 and row['handler_wait_ms'] == 2000
    assert row['stages'] == {'blender': {'duration_ms': 6000, 'succeeded': False}}
    assert row['status'] == 503 and row['request_id'] == scope['headers'][0][1].decode()
    assert messages[-1]['body'] == b'unchanged'
    assert 'private path' not in caplog.text
    assert current_health.get() is None


def test_health_timing_does_not_log_other_routes_or_untrusted_request_ids(caplog):
    import asyncio
    from converter_api.health_timing import HealthTimingMiddleware, health_handler_started, measured_health
    async def handler(scope, receive, send):
        health_handler_started()
        assert measured_health('tempdir', lambda: 7) == 7
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    async def send(message):
        pass
    async def receive():
        return {'type': 'http.request', 'body': b''}
    with caplog.at_level(logging.INFO, logger='standin.converter'):
        for path in ['/other', '/healthz']:
            asyncio.run(HealthTimingMiddleware(handler)(
                {'type': 'http', 'path': path, 'headers': [(b'x-request-id', b'private-path')]}, receive, send))
    assert len(caplog.records) == 1
    assert 'request_id' not in json.loads(caplog.records[0].message)
    assert 'private-path' not in caplog.text
