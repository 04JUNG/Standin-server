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
