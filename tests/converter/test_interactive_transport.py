"""Private HTTP boundary, distinct from the separately run real Blender gate."""
import json
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from converter_api.interactive import register_interactive, SnapshotStore, MAX_REQUEST
from converter.interactive_revision import LatestRevisionGate
from converter.interactive_export import Superseded


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('CONVERTER_INTERACTIVE_ENABLED','true')
    monkeypatch.setenv('CONVERTER_INTERACTIVE_TOKEN','t'*32)
    monkeypatch.setenv('CONVERTER_INTERACTIVE_ROOT',str(tmp_path/'private'))
    from converter_api.work_middleware import WorkMiddleware
    app=FastAPI();app.add_middleware(WorkMiddleware);register_interactive(app,None,None)
    with TestClient(app) as client:yield client,app.state.interactive


def test_transport_disabled_by_default(monkeypatch):
    monkeypatch.delenv('CONVERTER_INTERACTIVE_ENABLED',raising=False)
    app=FastAPI();register_interactive(app,None,None)
    assert TestClient(app).post('/interactive/v1/solve',json={}).status_code==404


def test_token_required_before_body_or_worker(client,monkeypatch):
    http,store=client;calls=[]
    monkeypatch.setattr(store,'solve',lambda value:calls.append(value) or {'snapshotId':'verified'})
    assert http.post('/interactive/v1/solve',json={}).status_code==401
    assert http.post('/interactive/v1/solve',headers={'Authorization':'Bearer '+'x'*32},json={}).status_code==401
    assert calls==[]
    response=http.post('/interactive/v1/solve',headers={'Authorization':'Bearer '+'t'*32},json={})
    assert response.status_code==200
    assert response.headers['x-standin-work-class']=='preview'
    assert calls==[{}]


def test_bounded_chunked_body_and_parse_errors(client):
    http,_=client;headers={'Authorization':'Bearer '+'t'*32}
    assert http.post('/interactive/v1/solve',headers=headers,content=iter([b'x'*MAX_REQUEST,b'x'])).status_code==413
    assert http.post('/interactive/v1/solve',headers=headers,content=b'{').status_code==400


def test_unknown_owner_path_and_missing_static_are_closed(client):
    http,_=client;headers={'Authorization':'Bearer '+'t'*32,'X-Standin-Owner':'a'*64}
    assert http.get('/interactive/v1/snapshots/unknown/model',headers=headers).status_code==404
    assert http.get('/interactive/v1/snapshots/unknown/model',headers={'Authorization':'Bearer '+'t'*32}).status_code==400
    assert http.post('/interactive/v1/static/resolve',headers=headers,json=[{'identity':{'sourceSha256':'b'*64,'characterSha256':'c'*64,'characterId':'registered','scope':'full'},'camera':{}}]).json()==[None]


def test_private_volume_is_exclusive_and_restarts_revoke_old_files(tmp_path):
    root=tmp_path/'private';first=SnapshotStore(root,None,None)
    old=root/'snapshot-crashed';old.mkdir();(old/'private').write_bytes(b'private')
    with pytest.raises(BlockingIOError):SnapshotStore(root,None,None)
    first._volume_lock.close()
    second=SnapshotStore(root,None,None)
    assert not old.exists()
    second._volume_lock.close()


def test_old_http_cancel_never_revokes_new_revision(tmp_path):
    store=SnapshotStore(tmp_path/'private',None,None)
    old=store.gate.select_binding('a'*64,'0',1,'b'*64)
    new=store.gate.select_binding('a'*64,'0',2,'c'*64)
    store.cancel('a'*64,'0',1);new()
    with pytest.raises(Superseded):old()
    store.cancel('a'*64,'0',2)
    with pytest.raises(Superseded):new()


def test_foreign_directory_is_never_cleared(tmp_path):
    root=tmp_path/'foreign';old=root/'body-important';old.mkdir(parents=True);(old/'keep').write_bytes(b'not-a-cache')
    with pytest.raises(ValueError):SnapshotStore(root,None,None)
    assert (old/'keep').read_bytes()==b'not-a-cache'


def test_partial_or_tampered_body_base_is_not_reused(tmp_path):
    from converter.interactive_adapter import runtime_revision
    from converter.solved_state import canonical,digest
    store=SnapshotStore(tmp_path/'private',None,None);base=tmp_path/'partial';base.mkdir()
    (base/'body.blend').write_bytes(b'interrupted')
    assert store._body_metadata(base,{'owner':'fixture'}) is None
    metadata=dict(schema='interactive-body-base-v1',binding={'owner':'fixture'},runtimeRevision=runtime_revision(),snapshotSha256=digest(b'interrupted'))
    (base/'base.json').write_bytes(canonical(metadata));assert store._body_metadata(base,{'owner':'fixture'})==metadata
    (base/'body.blend').write_bytes(b'changed');assert store._body_metadata(base,{'owner':'fixture'}) is None


@pytest.mark.parametrize('delta',[-1,7200])
def test_solve_rejects_expired_or_unbounded_bff_deadline(tmp_path,delta):
    from datetime import datetime,timezone
    import time
    store=SnapshotStore(tmp_path/'private',None,None)
    assert store.ttl==3600
    payload=dict(selection={},operation='operation',revision=1,person='0',bvhBase64='',expiresAt=datetime.fromtimestamp(time.time()+delta,timezone.utc).isoformat())
    with pytest.raises(ValueError,match='deadline'):store.solve(payload)
