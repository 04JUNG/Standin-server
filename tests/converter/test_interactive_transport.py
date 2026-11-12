"""Private HTTP boundary, distinct from the separately run real Blender gate."""
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from converter_api.interactive import register_interactive, SnapshotStore, MAX_REQUEST
from converter_api.work_queue import current_work, measured_slot
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


def test_interactive_timing_reports_queue_and_cumulative_worker_time(client, monkeypatch):
    http, store = client

    @contextmanager
    def worker_slot():
        # Model a new admission overwriting the slot's previous wait sample.
        current_work.get().queue_wait_ms = 0
        time.sleep(.005)
        yield

    def solve(_):
        for _ in range(2):
            with measured_slot(worker_slot(), worker=True):
                time.sleep(.01)
        return {'snapshotId': 'verified'}

    monkeypatch.setattr(store, 'solve', solve)
    response = http.post('/interactive/v1/solve', headers={'Authorization': 'Bearer ' + 't' * 32}, json={})
    assert response.status_code == 200
    handler = float(response.headers['x-standin-interactive-handler-ms'])
    wait = float(response.headers['x-standin-queue-wait-ms'])
    work = float(response.headers['x-standin-execution-ms'])
    assert wait >= 9 and work >= 19
    assert handler + .01 >= wait + work


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
    metadata=dict(schema='interactive-body-base-v1',binding={'owner':'fixture'},runtimeRevision=runtime_revision(),snapshotSha256=digest(b'interrupted'),model={'bones': []})
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

def test_gc_removes_only_idle_superseded_snapshots(tmp_path):
    store = SnapshotStore(tmp_path/'private', None, None)
    for sid, owner, revision, busy in [('old','a',1,False), ('working','a',2,True),
                                      ('current','a',3,False), ('other','b',1,False)]:
        directory = store.root / ('snapshot-' + sid)
        directory.mkdir()
        store.entries[sid] = dict(owner=owner*64, person='0', revision=revision,
                                  busy=busy, expiry=time.monotonic()+60, directory=directory)
    store._gc()
    assert set(store.entries) == {'working', 'current', 'other'}
    assert not (store.root/'snapshot-old').exists()


# --- Binary solved state (v2) negotiation and delivery -------------------------------

V2_HEADERS={'Authorization':'Bearer '+'t'*32,'X-Standin-State-Encoding':'standin-binary-f64le-v2'}


@pytest.mark.parametrize('header,env,expected',[
    (None,None,()),('typed-f64be-json-v1',None,()),
    ('standin-binary-f64le-v2',None,('standin-binary-f64le-v2',)),
    ('standin-binary-f64le-v2','0',())])
def test_solve_encoding_is_opt_in_and_operator_can_force_v1(client,monkeypatch,header,env,expected):
    http,store=client;calls=[]
    if env is not None:monkeypatch.setenv('CONVERTER_INTERACTIVE_BINARY_STATE',env)
    def solve(payload,*encoding):
        calls.append(encoding)
        return {'snapshotId':'s','packet':{'canonicalEncoding':encoding[0] if encoding else 'typed-f64be-json-v1'}}
    monkeypatch.setattr(store,'solve',solve)
    headers={'Authorization':'Bearer '+'t'*32,**({'X-Standin-State-Encoding':header} if header else {})}
    response=http.post('/interactive/v1/solve',headers=headers,json={})
    assert response.status_code==200 and calls==[expected]
    assert response.headers['x-standin-state-encoding']==(expected[0] if expected else 'typed-f64be-json-v1')


def test_unknown_state_encoding_is_rejected_before_body_or_worker(client,monkeypatch):
    http,store=client;calls=[]
    monkeypatch.setattr(store,'solve',lambda *a:calls.append(a))
    response=http.post('/interactive/v1/solve',headers={**V2_HEADERS,'X-Standin-State-Encoding':'gzip-json'},json={})
    assert response.status_code==400 and calls==[]


def test_invalid_binary_state_switch_fails_startup(tmp_path,monkeypatch):
    monkeypatch.setenv('CONVERTER_INTERACTIVE_ENABLED','true');monkeypatch.setenv('CONVERTER_INTERACTIVE_TOKEN','t'*32)
    monkeypatch.setenv('CONVERTER_INTERACTIVE_ROOT',str(tmp_path/'private'));monkeypatch.setenv('CONVERTER_INTERACTIVE_BINARY_STATE','maybe')
    with pytest.raises(ValueError):register_interactive(FastAPI(),None,None)


def _binary_entry(store,tmp_path):
    from converter.interactive_adapter import runtime_revision
    from converter.solved_state import SolvedPoseState,digest
    from tests.converter.test_interactive_state import identity,model
    state=SolvedPoseState.create(identity(),model(),encoding='standin-binary-f64le-v2')
    directory=tmp_path/'snapshot';(directory/'solved').mkdir(parents=True)
    (directory/'solved/state.canonical.bin').write_bytes(state.canonical_bytes)
    store.entries['binary']=dict(owner='a'*64,person='p',revision=1,request='r',busy=False,directory=directory,
        expiry=time.monotonic()+60,checkpoint=lambda:None,stateHash=state.state_hash,
        encoding='standin-binary-f64le-v2',modelFile='solved/state.canonical.bin',
        packetSha256=digest(state.canonical_bytes),runtimeRevision=runtime_revision())
    return state,directory


def test_binary_model_is_served_as_exact_state_bytes_with_v2_media_type(client,tmp_path):
    http,store=client;state,directory=_binary_entry(store,tmp_path)
    headers={'Authorization':'Bearer '+'t'*32,'X-Standin-Owner':'a'*64}
    response=http.get('/interactive/v1/snapshots/binary/model',headers=headers)
    assert response.status_code==200 and response.content==state.canonical_bytes
    assert response.headers['content-type']=='application/vnd.standin.solved-pose-state-v2'
    (directory/'solved/state.canonical.bin').write_bytes(state.canonical_bytes[:-1]+b'\1')
    assert http.get('/interactive/v1/snapshots/binary/model',headers=headers).status_code==409


def test_binary_snapshot_answers_variants_without_decoding(client,tmp_path,monkeypatch):
    http,store=client;_binary_entry(store,tmp_path)
    def forbidden(*a,**k):raise AssertionError('variants must not decode a binary state')
    monkeypatch.setattr('converter_api.interactive.load_solved',forbidden)
    response=http.get('/interactive/v1/snapshots/binary/variants',headers={'Authorization':'Bearer '+'t'*32,'X-Standin-Owner':'a'*64})
    assert response.status_code==409 and response.json()['detail']=='VARIANTS_UNAVAILABLE'
