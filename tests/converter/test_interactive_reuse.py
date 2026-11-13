import copy
import json
from pathlib import Path
import pytest
from converter.interactive_adapter import runtime_revision, load_solved
from converter.interactive_reuse import core_binding, rebind_solved, rebind_export
from converter.solved_state import SolvedPoseState, canonical, digest, decode, RECEIPT_SCHEMA
from tests.converter.test_interactive_state import identity, model


def bundle(root, supplied=None):
    root.mkdir(); state = SolvedPoseState.create(supplied or identity(), model())
    (root/'solved.blend').write_bytes(b'private snapshot')
    (root/'staged.fbx').write_bytes(b'verified immutable fbx')
    (root/'state.canonical.json').write_bytes(state.canonical_bytes)
    (root/'snapshot.json').write_bytes(canonical(dict(schema='interactive-snapshot-v1',
        stateHash=state.state_hash, runtimeRevision=runtime_revision(),
        snapshotSha256=digest((root/'solved.blend').read_bytes()),
        stagedFbxSha256=digest((root/'staged.fbx').read_bytes()))))
    (root/'packet.json').write_text(json.dumps(state.packet()))
    return state


def test_appearance_revision_rebinds_without_blender_and_survives_original_removal(tmp_path):
    source=tmp_path/'source'; old=bundle(source); new_id=identity()
    new_id['selection'].update(operation='new-op',revision=2)
    new=rebind_solved(source,tmp_path/'new',new_id)
    assert new.payload['model']==old.payload['model']
    assert new.state_hash!=old.state_hash
    assert new.packet()['operation']=='new-op'
    assert new.payload['safety']['handChecksPassed'] is False
    import shutil
    shutil.rmtree(source)
    assert load_solved(tmp_path/'new')[0].state_hash==new.state_hash


def test_bound_snapshot_read_rejects_changed_bytes_and_preserves_identity(tmp_path):
    root = tmp_path/'snapshot'
    state = bundle(root)
    identity_copy = state.identity
    identity_copy['selection']['revision'] = 2
    assert state.identity['selection']['revision'] == 1
    assert load_solved(root, verified_state=state)[0] is state
    (root/'state.canonical.json').write_bytes(state.canonical_bytes + b' ')
    with pytest.raises(ValueError, match='state changed'):
        load_solved(root, verified_state=state)


@pytest.mark.parametrize('field,part,value',[
    ('ownerBinding',None,'e'*64),('body','characterSha256','d'*64),
    ('source','originalBvhSha256','d'*64),('hands','left','other'),
    ('shape','controlsSha256','d'*64),('scope',None,'half'),
    ('solver','runtimeSha256','d'*64),('selection','person','other')])
def test_changed_geometry_or_owner_never_reuses(tmp_path,field,part,value):
    source=tmp_path/'source';bundle(source); changed=identity()
    if part is None:changed[field]=value
    else:changed[field][part]=value
    with pytest.raises(ValueError):rebind_solved(source,tmp_path/'new',changed)


def test_verified_export_rebinds_only_exact_model_and_checks_artifact_hash(tmp_path):
    old=bundle(tmp_path/'old'); cached=tmp_path/'cache';cached.mkdir()
    raw=(tmp_path/'old/staged.fbx').read_bytes();(cached/'artifact.fbx').write_bytes(raw)
    receipt=dict(schema=RECEIPT_SCHEMA,verified=True,runtimeRevision=runtime_revision(),
        artifactSha256=digest(raw),artifactBytes=len(raw),packetModelSha256=digest(canonical(old.payload['model'])),
        identity=old.payload['identity'],stateHash=old.state_hash,revision=1,operation='op')
    (cached/'verified.canonical.json').write_bytes(canonical(receipt))
    new_id=identity();new_id['selection'].update(operation='new-op',revision=2)
    new=rebind_solved(tmp_path/'old',tmp_path/'new',new_id)
    rebound=rebind_export(cached,tmp_path/'export',tmp_path/'new')
    assert rebound['stateHash']==new.state_hash and rebound['operation']=='new-op'
    assert (tmp_path/'export/artifact.fbx').read_bytes()==raw
    (cached/'artifact.fbx').write_bytes(b'tampered')
    with pytest.raises(ValueError):rebind_export(cached,tmp_path/'rejected',tmp_path/'new')


def test_appearance_cache_rebinds_receipt_without_repeating_worker():
    from converter_api.appearance_cache import verified_appearance
    raw=b'exact output'; source=b'base';calls=[]
    job=dict(source_artifact_sha256=digest(source),binding_sha256='one',skin={'id':'flesh'})
    def produce():
        calls.append(1)
        return raw,dict(file_verified=True,fbx_sha256=digest(raw),binding_sha256='one')
    first=verified_appearance('fixture',source,job,'runtime',produce,lambda:None)
    second=verified_appearance('fixture',source,{**job,'binding_sha256':'two'},'runtime',produce,lambda:None)
    assert len(calls)==1 and second[0]==raw and second[1]['binding_sha256']=='two'
    assert first[1]['binding_sha256']=='one'
    verified_appearance('fixture',source,{**job,'skin':{'id':'muscle'}},'runtime',produce,lambda:None)
    assert len(calls)==2


def test_snapshot_store_reuses_core_without_repeating_prepare_or_solve(tmp_path,monkeypatch):
    import base64, threading, time
    from types import SimpleNamespace
    from datetime import datetime,timezone
    from converter_api import interactive as api
    from converter.interactive_adapter import body_binding
    raw=b'unchanged source'
    resolved=SimpleNamespace(metadata=SimpleNamespace(sha256='c'*64,rig_profile='mixamo'),path=tmp_path/'body.fbx')
    runner=SimpleNamespace(settings=SimpleNamespace(blender_binary='unused'),_process_slots=threading.Semaphore(1))
    store=api.SnapshotStore(tmp_path/'private',SimpleNamespace(resolve=lambda cid:resolved),runner)
    monkeypatch.setattr(api,'resolve_shape',lambda *a:None)
    monkeypatch.setattr(api,'camera_for_model',lambda *a:identity()['camera'])
    calls=[]
    def child(stage,job,ip,base,destination,checkpoint,*encoding):
        checkpoint();calls.append(stage);current=decode(Path(ip).read_bytes())
        if stage=='prepare':
            base.mkdir(parents=True);(base/'body.blend').write_bytes(b'base')
            (base/'base.json').write_bytes(canonical(dict(schema='interactive-body-base-v1',
                binding=body_binding(current),runtimeRevision=runtime_revision(),snapshotSha256=digest(b'base'),model=model())))
        elif stage=='publish':bundle(destination,current)
        # 'solve' only freezes staged.fbx in production; the fake publish writes the bundle.
    monkeypatch.setattr(store,'_child',child)
    selection=dict(schema='interactive-selection-v1',ownerBinding='a'*64,candidateId='c',
        body=dict(characterId='registered-fixture',assetSha256='c'*64),shape=None,shapeSource=None,
        source=dict(variant='base',originalBvhSha256=digest(raw),outputBvhSha256=digest(raw)),
        camera=dict(source_bvh_sha256=digest(raw)),hands=None,scope='full',mirror=False)
    payload=dict(selection=selection,operation='one',person='p',revision=1,bvhBase64=base64.b64encode(raw).decode(),
        expiresAt=datetime.fromtimestamp(time.time()+600,timezone.utc).isoformat())
    first=store.solve(payload)
    second=store.solve({**payload,'operation':'two','revision':2})
    assert calls==['prepare','solve','publish']
    assert first['packet']['stateHash']!=second['packet']['stateHash']
    assert second['packet']['operation']=='two'
    store.cancel('a'*64,'p',1)
    assert store.read(second['snapshotId'],'a'*64,'model')
    (store.entries[second['snapshotId']]['core']/'staged.fbx').write_bytes(b'broken cache')
    third=store.solve({**payload,'operation':'three','revision':3})
    assert calls==['prepare','solve','publish','solve','publish']
    assert third['packet']['operation']=='three'


def test_identical_concurrent_appearance_has_one_producer():
    from concurrent.futures import ThreadPoolExecutor
    import time
    from converter_api.appearance_cache import verified_appearance
    calls=[];raw=b'concurrent verified';data=b'unique concurrent source'
    job=dict(source_artifact_sha256=digest(data),binding_sha256='first')
    def produce():
        calls.append(1);time.sleep(.05)
        return raw,dict(file_verified=True,fbx_sha256=digest(raw),binding_sha256='first')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda binding:verified_appearance('concurrent',data,
            {**job,'binding_sha256':binding},'v1',produce,lambda:None),['first','second']))
    assert len(calls)==1
    assert [r[1]['binding_sha256'] for r in results]==['first','second']


def test_unverified_result_is_not_cached_and_memory_is_bounded(monkeypatch):
    from converter_api import appearance_cache as cache
    monkeypatch.setattr(cache,'_MAX_BYTES',8)
    raw=b'12345678';source=b'bounded-source';calls=[]
    job=dict(source_artifact_sha256=digest(source),binding_sha256='binding')
    def invalid():
        calls.append(1)
        return raw,dict(file_verified=False,fbx_sha256=digest(raw))
    for _ in range(2):
        with pytest.raises(ValueError):cache.verified_appearance('unverified',source,job,'runtime',invalid,lambda:None)
    assert len(calls)==2
    def verified():return raw,dict(file_verified=True,fbx_sha256=digest(raw))
    cache.verified_appearance('bounded-one',source,job,'runtime',verified,lambda:None)
    cache.verified_appearance('bounded-two',source,job,'runtime',verified,lambda:None)
    assert sum(len(v[0]) for v in cache._cache.values())<=8


def test_hand_quality_screen_cannot_be_reenabled_by_stale_environment(monkeypatch):
    from converter.hand_contact import selected_hand_screen_enabled
    monkeypatch.setenv('CONVERTER_SCREEN_SELECTED_HANDS','1')
    assert selected_hand_screen_enabled('full') is False
    assert selected_hand_screen_enabled('half') is False
