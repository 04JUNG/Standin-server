"""Static release certificates cannot become a cache of user-generated poses."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from itertools import product
from types import SimpleNamespace

import pytest

from converter.hand_library import load_library
from converter.hand_screen_catalog import (
    LibraryScreenCatalog, PROFILE_VERSION, binding, make_catalog, runtime_profile)
from converter_api.runner import BlenderInfo, BlenderRunner, RunnerSettings
from src.config import Config
from src.hand_skin_screen import HandSkinGate
from src.hand_skin_screen import HandSkinClient
from tests.test_hand_skin_parallel import Checker, receipt
from tests.test_hand_skin_parallel_contracts import normalize


class Client(Checker):
    character_id = 'master'
    character_sha256 = 'a' * 64
    def __init__(self, statuses=None):
        super().__init__(statuses)
        self.profile = dict(version=PROFILE_VERSION, profile_sha256='c' * 64,
            character_id=self.character_id, character_sha256=self.character_sha256,
            shape_sha256=None,
            hand_library_sha256=load_library().sha256)
        self.state = SimpleNamespace(profile_calls=0)

    def screening_profile(self, timeout):
        self.state.profile_calls += 1
        return deepcopy(self.profile)


def catalog_file(tmp_path, statuses=None, hands=None):
    statuses = statuses or {'a':'clear_screen', 'b':'intersection', 'c':'clear_screen'}
    pairs = []
    hashes = {}
    for pid, status in statuses.items():
        text = 'BVH ' + pid
        path = tmp_path/(pid+'.bvh');path.write_text(text)
        hashes[pid] = hashlib.sha256(text.encode()).hexdigest()
        pairs.append((binding(pid, hashes[pid], 'master', 'a'*64, hands, 'c'*64), receipt(text, hands, status)))
    raw = json.dumps(make_catalog(pairs)).encode()
    path = tmp_path/'catalog.json';path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), hashes


@pytest.mark.parametrize('capacity', range(1, 6))
def test_static_results_identical_and_zero_live_calls(tmp_path, capacity):
    path, sha, hashes = catalog_file(tmp_path)
    client = Client({'b':'intersection'})
    g = HandSkinGate(client, tmp_path, hashes, concurrency=capacity,
        catalog=LibraryScreenCatalog(path,sha))
    s = g.session();actual=s.filter([{'pose_id':p} for p in hashes])
    other = HandSkinGate(Client({'b':'intersection'}),tmp_path,hashes,concurrency=capacity).session()
    expected=other.filter([{'pose_id':p} for p in hashes])
    assert normalize(actual)==normalize(expected)
    assert normalize(s.rejected)==normalize(other.rejected)
    assert not client.calls and s.checks == other.checks == 3
    assert client.state.profile_calls == 1
    assert all(e['source']=='prevalidated_library' for e in s.events)
    # Every request verifies the live identity; no shared per-user/receipt cache.
    g.session().filter([{'pose_id':'a'}])
    assert client.state.profile_calls == 2 and not client.calls


@pytest.mark.parametrize('capacity', [1,2,5])
@pytest.mark.parametrize('change', ['profile','character','hand_library','hands','bvh','pose_id','no_profile','shape','profile_shape'])
def test_mismatch_always_uses_live_checker(tmp_path,capacity,change):
    path, sha, hashes = catalog_file(tmp_path)
    client=Client();hands=None;pid='a'
    if change=='profile':client.profile['profile_sha256']='d'*64
    if change=='character':client.character_sha256='d'*64
    if change=='hand_library':client.profile['hand_library_sha256']='d'*64
    if change=='hands':hands={'left':'fist'}
    if change=='bvh':
        (tmp_path/'a.bvh').write_text('BVH changed')
        hashes['a']=hashlib.sha256(b'BVH changed').hexdigest()
    if change=='pose_id':
        pid='other';(tmp_path/'other.bvh').write_text('BVH a');hashes[pid]=hashes['a']
    if change=='no_profile':client.profile=None
    if change=='shape':client.shape_sha256='d'*64
    if change=='profile_shape':client.profile['shape_sha256']='d'*64
    s=HandSkinGate(client,tmp_path,hashes,concurrency=capacity,catalog=LibraryScreenCatalog(path,sha)).session(hands)
    assert s.filter([{'pose_id':pid}])
    assert len(client.calls)==1


def test_shape_recipe_cannot_borrow_a_base_body_receipt():
    def forbidden(*a,**kw):raise AssertionError('a base-body request must not be sent')
    client=HandSkinClient('http://local','master','a'*64,opener=forbidden,shape_sha256='d'*64)
    assert client.screening_profile() is None
    result=client.check('BVH a',None,15)
    assert result['status']=='unavailable' and result['reason']=='hand_screen_body_shape_unsupported'


@pytest.mark.parametrize('capacity', [1,2,5])
def test_generated_bvh_never_uses_library_certificate_even_when_bytes_match(tmp_path,capacity):
    path,sha,hashes=catalog_file(tmp_path)
    client=Client({'a':'intersection'})
    s=HandSkinGate(client,tmp_path,hashes,concurrency=capacity,catalog=LibraryScreenCatalog(path,sha)).session()
    assert not s.filter([{'pose_id':'a','bvh':'BVH a'}],generated=True)
    assert len(client.calls)==1 and client.state.profile_calls==0


@pytest.mark.parametrize('capacity',[1,2,5])
def test_generated_bytes_still_live_after_library_hit_in_same_session(tmp_path,capacity):
    path,sha,hashes=catalog_file(tmp_path)
    client=Client({'a':'intersection'})
    s=HandSkinGate(client,tmp_path,hashes,concurrency=capacity,catalog=LibraryScreenCatalog(path,sha)).session()
    assert s.filter([{'pose_id':'a'}]) and not client.calls
    assert not s.filter([{'pose_id':'a','bvh':'BVH a'}],generated=True)
    assert len(client.calls)==1 and s.rejected


@pytest.mark.parametrize('capacity',[1,2,5])
def test_runtime_failure_still_fails_closed_after_profile_miss(tmp_path,capacity):
    path,sha,hashes=catalog_file(tmp_path)
    client=Client({'a':'unavailable'});client.profile['profile_sha256']='d'*64
    s=HandSkinGate(client,tmp_path,hashes,concurrency=capacity,catalog=LibraryScreenCatalog(path,sha)).session()
    assert not s.filter([{'pose_id':'a'}]) and s.unverified


def test_bvh_integrity_is_checked_before_any_certificate_lookup(tmp_path):
    path,sha,hashes=catalog_file(tmp_path)
    (tmp_path/'a.bvh').write_text('BVH changed')
    client=Client();s=HandSkinGate(client,tmp_path,hashes,catalog=LibraryScreenCatalog(path,sha)).session()
    assert not s.filter([{'pose_id':'a'}]) and s.unverified
    assert not client.calls and client.state.profile_calls==0


@pytest.mark.parametrize('capacity',[1,2,5])
@pytest.mark.parametrize('statuses',list(product(['clear_screen','intersection','unavailable'],repeat=3)))
def test_mixed_certificates_and_live_failures_preserve_serial_decisions(tmp_path,capacity,statuses):
    status_map=dict(zip(('a','b','c'),statuses))
    # A transient failure is never certified. Keep an unused valid entry so
    # the all-unavailable case also exercises an enabled catalog.
    certified={p:s for p,s in status_map.items() if s!='unavailable'} | {'unused':'clear_screen'}
    path,sha,hashes=catalog_file(tmp_path,certified)
    for pid in status_map:
        text='BVH '+pid;(tmp_path/(pid+'.bvh')).write_text(text)
        hashes[pid]=hashlib.sha256(text.encode()).hexdigest()
    a=HandSkinGate(Client(status_map),tmp_path,hashes).session()
    b=HandSkinGate(Client(status_map),tmp_path,hashes,concurrency=capacity,
        catalog=LibraryScreenCatalog(path,sha)).session()
    candidates=[{'pose_id':p} for p in status_map]
    assert normalize(a.filter(candidates))==normalize(b.filter(candidates))
    assert normalize(a.rejected)==normalize(b.rejected)
    assert normalize(a.unverified)==normalize(b.unverified)
    assert a.checks==b.checks


def test_catalog_is_deepcopied_and_cannot_be_mutated_by_request_results(tmp_path):
    path,sha,hashes=catalog_file(tmp_path)
    g=HandSkinGate(Client(),tmp_path,hashes,catalog=LibraryScreenCatalog(path,sha))
    a=g.session().filter([{'pose_id':'a'}]);a[0]['hand_skin_screen']['status']='intersection'
    assert g.session().filter([{'pose_id':'a'}])[0]['hand_skin_screen']['status']=='clear_screen'


@pytest.mark.parametrize('capacity',[1,2])
def test_hand_presets_are_snapshotted_before_library_validation(tmp_path,capacity):
    hands={'left':'fist'}
    path,sha,hashes=catalog_file(tmp_path,hands=hands)
    client=Client();session=HandSkinGate(client,tmp_path,hashes,concurrency=capacity,
        catalog=LibraryScreenCatalog(path,sha)).session(hands)
    hands['left']='relaxed'
    assert session.filter([{'pose_id':'a'}]) and not client.calls
    assert session.hands=={'left':'fist'}


def test_request_cancellation_revokes_static_results(tmp_path):
    import threading
    from src.hand_skin_parallel import request_cancelled
    path,sha,hashes=catalog_file(tmp_path)
    client=Client();cancel=threading.Event()
    original=client.screening_profile
    def cancel_on_profile(timeout):
        cancel.set()
        return original(timeout)
    client.screening_profile=cancel_on_profile
    token=request_cancelled.set(cancel)
    try:
        session=HandSkinGate(client,tmp_path,hashes,concurrency=2,
            catalog=LibraryScreenCatalog(path,sha)).session()
        assert not session.filter([{'pose_id':'a'}]) and not client.calls
    finally:request_cancelled.reset(token)


@pytest.mark.parametrize('statuses',list(product(['clear_screen','intersection','unavailable'],repeat=3)))
def test_library_catalog_preserves_search_and_presentation_permissions(tmp_path,statuses):
    from api.pose_presentation import build_pose_presentation
    from tests.test_fullbody_routing import engine,decision,evidence
    from tests.test_hand_skin_parallel_contracts import asset
    statuses=dict(zip(('a','b','c'),statuses))
    path,sha,hashes=catalog_file(tmp_path,{p:s for p,s in statuses.items() if s!='unavailable'}|{'unused':'clear_screen'})
    for pid in statuses:
        text='BVH '+pid;(tmp_path/(pid+'.bvh')).write_text(text)
        hashes[pid]=hashlib.sha256(text.encode()).hexdigest()
    outputs=[]
    for enabled in (False,True):
        e=engine();e.hand_skin_gate=HandSkinGate(Client(statuses),tmp_path,hashes,concurrency=2,
            catalog=LibraryScreenCatalog(path,sha) if enabled else None)
        raw=e.search(decision(),evidence())
        outputs.append(build_pose_presentation(raw,output_scope='full',resolve_library=asset).model_dump())
    assert normalize(outputs[0])==normalize(outputs[1])


@pytest.mark.parametrize('problem',['wrong_pin','tamper','invalid_receipt','wrong_binding','transient','duplicate_key'])
def test_untrusted_catalog_fails_before_service_activation(tmp_path,problem):
    path,sha,_=catalog_file(tmp_path)
    data=json.loads(path.read_text());key=next(iter(data['entries']));row=data['entries'][key]
    if problem=='wrong_pin':sha='0'*64
    if problem=='tamper':path.write_text(path.read_text()+' ')
    if problem in ('invalid_receipt','wrong_binding','transient','duplicate_key'):
        if problem=='invalid_receipt':row['receipt']['intersection_pairs']=1
        if problem=='wrong_binding':row['binding']['character_sha256']='d'*64
        if problem=='transient':row['receipt'].update(status='unavailable',reason='timeout')
        raw=json.dumps(data)
        if problem=='duplicate_key':raw=raw[:-1]+',"version":"hand-screen-library.v1"}'
        path.write_text(raw);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):LibraryScreenCatalog(path,sha)


@pytest.mark.parametrize('settings',[{}, {'hand_skin_prevalidated_path':'x'}, {'hand_skin_prevalidated_sha256':'a'*64},
    {'hand_skin_prevalidated_path':'x','hand_skin_prevalidated_sha256':'bad'}])
def test_configuration_requires_both_path_and_pin(settings):
    if not settings:
        assert not Config().hand_skin_prevalidated_path
    else:
        with pytest.raises(ValueError):Config(**settings)


def test_profile_changes_with_code_runtime_and_geometry_options(tmp_path,monkeypatch):
    for name in ('converter/worker.py','converter/hand_contact.py','converter_api/app.py'):
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('# baseline')
    runner=BlenderRunner(RunnerSettings(blender_binary='unused',worker_path=tmp_path/'converter/worker.py'))
    runner.inspect_blender=lambda:BlenderInfo(version='5.2.0',build_hash='fbe6228777e7')
    baseline=runtime_profile(runner)
    file=tmp_path/'converter/hand_contact.py';file.write_text('# changed')
    assert runtime_profile(runner)!=baseline;file.write_text('# baseline')
    assert runtime_profile(runner)==baseline
    for env in ({'STANDIN_WRIST_TRANSFER':'0'},{'STANDIN_WRIST_TWIST_FOREARM_SHARE':'0.8'}):
        runner.settings=replace(runner.settings,process_env=env)
        assert runtime_profile(runner)!=baseline
    runner.settings=replace(runner.settings,process_env=None,force_exact_v324=True)
    assert runtime_profile(runner)!=baseline
    runner.settings=replace(runner.settings,force_exact_v324=False)
    monkeypatch.setattr('converter.hand_screen_catalog.platform.machine',lambda:'different-architecture')
    assert runtime_profile(runner)!=baseline
