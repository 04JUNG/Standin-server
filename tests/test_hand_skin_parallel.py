"""Bounded parallel screening and cross-request state isolation; no network/models."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from itertools import product, chain, repeat
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from converter.hand_contact import validate_screen, SCREEN_VERSION
from converter.hand_library import load_library, normalize_request
from converter.protocol import SOLVER_VERSION
from src.hand_skin_screen import HandSkinGate, SkinSession, unknown
from src.hand_skin_parallel import ParallelSkinSession, Outcome, ScreenSlots, request_cancelled
from src.logging_setup import request_id_var
from tests.test_fullbody_routing import engine, decision, evidence, Camera, Semantic
from api.hand_skin_middleware import HandSkinMiddleware


def gate(tmp_path,client,parallel=True,ids=('a','b','c'),**kw):
    hashes={}
    for pid in ids:
        data='BVH '+pid;(tmp_path/(pid+'.bvh')).write_text(data)
        hashes[pid]=hashlib.sha256(data.encode()).hexdigest()
    return HandSkinGate(client,tmp_path,hashes,concurrency=2 if parallel else 1,**kw)


def cancelled_session(g,cancel,hands=None):
    token=request_cancelled.set(cancel)
    try:return g.session(hands)
    finally:request_cancelled.reset(token)

def receipt(text, hands, status='clear_screen'):
    return dict(status=status, method=SCREEN_VERSION, scope='opposite_hands_only',
        complete_collision_certificate=False, triangle_counts={'left':12,'right':12},
        intersection_pairs=int(status=='intersection'), examples=[],
        source_bvh_sha256=hashlib.sha256(text.encode()).hexdigest(),character_id='master',
        character_sha256='a'*64,hands=normalize_request(hands),solver_version=SOLVER_VERSION,
        hand_library_sha256=load_library().sha256,screened_artifact_sha256='b'*64)

class Checker:
    def __init__(self,statuses=None,delays=None,started=None,release=None):
        self.statuses=statuses or {};self.delays=delays or {};self.calls=[]
        self.started=started;self.release=release
    def check(self,text,hands,timeout):
        pid=text.removeprefix('BVH ')
        self.calls.append((pid,hands,timeout,getattr(self,'_screen_optional',False),request_id_var.get()))
        if self.started:self.started.set()
        if self.release:self.release.wait(timeout)
        delay=self.delays.get(pid,0.)
        if delay:time.sleep(min(delay,timeout))
        if delay>timeout:return unknown('hand_screen_request_timeout')
        status=self.statuses.get(pid,'clear_screen')
        if status=='raise':raise RuntimeError('private details must not escape')
        if status=='malformed':return {'status':'clear_screen'}
        if status=='unavailable':return unknown('offline')
        return receipt(text,hands,status)

def signature(out):
    return (out['status'],out.get('reason'),
        [(r['pose_id'],r['rank_score'],r['release_allowed'],r['refine_allowed']) for r in out['selected']],
        (out.get('generation') or {}).get('status'))

@pytest.mark.parametrize('statuses',list(product(['clear_screen','intersection','unavailable'],repeat=3)))
@pytest.mark.parametrize('top_k',[1,2])
def test_fullbody_fusion_failures_and_generation_contract_same(tmp_path,statuses,top_k):
    outputs=[]
    for parallel in (False,True):
        client=Checker(dict(zip(('a','b','c'),statuses)))
        e=engine();e.top_k=top_k;e.hand_skin_gate=gate(tmp_path,client,parallel)
        out=e.search(decision(),evidence());outputs.append(out)
        assert all(r['hand_skin_screen']['status']=='clear_screen' for r in out['selected'])
        assert all(not r.get('release_allowed') and not r.get('refine_allowed') for r in out['selected'])
        if out.get('hand_skin_screen',{}).get('unverified'):assert e.generator.calls==0
    assert signature(outputs[0])==signature(outputs[1])

def test_completion_order_never_changes_ranking(tmp_path):
    client=Checker(delays={'a':.02,'b':.001,'c':.001})
    s=gate(tmp_path,client).session()
    assert [r['pose_id'] for r in s.filter([{'pose_id':p} for p in ('a','b','c')])]==['a','b','c']

def test_max_checks_and_target_count_bound_submissions(tmp_path):
    c=Checker();s=gate(tmp_path,c,max_checks=1).session()
    assert [r['pose_id'] for r in s.filter([{'pose_id':p} for p in ('a','b','c')])]==['a']
    assert len(c.calls)==1 and s.checks==1 and s.unverified
    c=Checker();s=gate(tmp_path,c).session()
    assert len(s.filter([{'pose_id':p} for p in ('a','b','c')],limit=1))==1
    assert len(c.calls)==1

def test_optional_priority_and_request_trace_are_session_local(tmp_path):
    c=Checker();g=gate(tmp_path,c);s=g.session();other=g.session()
    token=request_id_var.set('req_local_trace')
    try:
        with s.optional_phase():s.filter([{'pose_id':'a'},{'pose_id':'b'}])
        other.filter([{'pose_id':'c'}])
    finally:request_id_var.reset(token)
    assert [r[3] for r in c.calls]==[True,True,False]
    assert all(r[4]=='req_local_trace' for r in c.calls)

def test_duplicate_bytes_screened_once_and_no_bvh_leaks(tmp_path):
    c=Checker({'a':'intersection'});s=gate(tmp_path,c).session()
    rows=[{'pose_id':'a','bvh':'BVH a'},{'pose_id':'b','bvh':'BVH a'}]
    assert not s.filter(rows,generated=True)
    assert len(c.calls)==1 and len(s.rejected)==2
    assert all('bvh' not in r and not r['release_allowed'] for r in s.rejected)

def test_optional_failure_does_not_revoke_committed_main_matches(tmp_path):
    c=Checker({'c':'unavailable'});s=gate(tmp_path,c).session()
    accepted=s.filter([{'pose_id':'a'},{'pose_id':'b'}]);main=deepcopy(s.report())
    with s.optional_phase():assert s.filter([{'pose_id':'c'}])==[]
    assert len(accepted)==2 and not main['unverified'] and s.report()['unverified']


def test_default_is_original_session_and_factory_can_opt_in(tmp_path,monkeypatch):
    g=gate(tmp_path,Checker(),False)
    assert type(g.session()) is SkinSession
    from src.config import CFG
    from src.hand_skin_screen import build_hand_skin_gate
    monkeypatch.setattr(CFG,'hand_skin_screen_concurrency',2)
    configured=build_hand_skin_gate(SimpleNamespace(hashes=g.hashes),tmp_path)
    assert type(configured.session()) is ParallelSkinSession


@pytest.mark.parametrize('concurrency',[0,6,-1,True,1.5,'2'])
def test_invalid_concurrency_is_rejected(tmp_path,concurrency):
    with pytest.raises(ValueError):HandSkinGate(Checker(),tmp_path,{},concurrency=concurrency)


def test_no_receipts_or_optional_flags_shared_across_users(tmp_path):
    checker=Checker();g=gate(tmp_path,checker)
    a=g.session({'left':'fist'});b=g.session({'right':'relaxed'});c=g.session({'left':'fist'})
    first=a.filter([{'pose_id':'a'}])
    assert a.filter([{'pose_id':'a'}])==first and len(checker.calls)==1
    with b.optional_phase():assert b.filter([{'pose_id':'a'}])
    assert c.filter([{'pose_id':'a'}]) and len(checker.calls)==3
    assert a.cache is not b.cache and a.cache is not c.cache
    assert a.cancelled is not b.cancelled and a.events is not b.events
    assert [call[3] for call in checker.calls]==[False,True,False]
    first[0]['hand_skin_screen']['status']='intersection'
    assert a.filter([{'pose_id':'a'}])[0]['hand_skin_screen']['status']=='clear_screen'


def test_presets_snapshotted_per_person(tmp_path):
    hands={'left':'fist'};client=Checker();s=gate(tmp_path,client).session(hands)
    hands['left']='relaxed'
    s.filter([{'pose_id':'a'}]);assert client.calls[0][1]=={'left':'fist'}


def test_global_bound_and_trace_isolation_across_concurrent_requests(tmp_path):
    state=dict(active=0,peak=0);lock=threading.Lock();barrier=threading.Barrier(3)
    class BusyChecker(Checker):
        def check(self,text,hands,timeout):
            with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
            try:
                time.sleep(.01)
                return super().check(text,hands,timeout)
            finally:
                with lock:state['active']-=1
    client=BusyChecker();g=gate(tmp_path,client)
    def run(name,hands):
        token=request_id_var.set(name)
        try:
            s=g.session(hands);barrier.wait(1)
            return s.filter([{'pose_id':p} for p in ('a','b','c')])
        finally:request_id_var.reset(token)
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(run,'user-a',{'left':'fist'});b=pool.submit(run,'user-b',{'right':'relaxed'})
        barrier.wait(1);assert len(a.result(2))==len(b.result(2))==3
    assert state['peak']==2 and len(client.calls)==6 and state['active']==0
    assert all(trace==('user-a' if hands=={'left':'fist'} else 'user-b') for _,hands,_,_,trace in client.calls)


@pytest.mark.parametrize('entry',['filter','check'])
def test_active_wall_deadline_discards_late_success(tmp_path,monkeypatch,entry):
    s=gate(tmp_path,Checker(),budget_seconds=.5).session();ticks=chain([0.,.6],repeat(.6))
    monkeypatch.setattr('src.hand_skin_parallel.monotonic',lambda:next(ticks))
    monkeypatch.setattr(s,'execute',lambda text,budget,optional:Outcome(receipt(text,None),.01,budget))
    if entry=='filter':assert not s.filter([{'pose_id':'a'}])
    else:assert s.check('BVH a')['status']=='unavailable'
    assert s.failed and not s.cache


def test_late_transport_success_is_unavailable_and_not_cached(tmp_path,monkeypatch):
    monkeypatch.setattr('src.hand_skin_parallel.CALL_SECONDS',.005)
    class LateChecker(Checker):
        def check(self,text,hands,timeout):
            time.sleep(.01)
            return receipt(text,hands)
    s=gate(tmp_path,LateChecker()).session()
    assert not s.filter([{'pose_id':'a'},{'pose_id':'b'}])
    assert s.failed and not s.cache


def test_cancel_one_user_does_not_cancel_other_or_reuse_receipt(tmp_path):
    cancel=threading.Event();cancel.set();client=Checker();g=gate(tmp_path,client)
    a=cancelled_session(g,cancel);b=g.session()
    assert not a.filter([{'pose_id':'a'}]) and not client.calls
    assert b.filter([{'pose_id':'a'}]) and len(client.calls)==1


def test_cancel_inflight_discards_both_successes_and_starts_no_new_calls(tmp_path):
    entered=threading.Event();release=threading.Event();cancel=threading.Event()
    client=Checker(started=entered,release=release);s=cancelled_session(gate(tmp_path,client),cancel)
    with ThreadPoolExecutor(1) as pool:
        f=pool.submit(s.filter,[{'pose_id':p} for p in ('a','b','c')])
        assert entered.wait(1);cancel.set();release.set();assert not f.result(1)
    assert len(client.calls)<=2 and s.failed and not s.cache


def test_optional_leaves_capacity_for_required_work_and_never_queues_ahead():
    slots=ScreenSlots()
    assert slots.acquire(.01,optional=True)
    assert not slots.acquire(.01,optional=True)
    assert slots.acquire(.01)  # Required check can reach converter preemption.
    assert not slots.acquire(.001)
    slots.release();slots.release();assert slots.active==0


def test_changed_or_unsafe_path_cannot_reuse_a_session_safe_receipt(tmp_path):
    client=Checker();g=gate(tmp_path,client);s=g.session()
    assert s.filter([{'pose_id':'a'}]);before=len(client.calls)
    (tmp_path/'a.bvh').write_text('changed')
    assert not s.filter([{'pose_id':'a'},{'pose_id':'../escape'}])
    assert len(client.calls)==before and s.unverified


def test_disconnect_reaches_running_sync_screen_and_resets_context(tmp_path):
    async def scenario():
        incoming=asyncio.Queue();await incoming.put({'type':'http.request','body':b'fixture','more_body':False})
        entered=threading.Event();release=threading.Event();client=Checker(started=entered,release=release)
        g=gate(tmp_path,client);observed=[]
        async def app(scope,receive,send):
            assert (await receive())['body']==b'fixture'
            s=g.session();task=asyncio.create_task(asyncio.to_thread(s.filter,[{'pose_id':'a'}]))
            assert await asyncio.to_thread(entered.wait,1)
            # The middleware owns the only underlying receive reader.
            await incoming.put({'type':'http.disconnect'})
            assert (await receive())['type']=='http.disconnect'
            assert s.cancelled.is_set();release.set();observed.extend(await task)
        await asyncio.wait_for(HandSkinMiddleware(app)({'type':'http','path':'/analyze'},incoming.get,lambda _:None),2)
        assert observed==[] and request_cancelled.get() is None
    asyncio.run(scenario())


@pytest.mark.parametrize('failure',['unavailable','raise','malformed'])
def test_speculative_success_after_failure_never_committed(tmp_path,failure):
    client=Checker({'a':failure},delays={'a':.008});s=gate(tmp_path,client).session()
    assert not s.filter([{'pose_id':'a'},{'pose_id':'b'}])
    assert s.failed and not s.cache and s.report()['discarded_speculative']==1


def test_small_budget_reserves_one_full_call_at_a_time(tmp_path,monkeypatch):
    monkeypatch.setattr('src.hand_skin_parallel.CALL_SECONDS',.2)
    client=Checker(delays={'a':.005,'b':.005})
    s=gate(tmp_path,client,budget_seconds=.25).session()
    assert len(s.filter([{'pose_id':'a'},{'pose_id':'b'}]))==2
    assert all(.19<call[2]<=.2 for call in client.calls)


def test_budget_shared_by_library_generation_and_optional_phase(tmp_path,monkeypatch):
    monkeypatch.setattr('src.hand_skin_parallel.CALL_SECONDS',.2)
    client=Checker(delays={'a':.12,'b':.12});s=gate(tmp_path,client,budget_seconds=.2).session()
    assert s.filter([{'pose_id':'a'}])
    assert not s.filter([{'bvh':'BVH b'}],generated=True)
    count=len(client.calls)
    with s.optional_phase():assert not s.filter([{'pose_id':'c'}])
    assert s.failed and len(client.calls)==count and s.unverified


def test_broken_receive_sets_only_its_request_event_and_cleans_up():
    async def scenario():
        async def receive():raise OSError('private transport detail')
        async def app(scope,receive,send):
            assert (await receive())['type']=='http.disconnect'
            assert request_cancelled.get().is_set()
        await asyncio.wait_for(HandSkinMiddleware(app)({'type':'http','path':'/analyze'},receive,lambda _:None),1)
        assert request_cancelled.get() is None
    asyncio.run(scenario())
