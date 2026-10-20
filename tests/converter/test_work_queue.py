"""Scheduling contention and real subprocess interruption, without Blender assets."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from converter_api.work_queue import PrioritySlots, WorkControl, WorkStopped, current_work
from converter_api.work_middleware import WorkMiddleware
from tests.converter.test_runner_failures import (
    _runner, _convert, launch_python_fixture_portably,
)
from converter_api.runner import ThumbnailRequest


def wait_until(predicate, timeout=2):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('worker did not reach expected state')
        time.sleep(.002)


def run(slots, control, action):
    token = current_work.set(control)
    try:
        with slots:
            return action(control)
    finally:
        current_work.reset(token)


def test_required_preempts_running_preview_and_overtakes_queued_preview():
    slots = PrioritySlots(continuation_seconds=.01)
    preview = WorkControl('preview')
    order = []
    started = threading.Event()
    def render(control):
        order.append('preview_start'); started.set()
        while True:
            control.check('execution')
            control.cancelled.wait(.002)
    with ThreadPoolExecutor(3) as pool:
        active = pool.submit(run, slots, preview, render)
        assert started.wait(1)
        queued = pool.submit(run, slots, WorkControl('preview'), lambda _: order.append('queued_preview'))
        wait_until(lambda: slots.snapshot()['queued']['preview'] == 1)
        mandatory = pool.submit(run, slots, WorkControl('screen', budget_seconds=.5),
                                lambda _: order.append('screen'))
        mandatory.result(1)
        with pytest.raises(WorkStopped) as error: active.result(1)
        assert error.value.code == 'PREVIEW_PREEMPTED'
        queued.result(1)
    assert order == ['preview_start', 'screen', 'queued_preview']
    assert slots.snapshot()['active'] == 0


def test_consecutive_mandatory_checks_reserve_slot_between_http_calls():
    slots = PrioritySlots(continuation_seconds=.05)
    order = []
    with ThreadPoolExecutor(1) as pool:
        run(slots, WorkControl('screen'), lambda _: order.append('screen0'))
        preview = pool.submit(run, slots, WorkControl('preview'), lambda _: order.append('preview'))
        for i in range(1, 6):
            time.sleep(.003)  # Serial HTTP roundtrip while preview is already queued.
            run(slots, WorkControl('screen'), lambda _, i=i: order.append('screen'+str(i)))
        preview.result(1)
    assert order == ['screen'+str(i) for i in range(6)]+['preview']


@pytest.mark.parametrize('stop', ['cancel', 'deadline'])
def test_stopped_queued_request_never_starts_and_releases_ticket(stop):
    slots = PrioritySlots(continuation_seconds=0)
    entered, release = threading.Event(), threading.Event()
    queued = WorkControl('preview', budget_seconds=.07 if stop == 'deadline' else 1.)
    calls = []
    with ThreadPoolExecutor(2) as pool:
        active = pool.submit(run, slots, WorkControl(), lambda _: (entered.set(), release.wait(1)))
        assert entered.wait(1)
        pending = pool.submit(run, slots, queued, lambda _: calls.append('must_not_run'))
        wait_until(lambda: slots.snapshot()['queued']['preview'] == 1)
        if stop == 'cancel': queued.cancelled.set()
        try:
            with pytest.raises(WorkStopped) as error: pending.result(1)
            assert error.value.phase == 'queue'
            assert error.value.code == ('REQUEST_CANCELLED' if stop == 'cancel' else 'WORK_BUDGET_EXHAUSTED')
            assert slots.snapshot()['queued']['preview'] == 0 and calls == []
        finally:
            release.set(); active.result(1)
    assert slots.snapshot()['active'] == 0


def test_screen_never_preempts_export_and_queue_full_is_bounded():
    slots = PrioritySlots(max_pending=1, continuation_seconds=0)
    entered, release = threading.Event(), threading.Event()
    export = WorkControl('normal')
    with ThreadPoolExecutor(2) as pool:
        active = pool.submit(run, slots, export, lambda _: (entered.set(), release.wait(1)))
        assert entered.wait(1)
        waiting = pool.submit(run, slots, WorkControl('screen'), lambda _: 'screen')
        wait_until(lambda: slots.snapshot()['queued']['screen'] == 1)
        try:
            with pytest.raises(WorkStopped) as error:
                run(slots, WorkControl('preview'), lambda _: None)
            assert error.value.code == 'WORK_QUEUE_FULL' and not export.preempted.is_set()
        finally:
            release.set(); active.result(1)
        assert waiting.result(1) == 'screen'


def test_real_28_second_optional_process_is_killed_before_15_second_screen(tmp_path):
    runner, jobs = _runner(tmp_path, timeout=30)
    marker = tmp_path/'preview-started'
    fake = tmp_path/'fake_blender'
    source = fake.read_text().replace('if mode == "timeout":',
        'if not job.get("hand_contact_screen"):\n    pathlib.Path('+repr(str(marker))+').write_text("started")')
    source = source.replace('time.sleep(10)', 'time.sleep(28)')
    fake.write_text(source)
    preview = WorkControl('preview', budget_seconds=40)
    def convert(control, **extra):
        token = current_work.set(control)
        try: return _convert(runner, tmp_path, **extra)
        finally: current_work.reset(token)
    with ThreadPoolExecutor(2) as pool:
        active = pool.submit(convert, preview, thumbnail=ThumbnailRequest())
        wait_until(marker.exists)
        start = time.monotonic()
        screen = pool.submit(convert, WorkControl('screen', budget_seconds=15), hand_contact_screen=True)
        try:
            result = screen.result(3)
        finally:
            preview.cancelled.set()
        with pytest.raises(WorkStopped) as error: active.result(2)
        assert error.value.code == 'PREVIEW_PREEMPTED'
    assert time.monotonic()-start < 2
    assert result.report['hand_contact_screen']['status'] == 'clear_screen'
    assert result.queue_wait_ms < 1000 and result.execution_ms > 0
    assert list(jobs.iterdir()) == [] and runner._process_slots.snapshot()['active'] == 0


def test_real_running_process_cancelled_and_tempdir_cleaned(tmp_path):
    runner, jobs = _runner(tmp_path, 'timeout')
    control = WorkControl('preview')
    def convert():
        token = current_work.set(control)
        try: return _convert(runner, tmp_path)
        finally: current_work.reset(token)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(convert)
        wait_until(lambda: any(jobs.iterdir()))
        control.cancelled.set()
        with pytest.raises(WorkStopped) as error: future.result(2)
    assert error.value.code == 'REQUEST_CANCELLED' and list(jobs.iterdir()) == []


def test_disconnect_after_body_reaches_sync_worker_without_parallel_receive():
    async def scenario():
        incoming = asyncio.Queue()
        await incoming.put({'type': 'http.request', 'body': b'payload', 'more_body': False})
        worker_started = threading.Event()
        observed = []
        async def app(scope, receive, send):
            body = await receive()
            assert body['body'] == b'payload'
            def worker():
                control = current_work.get(); worker_started.set()
                try:
                    while True:
                        control.check('execution'); control.cancelled.wait(.002)
                except WorkStopped as error:
                    observed.append(error.code)
            await asyncio.to_thread(worker)
        task = asyncio.create_task(WorkMiddleware(app)(
            {'type':'http', 'path':'/convert-framed', 'headers':[]}, incoming.get, lambda _: None))
        assert await asyncio.to_thread(worker_started.wait, 1)
        await incoming.put({'type': 'http.disconnect'})
        await asyncio.wait_for(task, 1)
        assert observed == ['REQUEST_CANCELLED']
    asyncio.run(scenario())


def test_work_header_cannot_promote_preview_or_extend_screen_budget():
    async def request(path, headers):
        messages = []
        incoming = asyncio.Queue()
        await incoming.put({'type':'http.request', 'body':b'', 'more_body':False})
        async def send(message): messages.append(message)
        async def app(scope, receive, send):
            await receive()
            control = current_work.get()
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':control.kind.encode()})
        await WorkMiddleware(app)({'type':'http', 'path':path, 'headers':headers}, incoming.get, send)
        return messages
    assert asyncio.run(request('/screen-hand-contact', [(b'x-standin-work-budget-ms',b'15001')]))[0]['status'] == 400
    assert asyncio.run(request('/convert-framed', [(b'x-standin-work-class',b'screen')]))[-1]['body'] == b'preview'
    assert asyncio.run(request('/screen-hand-contact', [(b'x-standin-screen-optional',b'1')]))[-1]['body'] == b'preview'


def test_mandatory_can_enter_a_queue_full_of_optional_renders():
    slots=PrioritySlots(max_pending=1,continuation_seconds=0)
    entered,release=threading.Event(),threading.Event()
    with ThreadPoolExecutor(3) as pool:
        export=pool.submit(run,slots,WorkControl(),lambda _:(entered.set(),release.wait(1)))
        assert entered.wait(1)
        preview=pool.submit(run,slots,WorkControl('preview'),lambda _:pytest.fail('evicted work started'))
        wait_until(lambda:slots.snapshot()['queued']['preview']==1)
        screen=pool.submit(run,slots,WorkControl('screen'),lambda _:'checked')
        try:
            with pytest.raises(WorkStopped) as error:preview.result(1)
            assert error.value.code=='PREVIEW_PREEMPTED' and error.value.phase=='queue'
        finally:
            release.set();export.result(1)
        assert screen.result(1)=='checked'
    assert PrioritySlots().max_pending<40


def test_broken_transport_does_not_strand_body_consumer():
    async def scenario():
        replies=[]
        async def receive():raise OSError('private transport data')
        async def send(message):replies.append(message)
        async def app(scope,receive,send):
            assert (await receive())['type']=='http.disconnect'
            assert current_work.get().cancelled.is_set()
            await send({'type':'http.response.start','status':499,'headers':[]})
            await send({'type':'http.response.body','body':b''})
        await asyncio.wait_for(WorkMiddleware(app)({'type':'http','path':'/convert-framed','headers':[]},receive,send),1)
        assert replies[0]['status']==499
    asyncio.run(scenario())


def test_final_fbx_preparation_preempts_optional_image_without_preempting_exports():
    slots = PrioritySlots(continuation_seconds=0)
    entered = threading.Event()
    image = WorkControl('preview')
    def render(control):
        entered.set()
        while True:
            control.check('execution')
            control.cancelled.wait(.002)
    with ThreadPoolExecutor(2) as pool:
        preview = pool.submit(run, slots, image, render)
        assert entered.wait(1)
        ready = pool.submit(run, slots, WorkControl('normal'), lambda _: 'verified-fbx')
        assert ready.result(1) == 'verified-fbx'
        with pytest.raises(WorkStopped) as error:
            preview.result(1)
        assert error.value.code == 'PREVIEW_PREEMPTED'
    assert slots.snapshot()['active'] == 0
