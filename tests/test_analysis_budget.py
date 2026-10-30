import asyncio
import json
from time import monotonic

from src.analysis_budget import analysis_budget, analysis_deadline
from src.hand_skin_parallel import request_cancelled
from api.hand_skin_middleware import HandSkinMiddleware


def test_nested_budget_never_refreshes_request_deadline():
    with analysis_budget(120, clock=lambda: 0):
        with analysis_budget(120, clock=lambda: 80):
            assert analysis_deadline.get() == 120
    assert analysis_deadline.get() is None


def test_http_deadline_responds_504_and_cancels_optional_workers():
    sent, events = [], []
    async def app(scope, receive, send):
        events.append(request_cancelled.get())
        assert analysis_deadline.get() is not None
        await asyncio.sleep(1)
    async def receive():
        await asyncio.sleep(2)
    async def send(msg): sent.append(msg)
    started = monotonic()
    asyncio.run(HandSkinMiddleware(app, timeout_seconds=.025)(
        {'type': 'http', 'path': '/analyze'}, receive, send))
    assert monotonic() - started < .5
    assert sent[0]['status'] == 504
    assert json.loads(sent[1]['body'])['detail']['code'] == 'ANALYSIS_TIMEOUT'
    assert events[0].is_set()
    assert analysis_deadline.get() is None and request_cancelled.get() is None


def test_unrelated_route_has_no_deadline():
    async def app(scope, receive, send):
        assert analysis_deadline.get() is None
    asyncio.run(HandSkinMiddleware(app)({'type': 'http', 'path': '/healthz'}, None, None))


def test_real_sync_route_timeout_does_not_wait_for_worker():
    import threading
    import httpx
    from fastapi import FastAPI
    release = threading.Event()
    entered = threading.Event()
    request_events = []
    app = FastAPI()
    @app.post('/analyze')
    def analyze():
        request_events.append(request_cancelled.get())
        entered.set()
        release.wait(1)
        return {'done': True}
    app.add_middleware(HandSkinMiddleware, timeout_seconds=.05)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            started = monotonic()
            try:
                response = await client.post('/analyze')
                assert response.status_code == 504
                assert entered.is_set() and request_events[0].is_set()
                assert monotonic() - started < .5
                assert not release.is_set()
            finally:
                release.set()
    asyncio.run(run())


def test_dependency_timeout_is_not_reported_as_request_deadline():
    import pytest
    async def app(scope, receive, send):
        raise TimeoutError("dependency_failed_early")
    async def receive(): await asyncio.sleep(2)
    async def send(_): raise AssertionError("must not send a deadline response")
    with pytest.raises(TimeoutError, match="dependency_failed_early"):
        asyncio.run(HandSkinMiddleware(app)({'type': 'http', 'path': '/analyze'}, receive, send))
