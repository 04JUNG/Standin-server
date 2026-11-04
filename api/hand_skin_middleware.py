"""Bind the analysis deadline and screening cancellation to one /analyze lifetime."""
import asyncio
import threading
from time import monotonic
from src.analysis_budget import analysis_deadline
from starlette.responses import JSONResponse

from src.hand_skin_parallel import request_cancelled


class HandSkinMiddleware:
    def __init__(self, app, timeout_seconds=120.):
        self.app = app
        self.timeout_seconds = min(120., max(.001, timeout_seconds))

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/analyze':
            return await self.app(scope, receive, send)
        cancelled = threading.Event()
        token = request_cancelled.set(cancelled)
        deadline_token = analysis_deadline.set(monotonic() + self.timeout_seconds)
        response_started = False
        async def bounded_send(message):
            nonlocal response_started
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)
        inbox = asyncio.Queue(maxsize=1)

        async def pump():
            # One receive reader; preserve multipart streaming/backpressure.
            try:
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        cancelled.set()
                        await inbox.put(message)
                        return
                    await inbox.put(message)
            except Exception:
                cancelled.set()
                await inbox.put({'type': 'http.disconnect'})

        task = asyncio.create_task(pump())
        try:
            limiter = asyncio.timeout(self.timeout_seconds)
            try:
                async with limiter:
                    await self.app(scope, inbox.get, bounded_send)
            except asyncio.TimeoutError:
                if not limiter.expired():
                    raise  # A dependency timeout is not the 120s request deadline.
                cancelled.set()
                if not response_started:
                    await JSONResponse(status_code=504, content={"detail": {
                        "code": "ANALYSIS_TIMEOUT", "message": "분석 제한 시간(120초)을 초과했습니다."}})(scope, receive, send)
        finally:
            cancelled.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            analysis_deadline.reset(deadline_token)
            request_cancelled.reset(token)
