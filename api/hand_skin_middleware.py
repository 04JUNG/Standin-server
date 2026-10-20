"""Bind optional parallel screening cancellation to one /analyze lifetime."""
import asyncio
import threading

from src.hand_skin_parallel import request_cancelled


class HandSkinMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/analyze':
            return await self.app(scope, receive, send)
        cancelled = threading.Event()
        token = request_cancelled.set(cancelled)
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
            await self.app(scope, inbox.get, send)
        finally:
            cancelled.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            request_cancelled.reset(token)
