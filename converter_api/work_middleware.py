"""ASGI request lifetime -> thread-safe Blender cancellation/deadline control."""
import asyncio
import re
from .work_queue import WorkControl, current_work


class WorkMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path = scope.get('path', '')
        headers = dict(scope.get('headers', []))
        screening = path == '/screen-hand-contact'
        optional = headers.get(b'x-standin-screen-optional') == b'1'
        kind = ('screen' if screening and not optional else 'preview'
                if screening or path in ('/convert-framed', '/render-thumbnail') else 'normal')
        maximum = 15_000 if screening else 300_000
        value = headers.get(b'x-standin-work-budget-ms', str(maximum).encode())
        if not value.isdigit() or len(value) > 9 or not 1 <= int(value) <= maximum:
            await send({'type': 'http.response.start', 'status': 400,
                        'headers': [(b'content-type', b'application/json')]})
            await send({'type': 'http.response.body', 'body': b'{"error":{"code":"INVALID_WORK_BUDGET"}}'})
            return
        control = WorkControl(kind=kind, budget_seconds=int(value)/1000.)
        request_id = headers.get(b'x-request-id', b'')
        if re.fullmatch(rb'req_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}', request_id):
            control.request_id = request_id.decode('ascii')
        token = current_work.set(control)
        inbox = asyncio.Queue(maxsize=1)

        async def pump():
            # Exactly one reader of the underlying receive channel. Bounded
            # buffering preserves upload backpressure and StreamingResponse use.
            try:
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        control.cancelled.set()
                        await inbox.put(message)
                        return
                    await inbox.put(message)
            except Exception:
                # A broken transport must not strand a worker or body consumer.
                # Never log the transport exception's potentially private text.
                control.cancelled.set()
                await inbox.put({'type': 'http.disconnect'})

        async def monitored_send(message):
            if message['type'] == 'http.response.start':
                meta = control.diagnostics()
                extra = [(b'x-standin-work-class', control.kind.encode()),
                         (b'x-standin-work-phase', control.phase.encode()),
                         (b'x-standin-work-reason', control.reason.encode()),
                         (b'x-standin-queue-wait-ms', str(meta['queue_wait_ms']).encode()),
                         (b'x-standin-execution-ms', str(meta['execution_ms']).encode())]
                existing = {key.lower() for key, _ in message.get('headers', [])}
                if control.conversion_id and b'x-standin-conversion-id' not in existing:
                    extra.append((b'x-standin-conversion-id', control.conversion_id.encode()))
                if control.reason in ('PREVIEW_PREEMPTED', 'WORK_QUEUE_FULL'):
                    extra.append((b'retry-after', b'1'))
                message = {**message, 'headers': list(message.get('headers', []))+extra}
            await send(message)

        task = asyncio.create_task(pump())
        try:
            await self.app(scope, inbox.get, monitored_send)
        finally:
            control.cancelled.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            current_work.reset(token)
