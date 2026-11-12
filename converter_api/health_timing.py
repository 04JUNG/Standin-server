"""Readiness latency evidence without changing health checks or their results."""
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import os
import re
import time


@dataclass
class HealthTrace:
    started: float = field(default_factory=lambda: time.perf_counter())
    handler_started: float | None = None
    stages: dict = field(default_factory=dict)


current_health: ContextVar[HealthTrace | None] = ContextVar('health_trace', default=None)


def health_handler_started():
    trace = current_health.get()
    if trace is not None:
        trace.handler_started = time.perf_counter()


def measured_health(stage, action, *args, **kwargs):
    trace = current_health.get()
    if trace is None:
        return action(*args, **kwargs)
    started = time.perf_counter()
    succeeded = False
    try:
        result = action(*args, **kwargs)
        succeeded = True
        return result
    finally:
        trace.stages[stage] = {
            'duration_ms': round((time.perf_counter() - started) * 1000, 3),
            'succeeded': succeeded,
        }


class HealthTimingMiddleware:
    """Include sync-handler admission delay, which stage timers cannot observe."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/healthz':
            return await self.app(scope, receive, send)
        trace = HealthTrace()
        token = current_health.set(trace)
        status = None
        request_id = dict(scope.get('headers', [])).get(b'x-request-id', b'')
        safe_id = (request_id.decode('ascii') if re.fullmatch(
            rb'req_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}', request_id) else None)

        async def observed_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        finally:
            elapsed = round((time.perf_counter() - trace.started) * 1000, 3)
            payload = {
                'event': 'converter_health_probe',
                'version': os.getenv('DEPLOYMENT_VERSION', 'development'),
                'duration_ms': elapsed,
                'handler_wait_ms': (round((trace.handler_started - trace.started) * 1000, 3)
                                    if trace.handler_started is not None else None),
                'status': status,
                'stages': trace.stages,
                **({'request_id': safe_id} if safe_id else {}),
            }
            current_health.reset(token)
            logging.getLogger('standin.converter').log(
                logging.WARNING if elapsed >= 5000 or status != 200 else logging.INFO,
                json.dumps(payload, sort_keys=True))
