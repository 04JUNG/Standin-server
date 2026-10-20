"""One bounded Blender capacity pool; safety work can interrupt optional renders.

No BVH/asset/result caching or pose decisions live here. Interrupted outputs are
always discarded. Export/scene work is never preempted by another request.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time


class WorkStopped(RuntimeError):
    def __init__(self, code, status, phase):
        super().__init__(code)
        self.code, self.status, self.phase = code, status, phase


@dataclass(eq=False)
class WorkControl:
    kind: str = 'normal'
    budget_seconds: float = 300.
    started: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)
    preempted: threading.Event = field(default_factory=threading.Event)
    queue_wait_ms: float = 0.
    execution_ms: float = 0.
    conversion_id: str | None = None
    request_id: str | None = None
    phase: str = 'request'
    reason: str = 'ready'

    @property
    def remaining(self):
        return self.started + self.budget_seconds - time.monotonic()

    def check(self, phase):
        self.phase = phase
        failure = (('REQUEST_CANCELLED', 499) if self.cancelled.is_set() else
                   ('PREVIEW_PREEMPTED', 503) if self.preempted.is_set() else
                   ('WORK_BUDGET_EXHAUSTED', 504) if self.remaining <= 0 else None)
        if failure:
            self.reason = failure[0]
            raise WorkStopped(*failure, phase)

    def diagnostics(self):
        return dict(work_class=self.kind, phase=self.phase, reason=self.reason,
                    queue_wait_ms=round(self.queue_wait_ms, 3),
                    execution_ms=round(self.execution_ms, 3),
                    budget_ms=round(self.budget_seconds * 1000, 3),
                    **({'conversion_id': self.conversion_id} if self.conversion_id else {}),
                    **({'request_id': self.request_id} if self.request_id else {}))


current_work: ContextVar[WorkControl | None] = ContextVar('converter_work', default=None)


class PrioritySlots:
    """Semaphore-compatible FIFO priority pool, including scene runner callers.

    Only optional previews are preemptible. A small bounded continuation window
    keeps a just-completed required check's next HTTP check ahead of previews.
    Capacity is released only after its process has actually stopped.
    """
    PRIORITY = {'screen': 0, 'normal': 1, 'preview': 2}

    def __init__(self, capacity=1, *, max_pending=8, continuation_seconds=.2):
        self.capacity, self.max_pending = capacity, max_pending
        self.continuation_seconds = continuation_seconds
        self._condition = threading.Condition()
        self._pending = []
        self._active = {}
        self._continuation_until = 0.
        self._sequence = 0

    def snapshot(self):
        with self._condition:
            return dict(capacity=self.capacity, active=len(self._active),
                        queued={kind: sum(w.kind == kind for _, w in self._pending)
                                for kind in self.PRIORITY})

    def acquire(self, blocking=True, timeout=None):
        control = current_work.get() or WorkControl()
        thread_id = threading.get_ident()
        started = time.monotonic()
        with self._condition:
            control.check('queue')
            if len(self._pending) >= self.max_pending:
                # Keep blocked sync workers below Starlette's default 40-token
                # pool, and never reject safety work just because previews filled
                # those bounded waiting places.
                victim = next((row for row in reversed(self._pending)
                               if row[1].kind == 'preview'), None)
                if control.kind in ('screen', 'normal') and victim is not None:
                    victim[1].preempted.set()
                    self._pending.remove(victim)
                else:
                    control.reason, control.phase = 'WORK_QUEUE_FULL', 'queue'
                    raise WorkStopped('WORK_QUEUE_FULL', 503, 'queue')
            self._sequence += 1
            ticket = (self._sequence, control)
            self._pending.append(ticket)
            try:
                while True:
                    control.check('queue')
                    # Preempt only as many optional processes as capacity needs.
                    if control.kind in ('screen', 'normal'):
                        screens = sum(w.kind in ('screen', 'normal') for _, w in self._pending)
                        preempting = sum(w.preempted.is_set() for w in self._active.values())
                        needed = max(0, screens - (self.capacity-len(self._active)) - preempting)
                        for active in self._active.values():
                            if needed and active.kind == 'preview' and not active.preempted.is_set():
                                active.preempted.set()
                                needed -= 1
                    first = min(self._pending, key=lambda row: (self.PRIORITY[row[1].kind], row[0]))
                    held = control.kind != 'screen' and time.monotonic() < self._continuation_until
                    if first is ticket and len(self._active) < self.capacity and not held:
                        self._pending.remove(ticket)
                        self._active[thread_id] = control
                        return True
                    if not blocking or (timeout is not None and time.monotonic()-started >= timeout):
                        return False
                    self._condition.wait(min(.025, max(.001, control.remaining)))
            finally:
                control.queue_wait_ms = (time.monotonic()-started)*1000
                if ticket in self._pending:
                    self._pending.remove(ticket)
                self._condition.notify_all()

    def release(self):
        with self._condition:
            control = self._active.pop(threading.get_ident())
            if control.kind == 'screen':
                self._continuation_until = time.monotonic()+self.continuation_seconds
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()
