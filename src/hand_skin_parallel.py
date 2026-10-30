"""Ordered, bounded screening with request/session-local state only.

Receipts certify opposite-hand skin triangles, never export permission. No
receipt, BVH, candidate or ownership cache is shared across sessions/users.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from copy import copy, deepcopy
from dataclasses import dataclass
import hashlib
import threading
from time import monotonic

from converter.hand_contact import validate_screen
from .hand_skin_screen import SkinSession, unknown
from .hand_skin_capacity import validate_concurrency


# Cancellation is bound by ASGI lifetime, never by a client-provided request ID.
request_cancelled: ContextVar[threading.Event | None] = ContextVar('hand_screen_cancelled', default=None)
CALL_SECONDS = 15.


@dataclass
class Outcome:
    receipt: dict
    elapsed: float
    budget: float


class ScreenSlots:
    """Bound all sessions of one gate; optional work only starts on an idle gate."""
    def __init__(self, capacity=2):
        self.capacity = validate_concurrency(capacity)
        self.condition = threading.Condition()
        self.active = 0
        self.waiting = 0

    def acquire(self, timeout, *, optional=False):
        deadline = monotonic()+timeout
        with self.condition:
            if optional:
                if self.active or self.waiting:
                    return False
            else:
                self.waiting += 1
                try:
                    while self.active >= self.capacity:
                        remaining = deadline-monotonic()
                        if remaining <= 0:
                            return False
                        self.condition.wait(remaining)
                finally:
                    self.waiting -= 1
            self.active += 1
            return True

    def release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()


class ParallelSkinSession(SkinSession):
    def __init__(self, gate, hands):
        super().__init__(gate, deepcopy(hands))
        self.cancelled = request_cancelled.get() or threading.Event()
        self.wall_seconds = 0.
        self.optional = False
        self.discarded = 0

    @contextmanager
    def optional_phase(self):
        previous = self.optional
        self.optional = True
        try:
            yield
        finally:
            self.optional = previous

    def remaining(self):
        return max(0., min(self.gate.budget_seconds-self.spent_seconds,
                           self.gate.budget_seconds-self.wall_seconds))

    def stopped(self):
        if self.cancelled.is_set():
            return unknown('hand_screen_request_cancelled')
        if self.failed:
            return unknown('hand_screen_upstream_unavailable')
        if self.remaining() <= 0:
            return unknown('hand_screen_budget_exhausted')
        return None

    def execute(self, text, budget, optional):
        started = monotonic()
        acquired = False
        try:
            # Queue time consumes this reservation too. Across requests, keep
            # actual converter calls bounded by the gate, not by each executor.
            while not acquired:
                if self.cancelled.is_set():
                    return Outcome(unknown('hand_screen_request_cancelled'), monotonic()-started, budget)
                remaining = budget-(monotonic()-started)
                if remaining <= 0:
                    return Outcome(unknown('hand_screen_work_budget_exhausted'), monotonic()-started, budget)
                acquired = self.gate.parallel_slots.acquire(timeout=min(.05, remaining), optional=optional)
                if optional and not acquired:
                    return Outcome(unknown('hand_screen_optional_capacity_unavailable'), monotonic()-started, budget)
            remaining = budget-(monotonic()-started)
            if self.cancelled.is_set():
                receipt = unknown('hand_screen_request_cancelled')
            elif remaining <= 0:
                receipt = unknown('hand_screen_work_budget_exhausted')
            else:
                transport = copy(self.client)
                transport._screen_optional = optional
                receipt = transport.check(text, self.hands, timeout=remaining)
                validate_screen(receipt)
        except Exception:
            receipt = unknown('hand_screen_failed')
        finally:
            if acquired:
                self.gate.parallel_slots.release()
        elapsed = monotonic()-started
        if self.cancelled.is_set():
            receipt = unknown('hand_screen_request_cancelled')
        elif elapsed > budget:
            receipt = unknown('hand_screen_work_budget_exhausted')
        return Outcome(receipt, elapsed, budget)

    def commit(self, text, outcome, check_number):
        receipt = outcome.receipt
        if self.cancelled.is_set():
            receipt = unknown('hand_screen_request_cancelled')
        elif self.failed:
            self.discarded += 1
            receipt = unknown('hand_screen_upstream_unavailable')
        elif self.remaining() <= 0:
            receipt = unknown('hand_screen_budget_exhausted')
        if receipt['status'] == 'unavailable':
            self.failed = True
        else:
            # Only determinate receipts in this session; never cross-request.
            self.cache[hashlib.sha256(text.encode('utf-8')).hexdigest()] = deepcopy(receipt)
        timing = receipt.get('availability') or {}
        event = dict(check=check_number, status=receipt['status'], reason=receipt.get('reason'),
            elapsed_ms=round(outcome.elapsed*1000, 3), call_budget_ms=round(outcome.budget*1000, 3),
            optional=self.optional, **{key: timing[key] for key in
                ('queue_wait_ms','execution_ms','phase','cause','conversion_id') if key in timing})
        self.events.append(event)
        from .logging_setup import log_info
        log_info('hand_skin_screen_check', **event)
        return receipt

    def check(self, text, *, timeout_seconds=None):
        stop = self.stopped()
        if stop:
            return stop
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if key in self.cache:
            return deepcopy(self.cache[key])
        if self.checks >= self.gate.max_checks:
            return unknown('hand_screen_check_limit_exhausted')
        budget = min(CALL_SECONDS, self.remaining())
        if timeout_seconds is not None:
            budget = min(budget, timeout_seconds)
        if budget <= 0:
            return unknown('hand_screen_budget_exhausted')
        self.checks += 1
        started = monotonic()
        outcome = self.execute(text, budget, self.optional)
        self.wall_seconds += monotonic()-started
        self.spent_seconds += outcome.elapsed
        return self.commit(text, outcome, self.checks)

    def diagnose(self, candidate, receipt, generated):
        item = dict(pose_id=candidate.get('pose_id'), generated=generated,
            hand_skin_screen=receipt, review_required=True, release_allowed=False,
            refine_allowed=False, actual_mesh_validated=False)
        target = self.rejected if receipt['status'] == 'intersection' else self.unverified
        if len(target) < 20:
            target.append(item)

    def filter(self, candidates, *, generated=False, limit=5):
        accepted = []
        pending = iter(candidates)
        finished = False
        with ThreadPoolExecutor(max_workers=self.gate.concurrency, thread_name_prefix='hand-skin') as pool:
            while not finished and len(accepted) < limit:
                remaining = self.remaining()
                # Never buy more concurrency by shortening a mandatory call or
                # increasing the session's aggregate work allowance. With the
                # default 45s budget, at most three full 15s reservations fit.
                width = min(self.gate.concurrency, limit-len(accepted),
                            max(1, int(remaining // CALL_SECONDS)))
                wave, futures = [], {}
                wall_started = monotonic()
                for _ in range(width):
                    candidate = next(pending, None)
                    if candidate is None:
                        finished = True
                        break
                    text = None
                    receipt = self.stopped()
                    try:
                        text = candidate['bvh'] if generated else self.gate.library_text(candidate['pose_id'])
                        if not isinstance(text, str) or not text:
                            raise ValueError('missing BVH')
                    except Exception:
                        receipt = unknown('hand_screen_bvh_unavailable_or_changed')
                    key = hashlib.sha256(text.encode('utf-8')).hexdigest() if text else None
                    if receipt is None and key in self.cache:
                        receipt = deepcopy(self.cache[key])
                    if receipt is None and key not in futures:
                        if self.checks >= self.gate.max_checks:
                            receipt = unknown('hand_screen_check_limit_exhausted')
                        else:
                            budget = min(CALL_SECONDS, remaining)
                            if budget <= 0:
                                receipt = unknown('hand_screen_budget_exhausted')
                            else:
                                self.checks += 1
                                remaining -= budget
                                ctx = copy_context()
                                futures[key] = (pool.submit(ctx.run, self.execute, text, budget, self.optional), self.checks)
                    wave.append((candidate, text, key, receipt))
                if not wave:
                    break
                outcomes = {key: future.result() for key, (future, _) in futures.items()}
                self.wall_seconds += monotonic()-wall_started
                self.spent_seconds += sum(o.elapsed for o in outcomes.values())
                committed = {}
                for candidate, text, key, receipt in wave:
                    if receipt is None:
                        if key not in committed:
                            committed[key] = self.commit(text, outcomes[key], futures[key][1])
                        receipt = deepcopy(committed[key])
                    if self.failed and receipt['status'] == 'clear_screen':
                        receipt = unknown('hand_screen_upstream_unavailable')
                    if self.cancelled.is_set():
                        receipt = unknown('hand_screen_request_cancelled')
                        self.failed = True
                        accepted.clear()
                    if receipt['status'] == 'clear_screen':
                        accepted.append({**candidate, 'hand_skin_screen': receipt})
                    else:
                        self.diagnose(candidate, receipt, generated)
        # Close the gap between the last commit and the caller receiving rows.
        return [] if self.cancelled.is_set() else accepted

    def report(self):
        return {**super().report(), 'wall_elapsed_ms': round(self.wall_seconds*1000, 2),
                'discarded_speculative': self.discarded, 'policy': 'ordered-reserved-parallel-v1'}
