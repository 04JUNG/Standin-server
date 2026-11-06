"""Process-wide numerical source slots; selected work precedes candidate work."""
from contextlib import contextmanager
from functools import lru_cache
import itertools
import os
import threading
import time


class SourceAdmissionError(ValueError):
    pass


class SourceAdmission:
    def __init__(self, concurrency=1, capacity=20):
        if concurrency not in (1, 2) or capacity < concurrency:
            raise ValueError("source concurrency must be 1 or 2")
        self.concurrency, self.capacity = concurrency, capacity
        self.condition = threading.Condition()
        self.waiting = set()
        self.sequence = itertools.count()
        self.active = 0

    @contextmanager
    def slot(self, stage, deadline):
        with self.condition:
            if len(self.waiting) >= self.capacity:
                raise SourceAdmissionError("source_queue_full")
            ticket = (0 if stage == "selected" else 1, next(self.sequence))
            self.waiting.add(ticket)
            try:
                while self.active >= self.concurrency or ticket != min(self.waiting):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SourceAdmissionError("source_queue_timeout")
                    self.condition.wait(remaining)
                if time.monotonic() >= deadline:
                    raise SourceAdmissionError("source_queue_timeout")
                self.waiting.remove(ticket)
                self.active += 1
            except BaseException:
                self.waiting.discard(ticket)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


@lru_cache(maxsize=1)
def source_admission():
    # Fixed once per service process, never changed for an individual request.
    return SourceAdmission(int(os.getenv("CANDIDATE_REFINE_CONCURRENCY", "1")))
