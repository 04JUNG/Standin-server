"""Opt-in local diagnostic stage timings. No logging, storage or I/O."""
from contextlib import contextmanager
import os
import time


class StageTimings:
    def __init__(self, enabled=None, clock=time.monotonic):
        self.enabled = os.getenv('STANDIN_LAB_PREVIEW_TRACE') == '1' if enabled is None else enabled
        self.clock = clock
        self.records = {}

    @contextmanager
    def measure(self, stage):
        if not self.enabled:
            yield
            return
        start = self.clock()
        ok = False
        try:
            yield
            ok = True
        finally:
            self.records[stage] = {'duration_ms': max(0., (self.clock()-start)*1000), 'ok': ok}
