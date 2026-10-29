"""Request-local monotonic deadline, shared by foreground and copied contexts."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

analysis_deadline = ContextVar("analysis_deadline", default=None)

@contextmanager
def analysis_budget(seconds=120., *, clock=monotonic):
    existing = analysis_deadline.get()
    token = analysis_deadline.set(min(existing, clock() + seconds) if existing is not None else clock() + seconds)
    try:
        yield analysis_deadline.get()
    finally:
        analysis_deadline.reset(token)

def bounded_deadline(deadline, *, reserve=0.):
    shared = analysis_deadline.get()
    return min(deadline, shared - reserve) if shared is not None else deadline
