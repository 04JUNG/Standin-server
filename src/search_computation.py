"""Share intermediate search work within one rough analysis."""

from contextlib import contextmanager
from contextvars import ContextVar

from .tracing import span

_search_work = ContextVar('search_work', default=None)


@contextmanager
def share_search_computation():
    token = _search_work.set({})
    try:
        yield
    finally:
        _search_work.reset(token)


def cached_search_work(stage, key, compute, *, limit):
    work = _search_work.get()
    if work is None or key is None:
        return compute()
    cache = work.setdefault(stage, {})
    if key in cache:
        with span(f'{stage}_reuse'):
            return cache[key]
    with span(f'{stage}_compute'):
        result = compute()
    if len(cache) < limit:
        cache[key] = result
    return result
