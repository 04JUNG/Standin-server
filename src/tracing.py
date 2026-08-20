"""Request-local, dependency-free timing spans for diagnostics.

The tracer is inert unless ``capture_trace`` is active, so pipeline callers do
not need an evaluation flag and concurrent FastAPI requests do not share state.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class TimingTrace:
    spans: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed_ms: float) -> None:
        self.spans[name] = self.spans.get(name, 0.0) + float(elapsed_ms)
        self.counts[name] = self.counts.get(name, 0) + 1

    def to_dict(self) -> dict:
        return {
            "spans_ms": {name: round(value, 3) for name, value in self.spans.items()},
            "counts": dict(self.counts),
        }

    def server_timing(self) -> str:
        return ", ".join(
            f"{name};dur={elapsed:.3f}"
            for name, elapsed in self.spans.items()
        )


_ACTIVE_TRACE: ContextVar[TimingTrace | None] = ContextVar(
    "standin_active_timing_trace", default=None
)


@contextmanager
def capture_trace():
    trace = TimingTrace()
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE_TRACE.reset(token)


@contextmanager
def span(name: str):
    started = perf_counter()
    try:
        yield
    finally:
        trace = _ACTIVE_TRACE.get()
        if trace is not None:
            trace.add(name, (perf_counter() - started) * 1000.0)
