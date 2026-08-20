"""Thread-safe product wrapper around the internal semantic-search runtime."""
from __future__ import annotations

from collections import OrderedDict
import copy
from pathlib import Path
import threading
import time
from typing import Any

from .semantic_search import SemanticPoseSearch, discover_semantic_build


SEMANTIC_SERVICE_VERSION = 1


class SemanticBusyError(RuntimeError):
    """The bounded semantic encoder pool could not accept another request."""


class SemanticSearchService:
    """Bounded query execution and build-aware in-memory LRU cache.

    The ONNX session is shared, so the semaphore also prevents accidental
    unbounded concurrent calls into one encoder session. Cache entries are
    keyed by the immutable semantic build and complete request semantics.
    """

    def __init__(
        self,
        runtime: SemanticPoseSearch,
        *,
        max_concurrency: int,
        acquire_timeout_seconds: float,
        cache_size: int,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("semantic max_concurrency must be positive")
        if acquire_timeout_seconds < 0:
            raise ValueError("semantic acquire timeout must be non-negative")
        if cache_size < 0:
            raise ValueError("semantic cache_size must be non-negative")
        self.runtime = runtime
        self.max_concurrency = max_concurrency
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.cache_size = cache_size
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._busy_rejections = 0
        self._in_flight = 0

    @property
    def semantic_build_id(self) -> str:
        return self.runtime.manifest["semantic_build_id"]

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.strip().split())

    def _cache_key(
        self, query: str, top_k: int, view_hint: str | None
    ) -> tuple[Any, ...]:
        return (
            SEMANTIC_SERVICE_VERSION,
            self.semantic_build_id,
            self._normalize_query(query),
            top_k,
            view_hint,
        )

    def _read_cache(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        if self.cache_size == 0:
            return None
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return copy.deepcopy(cached)

    def _write_cache(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        if self.cache_size == 0:
            return
        with self._lock:
            self._cache[key] = copy.deepcopy(value)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        view_hint: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        key = self._cache_key(query, top_k, view_hint)
        with self._lock:
            self._requests += 1
        cached = self._read_cache(key)
        if cached is not None:
            cached["cache_hit"] = True
            cached["service_time_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            return cached

        acquired = self._semaphore.acquire(timeout=self.acquire_timeout_seconds)
        if not acquired:
            with self._lock:
                self._busy_rejections += 1
            raise SemanticBusyError("semantic search concurrency limit reached")
        try:
            # A request may have populated the same key while this caller was waiting.
            cached = self._read_cache(key)
            if cached is not None:
                cached["cache_hit"] = True
                cached["service_time_ms"] = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                return cached
            with self._lock:
                self._cache_misses += 1
                self._in_flight += 1
            try:
                result = self.runtime.search(query, top_k=top_k)
            finally:
                with self._lock:
                    self._in_flight -= 1
            result["cache_hit"] = False
            result["view_hint"] = view_hint
            result["service_version"] = SEMANTIC_SERVICE_VERSION
            result["service_time_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            self._write_cache(key, result)
            return copy.deepcopy(result)
        finally:
            self._semaphore.release()

    def readiness(self) -> dict[str, Any]:
        manifest = self.runtime.manifest
        with self._lock:
            stats = {
                "requests": self._requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "busy_rejections": self._busy_rejections,
                "in_flight": self._in_flight,
                "cache_entries": len(self._cache),
            }
        return {
            "ready": True,
            "reason": "ready",
            "service_version": SEMANTIC_SERVICE_VERSION,
            "semantic_build_id": manifest["semantic_build_id"],
            "semantic_db_schema_version": manifest["semantic_db_schema_version"],
            "semantic_unit_count": len(self.runtime.units),
            "pose_member_count": manifest["counts"]["pose_members"],
            "embedding_version": manifest["embedding"]["embedding_version"],
            "embedding_model": manifest["embedding"]["model_id"],
            "embedding_revision": manifest["embedding"]["revision"],
            "max_concurrency": self.max_concurrency,
            "cache_size": self.cache_size,
            "stats": stats,
        }


def load_semantic_service(
    *,
    build_dir: Path | None,
    builds_root: Path,
    profile_path: Path,
    models_root: Path,
    max_concurrency: int,
    acquire_timeout_seconds: float,
    cache_size: int,
) -> SemanticSearchService:
    resolved_build = build_dir or discover_semantic_build(builds_root)
    runtime = SemanticPoseSearch(
        resolved_build,
        profile_path=profile_path,
        models_root=models_root,
    )
    return SemanticSearchService(
        runtime,
        max_concurrency=max_concurrency,
        acquire_timeout_seconds=acquire_timeout_seconds,
        cache_size=cache_size,
    )
