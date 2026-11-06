"""Bounded source preparation; private immutable artifacts, no Blender threads.

BFF owns durable jobs/current revision. This cache is strictly an optimization;
source files and exact engine identity are revalidated on every entry/reuse.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import uuid

from api.candidate_refine_models import CandidateResult, PrepareResponse, Timing
from src.refine_v3_contracts import canonical_bytes, canonical_sha256

VERSION = "candidate-source-adapter-v1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024


class CandidateError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSnapshot:
    data: bytes
    set_id: str | None


class CandidatePreparer:
    def __init__(self, resolve, solve, identity, *, concurrency=1, capacity=20,
                 cache_capacity=64, ttl=600):
        if concurrency not in (1, 2) or capacity < concurrency:
            raise ValueError("candidate concurrency must be 1 or 2 with bounded admission")
        self.resolve, self.solve, self.identity = resolve, solve, identity
        self.concurrency, self.capacity = concurrency, capacity
        self.cache_capacity, self.ttl = cache_capacity, ttl
        self.pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="candidate-source")
        self.lock = threading.RLock()
        self.flights = {}
        self.cache = OrderedDict()
        self.prepared = OrderedDict()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _snapshot(self, item):
        snapshot = self.resolve(item.request.pose_id)
        if (not isinstance(snapshot, SourceSnapshot) or not snapshot.data
                or len(snapshot.data) > MAX_SOURCE_BYTES
                or hashlib.sha256(snapshot.data).hexdigest() != item.source_sha256):
            raise CandidateError("source_changed")
        return snapshot

    def _keys(self, batch, item, engine, snapshot=None):
        request = item.request.model_dump(mode="json")
        # Neither the shadow experiment nor PNG generation is an automatic engine.
        request.pop("render_thumbnail")
        request.pop("refine_v3")
        source = {"version": VERSION, "owner_id": batch.owner_id, "job_id": batch.job_id,
                  "person_id": item.person_id, "candidate_id": item.candidate_id,
                  "candidate_set_revision": batch.candidate_set_revision,
                  "source_sha256": item.source_sha256, "source_set_id": snapshot.set_id if snapshot else None,
                  "request": request, "engine": engine}
        key = canonical_sha256(source)
        return key, canonical_sha256({"source_fingerprint": key, "phase": "source_preselection"})

    def _evict(self):
        now = time.monotonic()
        for mapping in (self.cache, self.prepared):
            for key, entry in list(mapping.items()):
                if entry[0] <= now:
                    del mapping[key]
            while len(mapping) > self.cache_capacity:
                mapping.popitem(last=False)

    def _run(self, item, snapshot, deadline, budget_ms, enqueued):
        start = time.monotonic()
        if start >= deadline:
            raise CandidateError("queue_timeout")
        execution_deadline = min(deadline, start + budget_ms / 1000)
        # A single immutable bytes instance is shared by in-flight requests. The
        # legacy path-only solver receives a private read-only snapshot; library
        # mutations cannot race solver probes. No global CFG/camera mutation.
        with tempfile.TemporaryDirectory(prefix="candidate-source-") as directory:
            path = Path(directory) / "source.bvh"
            path.write_bytes(snapshot.data)
            path.chmod(0o400)
            result = self.solve(item.request, snapshot, str(path), execution_deadline)
        elapsed = time.monotonic() - start
        if time.monotonic() >= execution_deadline or result.reason == "timeout":
            raise CandidateError("execution_timeout")
        if result.refined and not result.bvh:
            raise CandidateError("artifact_missing")
        data = result.bvh.encode("utf-8") if result.refined else snapshot.data
        value = {"status": "ready_refined" if result.refined else "ready_base",
                 "reason": result.reason, "bvh": data.decode("utf-8"),
                 "artifact_sha256": hashlib.sha256(data).hexdigest(),
                 "refine_version": result.refine_version,
                 "queue_ms": (start - enqueued) * 1000, "execution_ms": elapsed * 1000}
        return canonical_bytes(value)

    def _completed(self, key, future):
        with self.lock:
            if self.flights.get(key) is future:
                del self.flights[key]
            if not future.cancelled() and future.exception() is None:
                raw = future.result()
                self.cache[key] = (time.monotonic() + self.ttl, raw, hashlib.sha256(raw).digest())
                self._evict()

    def _admit(self, key, item, snapshot, deadline, budget_ms):
        with self.lock:
            self._evict()
            cached = self.cache.get(key)
            if cached is not None:
                if hashlib.sha256(cached[1]).digest() == cached[2]:
                    self.cache.move_to_end(key)
                    return cached[1], "hit"
                del self.cache[key]
            if key in self.flights:
                return self.flights[key], "in_flight"
            if len(self.flights) >= self.capacity:
                raise CandidateError("queue_full")
            future = self.pool.submit(self._run, item, snapshot, deadline, budget_ms, time.monotonic())
            self.flights[key] = future
            future.add_done_callback(lambda completed: self._completed(key, completed))
            return future, "miss"

    def prepare(self, batch):
        # Detach mutable caller arrays/dicts before scheduling.
        batch = type(batch).model_validate_json(batch.model_dump_json())
        start = time.monotonic()
        deadline = start + batch.batch_budget_ms / 1000
        engine = json.loads(canonical_bytes(self.identity()))
        work = []
        snapshots = {}
        for item in batch.candidates:
            key, context = self._keys(batch, item, engine)
            try:
                snapshot = self._snapshot(item)  # revalidate even cache hits
                snapshot = snapshots.setdefault((item.source_sha256, snapshot.set_id), snapshot)
                key, context = self._keys(batch, item, engine, snapshot)
                if time.monotonic() >= deadline:
                    raise CandidateError("batch_timeout")
                task, cache = self._admit(key, item, snapshot, deadline, batch.candidate_budget_ms)
                work.append((item, key, context, task, cache, None))
            except CandidateError as exc:
                work.append((item, key, context, None, "miss", str(exc)))
            except (OSError, UnicodeError):
                work.append((item, key, context, None, "miss", "source_unavailable_or_changed"))
        results = []
        for item, key, context, task, cache, error in work:
            values = {}
            try:
                if error:
                    raise CandidateError(error)
                raw = task if isinstance(task, bytes) else task.result(timeout=max(0, deadline - time.monotonic()))
                if time.monotonic() >= deadline:
                    raise CandidateError("batch_timeout")
                values = json.loads(raw)
                # Same per-request engine config, never stale successful bytes.
                if self.identity() != engine:
                    raise CandidateError("engine_changed")
                # Recheck original source after solve as well as before admission.
                current_snapshot = self._snapshot(item)
                if self._keys(batch, item, engine, current_snapshot)[0] != key:
                    raise CandidateError("source_lineage_changed")
                status = values.pop("status")
                reason = values.pop("reason")
                prepared_id = None
                if status in {"ready_base", "ready_refined"}:
                    prepared_id = uuid.uuid4().hex
                origin_queue, origin_execution = values.pop("queue_ms"), values.pop("execution_ms")
                timing = Timing(queue_ms=origin_queue if cache == "miss" else 0,
                                execution_ms=origin_execution if cache == "miss" else 0,
                                origin_queue_ms=origin_queue, origin_execution_ms=origin_execution,
                                total_ms=(time.monotonic() - start) * 1000, cache=cache)
                actual_version = values.pop("refine_version")
                if engine.get("code_version", actual_version) != actual_version:
                    raise CandidateError("engine_version_mismatch")
                result = CandidateResult(person_id=item.person_id, candidate_id=item.candidate_id,
                    context_fingerprint=context, source_fingerprint=key, status=status, reason=reason,
                    source_sha256=item.source_sha256, engine=engine, prepared_id=prepared_id,
                    timing=timing, **values)
                if prepared_id:
                    with self.lock:
                        raw_result = result.model_dump_json().encode()
                        self.prepared[prepared_id] = (time.monotonic() + self.ttl, raw_result,
                            hashlib.sha256(raw_result).digest(), batch, item, engine)
                        self._evict()
            except Exception as exc:
                if isinstance(exc, TimeoutError) and cache == "miss" and task is not None:
                    # Queued obsolete work can be removed. Running work keeps its slot.
                    task.cancel()
                # Running work retains its slot until completion. Never mark a
                # timeout as safety retention; never cache a failed result.
                reason = (str(exc) if isinstance(exc, CandidateError) else
                          "batch_timeout" if isinstance(exc, TimeoutError) else "execution_failed")
                status = ("unsupported" if reason in {"pose_quarantined", "search_camera_requires_v2"}
                          else "failed_retryable")
                result = CandidateResult(person_id=item.person_id, candidate_id=item.candidate_id,
                    context_fingerprint=context, source_fingerprint=key, status=status,
                    reason=reason, engine=engine, source_sha256=item.source_sha256,
                    timing=Timing(total_ms=(time.monotonic() - start) * 1000, cache=cache))
            results.append(result)
        return PrepareResponse(**batch.model_dump(include={"contract_version", "batch_id", "owner_id",
            "job_id", "candidate_set_revision"}), results=results, elapsed_ms=(time.monotonic() - start) * 1000)

    def reuse(self, request):
        with self.lock:
            self._evict()
            entry = self.prepared.get(request.prepared_id)
        if entry is None:
            raise CandidateError("prepared_expired")
        _, raw, checksum, batch, item, engine = entry
        if hashlib.sha256(raw).digest() != checksum:
            raise CandidateError("prepared_corrupt")
        result = CandidateResult.model_validate_json(raw)
        if (any(getattr(batch, name) != getattr(request, name) for name in
                ("owner_id", "job_id", "candidate_set_revision"))
                or item.person_id != request.person_id or item.candidate_id != request.candidate_id
                or result.context_fingerprint != request.context_fingerprint
                or engine != self.identity()):
            raise CandidateError("prepared_context_changed")
        current_snapshot = self._snapshot(item)
        if self._keys(batch, item, engine, current_snapshot)[0] != result.source_fingerprint:
            raise CandidateError("source_lineage_changed")
        # No source or target solver, exporter, VLM or thumbnail call here.
        return result
