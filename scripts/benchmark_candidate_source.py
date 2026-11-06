#!/usr/bin/env python3
"""Offline synthetic source benchmark, scheduled by target/performance owner.

Run C=1 and C=2 sequentially under identical process/CPU limits. No body model,
Blender, PNG, provider call or target-safety claim. All environment/config is
fixed at process launch; the driver never changes CFG or BLAS thread settings.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.candidate_refine_models import PrepareRequest, ReuseRequest
from src.refine_candidates import CandidatePreparer, SourceSnapshot, VERSION
from tests.test_smoke import _synthetic_bvh
from tests.test_refine_v2 import _bvh_with_rotation, _target_kp
from api import app as runtime
from api.candidate_refine import source_runtime_identity
from src.refine_selector import SELECTOR_VERSION


def run(concurrency, output):
    with tempfile.TemporaryDirectory(prefix="source-benchmark-") as directory:
        base = Path(_synthetic_bvh(directory, "base.bvh")).read_bytes()
        digest = hashlib.sha256(base).hexdigest()
        fixtures = []
        for label, joint, angle in [("standing", "LeftArm", 0.), ("arm_bend", "LeftArm", 15.),
                                    ("leg_bend", "LeftUpLeg", 25.), ("occluded", "LeftArm", 15.),
                                    ("partial_unowned", "LeftArm", 15.)]:
            path = _bvh_with_rotation(directory, label + ".bvh", joint, angle)
            keypoints, scores = _target_kp(path)
            if label == "occluded": scores[9:] = 0
            fixtures.append((label, keypoints, scores))
        candidates = []
        for person in range(2):
            for i, (label, points, scores) in enumerate(fixtures):
                candidates.append(dict(person_id=str(person), candidate_id=str(i), source_sha256=digest,
                    request=dict(pose_id=f"synthetic-{i}", view="front", keypoints=points.tolist(),
                        scores=scores.tolist(), render_thumbnail=False,
                        refine_allowed=label != "partial_unowned", refinable_limbs=["left_arm"],
                        lower_body_observed=False, skeleton_state="partial" if label == "occluded" else "valid",
                        coverage_class="reduced" if label == "occluded" else "full", slot_origin="vlm",
                        skeleton_source="full_image")))
        batch = PrepareRequest(batch_id="synthetic-source", owner_id="local-benchmark", job_id="source-benchmark",
                               candidate_set_revision="1", candidates=candidates)
        lock = threading.Lock()
        calls = active = peak = 0
        def solve(req, snapshot, path, deadline):
            nonlocal calls, active, peak
            with lock:
                calls += 1; active += 1; peak = max(active, peak)
            try:
                return runtime._refine_impl(req, source_snapshot=({"set_id": snapshot.set_id}, path),
                                            execution_deadline=deadline, run_shadow=False)
            finally:
                with lock: active -= 1
        identity = lambda: {"adapter_version": VERSION, "selector_version": SELECTOR_VERSION,
                            **runtime._refine_capability(), **source_runtime_identity()}
        service = CandidatePreparer(lambda _: SourceSnapshot(base, None), solve, identity, concurrency=concurrency)
        start = time.monotonic(); cpu = time.process_time()
        try:
            cold = service.prepare(batch)
            cold_calls = calls
            # Warm process, empty result cache: exercise source computation again.
            with service.lock:
                service.cache.clear()
            warm = service.prepare(batch)
            warm_calls = calls - cold_calls
            cache_hit = service.prepare(batch)
            before_selection = calls; selected_start = time.monotonic()
            for result in cache_hit.results:
                if result.prepared_id:
                    service.reuse(ReuseRequest(**batch.model_dump(include={"contract_version", "batch_id", "owner_id",
                        "job_id", "candidate_set_revision"}), person_id=result.person_id, candidate_id=result.candidate_id,
                        prepared_id=result.prepared_id, context_fingerprint=result.context_fingerprint))
            selected_ms = (time.monotonic() - selected_start) * 1000
            report = dict(schema_version="candidate-source-benchmark-v1", fixture_type="synthetic",
                resource=dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine(),
                    cpu_count=os.cpu_count(), cpu_quota="not_enforced_by_driver", concurrency=concurrency,
                    blas_threads={k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    maxrss_unit="bytes" if sys.platform == "darwin" else "KiB"),
                sample_count=10, person_count=2, fixture_labels=[f[0] for f in fixtures],
                safety=dict(scope="source_only", target_mesh_checked=False, contact_safety_certified=False,
                    cold_statuses=dict(Counter(r.status for r in cold.results)),
                    warm_statuses=dict(Counter(r.status for r in warm.results)),
                    exact_retained_source=all(r.artifact_sha256 == digest for r in cold.results if r.status == "ready_base"),
                    source_bytes_stable=all(r.bvh == s.bvh for r, s in zip(cold.results, warm.results))),
                source_adapter_calls=dict(cold=cold_calls, warm=warm_calls,
                    cache_hit=before_selection-cold_calls-warm_calls, selected=calls-before_selection),
                peak_active=peak, selected_reuse_ms=selected_ms, process_cpu_seconds=time.process_time()-cpu,
                total_ms=(time.monotonic()-start)*1000,
                limitations=["Not production 1vCPU/p95/SLA", "synthetic BVH/2D observations only",
                    "no sitting/contact/real rough accuracy certification", "no Blender warm scene reuse"],
                cold_semantics="empty result cache; process launch/import/fixture creation excluded from timing",
                warm_semantics="same process, result cache cleared, source adapters rerun",
                cache_semantics="same process, exact source result cache hit",
                cold=cold.model_dump(), warm=warm.model_dump(), cache_hit=cache_hit.model_dump())
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"output": str(Path(output).resolve()), "concurrency": concurrency,
                "cold_ms": cold.elapsed_ms, "warm_ms": warm.elapsed_ms, "cache_hit_ms": cache_hit.elapsed_ms, "peak_active": peak,
                "source_adapter_calls": report["source_adapter_calls"], "safety": report["safety"]}))
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.concurrency, args.output)
