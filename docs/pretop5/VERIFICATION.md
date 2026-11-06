# Source preparation verification — 2026-11-06

Implementation base: server develop `f260b1e6843996935ab287cdfad63465c18f7266`. New branch `codex/pretop5-candidate-engine`; no unmerged dependency for source preparation. The [user-corrected design](../TOP5_CANDIDATE_REFINE_V3_V4_DESIGN_2026-11-06.md) takes priority: apply actual body/shape/hands after pose selection only.

## Executed local checks

- Python 3.12.13, macOS arm64; isolated `/tmp/pretop5-engine-py312` environment, repository-pinned rtmlib 0.0.15 / onnxruntime 1.28.0; `pip check` clean.
- `python -m pytest tests/ --ignore=tests/converter -q`: **3441 passed, 10 skipped, 31 subtests passed**, 69.59s. Existing tar-extraction deprecation warning only. SDK fault tests use local loopback servers, not external providers.
- After adding authoritative library `set_id` invalidation and final hardening: focused source/camera/deferred-preview/scene suite **53 passed**. Tests verify source metadata/config/hash/cache tamper, ownership/revision isolation, source failures/no-gain/timeouts, C<=2/single-flight, zero selected solver calls, and repeated HTTP cancellation holding actual scene-worker admission.
- `python tests/test_smoke.py`: **54/54 passed**. `python -m compileall -q src api scripts` and `git diff --check` passed.
- Default-gate authenticated ASGI HTTP sample: synthetic source/observations produce `ready_refined/ok` plus `ready_base/skeleton_policy`; retained source is byte-identical. Both actual JSON results parsed successfully by BFF owner's compiled `candidate-preparation/service.js::resultSchema` (2/2). [Wire fixture](wire-fixture.json) contains exact synthetic bytes and code/config/runtime hashes. Illustrative prepared IDs are expired test handles.

## Performance measurement handoff

Target/performance owner coordinates resource scheduling. `scripts/benchmark_candidate_source.py --concurrency 1|2 --output ...` provides the same synthetic two-person/ten-candidate input for sequential C1/C2 runs. Set BLAS limits once at process launch. Report separates cold empty result cache, warm process with result cache cleared, exact cache hit and selected reuse; includes code/config/resource/CPU/RSS/peak/call counts and safety outcomes. Driver does not impose a CPU quota and does not imply a 1vCPU container. No measured source latency is asserted here until owner-coordinated results are attached. `warm` never implies validated warm Blender scene reuse.

## Limits and boundaries

- Source-only scope: no actual target mesh/body/hands application, PNG/FBX generation, target skin/contact certification, v3 promotion or automatic v4 intent. Automatic profile allowlist empty. Existing accepted solver/selector/gates/normalization/fixed camera remain unchanged.
- Internal chunks <=10; BFF owns sequential chunks/one global job lease/revision/all-candidate barrier for 3+ people, without truncation. Engine admission is FIFO and bounded to 20 queued/running source candidates, C1 by default, C2 maximum per service process. Expired queued tasks are cancelled; running cooperative native work retains its actual slot.
- Process-local source/prepared caches: capacity64, TTL600s, one engine process required for prepared-ID affinity. Expiry/restart/capacity eviction yields 409 and requires BFF invalidation/reprepare. BFF retains durable job/ownership/current selection authority; engine receipt alone cannot authorize final target bytes.
- Actual selected target/FBX correspondence belongs to the converter/target PR. No unmerged target dependency in this engine PR. No real-rough sitting/contact accuracy or production p95/SLA claim from the synthetic sample.
- Local Docker image build unavailable because Docker daemon/socket is absent. CI image validation remains required. No deployment, feature activation, protected branch push/merge, paid provider evaluation or AWS mutation performed.
