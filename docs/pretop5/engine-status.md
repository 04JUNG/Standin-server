# Engine status — 2026-11-06 KST

- Worktree: `/Users/dowon/.codex/worktrees/d3aa/Standin-server`
- Branch: `codex/pretop5-candidate-engine`
- Latest fetched develop: `f260b1e6843996935ab287cdfad63465c18f7266`
- Ownership: api/new candidate source batch, src/refine candidate adapter, scene admission boundary, common CONTRACT.md/schema and corrected design copy. No target/converter or appearance files edited.
- Dependencies: develop only; no unmerged PR. SelectedTargetProvider is separate post-selection converter/BFF integration. Existing outline/skin/large-preview PRs untouched.
- Latest user correction implemented: pre-Top5 source only; body/shape/hands/appearance rejected by source API and excluded from cache. Actual target applies only after user pose selection. Source prepare never calls selected target provider.
- Implemented: private default-off authenticated capabilities/prepare/reuse; immutable read-only source snapshot, exact code/config/runtime/library/set metadata identity; global per-process bounded C1/2 FIFO admission + single-flight + TTL cache; exact base bytes on retention, retryable timeout/crash, zero source solver on selection; no PNG/FBX/VLM/shadow recompute. Repeated scene HTTP cancellation retains actual worker slot.
- Published contract: shared CONTRACT.md/schema.json. 10 is transport chunk bound; BFF must sequentially chunk 3+ people under one job/revision and whole barrier, never truncate 11+.
- Validation: Python3.12 full suite3446 passed/10 skipped/31 subtests; final focused56 passed; smoke54/54; compileall/diff check clean. Actual synthetic default-gate ASGI response refined/base accepted by BFF actual schema2/2. Shared `engine-wire-fixture.json` contains source-only evidence, not usable prepared/download authority. See worktree `docs/pretop5/VERIFICATION.md`.
- Measurement handoff: `/Users/dowon/.codex/worktrees/d3aa/Standin-server/scripts/benchmark_candidate_source.py`, Python `/tmp/pretop5-engine-py312/bin/python`. Target owner schedules C1 then C2 with identical quota/BLAS/resource conditions; report distinguishes cold, warm process without result cache, cache hit, selected reuse. Driver ready, source quant not independently run while target measurement is active. Synthetic two-person10-candidate standing/arm+leg bend/occlusion/unowned partial; no real sitting/contact or target safety claims.
- Limits: prepared/source capacity64+TTL600s, one engine process for ID affinity; eviction/restart means reuse409 and BFF invalidation/reprepare. Docker daemon absent locally; CI image check required. Empty automatic v3/v4 allowlist; target_refine/automatic_parts/scene_joint_validation false.
- PR: https://github.com/04JUNG/Standin-server/pull/171 (draft, develop base). Initial PR checks passed; mixed-admission hardening follow-up pending push. No deployment/protected push/merge/activation/paid call/AWS changes.

## PR and final admission review

- New draft PR: https://github.com/04JUNG/Standin-server/pull/171 (develop base). Initial commit `76641849720cf9d3dc3c51b51b690b0cc7fcfc20`; next hardening commit in progress.
- Final review added process-wide numerical slots shared by opted-in candidate batch and legacy `/refine`. Actual source C<=1/2 holds for mixed old/new clients; selected legacy work precedes queued auto candidates. Feature OFF preserves original legacy path; opted-in selected saturation returns bounded503/Retry-After1. Automatic batches never enter Blender scene/parts slots.
- 56 focused tests and final whole Python3.12 suite3446 passed/10 skipped/31 subtests after mixed-admission priority/deadline regressions; source wire fixture regenerated and BFF actual schema2/2 accepted.

- Source quantitative handoff remains pending target-owner resource scheduling; please run driver C1 then C2 (same Python3.12 env + fixed BLAS/quota) and publish outputs/source verdict in shared BENCHMARK.md. Source driver does not run target/model/Blender or VLM; it is independently safe to schedule after target measurement completes. No source latency target/SLA claimed before this evidence.
