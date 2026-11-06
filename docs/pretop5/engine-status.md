# Engine status — 2026-11-06 KST

- Worktree: `/Users/dowon/.codex/worktrees/d3aa/Standin-server`
- Branch: `codex/pretop5-candidate-engine`
- Latest fetched develop: `f260b1e6843996935ab287cdfad63465c18f7266`
- Ownership: api/new candidate source batch, src/refine candidate adapter, scene admission boundary, common CONTRACT.md/schema and corrected design copy. No target/converter or appearance files edited.
- Dependencies: develop only; no unmerged PR. SelectedTargetProvider is separate post-selection converter/BFF integration. Existing outline/skin/large-preview PRs untouched.
- Latest user correction implemented: pre-Top5 source only; body/shape/hands/appearance rejected by source API and excluded from cache. Actual target applies only after user pose selection. Source prepare never calls selected target provider.
- Implemented: private default-off authenticated capabilities/prepare/reuse; immutable read-only source snapshot, exact code/config/runtime/library/set metadata identity; global per-process bounded C1/2 FIFO admission + single-flight + TTL cache; exact base bytes on retention, retryable timeout/crash, zero source solver on selection; no PNG/FBX/VLM/shadow recompute. Repeated scene HTTP cancellation retains actual worker slot.
- Published contract: shared CONTRACT.md/schema.json. 10 is transport chunk bound; BFF must sequentially chunk 3+ people under one job/revision and whole barrier, never truncate 11+.
- Validation: Python3.12 full suite3441 passed/10 skipped/31 subtests; final focused53 passed; smoke54/54; compileall/diff check clean. Actual synthetic default-gate ASGI response refined/base accepted by BFF actual schema2/2. Shared `engine-wire-fixture.json` contains source-only evidence, not usable prepared/download authority. See worktree `docs/pretop5/VERIFICATION.md`.
- Measurement handoff: `/Users/dowon/.codex/worktrees/d3aa/Standin-server/scripts/benchmark_candidate_source.py`, Python `/tmp/pretop5-engine-py312/bin/python`. Target owner schedules C1 then C2 with identical quota/BLAS/resource conditions; report distinguishes cold, warm process without result cache, cache hit, selected reuse. Driver ready, source quant not independently run while target measurement is active. Synthetic two-person10-candidate standing/arm+leg bend/occlusion/unowned partial; no real sitting/contact or target safety claims.
- Limits: prepared/source capacity64+TTL600s, one engine process for ID affinity; eviction/restart means reuse409 and BFF invalidation/reprepare. Docker daemon absent locally; CI image check required. Empty automatic v3/v4 allowlist; target_refine/automatic_parts/scene_joint_validation false.
- PR: preparing new source-only PR to develop; CI pending creation. No deployment/protected push/merge/activation/paid call/AWS changes.
