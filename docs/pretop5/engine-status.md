# Engine status — 2026-11-06 KST

- Worktree: `/Users/dowon/.codex/worktrees/d3aa/Standin-server`
- Branch: `codex/pretop5-candidate-engine`
- Base/latest fetched develop: `f260b1e6843996935ab287cdfad63465c18f7266`
- New draft PR: https://github.com/04JUNG/Standin-server/pull/171
- Verified code commit: `9b6b76df5c63b0dc2b2bb2fd1e5eedd48a1e72c4`; CI and Scene worker CI both success. Evidence-only documentation update follows.
- Ownership: new api/source batch+reuse, src/refine source adapters/admission, scene cancellation admission, common CONTRACT.md/schema and user-corrected design. Converter/target/appearance files untouched.
- Dependencies: develop only, no unmerged PR for source. SelectedTargetProvider belongs to the separate post-selection converter/BFF integration. Existing outline/skin/private-model/large-preview PRs untouched.

## Implemented and verified

- Latest user flow: search → source-only candidate refine → Top5 → user selects pose → actual body/shape/hands/target/final FBX. Preselection source API rejects target/body/shape/hand/appearance and excludes them from cache. No preselection model/target/PNG/FBX/VLM calls.
- Private default-off authenticated capabilities/prepare/reuse; immutable source bytes/private read-only file; exact code/config/runtime/library/authoritative set metadata identity; single-flight/TTL; exact source bytes on normal base retention; timeout/crash remains retryable; selected source solver calls0.
- Source candidate admission20, C1 default/C2 maximum. Opted-in legacy `/refine` and batch share actual process-wide source slots; selected legacy work precedes waiting candidates. Feature OFF preserves original legacy behavior. Queue saturation/timeouts are bounded; actual running work retains its slot. Auto candidates never enter/retry Blender scene/parts slots; repeated scene HTTP cancellation retains admission until actual completion.
- Internal10 is transport chunk bound, not product person limit. BFF sequential chunks and one job/revision/all-candidate barrier, no truncation11+. BFF role has now explicitly capped public batch64 to match process-local prepared capacity.
- Python3.12 final full server suite (converter excluded):3446 passed,10 skipped,31 subtests; focused56 passed; smoke54/54; compileall/diff check/pip check clean. CI and Scene worker CI passed for implementation commit9b6b76df5.
- Actual default-gate authenticated ASGI synthetic refined/base wire accepted by BFF actual compiled schema2/2. Shared engine-wire-fixture.json includes exact bytes/versions. BFF owner additionally recorded real Hono+PGlite→engine HTTP/source solver integration with5candidates (1refined/4base), selected source calls0, PNG/body/target calls0; its own outputs path is in bff-status.md.

## Client display sample

- Shared **fully synthetic Neck-bearing** sample: `engine-synthetic-neck-source/synthetic-neck.base.bvh`, `synthetic-neck.refined.bvh`, `manifest.json`. Repo-authored scaffold + hand-authored Neck, all neutral rotations0; no licensed source/user data. Actual default-gate source prepare returned ready_refined/ok for synthetic LeftArm Z+15deg front observation.
- Synthetic source SHA `e9cfa2b18e26848dd5f3db26b90313a2b3a2f26247c40fbad471a6d828d19593`; output SHA `496ee278d8b5ac3bb61d38a970872b515369572b1a1f706cfce25a90235b9998`. Required Neck gate intact. Exact receipt in docs/pretop5/neck-wire-fixture.json; IDs illustrative/ephemeral, not download authority.
- Automatic approval review rejected copying a separate real licensed BVH/derived output to the shared folder because explicit disclosure authorization was absent. That copy did not happen. The wholly synthetic replacement was separately approved and shared; rejected asset was not indirectly released.

## Owner-coordinated source benchmark

- Target/performance owner ran source C1 then C2 sequentially; shared BENCHMARK.md + source-c1-raw.json/source-c2-raw.json inspected. Summary/raw version evidence attached to PR as docs/pretop5/source-benchmark.json.
- Resource: macOS arm64,15logical CPUs, no quota, BLAS threads1; synthetic2people×5candidates. **Not production1vCPU/p95/SLA**. Measured source code digest `00f635d84716b7cc76a8a0969d86f7908362ddf9fbbb0523dcdcff8623183b5f`, before final shared legacy admission hardening. Mixed-load measurement still required.
- C1: cold199.47ms, warm recompute192.63ms, cache2.97ms, selected10reuse2.17ms. C2:215.73/210.62/3.41/2.10ms. C2 slower here; default remainsC1.
- Both:8ready_base+2ready_refined, exact base restoration, identical cold/warm bytes; adapter calls cold/warm10/10, cache/selected0/0; actual peak1/2. All source_only; mesh/contact certificationfalse.
- Driver: scripts/benchmark_candidate_source.py, venv `/tmp/pretop5-engine-py312/bin/python`. Reports cold empty result cache, warm process with result cache cleared, cache hit, selected reuse; does not imply warm Blender mutation.

## Limits / remaining acceptance

Prepared/source cache capacity64+TTL600s, one engine process for prepared-ID affinity; eviction/restart gives409 and BFF invalidation/reprepare. No durable/multi-replica authority claim. No accepted automatic v3/v4 profile; target_refine/automatic_parts/scene_joint_validation false, original explicit v4 intent and shadow boundaries preserved. Actual selected target/final FBX/relationships belong to other roles. Production-equivalent1vCPU/6GiB and real-rough/contact acceptance remain unclaimed. Docker daemon absent locally; GitHub CI image validation passed.

No deployment/staging rollout/protected branch push or merge/global activation/paid provider/cloud evaluation/AWS changes.
