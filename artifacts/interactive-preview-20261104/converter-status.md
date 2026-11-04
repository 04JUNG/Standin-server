# Converter status — 2026-11-04 KST

State: **draft local milestone; full design incomplete; not activated**.

- Worktree: isolated Converter feature worktree.
- Branch `codex/interactive-pose-converter-v1`, based on freshly fetched origin/develop `592ab0f059089e64a5c88a3ad607539124e8e6c6`.
- Implementation source: commit `3fa41e71f` (full SHA/runtime source digest in converter-contract-v1.json). Original checkout/other sessions' dirty edits not imported or overwritten. Only this owner's shared converter contract/status files are published there.
- PR: pending publication; no merge/deployment/feature activation/AWS/paid inference.

## Implemented local boundary

- `hands.solve_hands` extracts the existing private-scene hand solve/bake/rig/skin/smoothing path. Legacy bake retains file import/export behavior. Frozen solver files, manifest SHA and static GLB validator are unchanged.
- Immutable `solved-pose-state-v1` + separate bounded `interactive-pose-model-v1`, canonical float-safe full-state hash, explicit source/body/shape/source-controller-controls/hand/scope/mirror/camera/runtime lineage.
- Hash-checked private body/shape `.blend` reuse for hand-only revisions, safe revision checkpoints, cancel/tombstone/restart handling, packet publication last. No simultaneous mutation of one live scene.
- Fresh exporter process then fresh FBX import/skin/pose/vertex/topology/normal/hash validation; no receipt on stale/exception/file-validation failure.
- Exact bounded hand-combination atlas builder and shared Client module `converter/shared/interactive_pose_v1.mjs`. Every atlas variant requires matching file correspondence and an approved registered character; personal FBX/CS3C remain disabled. **No actual approved atlas/capability is published.**
- Detailed schema, execution, validation and follow-up: `docs/NEXT_SPRINT/INTERACTIVE_CONVERTER_BOUNDARY_V1_2026-11-04.md`.

## Evidence — do not conflate layers

- 201 focused Python/JS contract/regression tests passed (26 new boundary tests); existing smoke 54/54; compileall `src api scripts converter` and git diff whitespace check passed.
- Real Blender `5.2.0 LTS`, build `fbe6228777e7`: **synthetic fixture** fresh export/reimport passed in `full`, `half`, `bust`, `shoulder_head`, `head`. Max world position error 3.48e-7 m, max matrix error 2.17e-6, max weight error 0. No browser/native/OS/CSP certification is implied.
- **Approved actual master-v2 male + female shape 0.3.4:** body/shape reuse and all five scope snapshots generated; comparison against legacy FBX **FAILED strict split-normal correspondence**. Actual male independent final export/reimport rejects and publishes no receipt. See `approved-*-failure-evidence.json`, `approved-real-export-rejected.json`. No actual-asset final certification/capability or complete stage 2/3 claim.
- A proposal to treat split-normal mismatch as diagnostic-only was rejected by automatic approval review because it conflicts with the user's “do not relax validators” instruction. The proposal was reverted; normal/pose/vertex/skin gates remain strict. Resolve preservation under the existing gate before promotion.
- Isolated existing-path local stage audit: retarget+initial FBX 676.9 ms; hand import/apply/export 128.5 ms; framing import 66.7 ms; final reimport 66.6 ms; PNG render/write 142.6 ms. This is one warm-OS-cache local asset run, **not** service p50/p95 or a speedup claim. Detailed call timings omit local filenames/paths.
- Read-only local inventory: 131 GLBs in approved artifact/data trees, zero matching current posed-mesh-v1 lineage/full candidate manifest. Remote cache was not queried. Production Top-5 coverage is **unproven**, not a claim that production has no models. No Converter calls were started for missing cards.
- Only sanitized JSON IDs/hashes/metrics are committed. Source/generated FBX/BVH/GLB/.blend/PNG and runtime/controller assets remain local under `/tmp` or existing read-only asset locations.

## Integration blockers / coordination

1. **Initial FBX dependency remains.** Frozen V3.2.5 needs parent FBX bytes/hash/reimport for its safety selector and exact rollback. Shape also writes neutral/solver/corrected FBXs. This adapter is `post-frozen-body-pre-final-export-v1`; it is not an initial-FBX-free solve. A separately versioned solver boundary and frozen-equivalence suite are still needed.
2. Approved actual-asset split-normal correspondence fails; no relaxed tolerance or bypass is introduced. Full preset combinations, extreme shapes, source fingers/bilateral/mirror and contact-dependent refine regressions are not certified.
3. **No Converter HTTP operation routes exist in this PR.** BFF cannot freeze a fabricated `/interactive-state` adapter. Contract lists actual local entrypoints and exact packet/receipt fields. HTTP scheduling/storage/idempotency/artifact/cancel integration remains to implement after real-asset validation.
4. Exact search-camera frame bridge remains unresolved: trusted descriptors provide rotation/projection/distance, but current contract additionally needs correct view/projection matrices, torso reference, crop/frame/referenceAspect and target in model coordinates. Do not invent physical focal length or silently refit search. `baked:false` only is supported by this adapter; orbit is display-only. Capability must remain absent until versioned projection/frame evidence is supplied.
5. Top-5 asset coverage, Client/BFF real files, UI/OS/save/CSP validation, scheduling priority and measured end-to-end performance/rollout review remain pending. Local fixture success is not integrated completion.

Schema remains **draft, frozen:false**. BFF selectionHash may precede solve; stateHash is the hash of Converter's entire canonical envelope and must not be a second independent BFF hash. A prior atlas `basisStateHash` is prediction provenance only, never authority for the new revision/file. BFF must recheck ownership/current selection/source/quarantine/review/expiry at accept/query/download.
