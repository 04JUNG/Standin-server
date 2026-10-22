# Refine v3 implementation and verification — 2026-10-22

Actual rough follow-up: [live Gemini + Human-Art + strict FBX pilot](REFINE_V3_LIVE_GEMINI_ROUGH_2026-10-22.md)
records 3 real roughs, 6 corrected-pilot runs, 28 provider requests (49 including
diagnosis/rechecks), two provider-schema fixes, and **zero newly certified actual
rough FBXs**. Both selected B0 and F25 from the qualified rough fail the same
strict shell-direction test. V3 regional authority remains unverified.

Current follow-up: **one exact offline female BASE derived asset, both open hands,
and neutral QA source** passes actual final-FBX mesh v2 safety and strict receipt.
Production profiles remain zero. [Root causes, fixes, positive/negative evidence
and final regression](REFINE_V3_MESH_SAFETY_FIX_2026-10-22.md). The initial blocked
male result below remains historical evidence and remains blocked under v2.

Base: clean `origin/develop` `7a3713f7cbd1e78925f6e78704f5ff706b96bfa2`, re-fetched before implementation. The original `/Users/dowon/dev/Standin-server` working tree was never reset, cleaned, switched, pulled or edited. Three independent managed worktrees implemented observation evidence, regional solver and target/Converter safety; integration owns the shared API/lifecycle contracts. Historical revision-3 design and occlusion audit are preserved alongside this report; their statements about no implementation describe their original design stage.

## Implemented behavior

| Area | Concrete implementation | Permission boundary |
|---|---|---|
| Observation lineage | Immutable per-joint original/current coordinates and scores, source, visibility, stable slot ID and update reason through crop/rescue/repair | Model scores and VLM visibility do not certify direct observation; old global source gate is unchanged |
| Existing VLM calls | Optional bounded observations/relations/contacts, explicit exact person IDs and hand subset mapping, malformed optional-field fallback, call/attempt/usage traces | Default off; diagnostic only; no new analyze/hand/solver call |
| Regional arms | New source/query/provenance/ownership/camera binding, fixed B0 shoulder normalization, bounded observed-arm solve, numerical observability, SO(3)/pole/proxy/endpoint and independent serialized-file audit | Explicit trusted direct observation and anatomical-side authority required. Root, torso, legs, inactive channels and subsequent frames stay byte-identical. Elbow-missing endpoint remains unsupported |
| Source rollback | Exact input bytes/hash on timeout/reject, atomic candidate output, source hardlink protection | A source gain always remains `mesh_unverified` |
| Target identity | Immutable exact character/shape/rest/rig/map/hand/controller/Converter-manifest/build identity; finite evaluation-content receipts and snapshot adapter | Unknown rotations abstain. No bone-length substitute or continuous FK certification. Production registry empty |
| Final safety | Actual evaluated full-body skin triangle intersection, closed-component containment, reference-normal fold checks, explicit unsupported/time/pair limits; FBX reimport integrity stays separate | No contact/region-wide penetration exemption. No current production body policy/fold reference certificate |
| Final receipt | Immutable final-byte SHA/size, profile/context/revision/source and mesh/reimport audit identity | Strict issuance only for exact audited full-body bytes. Framed/rotated derivation certification unsupported |
| Integration | Off/shadow API, signed private BFF relay, current-camera check, shared remaining deadline and diagnostic allowlist | Shadow never replaces F25 or exports experiment bytes. Existing auto selection/review/deferred preview remain |
| Cache/lifecycle contracts | Source versus target fingerprints, owner/job/person/selection identity, body/shape/hand/scope/version invalidation and exact artifact checks | Foundation tested; target cache/product state migration is not activated in BFF/client |

No new manual button, body selection step, extra approval or Top5 refinement/render was introduced. Existing five output scopes and FBX-ready without waiting for preview remain in place. No deployment, production flag activation or merge was performed.

## Verification evidence

Isolated Python 3.12 venv under `/private/tmp/standin-refine-v3-venv`; no original venv/model assets changed. Live VLM calls, production API traffic and large evaluations were not used. Existing SDK fault tests use a local loopback HTTP server, not a real provider.

- Before target integration: full inference regression **2,478 passed, 10 skipped**, 28 subtests passed. Required smoke **54/54 passed**; compileall passed. Initial sandbox-only failures were local listening-socket restrictions and a missing optional decoder dependency; after isolated dependency setup and loopback permission, those tests passed.
- Evidence/solver/runtime selected integration: **148 passed**. Additional signed-camera and successful-F25 equivalence regressions passed. Off/shadow preserves accepted F25 BVH and limb decisions exactly; stale source/query/signature/camera cannot execute the experiment.
- Target/mesh boundary: **13 new + 130 existing** relevant worker/API/final-integrity/shape/hand tests passed. Final integrated rerun is recorded below.
- Independent review found a shadow camera-context mismatch and source/target unit mixing; both were fixed with regressions. Target evaluation content is independently checked against its exact receipt; no finite sample is promoted to a continuous envelope.
- Synthetic seven-mask source-only benchmark: 70 cache-off solves, per-mask warm p95 approximately **3.6–13.8 ms** on this local runtime. One missing-elbow side retained the base (`insufficient_gain`). These runs use explicit synthetic direct-visibility fixtures. Cold process, shared F25+target/Converter end-to-end SLO, multi-person resource load, real rough quality and hidden-joint recovery were **not** established. Raw report: [synthetic source timing](../evaluation/refine_v3/synthetic_source_cache_off_2026-10-22.json); rerun with `PYTHONPATH=. python scripts/eval_refine_v3_synthetic.py --output <new-report.json>`.

Actual Converter QA used read-only local `male-base-v0-7-2`, native shape corrections, both relaxed hands (30 finger bones), the repository CMU BVH fixture, and pinned Blender 5.2.0 LTS build `fbe6228777e7`. The final FBX had 52 bones, 24,988 vertices and 50,000 polygons; skin reimport and additional topology roundtrip passed. Maximum roundtrip joint error was `4.0676e-7 m`, vertex error `7.0289e-7 m` (both under 0.001 mm). **This verifies that snapshot's transport/integrity, not continuous TargetFK equivalence or pose improvement.** The full mesh shadow recorded intersections and exceeded its 3-second budget, with uncertified policy/fold reference: `validation_blocked`, `complete_collision_certificate=false`. Near-touch positives have not been independently classified. **At this initial stage, no actual output was approved as safe or improved.** [Exact QA report](../evaluation/refine_v3/actual_converter_snapshot_2026-10-22.json), [reproduction and limits](../converter/REFINE_V3_SUPPORT.md).

## Product support that remains unavailable

Production certified profile count is **0**. Continuous retarget/FK, source-canonical coordinate transform, certified palm/back/sole surface patches, actual profile swing/twist/hinge envelope, seam/region/fold-normal policy, strict five-scope derivation validation, real rough holdout/blind evaluation, and cold/multi-person end-to-end performance certification remain unsupported. Parent/pelvis optimization, missing-elbow endpoint solve, contact optimization, multi-person/prop geometry are not enabled. Real provider prompt accuracy/latency/token nonregression is unmeasured, so evidence prompts stay off.

Current model observations are `model_inferred`, and independent side/visibility authority has not been certified. Therefore the new regional production gate correctly abstains even when synthetic tests can move a visible arm. This is an implementation of a bounded experimental path and safety foundations, not a claim that all revision-3 product completion criteria have been achieved.

BFF/client ownership remains user-side; inference is server-team-owned; Converter/profile/mesh certification is shared. BFF companion is a separate develop-based PR with a private default-off signed sidecar relay only. It does not change public eligibility/cache keys or need a client UI change. Target-result caching, body/hand revision adoption and strict blocked-save UI must be integrated only after a real supported profile/policy and negotiated product capability exist.

## Initial integrated validation (before mesh safety follow-up)

- Inference: `python -m pytest tests/ --ignore=tests/converter -q` — **2,495 passed, 10 skipped**, 28 subtests passed (27.92 s).
- Converter unit/contracts: **415 passed, 1 skipped** (5.38 s). Three Blender-only executable modules (`test_convert.py`, `test_v325_runtime.py`, `test_v32_parent_coherent_math.py`) were excluded from CPython collection because they require `bpy`/`mathutils` and separate CLI/asset parameters. The actual Blender shape/hand/reimport QA above ran separately; these three full executable suites were not claimed as run.
- Focused solver/runtime/target/mesh hardening: **67 passed**.
- Smoke **54/54 passed**, `compileall src api scripts converter converter_api` and `git diff --check` passed.
- BFF companion [draft PR #129](https://github.com/04JUNG/Standin-app-server/pull/129): **741 passed, 2 skipped**; build/typecheck passed. No client code change is required or included for private shadow relay.

Skipped/unsupported cases are not counted as validated output quality. No operating flag or deployment was changed.
