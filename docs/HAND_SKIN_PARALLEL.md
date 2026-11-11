# Bounded hand screening

`HAND_SKIN_SCREEN_CONCURRENCY` defaults to `1`, which selects the existing
`SkinSession`. Values `2` through `5` select ordered parallel screening with that
shared per-engine capacity. Other values are rejected. The converter endpoint, collision predicate, BVH, character,
hand presets, solver, search weights and permission schemas are unchanged.

## Execution and failure contracts

- Each wave reserves full 15-second calls against the existing session work
  budget. Its width is bounded by capacity, remaining display slots and the full
  reservations that fit. The default 45-second budget allows at most three calls
  per wave, even with capacity 4 or 5. Calls are never shortened to fill capacity.
- Results commit in candidate order. A failure can have speculative successors
  already running in that wave; their results are discarded. Unknown is never a clear result.
- Total work includes slot waiting and transport time. Active screening wall time
  includes wave scheduling and waiting. Both budgets are checked before commit.
- The existing maximum number of checks applies across library, generated and
  optional mirror phases of one session.
- One gate admits at most the configured capacity across all its sessions. This is a
  per-engine/per-process limit, not a distributed limit across inference workers.
- Optional mirror checks run only when the gate is idle and leave one slot free
  for mandatory checks. An unavailable optional check leaves already verified
  main matches unchanged. Its HTTP request retains the optional-priority header.
- In parallel mode, `/analyze` disconnect or broken receive cancels only that
  request's sessions. No new checks start and late results are discarded. An
  in-flight HTTP request still ends by its bounded transport/converter budget;
  this patch does not provide an immediate remote process interrupt.

## Identity and ownership

Receipts, BVHs, hand presets, diagnostics, failures and cancellation events belong
to a session/request. Presets are copied at session creation. The only gate-wide
state is capacity accounting, the existing library hash registry and an optional
read-only operator release catalog. A repeated
search, including the same input from another user, runs its own checks by default.
There is no runtime-learned shared receipt cache, persistent artifact cache,
user-ID cache key or client-provided identity/cancellation token. The separate
opt-in [operator-prevalidated library catalog](HAND_SCREEN_PREVALIDATION.md)
contains only explicit immutable release assets; it never learns from user requests.

`library_text()` still checks the registered pose ID, path containment, symlinks
and current BVH bytes before session-local reuse. `HandSkinClient` still binds
the receipt to BVH, character, hand library, solver and hands and validates the
screened artifact hash. The checker certifies opposite-hand skin triangles only.

The inference service remains internal and unauthenticated, as documented in
`QA_SECURITY_RELEASE.md`. The BFF retains installation authentication, owned-job
SQL scoping, person/candidate membership, private asset binding and hash checks.
Request IDs are tracing values, not ownership keys. This patch adds no export,
refine or release grant. Existing BFF manual export continues to require its
stored permission, owned selection and confirmation checks.

## Validation and activation

Preparation against develop `766ce81efdec452c5e58d3ad945477d0d4661e9e` verified:

- 468 Python tests with concurrency `1`, and the same 468 with `2`.
- 114 unchanged local BFF ownership/presentation/export tests.
- Eight real patched-server replay reports passed through the unchanged BFF
  mapper, installation middleware, SQL ownership and private asset route against
  an in-memory PostgreSQL schema. Missing/wrong tokens, foreign owners, wrong
  people, candidate IDs and hashes failed closed; changed BVH bytes were rejected.
- Frozen retrieval/synthetic 2D → actual ASGI/Blender replay, two hand presets
  and two crossed rounds: candidate order, receipts and presentation contracts
  matched. Reimported geometry fingerprints matched for all 48 FBXs; raw FBX
  bytes were not identical.

These are local tests using a copied BFF snapshot and fixture installations.
They do not validate live credentials, tenant topology, AWS capacity or initial
image/model inference. No external model or staging request is required to run
the tests. The hand screen always evaluates full skin; presentation scope does
not change its collision predicate.

Keep the default at `1` until converter CPU/memory, process count and concurrent
request queue behavior have been checked in the target environment. This patch
preserves existing ECS capacity unless an explicit workflow override is set. With a one-process
converter, inference-side concurrency alone may provide no speed improvement.
Activation is an explicit environment change to `2` followed by restart;
rollback is `1` followed by restart. Neither operation is part of this patch's
local preparation.

## Capacity comparison and deployment

`scripts/benchmark_hand_screen_capacity.py` compares capacities 1 through 5 on
fixed, hash-bound caller-owned BVHs and a character. It calls the real local
Converter HTTP API and Blender, writes every completed FBX and keeps timing
separate from independent geometry verification. `service` mode uses the actual
45-second session policy. `converter` mode is an isolated capacity stress test
with 15-second individual calls; it does not claim five calls fit one session's
45-second aggregate work budget. Source assets are never committed.

```sh
python scripts/benchmark_hand_screen_capacity.py --manifest /private/manifest.json \
  --output /private/new-run --blender /path/to/blender --rounds 2
blender --background --python-exit-code 1 \
  --python tests/converter/verify_hand_screen_capacity.py -- /private/new-run
```

The two deploy workflows accept explicit environment-scoped overrides:
`HAND_SKIN_SCREEN_CONCURRENCY` for inference and
`CONVERTER_MAX_CONCURRENT_PROCESSES` for Converter. Both accept `1..5`; empty
values preserve the existing task definition exactly. The helper changes no
image, CPU, memory, role, secret or other container. Default inference concurrency
remains 1. Raising the inference limit without Converter capacity can only add
queueing. Shared capacity across users still needs target-environment load tests.

## Local comparison, 2026-10-30

Develop base `121684a1a0b5fa699f8f9d946ae5a3abdd45277e`, macOS arm64,
Blender 5.2.0 (`fbe6228777e7`), six fixed BVHs, relaxed left/right hands and one
existing character. Two rounds reverse the capacity order. Each cell is the
median of two runs in seconds; process capacity is set to match inference capacity.

| Configured capacity | Actual session policy | Isolated Converter capacity | Peak processes in session |
| --- | ---: | ---: | ---: |
| 1 | 10.02 | 9.88 | 1 |
| 2 | 5.09 | 5.23 | 2 |
| 3 | 5.23 | 3.92 | 3 |
| 4 | 5.21 | 3.88 | 3 |
| 5 | 5.28 | 3.78 | 3 |

Two workers reduced this local screening stage by 4.93 seconds (49%). Increasing
capacity beyond two did not improve the actual session policy in this sample.
The isolated Converter test reached all configured capacities but bypasses the
aggregate session work limit. A 4/5 setting therefore does not establish a faster
single-cut generation path.

All ten capacity/mode groups had identical ordered business results to serial,
including rejected/unverified candidates and decision/source-binding fields.
Independent reimport of 120 FBXs found exact equality of mesh vertices, evaluated
skin, topology, bind and posed bone matrices and vertex weights in all six cases.
Raw FBX bytes contain changing exporter metadata and are not expected to match.
Contract tests additionally compare all 27 clear/intersection/unavailable outcome
combinations at capacities 2 through 5 against serial presentation results.
The full inference suite passed 2,827 tests (10 skipped, 28 subtests); the
Converter CI contract set plus deployment helper passed 393 tests (one skipped).

These measurements cover hand screening only. They exclude fresh image/model
inference, use two local rounds and do not reproduce the 2 vCPU / 4 GB ECS target.
Peak memory and concurrent-user load remain unmeasured. Equality under these
fixtures does not guarantee equality under timeouts or resource exhaustion.
Start with two only after target-environment capacity validation. Defaults remain
one; no AWS assets were uploaded and no service configuration was changed.

The sanitized timings, source hashes and geometry fingerprints are recorded in
[`experiments/hand_screen_capacity/2026-10-30/results.json`](../experiments/hand_screen_capacity/2026-10-30/results.json).
Private character/BVH/FBX files are excluded from the repository.
