# Bounded hand screening

`HAND_SKIN_SCREEN_CONCURRENCY` defaults to `1`, which selects the existing
`SkinSession`. Setting it to `2` selects ordered parallel screening. Other values
are rejected. The converter endpoint, collision predicate, BVH, character,
hand presets, solver, search weights and permission schemas are unchanged.

## Execution and failure contracts

- Each wave reserves up to two full 15-second calls against the existing session
  work budget. If two reservations do not fit, only one call is submitted.
- Results commit in candidate order. A failure can have one speculative successor
  already running; its result is discarded. Unknown is never a clear result.
- Total work includes slot waiting and transport time. Active screening wall time
  includes wave scheduling and waiting. Both budgets are checked before commit.
- The existing maximum number of checks applies across library, generated and
  optional mirror phases of one session.
- One gate admits at most two converter calls across all its sessions. This is a
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
state is capacity accounting and the existing library hash registry. A repeated
search, including the same input from another user, must run its own checks.
There is no shared receipt cache, persistent artifact cache, user-ID cache key,
or client-provided identity/cancellation token.

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
does not change ECS definitions or converter capacity. With a one-process
converter, inference-side concurrency alone may provide no speed improvement.
Activation is an explicit environment change to `2` followed by restart;
rollback is `1` followed by restart. Neither operation is part of this patch's
local preparation.
