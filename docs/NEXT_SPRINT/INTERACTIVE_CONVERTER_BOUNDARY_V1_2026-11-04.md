# Interactive Converter boundary v1 — local draft, not rollout approval

Date: 2026-11-04 KST. Base: `592ab0f059089e64a5c88a3ad607539124e8e6c6`.
Design: [interactive preview and async FBX](INTERACTIVE_POSE_PREVIEW_ASYNC_FBX_DESIGN_2026-11-04.md).
Contract: [converter-contract-v1.json](../../artifacts/interactive-preview-20261104/converter-contract-v1.json).

## What this change can and cannot claim

This is an **opt-in local adapter**, with no new HTTP route, environment default,
production worker branch, capability registry entry, deployment or inference call.
The current static GLB generator/validator remains unchanged and rotation-only.
Search, camera search hypotheses, skeleton extraction and VLM logic are unchanged.

`hands.solve_hands` applies the existing hand geometry, safety diagnostics, body
preservation checks, rig-root/skin repair, rigged-rest bake and fist surface
smoothing to a private scene without exporting. `bake_hands` delegates to it and
retains its old import/export behavior. The frozen solver and manifest are untouched.

**The initial-FBX-free solve goal is not implemented.** V3.2.5 `convert_safe`
constructs the exact V3.2.4 parent FBX, hashes it, reimports it, evaluates surface
safety/selection, and may restore its exact bytes on fallback. Shape additionally
exports neutral/solver/corrected FBXs and validates source-controller transfers.
Bypassing `export_fbx` would invalidate these gates and rollback guarantees.
The new boundary is explicitly `post-frozen-body-pre-final-export-v1`.

Approved actual asset tests found split-normal differences above the new strict
normal correspondence check. These runs are **failed**, not final-file certified,
and no local-hands capability is released. A proposal to make normals diagnostic
only was rejected by automatic approval review as conflicting with the user's
instruction not to relax validators; that proposal was reverted. The strict
normal gate remains. Real asset normal preservation and the earlier frozen
solve/export separation remain follow-up work, not silent exceptions.

## Local execution boundary

1. A trusted runner writes the existing schema-v3 job. `prepare_body(job_path,
   identity, new_directory)` revalidates it with `load_job`, checks frozen
   hashes/build/source/character/shape/runtime identity, and runs the existing
   worker with full scope, no hands, no PNG, no display rotation. Its normal
   solver/shape/final-skin checks are retained.
2. Capture the resulting accepted private body/shape as a hashed `body.blend`
   plus canonical model/binding metadata. Body cache identity includes owner,
   original/refined BVH, refine version, character/rig, shape source/control
   digests, mirror, solver manifest and adapter runtime digest. Hand changes and
   selection revision can reuse it. Hand-dependent refine must change its refined
   source or refine version; otherwise callers must not reuse this base.
3. `solve_from_body(..., checkpoint=...)` restores that snapshot, checks it against
   its model, applies the **latest requested** hands, checks body preservation,
   finite matrices and self-collision diagnostics, crops with existing skin-region
   rules, captures state, and saves private `solved.blend`, canonical state and
   snapshot hashes. It publishes `packet.json` last, after a cancellation fence.
4. `export_bundle(..., expected_state_hash, checkpoint=...)` opens a copy of the
   solved snapshot in a fresh Blender process. Another fresh process imports the
   actual FBX and runs existing final skin validation plus strict model correspondence.
   Only after another current-revision fence does the host publish `receipt.json`.
   An exception, timeout, stale selection, restart or normal mismatch publishes
   **no ready receipt**. Private partial files may be garbage-collected by the host.

Two consumers never mutate the same live scene. Snapshots are local server
artifacts, not client-uploaded `.blend` files. They must remain private and
hash-checked; Blender uses `--disable-autoexec` and `use_scripts=False`.
Body snapshot keys include ownership; this implementation does not create a
cross-installation cache. Restart invalidates in-memory revision tokens. Durable
operation recovery must restore selection authority from BFF before reissuing
work; an old packet or receipt is not proof of current selection.

`LatestRevisionGate` retains one revision/tombstone per owner/person, refuses
old/conflicting revisions, and revokes older work at safe checkpoints. Export
children are terminated on cancellation/timeout. No unbounded revision queue is
implemented. The BFF must additionally fence its persisted operation, selection,
review, expiry, source lineage and quarantine at accept/query/download time.
These local helpers do not grant download authorization or expose HTTP endpoints.

## Wire format and canonical identity

`solved-pose-state-v1` is the immutable envelope. `SolvedPoseState` stores only
canonical `bytes`; accessors decode fresh copies. `stateHash` is SHA256 of the
**entire envelope**, including schema, identity, model, safety and dependency
facts. `modelHash` hashes the model with the same serialization. The identity has
exactly `ownerBinding`, `selection`, `source`, `body`, `shape`, `hands`, `scope`,
`mirror`, `camera`, `solver`. Unknown fields are rejected.

- `ownerBinding`: opaque SHA256-sized BFF ownership binding, not raw installation
  identity. `selection`: operation, monotonic revision, opaque person/candidate.
- `source`: original/refined BVH SHA256 and refine algorithm/run version.
- `body`: registered character ID/SHA and rig profile.
- `shape`: requested shape SHA (nullable), rules version, combined approved
  asset/blend/controller SHA and applied-control SHA. Defaults still bind their
  source and controls; `null` does not erase default-shape lineage.
- `hands`: explicit nullable left/right IDs, library SHA and hand-bake version.
  `null` retains approved source fingers. Explicit preset replaces only that
  side. Fist repair/weight redistribution/smoothing is part of this same rule.
- `solver`: solver version, manifest SHA, adapter version and runtime source SHA.
  Runtime changes invalidate body/snapshot reuse even if geometry coincides.
- `scope`: actual latest server values `full`, `half`, `bust`, `shoulder_head`,
  `head`. The design's “four scopes” predates current `head`; no scope is silently
  renamed to `upper_body`. `mirror` is anatomical pose identity, not view orbit.

Canonical serialization is `typed-f64be-json-v1`: sorted ASCII object keys,
compact ASCII JSON; every finite Python float becomes
`{"$f64":"<16 lowercase big-endian IEEE754 hex digits>"}`. Integers are exact
within ±(2^53−1), booleans remain booleans. `$f64` is reserved in input objects.
This avoids float/negative-zero/exponent differences between Python and JS.
Hash the original canonical bytes, then decode tagged numbers for rendering.
Packet `canonicalState` is the ASCII canonical payload, not a separately
re-stringified identity. Time, local paths, file names and view orbit are absent.

## Model data and hands

`interactive-pose-model-v1` is a separate bounded JSON model, **not** a rigged
GLB accepted by the old static GLB validator. The strict validator in
`converter/solved_state.py` is authoritative for this draft schema. Maximum 64 MiB,
256 bones, 32 meshes, 500,000 vertices total, 1,000,000 triangles per mesh,
32 influences/vertex. All finite weights are retained, never truncated to four;
unbound, unknown-bone or non-normalized vertices fail. No external URI, arbitrary
extension, material program, animation, shape-key driver or runtime script exists.

Coordinates are right-handed **Y-up, metres, hip origin**. Blender `(x,y,z)` maps
to `(x,z,-y)` after subtracting the fixed packet `originBlender`. The conversion
applies to joints, geometry, bind matrices and normals once. Matrices are flattened
row-major; consumers using column-major libraries must explicitly transpose/import
accordingly. Bones are parent-before-child, with `restLocal`, `poseLocal` and
world-space `inverseBind` in this shared frame. All current deformation is baked
into mesh/rest (`rigged_rest`), no animation. Skinning at the current pose is
identity. Mesh positions/weights are vertex-indexed. Split normals are loop-indexed:
`triangles` indexes positions; `normalTriangles` indexes normals. A renderer must
split corners consistently rather than substitute averaged vertex normals.

Server/client hand agreement uses **exact server-generated variant data** rather
than an independent browser kinematics implementation. `interactive_variants.py`
builds at most 32 explicit left/right combinations for an allowlisted registered
character and a single bound source/body/shape/scope/camera. Every included state
requires a matching verified FBX receipt. Full corrected surface, rest/bind,
weights and bone matrices come from that state. Fists cannot be approximated by
rotations alone. No arbitrary combinations or per-side patches are inferred.
Unsupported combinations return `null`, preserving the prior authoritative model.
Personal FBX/CS3C receive no capability by profile/name resemblance.

Client can import `converter/shared/interactive_pose_v1.mjs` without porting hand
math. `readSolvedState(bytes, expectedSha)` verifies original canonical bytes;
`readHandVariants(bytes, expectedSha)` verifies inner state and model bindings;
`selectHandVariant(table, currentIdentity, hands)` returns **local-draft** only.
A historic variant's `basisStateHash` does not certify the new revision or file.
No production atlas is shipped in this PR. Offline/on-demand atlas size/cost,
all preset combinations and visual QA remain required before capability approval.

## Camera, readiness and tolerances

Camera is explicit: `projection` (`orthographic`/`perspective`), row-major
`viewMatrix` and `projectionMatrix`, positive `distanceTorso`, the body SHA used
as torso reference, and `baked`. These matrices are the already selected search
hypothesis expressed in the packet coordinate frame, including framing/crop.
This adapter supports `baked:false` only; it rejects baked-camera requests to
avoid rotating twice. It does not infer physical focal length or promise CSP
camera transfer. View orbit is external display state. Shape change requires a
new source-frame camera decision upstream, not an implicit scale approximation.

Packet says `previewState:authoritative`, `exportState:not-requested` and
`safety.finalFileVerified:false`. This means approved solve/display state within
existing adapter checks, **not** a full collision certificate or final FBX receipt.
Skin integrity alone does not prove penetration/containment/folding/joint safety.
Existing refine selector and shape contact/surface checks still run upstream.

Correspondence verifies exact bone/parent identity, topology, skin bone index
identity and finite geometry; positions/joint translations ≤0.0001 m (existing
scene round-trip bound), other matrix elements ≤1e-5 (existing hand body matrix
bound), weights ≤1e-6, split-normal vector difference ≤1e-4. None of the frozen
solver/shape tolerances or static GLB validators is changed. Known normal
mismatches fail the new boundary. Final receipt binds identical stateHash,
revision/operation/identity, runtime version, model and FBX hashes. File hash is
checked again after reimport. Model bytes and export bytes have separate hashes.

## Validation and unfinished work

See [converter status](../../artifacts/interactive-preview-20261104/converter-status.md)
and sanitized evidence in that directory. Unit/cross-runtime tests are distinct
from synthetic Blender file tests and approved actual-asset file tests. BFF/Client
integration and stage-0 coverage/performance acceptance are not claimed.

Follow-up to eliminate the first FBX requires a **new versioned solver boundary**:
retain the exact parent rollback snapshot, make parent/selector safety evaluation
operate on immutable pose/surface state, and compare pose/skin/vertex and fallback
bytes/semantics against the complete frozen chain on approved regressions before
activation. Do not monkeypatch away exports/hash checks or edit frozen SHAs.
Resolve real-asset split-normal preservation without weakening the gate, expand
shape/extreme/preset/mirror/camera fixtures, then cross-validate contracts and
actual files through BFF and Client before any rollout decision.
