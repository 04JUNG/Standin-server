# Refine v3 actual mesh safety follow-up — 2026-10-22

One exact **derived female BASE + both open hands + neutral QA source** now passes
actual final-FBX safety validation. This is a finite full-body snapshot, not an
arbitrary-pose, continuous deformation, or real rough/refine quality certificate.
Production certified profiles remain **0**. Nothing was deployed, merged, enabled,
or written to the original dirty checkout/released assets.

## Why the original actual mesh was blocked

1. The male CMU output contains real shoulder-area triangle crossings. Released
   neutral assets also have toe crossings: the female native BASE neutral has
   three crossing pairs under the v2 unshrunk test, involving six vertices.
   Native `correct_pose` is not a collision certificate: the final definitions in
   the pinned male/female controllers explicitly disable intersection checks and
   contact corrections. A successful native shape correction cannot clear this.
2. Scalar Python AABB/SAT exhausted the unchanged three-second limit on 50,000
   triangles. The male mesh has 340,939 broad-phase pairs. An incomplete audit
   correctly cannot certify, even independently of its observed crossings.
3. An exact asset/shape/hand policy and independent transported fold reference
   were missing. Copying output normals or resetting reference area after each
   stage would hide defects and is not an acceptable fix.

## Changes and unchanged limits

- `mesh_validation.py` v2 performs a complete vectorized AABB sweep and batched
  SAT. Incident triangles are cleared only by a separating-plane proof that their
  intersection is confined to the shared vertex/edge; adjacency alone never
  exempts a pair. Unproved triangles stay unshrunk and undergo conservative SAT.
  Pair-local projections avoid translation-dependent false clears. Unsupported
  world-coordinate precision abstains. Independent review reproduced thin-overlap
  and translated-overlap false clears in the prior inset/proof attempt; both were
  fixed before the recorded v2 run. Its successful v1 receipt is superseded.
- Distance epsilon **1e-7 m**, original area ratio **0.001**, **3 seconds**, **50,000
  triangles**, and **200,000 unresolved exact pairs** are unchanged. All broad-phase
  pairs are accounted for, including 313,034 geometrically certified common-boundary
  pairs in the successful final mesh. No face, body region, contact, or crossing
  is hidden. A normal's magnitude is not compared to a distance epsilon; normal
  direction and the original independent area/collapse test are checked separately.
- `neutral_mesh_repair.py` is an explicit offline asset-maintenance utility,
  absent from production conversion. It considers only intersecting-face vertices,
  at most 64, with an absolute **0.5 mm** movement bound. Every candidate requires
  a complete unchanged validator pass. The female candidate moves six toe vertices
  by at most **0.278593 mm**. Alpha 0.05 remains blocked; alpha 0.1 passes. The
  original asset and inactive vertices remain unchanged; the derived asset has a
  new SHA/profile identity. No user's selected pose is silently adjusted.
- `deformation_witness.py` retains original neutral areas and all three corner
  normal directors through neutral repair, real body conversion, and real hand
  baking. Normalized LBS uses independent rig deformation maps; every actual
  reimported vertex must match replay within the existing **1e-5 m** FBX tolerance.
  Positive determinants, nonsingular maps, all three director signs and cumulative
  original-area ratios are required. DQS, unknown modifiers and unsupported
  envelopes abstain. This is discrete shell orientation, not a continuous volume
  Jacobian proof. Output normals/positions are not fitted to create the reference.
- `refine_v3_safe_snapshot_qa.py` runs the native shape controller, exports a new
  derived neutral, runs the frozen Converter, applies both real hand presets,
  checks the full skin after every stage, reimports final bytes for structural
  integrity, and issues/verifies a strict immutable receipt for those exact bytes.
  The existing full-character shape importer retains weighted terminal bones.
  Frozen retarget files and product runtime/flags are unchanged.

## Recorded actual results

Pinned Blender **5.2.0 LTS**, build `fbe6228777e7`; native
`female-base-v0-3-4` BASE. `--neutral-source` explicitly generates a neutral QA BVH
on the real Female Standing hierarchy. This does **not** certify its original
Standing rotations. Actual frozen body conversion and both open hand presets
(30 finger bones) execute normally.

| Stage | Full mesh | Broad-phase / remaining SAT pairs | Seconds | Max vertex replay error |
|---|---|---|---|---|
| Derived neutral repair | mesh_validated | 325,846 / 12,812 | 2.0744 | 4.43e-7 m |
| Actual body Converter | mesh_validated | 325,865 / 12,831 | 2.0603 | 9.14e-7 m |
| Actual both open hands / final | mesh_validated | 326,239 / 13,205 | 2.0694 | 1.338e-6 m |

Every stage has zero crossings, contained components, folded faces, and unknown
reasons; a closed, consistently oriented, single full-body component. Final FBX
reimport verifies **52 bones, 24,994 vertices, 50,000 polygons**, one bound skin.

- Final FBX SHA256: `e503416f66201fc22b6d5959797b00954da13ec480b0fcf5be9bc518a7114b4e`
- Strict receipt SHA256: `832025c6caecc8b451dc4040ac9fe5f15a702fe081d6928856fad4d14ab524e0`
- Exact profile SHA256: `638a78d790f087c98de10ee893b809caa1ef86671ec9c7b3d1a260167f124e0b`
- [Complete actual v2 evidence](../evaluation/refine_v3/actual_safe_female_neutral_open_v2_2026-10-22.json)
- Local FBX: `/private/tmp/standin-refine-v3-safe-snapshot-v2-20261022/final.fbx`

Negative controls remain rejected:

- The original exact male CMU final FBX remains `validation_blocked` with triangle
  crossings; its complete collision sweep now finishes in **2.4034 s**. Missing
  policy/fold reference remain explicitly unknown. [v2 negative audit](../evaluation/refine_v3/actual_original_male_negative_v2_2026-10-22.json).
- The unmodified Female Standing BVH with the repaired female asset is rejected
  during body witness construction for `discrete_shell_orientation_reversed`.
  No hands/final receipt are issued. [Negative control identity](../evaluation/refine_v3/actual_female_standing_negative_v2_2026-10-22.json).

## Reproduction and regression

Run Blender with `--background --python-exit-code 1 --python
converter/refine_v3_safe_snapshot_qa.py -- --character <read-only-release-fbx>
--character-id female-base-v0-3-4 --shape-registry <read-only-registry>
--source-bvh <Female-Standing-BVH> --neutral-source --hand-left open
--hand-right open --output-dir <fresh-temp-directory>`. Omit `--neutral-source`
to test the actual source rotations; failure must remain a failure. FBX exports
can contain different metadata bytes across runs, so each run binds its own exact
asset/profile/witness/final receipt rather than reusing the recorded receipt.

- Converter unit/contracts: **454 passed, 1 skipped** (5.16 s). The same three
  Blender-only executable suites remain excluded from CPython collection; the
  actual Blender QA above ran separately.
- Inference: **2,495 passed, 10 skipped**, 28 subtests passed (24.17 s).
- Smoke **54/54**, compileall, diff whitespace checks, and the entire frozen
  Converter `SHA256SUMS.v325` manifest pass.
- New regressions include scalar/batch SAT parity, legal shared boundaries, real
  incident/duplicate/thin/translated crossings, world precision abstention,
  immutable witness lineage, hidden corner flips, cumulative collapse,
  singular/reflected transforms, and bounded repair rejection/rollback.

No runtime profile was registered. Arbitrary poses, the original male CMU and
female Standing failures, continuous FK/volume proofs, five framed scopes and
real rough holdout quality remain outside the single passing snapshot.
