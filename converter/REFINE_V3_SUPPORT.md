# Refine v3 target and final-artifact safety boundary

Status: implemented internal contracts and default-off diagnostic. No production
profile/mesh policy is certified or activated. Existing Converter endpoints,
automatic flow, five output scopes and independent preview remain unchanged.

2026-10-23: one actual non-neutral rough + derived female BASE + support/relaxed
hands now passes full-body final-FBX validation. The offline application command
also reproduced the result. See [actual application, measurements and scope](../docs/REFINE_ACTUAL_ROUGH_APPLIED_2026-10-23.md).
This does not activate a production profile or promote regional observation authority.

The offline `--depth-safe` path now fits the actual target rig under fixed-camera
depth-order, hinge, relative angle/twist and whole-mesh guards. An actual final
FBX passes all gates; visual depth improvement remains unconfirmed by Gemini.
See [depth and joint safety evidence](../docs/REFINE_DEPTH_SAFETY_2026-10-23.md).

Follow-up: one exact offline derived female BASE + both open hands + neutral QA
source passes actual full-body final-FBX v2 validation and immutable receipt.
See [root causes, fixes, exact evidence and limits](../docs/REFINE_V3_MESH_SAFETY_FIX_2026-10-22.md).
Original male CMU and unmodified female Standing poses still fail closed.

## Implemented

- `src/target_profile.py`: immutable exact asset, shape, neutral rig/rest, map,
  hand/revision, frozen Converter manifest and Blender build identity. Finite
  equivalence receipts bind source BVH and actual evaluation hashes. A certificate
  for sampled poses cannot authorize unseen rotations or a continuous slider range.
- `src/target_fk.py`: frozen `SnapshotTargetAdapter` provides bounded lookup of
  independently certified exact snapshots. Unknown BVH hashes abstain. No bone
  length substitution or generic fast target FK is presented as Converter equivalence.
- `mesh_validation.py`: evaluated world-space full-body triangles after actual
  shape/hand baking, before framing. Bounded broad phase, conservative triangle
  SAT including coplanar overlap, closed/oriented-component winding containment,
  collapse/transported-normal fold checks. Shared boundaries require a geometric
  separating-plane proof; unproved triangles remain unshrunk. An entire
  neighboring pair/region is never exempted. Contact never waives a
  crossing. Unknown policy, open surfaces, absent independent fold reference,
  triangle/pair limits and timeout cannot produce a certificate.
- `artifact_receipt.py`: immutable revision/context/profile/final-byte receipt.
  Structural skin reimport alone is insufficient. Strict issuance requires the
  exact same full-body audited bytes; cropped or camera-rewritten derivations are
  unsupported until an independent derivation verifier is supplied. Image rendering
  is not required to issue or verify the receipt.
- Worker diagnostic `CONVERTER_REFINE_V3_MESH_SHADOW=1` is opt-in and default off.
  It adds `refine_v3_mesh_shadow`; blocked/unverified findings do not silently change
  existing conversion success in shadow. It cannot authorize a product safety claim.
  Existing `final_artifact.skin_verified` means finite geometry and skin binding,
  not penetration/folding validation.

`TargetEvaluation` uses meters in right-handed Blender Z-up. The regional solver's
source-canonical projection needs a separately certified transform; this adapter
must abstain there. Production `TARGET_REGISTRY` is empty. Trusted server code is
responsible for registering independently reviewed certificates; hashes are identity,
not evidence that arbitrary geometry is safe.

## Initial actual local QA, 2026-10-22 (historical v1)

Executed `refine_v3_target_qa.py` against Blender 5.2.0 LTS build `fbe6228777e7`,
read-only local release `male-base-v0-7-2`, its native shape controller, both relaxed
hands, and repository fixture `cmu_135_01_04368.bvh`. Outputs stayed under a new
`/private/tmp/standin-refine-v3-target-qa-boundary-20261022` directory. No service
call, deployment, production flag change or original checkout change was requested.

- Actual frozen Converter/shape correction/30 finger bones applied successfully.
- Final FBX reimport: 52 bones, 24,988 vertices, 50,000 polygons, skin binding valid.
- Additional export/reimport topology matched; maximum joint error
  `4.0676093186504025e-7 m`, vertex error `7.028863677424643e-7 m`.
- Final immutable artifact SHA256:
  `16b386e5a7c831c2c7a6f31f48b6a8b0bf2b2b65899fe269681476747860e6b9`.
- Full-body shadow detected triangle intersections and exhausted its 3 second
  budget. Profile policy and transported fold normals remain uncertified.
  Result is `validation_blocked`, `complete_collision_certificate=false`.
  Conservative contact/near-touch positives have not been independently classified;
  this result must not be reported as safe or a quality improvement.

Reproduction: run Blender with `--background --python-exit-code 1 --python
converter/refine_v3_target_qa.py -- --bvh <fixture> --character <read-only-fbx>
--character-id <id> --shape-registry <read-only-registry> --output-dir <new-temp-dir>`.
The script records a new exact profile from neutral shape rest, runs the real worker,
reimports the final file, and writes `qa-report.json`. It never registers that snapshot
as a certified fast FK or changes any downloaded artifact.

## Verification and remaining support

13 new tests cover changed asset/shape/hand/revision/implementation identity,
nonfinite or stale certificates, immutable snapshot payloads, unknown rotations,
triangle and adjacent/coplanar crossing, component containment without crossing,
folding, open-mesh/time-budget abstention, final byte/revision mismatch, and unsafe
receipt reuse across framed/replacement bytes. 130 relevant existing worker,
HTTP/ready, final-integrity, shape and hand contract tests passed in an isolated venv.
Synthetic closed tetrahedra demonstrate the policy code, not a certified body asset.

Continuous fast retarget/FK, source-canonical transform, certified surface patches,
anatomical joint-limit envelope, production mesh region/seam policy, generic
fold-normal transport, five-scope strict derivation certification, and real asset/rough holdout
quality/latency certification remain unsupported. Solver and strict final approval
must abstain on those missing capabilities. Existing F25/proxy or skin integrity
passes cannot fill them in.
