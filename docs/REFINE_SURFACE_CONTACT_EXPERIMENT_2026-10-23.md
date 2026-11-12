# Separate surface-contact experiment — 2026-10-23

The requested baseline was backed up before edits. The experiment applies a local
left-hand skin contact to the actual rough's validated female BASE FBX. It passes
the unchanged full-mesh, joint/depth, export and independent-skin gates plus a new
contact audit. **It does not achieve natural full-palm adherence.** Gemini sees no
clear contact improvement; fixed-camera 2D joint error increases 1.71%.

## Backup and isolation

- Baseline code commit: `814ea9730`, branch `codex/refine-depth-safe-backup-20261023`.
- Experiment branch: `codex/refine-surface-contact-20261023`.
- Baseline FBX SHA-256: `d0476ba2306700b5e66865682a05aefba0720781d28ccfbf7b935882ee810099`.
- Durable backup: `/Users/dowon/.codex/visualizations/2026/10/22/01a162e8-7b84-7af3-995a-87b9efbe7b26/refine-depth-safe-backup-20261023`.
  Its manifest hashes 116 copied viewer/run files. No environment files or keys.
- Existing viewer `http://127.0.0.1:18773/` remains unchanged.
- Separate viewer `http://127.0.0.1:18773/surface-contact/`.
- No deployment, push, production registration, original FBX edit or endpoint change.

## Contract and executed behavior

`converter/surface_contact.py` adds a contact requirement on top of existing
collision safety. The local experiment explicitly selects the left hand and
pelvic skin. It does not infer observed contact intent or side from a VLM.

The target includes Hips/Spine skin and proximal thigh blend weights within 0.35
current torso lengths of pelvis height. Selecting Hips weights alone truncates
real pelvic skin at a weight boundary and produces misleading surface distances.
The whole hand includes finger groups; the nearest connected patch is fixed from
the baseline before optimization. The optimizer cannot reselect an easier patch.
In this run the patch lies at `LeftHandThumb2`, not the full palm.

`plan_surface_contact.py` uses two shoulder swing directions, a transported elbow
hinge, and bounded arm/forearm twist. It balances point-to-triangle surface gap
against the exact saved rough's fixed-camera 2D observations. The existing
`pose_safety.audit_pose` must pass. Bone lengths and protected joints remain
restricted by the same policy. Final wrist/finger poses come from the existing
support/relaxed presets; no new anatomical certificate is claimed.

`realize_contact_rig.py` edits only the selected arm's rest-bone matrices in a
baseline planning carrier. This avoids unrelated toe roll numerical drift when
rebaking every rest bone. Its old mesh is discarded. Existing realization,
independent rest skin correction, full-body QA and byte-bound receipts still run.
The 1e-5 bone-matrix contract and 1e-5 m vertex replay contract are unchanged.

`audit_surface_contact.py` verifies the full-mesh receipts for both actual FBXs,
rebuilds the frozen contact region from the actual baseline, checks source/asset
bindings, and measures the delivered final mesh. A final status requires:

- Full-body mesh and final pose safety validated, with matching artifact bytes.
- Positive sample clearance at least 0.5 mm. Target is 2 mm, contact band ends at 5 mm.
- At least 25 mm² and 20% of the frozen patch's area.
- A face counts only when all three vertices and its centroid are within the band.
- Duplicate faces, nonfinite/degenerate geometry and mismatched topology fail closed.

The reported area is a sum of qualifying mesh faces, not a continuous skin-pressure
or exact triangle-pair minimum-distance certificate. The pre-existing full-mesh
certificate independently rejects crossings, contained components and folds.
A finite positive gap is used rather than mathematical zero or overlapping skin.

## Actual run and evidence

Input rough: `124629.png`, image SHA `ba92e47ea1600905a81b5973813f5c711087e479898dec0b230ae5716de3ccc6`.
Existing selected candidate: `rokoko_FootTapping_mixamo_00040`, actual source SHA
`835b3eb6ab103385ce8b9ee1627014df950ec36671e20108f80e63008f3c0d17`.
Character: `female-base-v0-3-4`; left `support`, right `relaxed`.

Accepted run: `/private/tmp/refine-contact-applied-v4-20261023`.
Durable copy: `/Users/dowon/.codex/visualizations/2026/10/22/01a162e8-7b84-7af3-995a-87b9efbe7b26/refine-surface-contact-run-20261023`.

| Measurement | Baseline | Contact experiment |
|---|---:|---:|
| Minimum frozen-patch sample gap | 58.8996 mm | 1.3468 mm |
| Median patch centroid gap | 59.2957 mm | 1.9842 mm |
| Maximum patch sample gap | See contact audit | 3.3779 mm |
| Qualifying patch area | 0 mm² | 111.7503 mm² |
| Qualified fraction of fixed patch | 0% | 100% |
| Fixed-camera actual-FBX body joint NME | 0.10723047 | 0.10906306 |
| Wrist NME | 0.15249300 | 0.15442357 |
| Final body and hands matrix error | — | 0 / 0 |

Final SHA: `06acf7951a1da6c70b1488e4c5b5fdbbfc33277f45557a711ca7d2441ba5b8d0`.
Full mesh receipt SHA: `afc013b424104dbdbef6db6cae1a01e53f97b0b55bd64b72afb13db437f7c22b`.
The receipt for unchanged geometry remains valid after adding the stricter
all-corner contact-area audit; `applied.json` binds the resulting contact audit SHA.

Neutral/body/hands mesh gates all pass in 2.18/2.36/2.41 seconds (existing 3 s
limit). Independent vertex replay errors are 0.44/1.73/2.38 µm (existing 10 µm
limit). Actual final FBX has 52 bones, 24,994 vertices and 50,000 triangles.

Early palm-only proposals failed angular/depth bounds or caused actual thumb–hip
triangle intersections. These were rejected. Expanding the effector to the whole
hand corrected the incomplete geometry model; no rejected FBX was adopted.

## Live Gemini and remaining limitations

`review_surface_contact_gemini.py` made one actual `gemini-3.8-flash` request with
the original rough plus six before/after full-body/close-up renders. Image hashes,
model version, usage and response are saved in `gemini-review.json`.

- Preferred: `uncertain`.
- Hand-body contact change: `same`.
- Rough pose match change: `same`.
- It reports visible spacing, hand at the lower abdomen rather than side waist,
  and remaining elbow, leg support and ankle differences.

The numerical contact pass concerns a small partly occluded thumb patch; it does
not contradict the lack of visible full-hand/palm adherence. Fixed-camera 2D error
is 1.71% worse. That metric uses saved observations, not independent 3D ground truth.
This experiment therefore remains separate from the validated depth baseline.
Automatic intent extraction, other body regions, full-palm alignment and skin
compression are unimplemented. Next work must model the intended contact surface
and hand orientation while respecting the same gates, rather than simply pulling
the palm toward the nearest point or expanding the safety limits.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 /private/tmp/standin-refine-v3-venv/bin/python scripts/apply_surface_contact.py \
  --analysis /private/tmp/standin-refine-v3-live-rough-fixed-20261022/evidence_shadow/124629/result.json \
  --baseline-run /private/tmp/refine-depth-one-command-v2-20261023 \
  --character /Users/dowon/dev/Standin-server/artifacts/converter-registration-check-2026-10-16/release/fbx/female-base-v0-3-4.fbx \
  --character-id female-base-v0-3-4 \
  --shape-registry /Users/dowon/dev/Standin-server/artifacts/converter-registration-check-2026-10-16/release/shape-bakes.local.json \
  --side Left --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --output-dir /private/tmp/refine-contact-reproduction-FRESH
```

The command requires a fresh output directory and writes `contact_applied: false`
on any downstream rejection. It does not perform live VLM requests implicitly.
`review_surface_contact_gemini.py` is the separate explicit VLM review command.

## Validation

- Converter suite: **495 passed, 1 skipped** (same three Blender-only test-module
  exclusions as the baseline; actual Blender pipeline executed separately).
- New contact tests cover exact triangle interior/edge distances, positive gap,
  penetration/coincidence rejection, tiny area, duplicate-face inflation, nonfinite
  and degenerate geometry, topology mismatch, untrusted threshold overrides and
  the centroid-only area overclaim regression.
- Frozen converter manifest: all 13 hashes unchanged.
- New Python modules compile; `git diff --check` passes.
- Baseline backup: all 116 file hashes verified; original viewer HTML unchanged.
- Separate browser page inspected; view buttons switch before/after render pairs.
  Downloaded localhost FBX bytes match the final artifact SHA above.
