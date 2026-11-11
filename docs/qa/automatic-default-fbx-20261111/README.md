# Automatic default-model FBX preparation (2026-11-11)

The Blender probe ran with develop `5999fe397a1b81cd8c2cbc9b71166a978d7c58c1`
(the posed-finger and anatomical-clay fixes). The PR is also integrating newer
develop `b258bae2e0d15c8c1ac3cd59573d083eec3c06f3`; that upstream isolated RC4
change does not modify the Converter runtime/policy used in this probe. The earlier manual-pilot artifacts
record their original policy and are historical. This report covers automatic
preparation with the current PR policy.

## Current actual defaults

The staging ECS BFF still pins catalog manifest
`0897cb6c18fdef3e24b945bc2df9b09e8e7f65579130a98127e97c72264a97cc`,
and the approved Converter anatomy manifest remains
`f590cdb9f4ed0a78fb331bdde90c715c526c1314ce03d046432ac8eeaf3bf973`.
The catalog was downloaded again read-only for this update:

| Role | Character | Height cm | Heads | head_scale |
|---|---|---:|---:|---:|
| Common default and male default | male-base-v0-8-0 | 168.684899 | 7.943109 | 1 |
| Female default | female-base-v0-4-0 | 169.893087 | 8.000001 | 1 |

Those defaults and their original asset/source SHA-256 values were not changed.
The ideal83 selection below exists only in a private local test catalog.

## Actual lifecycle/change test

The test uses one real pose (`2hand Idle_01`) from the active library and the
approved male-base, female-base, and male-ideal83 FBXs/source recipes. It runs the
real API lifespan and background watcher with TestClient; it never calls a batch
trigger or run_once. The real Blender runner counts `_convert_one` invocations.
Both full-body camera forms (semantic identity rotation baked, and unbaked)
are required for every default. All files use original source hands and complete
the existing worker/shape/final-FBX/finger-reload gates before registration.

The test sequence is startup → local default change to male-ideal83 → revert to
male-base → API restart. The changed catalog is atomically replaced on disk.
Only the two new male outputs should be calculated after the change; female
outputs should be verified cache hits. Returning to the original default and
restarting should add no conversions. Fresh Converter requests after each step
must return hit receipts and matching final byte hashes. See report.json for
observed times, per-generation ready/failed/pending counts and exact file keys.

This is a one-pose local integration test, not full-library preparation or a
production deployment. The first manual pilot's full 3,503-pose inventory remains
useful as an input list, but its old output keys are invalid under the newer
solver/material/cache policy. A full automatic generation includes 3,503 poses ×
two current genders × two camera forms = 14,012 pose/model/camera combinations
(14,004 distinct keys with source-hash deduplication). Full generation was not
started in this task. At the earlier planning assumption of 61.9 seconds/10 MiB
per cold output, that is an upper estimate of 240.9 serial hours/136.8 GiB before
exact existing-cache reuse; target-environment measurement remains necessary.

## Observed results

| Step | Elapsed | Cumulative Blender conversions |
|---|---:|---:|
| Startup, four original default/camera combinations | 63.141 s | 4 |
| Catalog change, two new male combinations | 43.360 s | 6 |
| Catalog reversion, all originals reused | 0.540 s | 6 |
| API restart, all originals resumed | 0.423 s | 6 |

All five final-file requests returned verified cache hits in 162–179 ms with no
additional Blender conversions. Six exact final FBXs passed checksum, skin,
54-bone skeleton and original-finger reimport checks. Both catalog generations
have a complete pre-conversion plan with four expected combinations each; the
file audit contains all six distinct outputs. These are local measurements and
should not be presented as staging deployment latency or full-library coverage.

The current tests are 784 ordinary-Python Converter cases passed, one skipped;
server smoke 54/54 and compileall passed. The 36 focused persistent-cache/
automatic-worker cases include API lifecycle startup/change monitoring. BFF
build/typecheck and 93 focused ownership/framing/receipt contracts pass. The
Blender-only native suites are separately exercised by remote CI.

## Operational behavior and checks

- Configure the private cache, read-only library bundle and authoritative catalog
  paths once at a separately authorized rollout. Startup and catalog/approved
  registry revisions then reconcile automatically; no per-model manual batch or
  deployment is triggered by this worker.
- Complete plans are written before conversion. Atomic per-generation checkpoints
  contain verified ready/failed/pending counts. Failed outputs retry; successful
  files are checked and reused after a restart. A stale generation cannot declare
  a changed default ready. Requests preserve safe fallback during preparation.
- Background conversions share the API's priority pool; foreground/safety work
  may preempt them. No new hand-only solver, client selection change or refinement
  ordering change is included. The selected-hand export gate remains enabled
  whenever the existing server policy enables it.
- Input manifest files are immutable by pin. Updating the authorization pin or
  prewarm intervals does not invalidate an unchanged model's output key. Source,
  model bytes, actual numeric recipe, camera, solver and validation policy still do.
- Unit contracts exercise automatic startup and observed catalog changes,
  changed numeric defaults/policy, independent process locking (existing cache
  suite), safe retry, corrupt-file repair, late-generation fencing, preemption,
  complete plan publication and cache-storage failure without API startup failure.
- Read-only `/final-cache/prewarm-status` exposes internal operational status;
  no endpoint accepts a catalog path or starts a batch from client input.

This preserves the existing validation scope: final skeleton/skin/finite geometry,
approved anatomical PSD/roundtrip checks and source-finger preservation. It does
not add a complete whole-body collision/folding certificate. No deployment,
shared-S3 write or actual staging/default-model change was performed.

Evidence: [automatic lifecycle report](report.json), [six final file audits](file-audits.json),
[full pre-conversion plans](plans.json), [actual current default models](current-defaults.json).
