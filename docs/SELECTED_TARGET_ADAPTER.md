# Selected source → target evidence adapter

2026-11-06. Local implementation, opt-in library only. No HTTP route, deployment,
global activation, accepted v3 target profile or automatic v4 intent is added.

The user corrected the flow: search → source candidate refinement → Top5 → user
pose selection → body/shape/hands → target/file checks. `SelectedTargetProvider`
must never run from candidate preparation. Changing body/hands leaves source
prepared IDs, observations, camera, VLM and source solver caches intact.

## Dependency

Stacked on server #170 branch `codex/outline-fbx-csp-20261105`, pinned commit
`70b8a6f37d748d5a81801e0c1d632f7c512ba517`; initial fetched develop was
`f260b1e6843996935ab287cdfad63465c18f7266`. This consumes its exact-ID+SHA approved
terminal-bone importer through `_run_body_job`. No inherited appearance/outline
or frozen solver file is edited. Existing deployment gates are not activated.

## Trusted caller contract

`converter_api.selected_target.SelectedTargetProvider` takes one shared
`SelectedBlenderRunner`, a server-owned mapping of opaque IDs to immutable
`TargetSelection`s, and an `is_current(binding)` callback. The caller must resolve
and check ownership, current selection, source reuse, body/pose eligibility,
quarantine, actual revisions and expiry before calling and again before any
publication/download. A claimed phase string from a public client is insufficient.

`prepare(selected_source_bvh_bytes, binding, target)` requires:

- Binding: `phase=selected_target`, `owner_id`, `job_id`, `person_id`,
  `candidate_id`, `candidate_set_revision`, `context_fingerprint`,
  `selection_revision`, `prepared_source_id`, and exact `output_sha256`.
- Target: exactly `{selection_id, revision}`. Revision is the SHA256 of the
  canonical `TargetSelection.identity()`. Only trusted registration may contain
  local asset paths or resolved approved shape recipes. Unknown profiles and
  non-default appearance are unsupported by this adapter.
- The provider calls the callback before work and before publishing; late results
  and expired/corrupt assets cannot become ready. The source engine is never
  imported or rerun. Camera stays display-only; no search camera is baked into FBX.

The returned frozen `PreparedTarget` contains exact final FBX bytes, a static GLB
derived from their evaluated surface, and an immutable JSON receipt. It carries
the complete binding, input/source identity, geometry/FK, model/profile/shape/hand
versions, actual artifact/preview hashes, correspondence, independent structural,
mesh and opposite-hand checks, and a new `preview_revision`.

`quality_scope=target_structural_checked`, `skin_safe=false`,
`export_authorized=false`, `target_refine=false`, `automatic_parts=false` and
`scene_joint_validation=false` remain explicit. Structural skin binding is not
collision certification. The existing full mesh policy registry has no accepted
production profile/fold reference. Hard mesh/contact detections block this strict
adapter; unsupported audits remain unverified and never become a safety claim.

`prepare_or_base` requires `binding.source_sha256` to match the original bytes
and independently prepares that authoritative original source with
the **same** body/shape/hands/scope on a geometry failure. It never retries a
timeout as success. A failed original remains failed. Fallback bytes receive a
different binding/preview revision. The caller must present that changed preview
before approving a download; it must not silently save against the earlier pose.

`reuse` only returns a valid same-context cached result. It never computes on a
miss/expiry. It rechecks current selection and model/controller/file/code hashes.
No source solver, target solver, PNG or FBX export runs on a hit. This does not
grant export permission or bypass selected-scene relation/contact checks.

## Bounded execution and reuse

All children use the runner's existing bounded priority slots and process-group
termination. Required target checks share that budget with converter work; they
never modify one Blender scene from multiple threads. A request cancellation
holds its slot until its actual child has stopped. BLAS limits are set once in
the dedicated runner environment. No global request-time environment mutation.

Prepared results have private full-binding single-flight and byte-bounded TTL
caches. Failures do not enter the success cache. A verified pre-hand body FBX
cache is separately partitioned by owner/job/person/candidate-set and exact
source/model/runtime. Hand/scope changes can reuse it via private byte copies,
without retargeting. It does **not** strip shape correctives: shaped conversions
stay isolated; the existing interactive adapter's verified `.blend` path remains
the shaped body reuse mechanism. No new shared mutable body scene is enabled.

Inspection resets to factory-empty before/after and independently hashes all
resident geometry/rig/shape/material/action/transform datablocks. The A/B/A test
poisons resident geometry, weights, shape keys, transform and material, then
proves exact surface/joint/preview recovery. This certifies sequential isolated
inspection reset, not warm solved-target mutation or all Blender globals.

## Final byte/preview evidence

The unchanged frozen converter still writes intermediate FBX files. This cost is
reported rather than hidden as in-memory preparation. No PNG is generated.
`selected_snapshot` independently reimports final bytes, retains actual terminal
bones, checks structural skin binding, extracts actual evaluated triangles and FK,
and decodes its serialized GLB independently. Triangle winding/multiplicity must
match the exact float32 surface; coordinate error must remain within the existing
0.1 mm scene tolerance. Inspection never rewrites the final file.

Default GLB material is a display surface only. Non-default skin/outline needs
the existing appearance adapter and identity; this PR does not flatten or relabel
those assets. Scene contacts, native CSP import, full collision certification,
source Top5 rendering and live BFF wiring are outside this library's authority.

## Reproduction

Run `python -m pytest tests/converter/test_selected_target.py -q` and the existing
worker/API/final-artifact/shape-hand/interactive/retarget-cache regression tests.
`tests/converter/measure_selected_target.py --help` measures selected-only actual
targets with local manifests, stored source BVH, C=1/2 cold/warm and actual cache
reuse. `verify_selected_restore.py` is a Blender-only A/B/A reset observer. No
models, BVH, user images, tokens or generated FBXs belong in Git.

See `docs/qa/selected-target-20261106/BENCHMARK.md` for actual measurements and
their resource/safety limits. A local multi-core comparison is not a 1 vCPU SLA.
