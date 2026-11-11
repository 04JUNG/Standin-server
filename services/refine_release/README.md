# New Refine Release RC4 — staging integration

The pose solver, VLM prompt, selection policy and safety gate in `runtime/` are frozen.
`runtime-manifest.json` binds every Python source file. Do not tune these files in
this integration PR. The endpoint verifies hashes before any work or download.

## Boundary

- Internal `POST /refine/release`: PNG `image`, library `pose_id`.
- The main API reads the current library BVH and sends its SHA to the private worker.
- The API resolves the selected pose in its current library and sends the **unedited BVH bytes** plus their hash over the private worker connection.
- The worker automatically retargets frame 0 to the unchanged RC4 source body, extracts the rig and evaluated skin, and binds every generated file. There is no pose registration list.
- One new `gpt-6-luna` / `low` call; automatic skeleton and conditions; no manual targets.
- `accepted`: verified FBX hash, selected mode, separate camera, display GLB hash.
- `withheld`: no corrected download. No rejected candidate is released.
- Internal `GET /refine/release/{handle}/fbx` and `/preview`: verified bytes only.
- BFF must immediately persist bytes and metadata under authenticated job ownership.
  Worker handles are task-local and are not durable public URLs.

The GLB is a static display surface baked from the **final FBX** with Blender's full
skinning evaluation. It does not change the FBX or the solver. Browser FBXLoader
truncates vertex influences to four, so it must not be used as evidence of the
verified final surface. Camera remains separate: Blender Z-up metres, right-handed
basis columns, +Z toward viewer. The client reverses GLB's Y-up conversion and
applies the returned orthographic/pinhole projection without reframing.

## Switches and rollback

Default is off. Keep production unchanged.

Main API staging only:
```
REFINE_RELEASE_ENABLED=1
REFINE_RELEASE_WORKER_URL=http://127.0.0.1:8002
```

BFF staging only: `REFINE_RELEASE_ENABLED=1`. The current job response advertises `refineRelease` only for a single target.
Only clients sending
`releaseVersion: RC4` use this path. Existing clients and legacy `/refine` remain
unchanged. Turn this flag off first to stop new RC4 calls. Prepare rollback from the current staging task definitions; do not restore an
old revision over unrelated concurrent deployments.

The isolated worker runs `services.refine_release.worker:app` on port 8002 inside
the inference task. It needs Blender 5.2.0 LTS, Git, the locked Python requirements,
`OPENAI_API_KEY` from the existing staging secret, and the settings documented in
`service.settings`. Source/asset/model bundles are private deployment artifacts,
not checked into Git. Do not expose the worker directly to the Internet.

## Scope

Every pose resolvable in the deployed library can enter the same automatic source
preparation path. No case ID, prepared experiment asset, manual coordinate, camera
or adjusted pose is used. Invalid/missing originals or incompatible conversion
fail closed; successful preparation is not a promise of passing the safety gate.

The converter used for the original RC4 fixtures is pinned separately in
`asset_runtime/` and `asset-runtime-manifest.json`. It is not the pose solver.
Configure `REFINE_RELEASE_BODY` and `REFINE_RELEASE_BODY_SHA256` with the same
unchanged source-body template. The cache identity includes original BVH hash,
body hash, converter/build-script hash and Blender version. Cached file hashes
are rechecked; each request gets a fresh rough binding outside the asset cache.
The old `register_assets.py` utility and `service.prepare` are retained solely
for historical fixture verification tests; the request path never calls them.

The fixed source mesh is displayed/exported; later body, hand and crop choices
are not silently applied to it. Single-target extraction and the unchanged
release safety gate still apply.

## Validation

```
PYTHONPATH=services/refine_release/runtime/experiments/refine_final_gate:. \
  python -m unittest discover -s tests/refine_release -v
python tests/test_smoke.py
```

54 release/adapter tests and 54 smoke tests passed in the clean develop worktree.
Actual staging/native validation and remaining limits are recorded in
[the staging report](../../docs/qa/refine-rc4-staging-20261111.md).
