# New Refine Release RC4 — staging integration

The pose solver, VLM prompt, selection policy and safety gate in `runtime/` are frozen.
`runtime-manifest.json` binds every Python source file. Do not tune these files in
this integration PR. The endpoint verifies hashes before any work or download.

## Boundary

- Internal `POST /refine/release`: PNG `image`, library `pose_id`.
- The main API reads the current library BVH and sends its SHA to the private worker.
- The worker accepts only registered, hash-pinned **unedited** BVH/FBX/rig/mesh assets.
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

BFF staging only: `REFINE_RELEASE_ENABLED=1`. Only clients sending
`releaseVersion: RC4` use this path. Existing clients and legacy `/refine` remain
unchanged. Turn this flag off first to stop new RC4 calls. Restore the recorded
staging BFF/inference task definitions to roll back the deployment.

The isolated worker runs `services.refine_release.worker:app` on port 8002 inside
the inference task. It needs Blender 5.2.0 LTS, Git, the locked Python requirements,
`OPENAI_API_KEY` from the existing staging secret, and the settings documented in
`service.settings`. Source/asset/model bundles are private deployment artifacts,
not checked into Git. Do not expose the worker directly to the Internet.

## Scope

The first staging bundle contains 45 distinct original library poses from the
validated cohorts. It contains no rough images, manual conditions, adjusted poses
or expected results. Other poses fail closed until their original assets have
been registered and bound to the deployed BVH. This is **not** an all-library
promotion. The fixed source mesh is displayed/exported; later body, hand and crop
choices are not silently applied to it.

## Validation

```
PYTHONPATH=services/refine_release/runtime/experiments/refine_final_gate:. \
  python -m unittest discover -s tests/refine_release -v
python tests/test_smoke.py
```

50 existing RC4 contracts and 54 smoke tests passed in the clean develop worktree.
Fresh staging execution, authenticated persistence/download, camera rendering and
production task-definition comparison must be recorded before declaring staging
end-to-end validation complete. Do not equate unit tests with that evidence.
