# Verified final artifact before preview (2026-10-21)

`POST /convert-framed-export` accepts `preview_format=none`. It runs the same retarget, shape, hand, framing/camera and worker integrity checks, then returns a binary-FBX/hash/lineage envelope with `artifact_contract=verified-fbx-v1`. No PNG or GLB job is requested. Full/cropped framing reports remain required, including full output with a camera. Existing PNG/model contracts are unchanged.

`POST /render-final-preview` accepts bounded final binary FBX bytes and their SHA256. A separate Blender process imports those bytes and renders a PNG; the parent verifies the input remained unchanged and checks preview hashes. `scene_camera=true` uses the already verified camera embedded in a scene FBX. This endpoint never retargets, crops, bakes hands, or exports another FBX. It uses the shared low-priority preview queue, is cancellable, and yields to final preparation and safety work.

The optional scene worker's `build` accepts `artifact_only=true`. Composition, layout and camera roundtrip validation still run; only final/raw/angle output renders are skipped. Earlier visual evidence required for model decisions remains. Legacy requests still return paired previews. No search or `/analyze` policy is changed.

## Validation

- Converter and scene-worker Python tests exclude three Blender-only modules from the normal Python runner; those require Blender imports (`bpy`/`mathutils`).
- Tests cover all five scopes, no visual work in artifact-only requests, corrupt FBX/source rejection, separate preview lineage, queue cancellation/preemption, shape/hands, cache publication integrity, scene artifact-only building, and existing contracts/deployment COPY allowlist.
- `scripts/verify_final_artifact_ready.py --bvh PATH --character PATH --character-id ID --output /tmp/ready-qa` runs a real local API/Blender probe. Add `--shape-registry PATH --shape-control muscle=0.5` for approved source baking. It checks first half-body output, hand change, hand-free head output and optional shaped output. The later PNG must name the same unchanged FBX hash and contain nonuniform pixels.
- `docs/qa/fbx-ready/real-converter.json` records the local Blender 5.2.0 result. This is a single local probe with an existing generated BVH and approved male model, not a before/after benchmark or an estimate of production latency reduction.

No installed Standin application, GUI E2E, CSP import, paid scene-model run, or live production S3 test was performed. Merge/deploy this PR (Converter **and** scene worker), then BFF, then client. Existing clients remain compatible. The converter preview revision changes intentionally, invalidating old runtime-bound caches.
