# Scene refinement v4 (local opt-in)

This worker ports the generic v4 pipeline into the save path. There are no per-cut
pose, yaw, coordinate or relation overrides. It never changes the frozen retarget
solver. It requires real fresh VLM + CLIP + HumanArt evidence; model output files
from an old experiment are not accepted by the production API.

The BFF first reauthorizes each reviewed scoped FBX and loads the owned raw image.
`POST /scene/plan` derives the preservation baseline from those exact FBXs,
reanalyzes the image, validates person identities, and proposes registered library
poses. The BFF obtains every proposed BVH through its existing library gate and
converts full rigs with the same body, shape and hands. `POST /scene/build` checks
the plan/body binding, applies independent head direction and fixed-length arm IK,
then crops the output and fits the original head anchors and depth order. Only
silhouette conflicts receive additional separation. The final FBX is reimported
and rendered, and mesh/bone/camera gates must pass.

The client must review this changed scene. The BFF pins it with an owner/job/source
bound 15-minute receipt; download reauthorizes the original reviews and returns
identical bytes. No hidden regeneration or replacement on download. Restart,
receipt expiry or changed body/hands require another scene review.

Launch separately using `python -m src.scene_refinement`. The entrypoint lists
required operator environment variables. `SCENE_REFINEMENT_ENABLED=1` and a private
32+ character token are mandatory. The BFF uses its own default-off
`SCENE_REFINEMENT_ENABLED=true`, private URL/token, and separate 900-second request
budget. This does not extend `/analyze` or modify startup routing configuration.
The private port must remain internal. One request runs at a time; a busy worker
rejects additional work. Operator-provided model/rig/library paths are never read
from public client input.

Currently supports the calibrated 52-core-bone Mixamo rig (preserving its 13 unweighted terminal nodes) matching
`SCENE_CHARACTER_SHA256`. A different rig/body is rejected; it is never silently
replaced with the master. Per-body calibrated axes and broader rig support remain
necessary before general rollout. The camera is embedded plus camera.json; CSP's
camera adoption remains separate from scene world transforms and this pipeline
makes no CSP pixel-match claim.


## Dedicated staging runtime (Draft)

See [the staging deployment and BFF handoff](../../docs/scene-worker-staging.md)
for the dedicated image, approved release inventory, required ECS settings,
private URL/auth configuration, error contract and smoke-test sequence.

`python -m src.scene_refinement` now loads assets through
`SCENE_ASSET_MANIFEST` + `SCENE_ASSET_MANIFEST_SHA256`, rather than independently
supplied model/library paths. `SCENE_REFINEMENT_ENABLED=1` and a private 32+ character
`SCENE_REFINEMENT_TOKEN` remain required. Only the dedicated Gemini/CPU deployment
is supported by this entrypoint; research callers can still construct Engine
explicitly. `/healthz` is liveness; `/readyz` passes only after offline assets,
Blender and model startup checks. Remote Gemini access requires a separate smoke.

The worker remains default-off. Deployment resources and approved assets still
need provisioning; this code does not activate the BFF or change `/analyze`.
