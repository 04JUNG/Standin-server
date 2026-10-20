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


## Runtime configuration

Run from the repository root in a dedicated worker environment. Provision Python
API/core dependencies, PyTorch, Transformers CLIP, the pinned HumanArt runtime
and its manifest/weights, and Blender 5.2 matching the Converter calibration.
The default inference and Converter containers are not wired to launch this worker.
Use Gemini for the tested configuration; optional MLX requires its own runtime.

Required operator environment variables:

- `SCENE_REFINEMENT_ENABLED=1`, `SCENE_REFINEMENT_TOKEN` (32+ characters).
- `SCENE_REFINEMENT_ROOT`: private scratch directory, retained briefly for diagnostics;
  restrict access because raw images and derived artifacts are stored there.
- `SCENE_BLENDER_BINARY`: trusted Blender executable.
- `SCENE_POSE_CATALOG`, `SCENE_POSE_LIBRARY`: matching catalog and registered BVH directory.
- `SCENE_CHARACTER_FBX`, `SCENE_CHARACTER_SHA256`, `SCENE_HEAD_AXES`: matching calibrated asset and axes.
- `SCENE_VLM_PROVIDER`, `SCENE_VLM_MODEL`: explicit real provider/model, plus its secret credential.
- `SCENE_POSE_MODEL_MANIFEST`: pinned HumanArt model bundle.
- Optional `SCENE_CLIP_MODEL` (model ID or local snapshot), `SCENE_REFINEMENT_HOST`
  (default `127.0.0.1`), `SCENE_REFINEMENT_PORT` (default `8002`).

Catalog, axes, character and weights are external operator assets, not bundled by
this PR. Production packaging, Linux/container model execution, capacity/load
behavior, multi-replica BFF receipt affinity, installed desktop native saving,
and CSP camera/framing require validation before enabling for users. BFF receipts
are process-local: requests reaching another replica must be routed consistently
or a shared receipt store added. Expired/missing receipts fail closed and require review.
