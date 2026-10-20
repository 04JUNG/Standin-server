# v4 scene worker: staging handoff (Draft)

This service runs `src.scene_refinement` separately from inference `/analyze`
and Converter. It is NOT deployed by the existing inference/Converter workflows.
This change prepares a dedicated image and an explicit staging workflow. It does
not provision AWS resources or enable the BFF feature.

## What this PR provides

- `Dockerfile.scene-worker`: Python 3.12 CPU/Gemini worker, offline CLIP/HumanArt,
  reusing the frozen Blender/Converter base built at the **same commit**.
- Asynchronous startup with `/healthz` (process alive) and `/readyz` (assets,
  Blender version/build/rig, CLIP inference and HumanArt startup probe passed).
- Hash-bound release inventory covering catalog, **all referenced BVHs**, rig,
  head calibration, CLIP snapshot and HumanArt manifest/weights.
- Dedicated ECS task/service examples and a manual **develop → staging only**
  build/deploy workflow. No automatic push/main/production rollout.
- Authenticated plan/build admission: one job including upload at a time; HTTP
  cancellation keeps the slot reserved until its native thread actually stops.

`readyz` does not call Gemini and **does not verify remote quota, account access,
model availability or scene accuracy**. A fresh paid staging plan/build smoke is
still required before BFF activation. The existing success request/response
contract is unchanged; see `src/scene_refinement/README.md`.

## Release assets and actual remaining blocker

The locally tested HumanArt bundle is `candidate`, its `license_review` is
`pending`, and its calibration profile hash is not pinned. It remains experimental.
**It cannot pass this staging preflight.** Do not relabel it to make startup pass.
The owner must supply an approved, calibrated HumanArt bundle and review evidence,
plus an approved rig/head-axis/catalog/CLIP release. No remote asset location or
worker DNS address has been verified in this PR.

Prepare a new, immutable directory outside Git with this layout (names may vary):

```text
v4-release-001/
  catalog.json
  library/bvh/*.bvh
  character.fbx
  head-axes.json
  humanart/manifest.json
  humanart/pose.onnx
  humanart/detector.onnx
  clip/config.json
  clip/preprocessor_config.json
  clip/pytorch_model.bin
  clip/tokenizer.json
  clip/tokenizer_config.json
  clip/special_tokens_map.json
  clip/vocab.json
  clip/merges.txt
```

Copy HF snapshot bytes with symlinks resolved; cache symlinks are rejected.
HumanArt `artifact.path` and `detector.path` must be relative, with no `..`, inside
the same release. When materializing a manifest from another environment, preserve
model identity, weight hashes, runtime contract, calibration and approval metadata;
path normalization is a new release artifact and needs its own manifest hash.
Runtime versions must match `rtmlib==0.0.15` and `onnxruntime==1.28.0`.
HumanArt license review must be `approved`, status `verified`/`shadow`/`canary`/
`promoted`, and calibration `profile_sha256` a real reviewed 64-digit SHA256.

Create a `paths.json` **outside** the release:

```json
{
  "catalog": "catalog.json",
  "library": "library",
  "character": "character.fbx",
  "head_axes": "head-axes.json",
  "pose_manifest": "humanart/manifest.json",
  "clip": "clip"
}
```

```sh
python scripts/inventory_scene_release.py \
  --root /releases/v4-release-001 --paths /operator/paths.json \
  --release-id v4-release-001
```

The tool emits `release.json` with review status **pending**, never an approval.
After the asset owner reviews the release, record `review.status="approved"`
and its actual review reference. Pin the final SHA256 of `release.json` in the task.
Review references are operator attestations; the runtime checks integrity and
metadata, not licensing/legal provenance. Every regular file must be inventoried;
extra files, changed weights, stale BVHs, path escapes and symlinks fail startup.
The release owner must keep the directory immutable during the task lifetime.

Provision a dedicated EFS access point with read-only IAM access for the task role
and mount `/scene-assets` read-only. UID 10001 must be able to read the release.
The task execution role needs permission to fetch the two configured secrets,
ECR image and CloudWatch logs. Do not put keys in GitHub Variables or image layers.
No runtime asset downloader or automatic asset upload is included.

## Task configuration

Start with `deploy/ecs/scene-worker-task-definition.example.json` and replace all
placeholders in the **infrastructure repository**. Values are NOT automatically
injected from GitHub Variables into ECS environment fields.

| Task setting | Value |
| --- | --- |
| `APP_ENV` | `staging` |
| `SCENE_REFINEMENT_ENABLED` | `1` (worker opt-in; Docker default is `0`) |
| `SCENE_ASSET_MANIFEST` | `/scene-assets/<immutable-release-id>/release.json` |
| `SCENE_ASSET_MANIFEST_SHA256` | SHA256 of approved release.json |
| `SCENE_VLM_PROVIDER` | `gemini` |
| `SCENE_VLM_MODEL` | explicit approved model ID |
| `SCENE_REFINEMENT_TOKEN` | ECS secret, random 32+ chars, same value used by BFF |
| `GEMINI_API_KEY` | separate ECS secret |

Image defaults bind `0.0.0.0:8002`, set the Blender path and private scratch root,
force offline HF loading and use one API process. The worker sets its own
`POSE_CANARY_STAGE=shadow` **before importing CFG**, using manifest-owned HumanArt
calibration. This does not change `/analyze` routing or require changing its
`POSE_BACKEND`/`POSE_MODEL_VARIANT`. The prior individual SCENE_* asset paths are
replaced by the single release manifest in the dedicated entrypoint.

Initial sizing is **4 vCPU / 8 GiB, not load validated**. Use private subnets;
allow inbound TCP8002 only from BFF's security group. EFS needs its own NFS rules;
Gemini needs outbound access through the existing approved egress. No public ALB
or client access. Register a private Cloud Map DNS name, for example
`scene-worker.<your-staging-namespace>`; the namespace/address must be created
and verified by infra, not guessed from this example.

Use `scene-worker-service.example.json` as input to infra. Desired count **1**,
no autoscaling, one process, deployment maximum100/minimum0 initially. Plans live
in memory; plan and build must reach the same process. A rolling replacement or
restart invalidates outstanding plans and requires a new review. This deployment
configuration deliberately has a short outage during replacement. BFF receipts
also need their own affinity/shared storage, which this PR does not change.
ECS health check must use **`/readyz`**, not `/healthz`. Set readiness grace/start
period300s; measure cold EFS hashing/model load before choosing rollout thresholds.

## Build and deploy

Local Linux Docker build (no assets/credentials in context):

```sh
docker build --platform linux/amd64 -f Dockerfile.converter -t standin-converter:scene-base .
docker build --platform linux/amd64 -f Dockerfile.scene-worker -t standin-scene:local .
```

GitHub staging Environment variables:

- `SCENE_AWS_DEPLOY_ROLE`: dedicated OIDC deploy role.
- `AWS_REGION`: staging region.
- `SCENE_ECS_CLUSTER`, `SCENE_ECS_SERVICE`: dedicated, already provisioned service.

Create ECR repository `standin/scene-worker`, IAM roles, logs, EFS release, secrets,
private DNS and ECS service first. Keep infrastructure configuration in its owning
repo. The workflow checks the existing task/service and refuses an inference or
Converter task, missing required settings or non-singleton deployment.

After this Draft is reviewed and merged, manually run **Build or deploy scene
worker (staging only)** on `develop`. `deploy_staging=false` uploads only the
commit-tagged image. `true` requires the ECS target and updates only that service,
then waits for ECS health/stability. It preserves task environment/secrets/mounts.
There is no automatic production deployment. Rollback uses the previously approved
task/image and matching immutable release; outstanding plans must be recreated.
Do not run deploy before the asset/infra checklist above is complete.

## BFF connection and smoke

From a host/task on the same private network:

```sh
curl --fail "$WORKER_URL/healthz"
curl --fail "$WORKER_URL/readyz"
# No auth should fail with 401, even while loading.
curl -i -X POST "$WORKER_URL/scene/plan" -H 'Content-Type: application/json' -d '{}'
```

Expected readiness: `{"ok":true,"phase":"ready","busy":false,"reason":null}`. Loading/failed
returns503 with a safe phase/reason code, never raw model errors, local paths or credentials.
Reason identifies the failed asset/config/policy/Blender/model/engine startup stage.
If startup fails, reproduce the preflight in the private operator environment;
do not paste model logs or customer images into public issue/CI output.

Set BFF private URL to the **verified** worker base URL (no `/scene` suffix), bearer
secret to the same `SCENE_REFINEMENT_TOKEN`, and its separate scene request budget
to900000ms. Any proxy in the path must support that timeout and160MiB JSON bodies.
Worker upload timeout is60s; per-Blender-stage timeout300s; fresh evidence has its
existing bounded model budget. There is no global worker completion SLA yet.
`503 busy` is capacity pressure, not failed readiness. Do not blindly retry build:
a failed build consumes its plan, and network ambiguity requires a fresh review.

Keep BFF `SCENE_REFINEMENT_ENABLED=false` until contract/UX work and real smoke
complete. With the feature enabled in a controlled test, use the BFF's owned-image
and reviewed-FBX route (not public worker access):

1. Analyze p060, p220 and p0718 freshly; select/review each original candidate.
2. Submit scene review → BFF calls `/scene/plan` with image SHA, original bytes,
   unique person indices, observations, reviewed scoped FBXs and their SHAs.
3. BFF authorizes proposed BVHs and converts full rigs; `/scene/build` checks
   plan/source/body identity and produces the integrated FBX and preview.
4. Verify people count2/2/3, body/scope/arm/relation changes, preserved layout,
   downloaded-FBX SHA equal to reviewed SHA, and expiry/restart/cancel behavior.
5. Import the actual downloaded files into CSP without manually changing transforms;
   compare position/size/camera adoption separately. No CSP pixel-match claim yet.

Worker failures now use `{"detail":{"code":"..."}}`:

| HTTP | Code | Meaning |
| --- | --- | --- |
| 401 | `detail="unauthorized"` | invalid/missing bearer |
| 413 | `SCENE_INPUT_TOO_LARGE` | streamed body exceeds160MiB |
| 422 | `SCENE_EVIDENCE_REJECTED` | invalid JSON/artifacts/evidence/plan binding |
| 503 | `SCENE_WORKER_NOT_READY` | startup not complete or failed |
| 503 | `SCENE_WORKER_BUSY` | slot occupied, Retry-After5 |
| 504 | `SCENE_WORKER_TIMEOUT` | upload/model/Blender timeout |
| 502 | `SCENE_WORKER_FAILED` | upstream/worker failure, details withheld |

BFF should map these to its own typed public errors. No success payload changes
or async job API are introduced here; BFF/client contract/UX changes remain owned
by their teammates. Unknown/expired plans still reject422 rather than falsely
claiming a complete typed expiry contract.

## Verification and remaining work

Local runtime/API/asset/deploy tests and a real macOS Blender5.2.0 build fbe6228777e7
preflight are recorded in the PR. Synthetic tests do not claim pose accuracy.
CI builds the Linux CPU image and checks dependency imports, authentication and
fail-closed readiness with missing assets, with no paid model calls.

Still required before rollout: approved release, provisioned staging resources,
Linux full model execution/cold-start sizing, remote Gemini credential/quota/model
smoke, BFF plan/build/review/download end-to-end test, BFF/client UX/contract fixes,
receipt affinity, load/restart/retention policy and CSP framing verification.
Raw inputs/artifacts remain on task-private ephemeral scratch until engine cleanup
or task deletion; existing cleanup is request-driven, not a strict idle TTL janitor.
Do not scale or declare user readiness based on health alone.
