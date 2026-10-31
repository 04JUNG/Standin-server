# Staging hand-screen activation

The serving configuration uses two screening workers and two Converter process
slots. Keep these values scoped to the GitHub `staging` environment:

- `HAND_SKIN_SCREEN_CONCURRENCY=2`
- `CONVERTER_MAX_CONCURRENT_PROCESSES=2`

The 2026-10-31 configuration rollout preserves inference image `3f04e70` and the
existing Converter CSP/body export image `csp-shape-ea4f57962-staging`. It changes
neither body releases, API credentials, CPU/memory nor production configuration.

Both serving services reached ECS `COMPLETED`/healthy state. An internal AWS smoke
used six library candidates and selected the same five, with identical order,
hand status, triangle counts and source/body/preset bindings. One worker took
32.458 s and two took 31.428 s. This single original-hand run saved only 3.2%; it
is not the earlier Mac result or a controlled multi-round workload benchmark.
It made no VLM calls and did not move private BVHs outside AWS. Parallel screening
is active, but this result does not establish a material generation-time win.

## Prevalidation prerequisite

The currently installed CSP image has no `/hand-screen-profile` route. Enabling a
catalog against it would fall back to live screening, so do not enable an empty
or incompatible catalog just to set a flag.

`Dockerfile.converter.hand-screen-staging` derives from that exact installed image
and adds only the read-only profile API and catalog definitions. It checks the
original API/CSP source hashes before copying the API; all geometry workers,
Blender, base layers and CSP export modules remain unchanged. CSP source additions
in this branch retain the version already installed in staging when this branch
is eventually integrated into develop.

The manual `Deploy converter` target `hand-screen-staging-image` builds an immutable
image only. It runs solely from `develop` in the protected `staging` environment;
it does not update ECS services or replace `latest`/`develop` image tags. The
feature branch was rejected by GitHub's existing environment branch protection.
Merge through the normal develop review/checks before invoking this target.

## Catalog release

The fixed staging library metadata contains 5,862 source poses, not the six poses
used by the earlier local experiment. The archive also contains thumbnails and
is about 3.97 GB. Read it inside AWS and stream only BVHs for preparation; do not
copy private library assets to a developer machine.

Prepare an operator-pinned catalog using the exact target code, Blender,
registered base body hashes and hand presets. An isolated job may prepare it
before serving deployment when its target source bytes and base image are pinned.
The serving profile must still match after deployment. Store the catalog as a new
immutable staging release. A startup initializer must verify its
SHA256 and mount it read-only before setting both `HAND_SKIN_PREVALIDATED_PATH`
and `HAND_SKIN_PREVALIDATED_SHA256` in the inference task. Do not reuse the Mac
certificates. Body-specific/custom-shape and partial-coverage limitations in
`HAND_SCREEN_PREVALIDATION.md` still apply.

Verify actual certificate hits and unchanged selected/rejected/unverified results
before describing the live screening stage as removed. Two workers alone do not
activate prevalidation, and no production speedup has been measured by this rollout.

## Full offline preparation and startup connection

The current serving hand-shape VLM flag is off, so descriptor defaults pass
`left=relaxed,right=relaxed`. The first full preparation therefore covers all
5,862 pinned source poses for both `standin-master-v2` and
`standin-female-v2-lbs`: 11,724 distinct base-body/pose/hand bindings. It does not
certify continuous body shapes, other hand presets, or the later selected body.

`scripts/launch_hand_screen_prevalidation.py` runs at most eight isolated staging
jobs. They use the exact installed CSP image and proposed profile/catalog source
bytes; only the read-only profile additions differ from the serving distribution.
They do not expose HTTP ports or update a service. The proposed target runtime
profile observed in AWS is
`6353976c30ddcb76e5fd0fc62f596c8cea43a03c938b2e177025bbf572970d0a`.
The eventual Converter must report this exact profile before any certificates hit.

Preparation verifies the entire 3.97 GB archive SHA256 inside AWS, streams only
the safe BVH files out of it, and checkpoints actual validated receipts in the
existing private bucket. No raw BVH/FBX leaves AWS. An initial pass used the live
14.85 s work budget; the full pass uses a bounded 60 s offline budget to avoid
repeating slow inspections. Timeouts never become certificates. A separate, isolated
single-process worker (`launch_hand_screen_retries.py`) reinspects those fixed
failures with a longer offline budget, without changing serving timeouts or any
geometric safety threshold. Completed receipts must retain exactly the original
pose/body/hands/runtime binding. This can eliminate former timeout-dependent
unavailability; it does not claim identical responses under arbitrary timing.

The profile image build checks both the original CSP sources and the exact added
API/catalog bytes used by preparation. An unrelated later source change cannot
silently reuse this release: it requires preparing/reviewing a new target profile.

`publish_hand_screen_prevalidation.py` waits for the fixed shard/retry summaries,
then runs assembly and startup-install verification inside AWS. Assembly rejects
pilot data, missing/duplicate shards, altered checkpoints, unmatched retries,
remaining unavailable inspections and incomplete body/hand pose coverage. It
publishes `catalog.json` first and `manifest.json` last, each to a fresh immutable
release key. Job-scoped PUT URLs give no general S3 write permission to workers.
Publication can be retried after a lost acknowledgement only when the catalog
and manifest contents are identical; it keeps the first publication timestamp.

The inference workflow has a **develop/staging-only** release connection:

- `HAND_SCREEN_RELEASE_MANIFEST_KEY`: pinned `hand-screen/releases/.../manifest.json`.
- `HAND_SCREEN_RELEASE_MANIFEST_SHA256`: exact manifest SHA256.
- `HAND_SCREEN_RELEASE_MANIFEST_JSON`: exact small manifest bytes as UTF-8 JSON;
  the deploy runner verifies its pin without requiring S3 read permission.
- `HAND_SCREEN_PREVALIDATION_ENABLED=0`: prepare/mount the file, retain live screening.
- Change the last value to `1` only for the separately verified activation step.

`prepare_hand_screen_release_deployment.py` adds an owned startup initializer and
shared volume without changing images/resources/secrets of other containers.
`install_hand_screen_release.py` checks manifest and catalog SHA256, every receipt,
entry count and profile binding before an atomic read-only installation. Inference
waits for successful installation and mounts that volume read-only. Empty rollout
variables preserve the existing task; explicit `0` after clearing the release
key, pin and inline JSON removes
this owned connection and runtime flags. No startup installer accepts a user path.
The initializer independently downloads the manifest and catalog from S3 using
the existing task read role; inline deployment metadata grants no extra AWS access.

These sources remain in PR #163 until the normal protected develop review/merge
path is complete. Preparing data and a task definition does not deploy that code
or activate certificate lookup in the serving process.
