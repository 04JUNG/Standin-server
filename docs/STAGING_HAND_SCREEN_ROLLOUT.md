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

After the profile image is deployed, prepare an operator-pinned catalog using the
actual target code, Blender, registered base body hashes and hand presets. Store
it as a new immutable staging release. A startup initializer must verify its
SHA256 and mount it read-only before setting both `HAND_SKIN_PREVALIDATED_PATH`
and `HAND_SKIN_PREVALIDATED_SHA256` in the inference task. Do not reuse the Mac
certificates. Body-specific/custom-shape and partial-coverage limitations in
`HAND_SCREEN_PREVALIDATION.md` still apply.

Verify actual certificate hits and unchanged selected/rejected/unverified results
before describing the live screening stage as removed. Two workers alone do not
activate prevalidation, and no production speedup has been measured by this rollout.
