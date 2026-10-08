# Precomputed candidate surfaces

Candidate angle fitting stays in inference. The expensive part was rebuilding a
posed character and rendering PNG for every candidate camera. `posed-mesh-v1`
moves that work to an offline library bake. Search, articulation, BVH downloads,
refine gates and the frozen solver are unchanged.

## Data and geometry

`scripts/precompute_pose_previews.py` enumerates a published bundle's `poses.db`
and its own `bvh/` directory. It uses `BlenderRunner.convert` with the pinned
character hash and frozen Blender/solver, requiring the normal success/integrity
checks. A second Blender process imports the resulting FBX and extracts the
evaluated surface, including finger geometry and split normals. There is no
new retargeting, decimation or source-pose repair in this step.

GLB vertices are `(Blender.x, Blender.z, -Blender.y)` relative to world Hips.
The client applies the stored `candidate-camera-v1.rotation` exactly once,
without another facing heuristic. A fixed +Z orthographic camera uses the same
rotated AABB center and `2.25 * vertex radius` frame as the aligned server PNG.
The client material/lighting differs; geometry and camera must agree.

Keys hash the BVH digest, immutable character digest, exporter source revision,
frozen solver manifest and Blender build. GLB and JSON manifest contain the
identity and FBX digest. Files changed under an old pose ID cannot silently reuse
its surface. Source line endings are normalized for Windows/Linux parity.

## Build / resume / publish

Use normal `BLENDER_BINARY`, `STANDIN_MASTER_V2_URI` and
`STANDIN_FEMALE_V2_LBS_URI` converter settings (local file or trusted registry S3).
Install `requirements-converter.txt` in the Python environment.

```powershell
python scripts/precompute_pose_previews.py `
  --library data/bundles/release --out data/pose-previews `
  --character standin-master-v2 --character standin-female-v2-lbs `
  --workers 4 --publish s3://STAGING_ASSETS/pose-previews
```

The destination is explicit; there is no public ACL. Each asset is published
before its checksum manifest. Rerunning validates and reuses successful local
assets, and retries publication. `progress.jsonl` records per-pose/character
results; failures cause nonzero exit. `--pose ID` can validate a small canary.
Worker concurrency is bounded to 1–8. User inputs/refined jobs are never read.
Run this as part of library preparation for every newly published bundle and
after exporter/solver/character updates; unprepared entries safely use PNG.

## Runtime and rollback

Set `POSE_PREVIEW_URI` on the internal converter to the dedicated S3 prefix or
absolute local directory. `GET /pose-preview/{source_sha}?character_id=...`
never starts Blender. It validates the registry character, manifest and GLB
digest, then keeps at most 128 MiB in an LRU memory cache. Missing entries return
404; corrupted or unavailable entries return 503. Internal API remains private.

BFF `/v1/pose-candidates/:id/preview-model` enforces job ownership, candidate
membership, current source digest, quarantine and character availability on
every request. It bounds the payload to 8 MiB and verifies converter headers
and checksum. The app uses a single lazy-loaded Three.js renderer to snapshot
cards and rejects external GLB resources. Navigation cancellation stops fallback
work. Missing assets/WebGL use the existing authenticated aligned PNG.

Only a selected pose proceeds to the existing FBX/refine/crop conversion. Final
FBX verification still uses its paired converter PNG; BVH remains unrotated.
Unset `POSE_PREVIEW_URI` or roll back the BFF/client to use the original path.

## Verification (2026-10-07)

Five actual staging candidates from an owned synthetic frontal-image test were
baked for both characters. Initial browser render/load on localhost: **510 ms**
for all five; reload: **139 ms**. These are local measurements, not deployed
latency guarantees (network and S3/BFF authorization are additional).

The first GLB's rotated surface was independently compared against the previous
actual staging FBX: 24,988 vertices, max relative nearest-surface error
`5.69e-7`, RMS coordinate error `2.02e-7`. Both use the same stored camera.
Further deployed latency and coverage must be recorded before rollout is called
complete. Large library bakes report their own coverage; fallback is intentional.
