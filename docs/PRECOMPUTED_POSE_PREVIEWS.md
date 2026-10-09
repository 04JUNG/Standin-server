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

Only a selected pose proceeds to FBX/refine/crop conversion. Clients supporting
`modelPreview` can now review the paired `framed-mesh-v1` surface without a PNG;
older clients retain paired PNG verification. BVH remains unrotated.
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

## Final review and export reuse (2026-10-08)

The converter's opt-in `preview_format=model` performs the existing retarget,
crop and camera transform, then bakes the actual final FBX in the same worker.
The BFF validates both hashes and model lineage, coalesces pending requests, and
retains the FBX/GLB pair for up to 10 minutes within a 128 MiB process-local cache.
Keys include ownership, BVH digest, character digest, scope, camera and exporter
revision. This is a per-instance cache; restart, eviction or another BFF instance
can still cause a conversion. No production publishing is implied by a local bake.

The app retains the static GLB on the GPU for review and redraws on resize. An
unrefined full-body selection can immediately use its precomputed surface while
the paired FBX prepares. A refined or cropped selection waits for its final model.
The BFF decides library eligibility from its stored refine artifact, even if the
client missed the refine response. Saving pins the reviewed source and character
hashes (plus final exporter revision for framed models); changes require review.
WebGL failure uses the exact scoped PNG and exports that PNG's paired FBX.
Refine requests in the new FBX flow skip the intermediate thumbnail only; the
solver, acceptance policy and accepted BVH are unchanged.

Local Windows validation used Blender 5.2.0/fbe6228777e7 and actual rig assets:

| Check | Observed result |
| --- | --- |
| Real FBX/GLB generation | Male full/half/bust/head, female full, Mixamo/100STYLE/CMU and refined input passed |
| Refined sample, final model vs final PNG | 9.61 s vs 14.88 s; one sample each during concurrent library baking |
| Real BFF → converter HTTP → Blender | 9.69 s preparation; 78.74 ms subsequent FBX response; exactly one conversion |
| Paired file identity | Downloaded FBX SHA equals GLB `base_fbx_sha256`; unauthorized cache hit rejected |
| Actual review component in Chromium | Six cases rendered; warm fetch + first frame 269–585 ms; close/reopen passed |
| Initial development module load | 3.73–3.96 s while Vite transformed modules and library baking ran |

These are local probes, not deployment latency guarantees. Browser verification
used the actual React review component with real generated models. Tauri native
save, CSP import, macOS and production traffic require release validation.
The BFF probe used fixture job/ownership storage with the actual framed route,
converter client and Blender; production database/S3 latency was excluded.
Roll out converter/inference first, BFF second, client last; absent capability
retains the legacy PNG path. The new exporter does not invalidate posed-mesh-v1
keys or modify the frozen solver manifest.

The isolated local library cache was completed and audited against the registered
2026-10-08 library snapshot: **1,968 poses × 2 characters = 3,936 valid GLBs**,
4,728,087,864 GLB bytes. This includes 520 newly baked models and 3,416 verified
existing models linked into the isolated cache. Four female-model timeouts were
retried successfully; the final `coverage.json` has no missing entries. Each entry
was checked through the serving store for source/character lineage and content
digest. Cache assets remain local and are not included in Git or published to S3.

Final checks: server converter/refine suite 173 passed and 1 skipped; application
smoke 54/54; BFF 270 tests; client review/export 105 tests, production build and
changed-file lint passed. Source-change, stale-character and lost-refine-response
cases are covered. The Blender-specific Python scripts were excluded from normal
pytest collection and exercised through the real Blender probes above.
