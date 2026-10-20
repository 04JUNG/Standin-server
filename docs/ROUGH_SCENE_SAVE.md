# Rough scene save — local validation, default off

`POST /compose-scene` accepts optional `rough_layout` and per-person observations,
or the existing explicit world placements. These modes cannot be mixed.
`response_format=fbx` returns the raw artifact; the default remains the ZIP bundle.

Automatic manifest example (COCO17 arrays abbreviated):

```json
{
  "schema_version": 1,
  "cut_id": "job-id",
  "coordinate_system": "threejs-y-up-m",
  "rough_layout": {"version":"rough-layout-v1","width":765,"height":714},
  "persons": [{"person_index":0,"fbx_sha256":"<64 lowercase hex>",
    "observation":{"box":[0,0,765,714],"keypoints":"17 pixel pairs","scores":"17 confidences"}}]
}
```

The converter fits head position and projected skin height under one perspective
camera. Only root translation changes: reviewed pose, body shape, hand articulation,
skin topology, scope, rotation and physical scale remain fixed. Depth assumes
comparable physical humans; it is an estimate. Low-information observations fall
back to visible skin bounds. There is no per-cut coordinate table or extra spreading.
Fit residuals describe chosen anchors, not pose correctness or CSP pixel alignment.
All inputs retain one-rig/static-skin/size/hash validation. Reimported world skin and
bone heads must survive within 0.1 mm. Raw output binds artifact SHA, person count,
protocol and exact submitted manifest SHA; the ZIP includes the camera/fit report.

Companion BFF/client branches connect normal FBX saving. The client sends reviewed
download paths, person/candidate IDs and hashes. The BFF reuses existing authorized
export handlers before/after composition and gets observations from the owned job.
The desktop verifies the scene hash before its existing native writer. Semantic
jobs and BVH keep their old behavior; enabled scene failures cannot silently fall
back to individual files. The BFF flag is `SCENE_EXPORT_ENABLED`, default false.

## Actual CSP validation (2026-10-19)

T001 and p0718 observations were extracted again with local Human-Art. Existing
selected poses were frozen; previous automatic p0718 rotations were retained.
The local BFF handler called actual HTTP Converter, Blender and FBX roundtrip.
Owned-job and reviewed-download authorization were fixtures, not production DB or
installed Tauri E2E. No production services were changed.

Actual saved FBXs were imported into separate rough copies in CSP, with no manual
model/camera transforms. Three people remain in one file with relative placement.
However, the tested import path recenters/reframes the scene and did not honor a
side-view FBX camera probe. T001 before/after therefore appears at the same default
size despite different serialized placement; p0718 also differs from the specified
camera. A transparent-bounds probe was insufficient and is not production code.

**CSP framing remains unresolved.** Keep the flag off. Do not claim completed
CSP-aligned export, deployment readiness or successful framing from Blender alone.

## Excluded experiments

An isolated experiment explicitly named the default FBX camera through a private
Blender exporter hook. It did not improve actual CSP framing, so that hook is not
included here. Transparent framing meshes and LWS/LWO companions also failed to
provide usable CSP framing and are excluded. This branch uses the public camera
and FBX export APIs only. It does not change hand/body collision correction or
analysis timeouts, enable the feature flag, or deploy any service.

## Branch validation (2026-10-20)

Matching branch in `Standin-server`, `Standin-app-server`, and `Standin-client`:
`feat/04j-289-multi-person-scene-save`. Deploy compatible Converter code before
considering the BFF flag; this branch does not turn it on.

- Converter Python regression: 373 passed, 1 skipped (Blender-runtime-only tests
  excluded from Python; the integration checks below use actual Blender).
- BFF TypeScript/build and full configured test list: 586 passed, 2 skipped.
- Client TypeScript, production web build, export/API regression: 44 passed.
- Blender 5.2.0: one/two/three rig composition, independent XYZ/yaw/scale and
  serialized skin/bone roundtrip passed; nested multi-rig sources rejected.
- T001 and p0718 automatic manifests: actual saved FBX reimport passed; world
  geometry matches the earlier CSP-tested artifacts within 0.1 mm. This is
  geometry preservation, not a new CSP framing pass.

No rough images, source FBXs, credentials, or local experiment binaries are
included in the feature commits.
