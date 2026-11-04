# Real-model interactive transport v2

This opt-in internal service connects the existing frozen solver/shape/hand adapters to the BFF. The canonical state/model/receipt schemas remain v1; the adapter is `post-frozen-file-canonical-snapshot-v2`. No frozen solver, manifest SHA, static GLB validator, final file safety check or correspondence tolerance is relaxed.

## Display/file correspondence

FBX imports quantize custom split normals. Re-exporting a saved Blender scene can produce a different split-normal surface despite unchanged joint and vertex data. The adapter therefore applies source/refine/shape/hands/crop, exports a private staged FBX once, imports it and captures the authoritative display packet. Its manifest binds the staged file SHA and .blend SHA. Export copies the staged bytes, independently reimports them in a fresh process and runs the unchanged full correspondence comparison. Only the successful verifier publishes a ready receipt. Model packets never certify a final file or full collision safety.

Identical corner normals are indexed without rounding; real split edges remain distinct. Canonical floats preserve their original IEEE754 bits. Packet scalar metadata is cached without exposing mutable decoded state. Initial/intermediate FBX dependencies remain; this change does not claim to eliminate synchronous FBX work or accelerate every uncached request.

## Internal HTTP

Disabled unless `CONVERTER_INTERACTIVE_ENABLED=true`, `CONVERTER_INTERACTIVE_TOKEN` contains at least 32 characters and `CONVERTER_INTERACTIVE_ROOT` names a private writable directory. Requests use `Authorization: Bearer <service token>`. Tokens never appear in URLs or public packets. Keep this service behind the existing internal network boundary.

| Endpoint | Meaning |
|---|---|
| `POST /interactive/v1/solve` | BFF-authored selection, operation/revision/person/expiry and base64 BVH; returns opaque snapshot ID and authoritative packet |
| `GET /interactive/v1/snapshots/{id}/model` | Exact bounded packet bytes |
| `POST /interactive/v1/snapshots/{id}/export` | Independent same-file export/reimport verification and receipt |
| `GET /interactive/v1/snapshots/{id}/file` | Verified staged FBX bytes only |
| `GET /interactive/v1/snapshots/{id}/variants` | Bounded exact, previously verified owner-bound hand combinations |
| `POST /interactive/v1/cancel` | Cancel matching owner/person/revision; a delayed cancellation cannot revoke a newer revision |
| `POST /interactive/v1/static/resolve` | Resolve up to 50 prebuilt source/character/scope assets and exact display camera; no Blender |
| `GET /interactive/v1/static/{sha256}` | SHA-checked GLB; no Blender |

Snapshot requests also carry `X-Standin-Owner`. Owner binding is opaque, not an installation ID. Only the BFF is trusted to submit selections; the service does not grant public release permission. Source bytes, approved registry character SHA, shape source/controls, hand library and frozen runtime are independently checked. Refined inputs must carry the actual persisted inference version/config/base hashes. Clients cannot provide filesystem paths, character meshes or executable code.

The stream limit is 3MiB (BVH at most 2MiB); artifacts are bounded at 64MiB. Existing process scheduling, request disconnect and deadline cancellation apply to child work. Private state is single-process/single-volume, locked exclusively, with the BFF operation deadline (at most one hour), 64 operation entries, bounded body/variant caches and 2GiB storage admission budget. Restart invalidates old private handles and removes crashed owned work. An ownership marker prevents cleanup of foreign nonempty directories; incomplete body-cache preparation is discarded before retry. BFF revokes durable readiness on a missing/expired snapshot and Client retry prepares a fresh revision. BFF remains the durable revision/ownership/quarantine/review/expiry authority and rechecks before public delivery. Do not enable multi-worker/shared-volume deployment without replacing this local store.

## Search camera and framing

`interactive_camera.camera_for_model` consumes trusted `candidate-camera-v2` continuous rotation and torso-normalized perspective distance. It preserves search XY/depth projection, including the +Z search versus -Z WebGL convention. Proper view rotation plus a negative horizontal projection preserves screen coordinates; interactive rendering must handle reversed winding. It never rounds yaw or fits changed hand/mesh bounds. Framing v2 is square, hip origin, target `[0,0,0]`; scope-specific fixed torso scale/center is already in the projection. This preserves normalized search geometry; it does not claim pixel-overlay alignment to the original input image. Fifty orthographic/perspective continuous-angle controls verify the mapping.

## Offline Top5 publication

`publish_interactive_static.py` consumes an immutable canonical state, matching independently verified receipt and approved character SHA. It keeps every skin influence and split normal and publishes the GLB before its source/character/scope manifest. Set `CONVERTER_INTERACTIVE_STATIC_ROOT` to that private store. A missing entry returns null and never schedules conversion. Actual local verification covers five distinct candidates, not the entire production library or every scope.

## Hands and acceptance

Recent verified hand combinations are cached separately from cancelled operations, owner-bound and TTL-limited. Atlas packets are bounded and contain exact canonical state hashes; personal FBX/CS3C and arbitrary preset combinations are not given a local capability. The client compiles whole surfaces before switching cached variants. First-use/unsupported combinations wait for authoritative solve; each new revision needs its own current final-file receipt.

Actual local checks: base / left fist / accepted v2.5.4 refine+fist; male and female all five scopes; female fat/muscle extremes; mirrored orthographic male. All independent joint/vertex/skin/normal comparisons are zero. Five Top5 GLBs verify their SHA. Client cached hand input to GPU completion p95 is 19.6ms (QA-only `gl.finish` wait) in 40 samples after warm-up (one local Chrome run, no display presentation/server/network). The actual Client HTTP/BFF Hono+PGlite/live Converter/file hash/cancellation path passes. Licensed assets, BVHs, FBXs, GLBs, .blend files, source controllers and credentials remain local and are not committed.

Native OS/CSP import, production PostgreSQL/distributed behavior and fleet performance are deployment gates. No merge, deployment or production activation is performed by this PR.
