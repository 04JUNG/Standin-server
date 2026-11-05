# Final FBX outline preparation and CSP acceptance

Current status: **Draft infrastructure; no CSP-supported nonzero export.** The tested inverted hull is rejected. The default remains off and width zero uses the existing artifact and path. No deployment or global activation is part of this change.

## Observed representation result

CSP 5.1.1 automatic import displayed the original controls correctly but displayed solid black surfaces for all three hull culling variants, on both the small control and the pinned male/female bodies. The client/CSP session performed these imports in a new test canvas without enabling native outlines or changing culling. This is an appearance failure despite valid FBX geometry and skin. `.clip` save/reopen is independently blocked by that installation's Unregistered license state. Rotation and CSP pose-edit acceptance remain unpassed.

| Representation | Actual transport / import result | Decision |
|---|---|---|
| Existing body, width 0 | Original bytes/path; normal white/grey body in CSP | Preserve |
| Reversed skinned hull, CullingOff | Black body in actual CSP import | Reject |
| Same hull, CullingOnCCW | Black body in actual CSP import, including real male/female | Reject |
| Same hull, CullingOnCW | Black body in actual small-control CSP import | Reject |
| Backfacing → transparent/black Blender graph | Geometry/Mix/Transparent shader nodes absent after actual FBX reimport | Reject as portable FBX graph |
| CSP native Add outline / width | Application state; no verified FBX width transport | Not an automatic FBX solution |

The installed official Blender 5.2 exporter writes Model `CullingOff` unconditionally. The candidate writer therefore used explicit FBX node culling values as separate controls. Blender reimport also loses the material backface flag; the verifier reports this fact and does not repair it or equate Blender success with CSP success.

Official references checked on 2026-11-05: [CSP FBX import and Basic/Realistic modes](https://help.clip-studio.com/en-us/manual_en/660_3d/Importing_3D_Files.htm), [CSP application outline/backface settings](https://help.clip-studio.com/en-us/manual_en/810_subtools/P.htm), [Autodesk FBX culling tokens](https://help.autodesk.com/cloudhelp/2020/ENU/FBX-API-Reference/cpp_ref/fbxfiletokens_8h.html), [Blender supported material-node export](https://docs.blender.org/manual/en/4.1/addons/import_export/node_shaders_info.html). These describe supported settings/data; they do not certify preservation in CSP.

## Boundary and integrity

- `outline_spec.py` strictly validates `{version:"fbx-outline-v1",width:0..2}`; bools, nonfinite numbers, unknown versions/keys and out-of-range widths fail. Black and full opacity are fixed. Missing/zero appearance stays out of legacy identity and output handling.
- `outline_fbx.py` parses the original FBX and adds a named hull, black material, copied skin/cluster records and separate bind pose. Original object values, geometry, normals, materials, UVs, morph records, skin and bones are not exported again through Blender. Unrecognized skin/normal/instance structures fail. Animated source files are outside the rigged-rest contract.
- The independent verifier rebuilds an exact allowlist from the original source/settings and compares every scene-affecting typed FBX record. Arbitrary extra meshes, missing hulls, material/skin changes and stale width/culling fail. It then imports the file, checks the complete skin, removes only the exact verified hull set and runs the existing strict body comparator unchanged.
- The immutable solved-state/display model, frozen solver, retarget implementation, bone/weight tolerances and preview outline renderer are unchanged. This is a postprocess of a previously verified individual FBX. It grants no BFF ownership/release authority.
- Personal models, combined-scene FBX and VRM ZIP are outside nonzero support. No source model/controller/key/private screenshot is committed.

## Physical mapping remains experimental

`stature-relative-v1-experimental`: displacement in world metres = width / 2 × full stature in metres × 0.0015. At 1.75m, widths .5/1/2 correspond to .65625/1.3125/2.625mm. The same full stature is used for cropped outputs. Width is the logical UI choice, never a direct CSS-px-to-mm conversion. Pose or crop bounds cannot supply stature.

The pin-specific baseline metadata is 1.75m for both character SHAs below; shaped requests must use the resolved approved height control. Additional real 160/190cm inputs use the original approved controllers, their unchanged field ranges and fixed-source hashes. Because the representation fails CSP rendering, this mapping has no accepted visual calibration and must not be advertised as constant pixels or supported thickness. A replacement expression requires a new version and fresh visual acceptance.

## Internal API

`GET /outline-contract` reports version, expression, mapping, runtime revision, enabled state, CSP certification and supported targets. Current supported characters/scopes/expressions are empty; candidate character metadata is not support. `CONVERTER_OUTLINE_ENABLED=false` by default. Setting it true cannot bypass the empty compile-time exact `(expression,mapping,character SHA,scope)` certification set.

`POST /outline-fbx` uses the existing private interactive Bearer token and multipart fields: `fbx`, `source_artifact_sha256`, `appearance` JSON, `output_scope`, `character_id`, `character_sha256`, `binding_sha256`, `stature_m`. Files are binary FBX bounded to 30 MiB. Width zero returns `OUTLINE_ZERO_USE_LEGACY` instead of minting a new file. Current nonzero requests fail with `OUTLINE_DISABLED`, `OUTLINE_UNSUPPORTED`, or `OUTLINE_NOT_CSP_CERTIFIED`; no file-ready marker or fallback success is issued.

When a future expression is accepted, the JSON envelope is `{artifact_contract:"outlined-fbx-v1",fbx_base64,fbx_sha256,receipt}`. The receipt binds source/final hashes, normalized appearance/hash, BFF selection binding, character SHA, scope, stature, physical displacement, expression/mapping/runtime, exact outline manifest, `outline.displacement_verified`/`max_displacement_error_m`, base correspondence and separate file/CSP booleans. Only the verifier writes the candidate receipt. Public delivery additionally requires a true exact-target CSP certification. Candidate CLI receipts are false.

BFF must check the capability, maintain appearance in its operation/revision/cache identity, independently hash returned bytes and recheck all bindings/authorization/expiry/current-selection fences at build and delivery. Slider motion does not start Blender per frame. BFF owns verified-byte reuse and latest-operation rebinding; the converter has no persistent artifact cache. Cancellation/deadline/worker/integrity failures clean the private temporary directory and retain the previous successful download.

Errors: 400 `INVALID_OUTLINE` / `OUTLINE_ZERO_USE_LEGACY`; 401 `UNAUTHORIZED`; 409 `SOURCE_FBX_SHA256_MISMATCH` / `OUTLINE_BINDING_MISMATCH`; 422 `OUTLINE_UNSUPPORTED`; 503 `OUTLINE_DISABLED` / `OUTLINE_NOT_CSP_CERTIFIED`; 500 `OUTLINE_INTEGRITY_ERROR`; 504 `OUTLINE_TIMEOUT`. BFF maps these to brief retry/unsupported user text.

## Verification evidence

- Python 3.12 and pinned `requirements-converter.txt`: 117 focused appearance/API/final/interactive tests passed. Repository smoke: 54/54. Python compilation passed. Converter CI runs the new contract tests and the real synthetic FBX tamper verifier.
- 90/90 baseline files: 2 pinned bodies × neutral/bend/sit90 × full/half/bust/shoulder_head/head × .5/1/2. 30 width-zero records preserve the existing source/path/hash. All original body position, normal, matrix and weight comparator maxima are exactly zero.
- 42 maximum-width deformation checks move real elbow/finger/knee/spine/neck joints. Maximum measured outline skin-transport error: 3.67779365048e-07m (0.000368mm). This proves Blender skin following, not CSP editing.
- Additional 36/36 real 160/190cm files passed (both approved controllers × full/half/head × .5/1/2), plus 12 zero path records. Across all 126 files, world-metre displacement error after independent import is at most 1.74580918136e-07m (0.000175mm).
- Nine real binary corruption/stale cases rejected: base position/material/weight; hull displacement; extra mesh; missing hull; connection; different width; different culling.
- Isolated local host preparation, including two fresh Blender processes: 2.949s first / 2.913s repeated. Repeat output SHA matches. These exclude network, solver and CSP. In-process matrix medians are .096s export / .392s verify and exclude process startup; they are not endpoint latency. Cancellation before work propagates without publication.
- BFF session separately reports live TCP → Hono/PGlite → converter/Blender zero-byte reuse and cancellation tests, plus .5/1/2 fail-closed before superseding the previous zero operation. See companion PR evidence; synthetic positive outline fixtures are not product acceptance.

Local evidence stays Git-ignored under `out/outline-fbx-csp/`: `summary.json`, `matrix/verification-results.json`, `heights/verification-results.json`, `host-qa.json`, `tamper/report.json`, `images/csp-approved-comparison.png`, and original screenshot path/hash index `images/screenshots.json`. Shared `artifacts/outline-fbx-csp-20261105/{CONTRACT.md,converter-status.md,csp-probes.json}` in the primary checkout records handoff paths. The comparison photo is the actual CSP zero vs rejected hull output, without changing the model pixels.

Pinned source and candidate hashes:

| Body | Original source SHA-256 | OnCCW candidate SHA-256 |
|---|---|---|
| male | `809a131510434089888429605bf7cd2e3bc75b63109e4ac5c9734b211ee52365` | `d15773b775e5ed4637969ed4a1aa4fc86c19c61a5c2a7c7907ef2d538d255c1d` |
| female | `e6947d8e2aeb1014e9505a61d2f42e6bd9d34965cc7bb3a6c2fcd2722d4f280a` | `29b7e2cfc0c1f6d94ca7e5fc84d449039cfa8544ce28d6246318188c0a48b2da` |

## Remaining acceptance

No accepted replacement FBX expression exists in this change. Do not merge/enable it as a completed CSP outline feature. A viable representation must first pass automatic import without black-body/internal-face/finger-blockage defects at all requested widths/views/scopes, then real CSP pose editing and `.clip` save/close/reopen on a legitimately registered test installation. The license blocker and the rejected representation are separate issues. Blender checks cannot replace either acceptance step.

Related: [client #162](https://github.com/04JUNG/Standin-client/pull/162), already merged [base server #167](https://github.com/04JUNG/Standin-server/pull/167), and [BFF companion #161](https://github.com/04JUNG/Standin-app-server/pull/161). No staging branch or personal-model #142 code was imported.
