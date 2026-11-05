# Final FBX outline preparation and CSP acceptance

Current status: **Replacement final FBXs display grey body surfaces and black contours in actual CSP imports. Full product acceptance remains incomplete.** The opaque inverted hull is replaced by an outward-facing, fully transparent skinned guide. The exact implementation bytes for both current approved defaults were imported by the exclusive client/CSP owner without manually enabling native outlines or changing culling. Width zero retains the existing file/path. The feature remains default-off and uncertified pending CSP joint editing and licensed `.clip` save/close/reopen. No merge, deployment or global activation is included.

## Actual CSP observations

CSP EX 5.1.1 build 202607291835 was tested in separate new canvases using File > Import > 3D Data, automatic import, with no mode-selection prompt. The exact Basic/Realistic property was not read. These observations do not certify every mode or renderer.

| Representation | Actual CSP result | Decision |
|---|---|---|
| Original body, width 0 | Grey surfaces with CSP's existing native contours | Preserve original bytes/path |
| Opaque reversed skinned hull, Off/OnCCW/OnCW | Solid black surfaces | Reject v1 |
| Transparent reversed hull | Grey body without useful extra contour | Reject |
| Transparent outward-facing skinned guide | Grey body plus darker contours on actual male/female final files | Implement v2; retain certification gate |
| Fixed-view black contour tubes | No black body initially; internal lines after orbit | Diagnostic only |
| Blender Backfacing/Mix/Transparent graph | Required nodes absent after FBX reimport | Reject as portable FBX expression |

The new guide keeps outward winding/normals and `CullingOff`. Its black material has `Opacity=0`, `TransparencyFactor=1`, `TransparentColor=(1,1,1)`. CSP's automatic outline pass visibly outlines the expanded transparent guide while retaining the original body. This is observed CSP behavior, not an assumption about shader graphs or backface culling surviving transport. Other renderers need not draw the transparent guide.

Actual CSP evidence:

- Current male/female full-body comparisons at 0/.5/1/2, male crouch, preserved male bend/female sit: grey surfaces and progressively stronger contours. This is visual comparison, not measured pixel width.
- Exact final implementation male/female width-2 imports and 200% front detail: no large black patch, obvious white halo or doubled exterior silhouette in these views. Fine dotted interior occlusion lines remain near some knee/shin views.
- Earlier male candidate front/three-quarter/side/back orbit and exact final female near-side/rear-three-quarter orbit. These are distinct hashes/evidence sets. The female screenshot named `back` is not an exact-back observation. Extra exact final male side/back testing stopped when the Mac locked; candidate orbit does not certify final-byte orbit.
- Both sexes at half/bust/shoulder_head/head: eight paired 0/2 comparisons, 16 exact hashes checked; grey bodies and stronger contours in every initial view. Both head width-2 files were orbited near-side/profile with a clean visible lower cut edge. Head zero already has native contour; width two is modestly stronger. Cropped .5/1 are file-tested, not CSP UI-tested. Male bust/shoulder cut jaggedness already exists in zero controls.

Save As on the exact final female file produced the actual license acquisition/registration dialog and was cancelled. The installation is Unregistered; no `.clip` was saved/reopened. CSP joint editing remains unverified: available object-edit controls did not expose demonstrated bone manipulation. Arbitrary views and every occluded cut face remain uncertified. These limits are distinct from successful initial import.

Official references checked during the investigation: [CSP FBX import and modes](https://help.clip-studio.com/en-us/manual_en/660_3d/Importing_3D_Files.htm), [CSP outline/backface settings](https://help.clip-studio.com/en-us/manual_en/810_subtools/P.htm), [Autodesk culling tokens](https://help.autodesk.com/cloudhelp/2020/ENU/FBX-API-Reference/cpp_ref/fbxfiletokens_8h.html), [Blender material-node export](https://docs.blender.org/manual/en/4.1/addons/import_export/node_shaders_info.html). These describe settings/data, not acceptance of this expression; screenshots and hashes establish the observations above.

## Integrity and versioned mapping

`outline_spec.py` strictly validates `{version:"fbx-outline-v1",width:0..2}`. Bools, nonfinite values, unknown versions/keys and invalid ranges fail. The logical schema stays stable while v2 changes expression/mapping/runtime identity. Missing/zero appearance stays outside legacy identity/output handling.

`outline_fbx.py` preserves original typed FBX object values and adds named guide geometry/material, copied skin/cluster records and a separate bind pose. Original geometry/normals/UVs/morphs/materials/skin/bones are not re-exported through Blender. Unsupported source structures and animated sources fail. The independent verifier reconstructs an exact allowlist and compares every scene-affecting typed record. It checks actual imported Principled Base Color/Alpha sockets, not viewport diffuse alpha, then removes only the exact validated guide set before the unchanged strict body comparator runs. Extra/missing meshes, material/skin changes, stale width/culling and opaque-guide corruption fail.

Frozen solver, retarget, immutable solved state/display model, bone/weight tolerances and preview outline renderer remain unchanged. Personal models, combined-scene FBX and VRM ZIP remain outside nonzero support. Models/controllers/keys/binary screenshots stay Git-excluded.

Expression: `skinned-outward-transparent-native-outline-v2`.
Mapping: `csp-native-scope-stature-v2-experimental`.

World-metre displacement = width / 2 × supplied full stature in metres × 0.0015 × scope factor.

| Scope | Factor |
|---|---:|
| full | 1.00 |
| half | 0.60 |
| bust | 0.35 |
| shoulder_head | 0.27 |
| head | 0.15 |

The factors reduce excessive expansion in close framing. Full stature is explicit, never inferred from cropped/posed bounds. At 1.75m, full .5/1/2 gives .65625/1.3125/2.625mm; head width two gives .39375mm. Current-default jobs use resolved stature. Height variants use approved hash-pinned sources/controllers within original ranges. This is experimental style mapping, not CSS-pixel conversion or constant pixels across cameras.

Legacy diagnostics retain `skinned-inverted-hull-fbx-culling-v1` / `stature-relative-v1-experimental` and original unscaled opaque behavior. API jobs explicitly select v2. Cache identity binds source, normalized appearance, stature, scope, character, expression, mapping and runtime; old representation bytes cannot masquerade as v2.

## Internal API and release gate

`GET /outline-contract` reports appearance version, expression, mapping, runtime revision, enabled state and supported targets. `CONVERTER_OUTLINE_ENABLED=false` remains the default. `CSP_CERTIFIED` remains empty: an environment flag cannot bypass exact `(expression,mapping,character SHA,scope)` acceptance. Candidate character metadata is not support; supported lists are empty.

`POST /outline-fbx` uses the existing private interactive Bearer token and multipart `fbx`, `source_artifact_sha256`, `appearance`, `output_scope`, `character_id`, `character_sha256`, `binding_sha256`, `stature_m`. Binary FBX is bounded to 30 MiB. Zero returns `OUTLINE_ZERO_USE_LEGACY` for existing-byte reuse. Current public nonzero requests fail closed without fallback success/file-ready markers.

The prospective positive envelope is `{artifact_contract:"outlined-fbx-v1",fbx_base64,fbx_sha256,receipt}`. Verifier receipts bind source/final hashes, appearance/hash, selection binding, character SHA, scope, stature, displacement, expression/mapping/runtime, exact manifest, skin/displacement checks, unchanged-body correspondence and separate file/CSP booleans. Internal QA receipts are `file_verified:true,csp_certified:false`; direct preparation measurements do not bypass the public gate.

BFF validates capability, operation/revision/cache identity, returned byte hashes and ownership/authorization/expiry/current-selection fences at build/delivery. BFF owns byte reuse/latest-operation rebinding. Slider motion does not start Blender per frame; converter has no persistent artifact cache. Failure/cancellation cleans temporary files and preserves the previous successful download.

Errors: 400 `INVALID_OUTLINE` / `OUTLINE_ZERO_USE_LEGACY`; 401 `UNAUTHORIZED`; 409 `SOURCE_FBX_SHA256_MISMATCH` / `OUTLINE_BINDING_MISMATCH`; 422 `OUTLINE_UNSUPPORTED`; 503 `OUTLINE_DISABLED` / `OUTLINE_NOT_CSP_CERTIFIED`; 500 `OUTLINE_INTEGRITY_ERROR`; 504 `OUTLINE_TIMEOUT`.

## Verification evidence

- Python 3.12/pinned converter requirements: exact CI Python list **503 passed / 1 existing skip**; repository smoke **54/54**; compileall, frozen lineage and whitespace pass. Docker's explicit copy list includes every outline module, tested via isolated capability/runtime import. Linux CI also runs the synthetic real-binary verifier.
- Final scoped v2 matrix **240/240 passes**: 16 sources × five scopes × .5/1/2. Distinct source sets: preserved approved baselines neutral/bend/sit (6), approved-controller 160/190cm variants (4), current approved defaults neutral/crouch/foot-tap (6).
- **80 zero** path/hash records and **80 actual Blender pose-edit checks** moving elbow/finger/knee/spine/neck. Original position/normal/matrix/weight comparator maxima **all zero**. Maximum imported displacement error **1.6799306554291134e-7m**; maximum dynamic skin transport error **4.643455606442997e-7m**. Blender editing does not substitute for CSP editing.
- **10 binary corruption/stale cases rejected**, including native opacity. Native synthetic import checks shader Alpha=0 and exact original object fingerprints.
- Internal host preparation with two fresh Blender processes: **3.057s first / 2.575s repeated**, identical repeated SHA. Warm local samples exclude solver/network/CSP and are not production latency promises.
- Earlier v1 file results are historical; CSP rejected v1. BFF/client zero/stale/cancellation/disabled contracts and synthetic positives do not establish product acceptance.

Exact current-default implementation files imported into CSP, width 2:

| Body | Original source SHA-256 | Final FBX SHA-256 | Bytes |
|---|---|---|---:|
| male-default83-v0-7-6-p1 | `085dd9c65a3a66035f7e0ce9e0a9171b9abb0e33dada92194676d92e00c8b026` | `02799598d9f60899e83627446c80bf9c6f7f737a47b8b2de7e7f92612200a4e3` | 3,881,916 |
| female-default80-v0-3-9-p1 | `712ac75e58efd4d6ccded0b1e931590900015de023327f1320cf6d7d94026ffc` | `07295db35bb6a148f14da4fad91d459ee05d1f5859225b29638b71dce9754e3b` | 2,975,260 |

Git-ignored local evidence: `out/outline-fbx-csp-v2/native-scope-qa/{inputs,export-results,verification-results}.json`, `host-qa.json`, `final-tamper/report.json`, `summary.json`, `images/csp-final-comparison.png`, `images/photo-index.json`. Exact final files: `native-scope-qa/cases/{male,female}-current-neutral/full/2/artifact.fbx`. The comparison crops actual CSP canvas screenshots without retouching model pixels. Shared original photos/hashes and reports: primary checkout `artifacts/outline-fbx-csp-20261105/csp-v2/{results.json,report.html,scope-results.json,scope-report.html}`.

## Remaining acceptance

Initial automatic final-FBX import now has positive visual evidence. Full acceptance needs demonstrated CSP joint manipulation, licensed `.clip` save/close/reopen and remaining widths/views/scope inspection. Fine interior occlusion lines and untested arbitrary angles remain explicit limits. Keep PR Draft and server/client certification lists empty until these pass. Purchase/settings reset/source-document changes/merge/deployment are outside scope.

Related: [client #162](https://github.com/04JUNG/Standin-client/pull/162), [BFF #161](https://github.com/04JUNG/Standin-app-server/pull/161), merged [base server #167](https://github.com/04JUNG/Standin-server/pull/167). No staging branch or personal-model #142 code imported.
