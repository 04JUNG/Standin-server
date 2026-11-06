# Final FBX skin postprocess

A nondefault skin is embedded in the final individual FBX as a new material, per-corner neutral UV and PNG Video content. Applying outline afterward preserves the skin and existing transparent black guide. Default skin omits this postprocess and retains the legacy artifact path and bytes. Global enablement remains off.

## Dependency and support

This is a separate skin branch stacked on outline PR #170, pinned at `70b8a6f37d748d5a81801e0c1d632f7c512ba517`. Review only the skin diff against `codex/outline-fbx-csp-20261105`; after that dependency merges, retarget the skin PR to develop and recheck CI. Latest develop at worktree creation was `f260b1e6843996935ab287cdfad63465c18f7266`. No merge, deployment or global activation is included.

| Target | Support |
| --- | --- |
| Current male `male-default83-v0-7-6-p1` / female `female-default80-v0-3-9-p1` | Full individual static FBX; exact source hashes in manifest |
| Default | Existing no-op path; no skin request |
| Gray guide, flesh, flesh guide, muscle | Approved prebuilt neutral UV + 4096 PNG; 2048 preview derivative |
| Outline combination | Skin then outline; representative actual CSP width 2, skin-only logical width 0 |
| Partial output scopes | Refused: crop UV lineage is not established |
| Old defaults, personal models, VRM, integrated scene, animation | Unsupported by this skin contract |
| Pose / shape / hands / refine | Preserve accepted canonical body data; all upstream gates still apply. No universal combination certification |

CSP supports single-FBX embedded texture transport in the observed version. No manual PNG selection or sidecar is needed for the accepted samples. Details, representative comparisons, exact hashes and limitations: [actual CSP evidence](qa/skin-fbx-csp-20261106/README.md).

## Contract and ownership

`skin` is a separate strict JSON object, `{"version":"fbx-skin-v1","id":"flesh-guide"}`. IDs are `default`, `gray-guide`, `flesh`, `flesh-guide`, `muscle`. Existing outline JSON is unchanged. Canonical skin hashing is UTF-8 JSON, sorted keys, compact separators. Product default is omission; posting default directly returns `SKIN_DEFAULT_USE_LEGACY`.

- `GET /skin-contract`: no-store capability, runtime/manifest revision, exact supported model/scope/skin targets and approved asset hash descriptors. Availability requires both local enablement and correct on-disk assets; a public certificate boolean alone is insufficient for downstream acceptance.
- `GET /skin-assets/{sha256}`: private Bearer authentication, immutable approved bytes only, bounded relative manifest paths and SHA/byte verification. Raw gzip UV bytes are returned without Content-Encoding. Arbitrary paths, URLs and unapproved digests are not accepted.
- `POST /skin-fbx`: private Bearer multipart `fbx`, `source_artifact_sha256`, `skin`, `binding_sha256`, `character_id`, `character_sha256`, `output_scope`. Maximum binary FBX 30 MiB. Exact registry/model/source checks precede work.
- Success: `artifact_contract: textured-fbx-v1`, `fbx_base64`, `fbx_sha256`, strict `skin-fbx-receipt-v1` receipt. Receipt binds source/output/choice/binding/model/scope/runtime/manifest/UV/texture, approved mutation fingerprints, fresh body/import verification and exact CSP acceptance row. BFF separately binds final state, authorization, caches and download; the client separately checks selected state and proof.
- Combined flow: original canonical FBX → skin proof/output → outline source/output. Outline source SHA must equal skin output SHA. Never apply skin to an existing skin/outline guide or recolor its Alpha0 material.

Error boundaries: 400 invalid/default/bounded binary; 401 service auth; 409 changed source/registry binding; 422 unsupported target, incompatible UV/topology or wrong pipeline order; 503 flag/assets/certificate unavailable; 504 timeout; 500 independent integrity failure. Existing queue cancellation checkpoints terminate owned subprocesses and clean private temporary files before any result is returned.

## Preservation and asset preparation

`skin_fbx.py` uses the existing typed FBX parser/encoder, not a Blender body re-export. It replaces one body material connection, adds one UV layer/link and three material/texture/video objects, and updates definition counts. Every other original object/connection/mesh value must equal the expected source tree. The verifier rebuilds that expected mutation independently and rejects extra/missing/changed values, except standard exporter header metadata. A separate fresh Blender process imports source and output, validates 52 bones, geometry, normals, matrices, weights, exact per-corner UV, opaque material, image linkage and embedded PNG hash. Existing comparator tolerances are unchanged; observed four maximum body errors are all zero.

Current female topology is 24,999 vertices / 50,010 triangles, unlike the earlier preview model. Current male is 24,988 / 50,000. `scripts/body_skins_preview/prepare_model.py` prepares each exact current neutral once. `scripts/skin_fbx/generate_textures.py` reuses approved guide style and latest artist-authored anatomy V2 2D paths, reprojects them on the current neutral surface and bone/stature stations, then bakes ordinary texture pixels. No old UV, vertex IDs or barycentric anchors are reused. Viewer-only dynamic line corrections are not represented as final-FBX features.

One-time local order: prepare neutral blends/source records → generate current textures → `prepare_preview_textures.py` → `package_assets.py`. Packaging asserts unchanged protected neutral geometry/rig data and exact raw triangle/corner correspondence. Preview data carries triangle IDs/digest plus UVs; no nearest-position mapping is allowed. Asset root is a separately provisioned private directory with the manifest-relative files. Assets are not generated during saves and are not included in the Docker image or Git. The tracked manifest binds all approved bytes and authoring inputs. Any asset/manifest replacement revokes the current CSP acceptance; do not rewrite pins and inherit old photos.

Set `CONVERTER_SKIN_ENABLED=true` and `CONVERTER_SKIN_ASSET_ROOT` only in the intended process after provisioning exact approved assets. The default example remains false. The existing interactive private token is also required. No request chooses filesystem roots or arbitrary image URLs.

## Validation and limits

- Exact converter CI Python list: 547 passed, one existing skip. Core smoke: 54/54. Compile and diff checks passed.
- Synthetic Blender binary test: valid round trip plus 11 rejected mutations (body position/weight, UV, image bytes/path/missing image, material, connection, extra mesh, source topology and wrong order). Synthetic fixtures never obtain CSP certification.
- Actual fresh-process neutral 8 skin-only + 8 combined, and current crouch 8 skin-only + 8 combined: file verification passed with zero body errors.
- Actual loopback HTTP: 10 skin + 10 outline outputs, including two approved shape-controller cases (female 160cm / male 190cm, muscle 0.5), plus two accepted relaxed-hand sources through both postprocesses. Source/skin/outline hash chains and all 18 approved asset downloads checked. Neutral HTTP output bytes exactly match the 16 CSP-observed files.
- Actual cancellation during Blender: no file returned, owned temporary files cleaned and source unchanged. Wrong-order and a cropped source falsely declared full both return 422.
- Actual CSP: 16 neutral imports, two representative posed flesh-guide + outline 2 imports, and one male combined camera orbit. No `.clip`, CSP joint editing, all-angle or all-pose claim. CSP's strong specular highlight can obscure fine guide lines; viewer lighting is not pixel-identical.
- A fist-hand input was rejected by the existing hand self-collision gate before skin. That gate was retained. Relaxed accepted hands passed. Refine service generation and every upstream shape/hand/pose combination were not exhaustively exercised.

Measured warm local HTTP for the ten neutral/shape cases: skin 3.263–7.364 seconds (median 3.476), outline 2.906–6.594 seconds (median 3.391). These include loopback and fresh Blender postprocesses under concurrent local work, exclude base solve and production network/storage, and are not production latency guarantees. No texture rebake occurs on save.

`tests/converter/verify_skin_roundtrip.py` is CI-safe without private models. `scripts/verify_skin_http.py --help` describes the local real HTTP fixture runner; token files, registry fixtures, original models, UV/PNG data and raw screenshots remain ignored. Companion BFF/client PRs own full-wire/native product acceptance. Shared local test wiring uses controlled auth/SQL/source/store drivers and must not be described as a production deployment.
