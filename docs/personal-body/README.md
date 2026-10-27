# Personal character local converter — draft implementation

Base: `origin/develop` at `c8b04c78af7db779564f8bb4e69d39447c8295bf` (fetched 2026-10-28 KST).
Implements the local-compute direction in `PERSONAL_BODY_LOCAL_REGISTRATION_STRATEGY_2026-10-28.md`, read from the existing checkout without copying any unrelated changes. This is a development foundation and **does not complete the FBX + CS3C product or its release gates**.

## Current behavior

A separate stdlib IPC host launches one Blender child per heavy operation. It never uploads assets, opens an HTTP listener, fetches poses, or falls back to the server. The app owns registration cards/names; the worker owns immutable original copies, normalized characters and verified conversion caches.

- Binary FBX adapters retain terminal/helper bones, mesh/object transforms, original weights, rest transforms and solid Principled material values. Multiple meshes bound to one armature are accepted.
- The common character representation is a private, hash-checked `.blend` plus a JSON inspection manifest. It records hierarchy, rest/inverse bind matrices, mapping, mesh transforms/counts/weight hashes, material descriptors and units. `.blend` retains the actual vertices, normals, weights and Blender's imported bind state. GLB is an optional registration preview; it is not the source of truth.
- Known canonical/Mixamo/CMU/100STYLE mappings must pass complete required joints, ancestry, nondegenerate joints, upright T-pose geometry, finite transforms and normalized original skin checks. Unknown rigs, A-pose, multiple armatures, reflections/nonuniform object scaling, shape keys, constraints and unsupported modifiers are rejected explicitly. A supported profile name alone never approves a model.
- Texture/Video records are rejected **before** Blender resolves image paths or extracts embedded images. External textures and procedural/layered materials are not certified. There is no silent material substitution.
- Body pose application reuses the unchanged current `converter.retarget` solver, `import_bvh`, `apply_output_mode('rigged_rest')` and `export_fbx`. Personal import/inspection is separate because the server importer intentionally drops leaf bones and applies object transforms.
- Both hands are currently unchanged. Each side reports `NO_FINGER_RIG` or `UNVALIDATED_FINGER_RIG` and its own detected finger count. A left-only finger fixture is tested. Finger presence does not imply validated hand-library support. Non-null hand requests fail `HAND_UNSUPPORTED`.
- Every registration and final FBX is exported and reimported. Gates compare mesh identities/counts/topology, original bone hierarchy and world positions, solid material values, bidirectional world-space vertex error, and deformation under five FK probes (both forearms, both shins, spine). This is stronger than count-only validation; it still does not prove CSP appearance or arbitrary skinning equivalence.
- Physical units are recorded as unconfirmed. The worker preserves imported scale; a user height-calibration flow is not implemented here. Output is full-body `rigged_rest` in the source orientation, with root translation disabled. Other output modes/orientations are not silently approximated.
- No automatic rigging, personal shape sliders, collision prevention, or guaranteed preservation of CSP expressions/clothing physics is offered.

## IPC and trust boundary

[converter-client-contract.v1.json](converter-client-contract.v1.json) is the strict versioned contract. UTF-8 NDJSON uses `protocolVersion: "standin.local-converter/1"`, unique `requestId`, and `action`. Stdout contains events only. Actions: `hello`, `import`, `inspect`, `apply`, `export`, `cancel`, `delete`. Only one heavy operation runs at a time; another receives `BUSY`. `cancel.targetRequestId` identifies the operation and the target receives `CANCELLED` if cancellation wins before atomic commit. Late cancellation reports `accepted:false`. EOF cancels and waits for child cleanup.

`import` requires a new `localModelId`, an absolute `sourcePath`, and the expected SHA256 `sourceHash`. Replacement uses a new ID and does not replace a successful old registration on failure. The client handles duplicate source hashes and atomically changes its card after import succeeds. Renaming is a client-card operation.

`apply` requires the registered source hash, absolute `posePath`, expected final BVH `poseHash`, `selectionId`, and `hands:{left:null,right:null}`. **The client must obtain final BVH bytes through existing authenticated BFF selection/download validation.** `selectionId` is correlation, not a token verified by this worker. Arbitrary local files are not evidence of BFF authorization. No server endpoint or security gate is changed by this PR. The existing client must confirm its authenticated path supplies the final BVH; a dedicated BFF endpoint, if needed, must retain candidate ownership/validity checks and must not accept a local model ID as a server character ID.

The immutable snapshot includes local ID, original SHA, importer/normalization/mapping/converter versions, a content fingerprint of all shipped converter Python/JSON, final pose SHA, left/right hand state, output mode and orientation. A changed runtime requires reimport rather than reusing stale normalized data. `export` requires the exact `snapshotHash`; it never uses a mutable “latest output.” Previously saved files remain unchanged after later requests or model deletion. Existing destination files are never overwritten. Cached BVH lookup still verifies the selected file hash.

The client native implementation was read without editing its checkout. Its actual resource lookup is `resource_dir/local-converter/`; the contract matches this location. Supply the chosen target-specific builder output at that location. A debug-only `STANDIN_LOCAL_CONVERTER` override points to the packaged launcher. Launch it with only `--storage <absolute app-owned directory>`. Keep stdin open until work finishes.

The current client integration requests `GET /v1/pose-candidates/{id}/export?jobId=<job>&personIndex=<index>&candidateId=<id>&format=bvh`. BFF code for that external service is not in this worktree, so its live behavior is not certified here. Before enabling the feature, the BFF owner must verify installation/job ownership, candidate-to-person membership, path/query candidate equality and the existing selected-candidate export grants; return the final persisted refined BVH when refined, otherwise the validated base BVH. Preview-only BVH, inference `/pose` URLs, expired/deleted candidates and unapproved generation must not substitute for this response. Local IDs, model originals and hand/body catalog parameters are not sent. The client hashes the returned bytes and passes that SHA to this worker. If this exact final-BVH response is unavailable, extend the existing authenticated selected-candidate endpoint with `format=bvh` under the same checks; do not add an unvalidated download route. Such BFF work and live qualification remain outside this PR.

## Isolation, limits and storage

Defaults are provisional regression bounds, not performance promises: 128 MiB input, 8 MiB BVH, 64 MiB declared/cumulative expanded FBX arrays or CS3C catalog, 1 million vertices, 512 bones, 128 meshes, 120 seconds per Blender child and a 2 GiB resident-memory stop threshold. Bounded zlib decoding verifies actual output against declarations before creating FBX arrays. Tests include both an oversized declaration and a compressed stream that expands beyond its declaration.

Child RSS is sampled every 100 ms using Darwin `libproc`, Windows `K32GetProcessMemoryInfo` or POSIX `ps`; exceeding the threshold kills the child. This is a monitored stop threshold, **not a hard OS memory sandbox**: a transient allocation may overshoot between samples. Windows process-tree cleanup uses `taskkill /T /F`; POSIX uses a dedicated process group. Native Windows execution and crash/limit behavior remain unverified. Parser isolation is not an OS filesystem/network sandbox. Auto-execution is disabled, only application-authored job files run, original FBX image references are rejected, and imported `.blend` files are app-generated and hash checked.

Original and BVH snapshots are copied with incremental hashes and cancellation checks. Writes go into a private staging directory and become visible through an atomic rename only after validation. A per-storage OS lock prevents two hosts from cleaning each other's staging. Restart removes only stale staging. Cancellation/failure preserves previous cards and user exports. Worker errors do not persist Blender traces, model names or source paths in logs/telemetry. Manifest/IPC paths are intentionally device-local and must not enter server analytics.

## Runtime packaging

The builder copies a complete vetted Blender 5.2.0 distribution, its Python 3.13, native libraries and existing license files, plus converter source/policies. End users do not install Python, Blender or Modeler. The launcher uses the **bundled** Python with isolated imports and bytecode writes disabled. The entry verifies every manifest file before running the host. It fixes the Blender executable; clients cannot pass a different runtime via IPC.

```sh
python scripts/local_converter/package_runtime.py \
  --distribution /path/to/Blender.app \
  --target darwin-arm64 --output /path/to/new/local-converter

# Windows builder input is the unpacked official portable distribution.
python scripts/local_converter/package_runtime.py \
  --distribution C:/build/blender-5.2.0-windows-x64 \
  --target windows-x64 --output C:/build/local-converter
```

Supported package layouts: `darwin-arm64`, `darwin-x64`, `windows-x64`. Only **macOS arm64** was assembled/executed here. Windows and macOS x64 are layout implementations, not tested packages. `runtime-manifest.json` identifies `target`, `launcher`, `blender`, `python`, per-file SHA256 and total size. Current candidates deliberately set `releaseReady:false`.

The measured macOS resource is about 935 MB / 892 MiB uncompressed. Full-byte manifest verification made the measured first IPC startup about 9 seconds on this machine; warm timings are in the report. This cost must be addressed before claiming responsive production behavior. No downloader or runtime installation UX is shipped.

Release still requires a pristine approved runtime input, macOS signing/notarization, Windows signing/native QA, and delivery of corresponding Blender source and all dependency notices. The existing local Blender input fails `codesign --verify --deep --strict`; copying it does not make a signed release. The builder does not sign, notarize, publish, or claim licensing clearance. Blender's [license](https://www.blender.org/about/license/), [redistribution FAQ](https://www.blender.org/support/faq/) and [source license inventory](https://github.com/blender/blender/blob/main/release/license/license.md) are the upstream review sources. The background/auto-execution switches follow the [official command-line manual](https://docs.blender.org/manual/en/5.0/advanced/command_line/arguments.html).

## CS3C status and fixture policy

The adapter recognizes a bounded, read-only SQLite character catalog and the observed `CLIP_STUDIO_3D_DATA2` catalog version 2. Both supplied samples are recognized, but production import returns `CS3C_DECODER_UNAVAILABLE`; neither becomes a successful card. Unknown containers/versions return `UNSUPPORTED_CS3C_VERSION`. This is **format detection and an explicit missing-adapter boundary**, not CS3C conversion support.

The supplied experiment and independent revalidation demonstrate original hierarchy/weights, test FK, GLB→FBX→reimport for two 11-mesh/8,517-vertex/21-node bone-shaped samples. Those facts are experimental evidence, not proof of our search retarget or general character appearance. The experimental decoder reads parameters from a hash-pinned Modeler EXE (`2ebe2d90f8609496cb2e81a7c9defae4e851479b8e5db76eb9dd8ec05d943150`) and its GLB uses diagnostic colors. This PR does not distribute that EXE, extracted key/table parameters, third-party sample originals or generated sample binaries. It does not ask end users to supply Modeler. A reviewed redistributable decoder, schema/version coverage and original appearance preservation are still needed to complete CS3C.

The real regression FBX/BVH files remain local. Portable evidence identifies them by SHA256. `make_fixture.py` creates authored geometry instead of checking third-party assets into Git. It covers two meshes, a transformed skinned accessory, a helper joint and left-only fingers. It does not count as an independent artist model.

## Reproduce checks

```sh
python -m unittest discover -s tests/converter -p test_local_converter.py -v
python tests/test_smoke.py
python -m compileall src api scripts converter

/path/to/blender --background --factory-startup --disable-autoexec --threads 2 \
  --python-exit-code 1 --python tests/converter/verify_local_blender.py

python scripts/local_converter/verify_runtime.py \
  --launcher /path/to/local-converter/standin-local-converter \
  --source /local/permitted-character.fbx \
  --pose /local/final-pose-1.bvh --pose /local/final-pose-2.bvh \
  --cs3c /local/sample.cs3c --report /local/report.json
```

See [verification-macos-arm64.json](verification-macos-arm64.json) for actual public IPC evidence. Validation covers import, optional preview, restart reuse, CMU/Mixamo library BVH apply/export, cache reuse, changed-pose rejection, unsupported hands, stale snapshots, destination preservation, active cancellation, EOF cleanup, deletion and export preservation. Library BVH use is converter QA; it is not authenticated live BFF end-to-end QA.

Remaining product gates: redistributable CS3C conversion, general human CS3C and at least three independent permitted models per format, textured appearance, validated personal finger mapping/deformation, physical scale confirmation, broader rest poses/transforms, signed native OS packages, real client registration/preview/reuse, authenticated BFF final-pose integration and supported CSP import. No deployment, merge, release tag or production configuration was performed.
