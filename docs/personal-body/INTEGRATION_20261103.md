> Superseded for working storage and runtime lifecycle by [2026-11-04](FBX_RUNTIME_20261104.md). This file retains the dated revision-3 evidence.

# Personal camera integration — 2026-11-03

This is a disabled-base merge candidate, not approval to enable or release personal models. Original #142 ancestry is preserved; develop `592ab0f059089e64a5c88a3ad607539124e8e6c6` is integrated. Client companion: #159.

## Contract and behavior

The NDJSON envelope stays `standin.local-converter/1`; the strict contract revision is **3**. `hello.cameraContractVersion` is `personal-camera/1`. `apply.camera` is mandatory: `{version, sourceBvhSha256, rotation}`. Rotation is null for source orientation or a finite proper 3×3 rotation, with original BVH lineage. Missing, reflected, scaled or unknown camera input is rejected. The client verifies the selected BFF camera/source; the local worker verifies the explicit matrix and records it with the final BVH SHA.

Normalization and converter versions advance to `/3`; importer and rig stay `/2`. Camera participates in snapshot/cache identity. Retarget and `rigged_rest` run first, then one rigid Y-up rotation about target Hips, converted into Blender basis. Original skin weights and scales are retained. Each posed surface is checked against the requested rigid transform before export/reimport. No server retarget algorithm or download authorization route is changed.

The existing authenticated, installation-owned BFF final-BVH export remains the client boundary. Personal names, models and source paths are local. Local fixture runs are not evidence of authenticated live BFF end-to-end success.

## Actual evidence

- [Packaged IPC report](verification-camera-macos-arm64-20261103.json): two approved original CS3C files and one permitted FBX, each with source/yaw/roll variants. Nine real FBX exports, separate snapshots, cache/restart reuse, invalid/missing camera and stale export rejection, and export preservation after deletion.
- [Independent geometry report](verification-camera-geometry-20261103.json): reimport all nine exports, compare world-space vertices/bones to a separate rigid-transform calculation, verify exact original skin weights/object scales. Maximum vertex error `7.52e-7`, bone error `7.47e-7` (threshold `2e-5`). This proves one rotation in this fixture scope, not CSP acceptance or physical unit calibration.
- [Runtime and test inventory](integration-evidence-20261103.json): decoder-included and decoder-free packages have different pinned manifests. Both are macOS arm64 and `releaseReady:false`. No original models, decoder payload, executable or runtime is committed.
- Latest local converter tests: 11; decoder tests: 6. Native client tests separately cover included-CS3C success, missing-decoder rejection, original FBX success and cancellation with real packaged IPC.
- The actual macOS debug client native picker imported original-byte CS3C and FBX, displayed real previews and renamed a card inline. The custom-library plus card and selection-only personal tab are actual product components. The app API was in mock mode; this is not live BFF qualification.

Reproduce `verify_camera.py` with a packaged launcher, repeated `--source`, an approved `--pose`, and a new `--output` directory. Reimport with `verify_camera_geometry.py` using Blender `--background --factory-startup --disable-autoexec --python-use-system-env --threads 2 --python-exit-code 1 --python <script> -- <output>`, with `PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1`. Verify the full pinned runtime manifest again afterwards; do not relax the bytecode integrity gate.

## Installer and activation gates

Client ordinary build entry now leaves the feature disabled and omits runtime resources unless given an explicitly pinned package. Configured builds validate every runtime byte, platform and camera contract before bundling. `releaseReady:false` is debug-only. A macOS arm64 review app/DMG does not certify Windows, macOS x64/universal, signing, notarization, decoder redistribution or public release.

**Actual CSP failure blocks activation.** CSP 5.1.1 exited during File → Import → 3D data of `model-0-camera-1.fbx` into a separate new canvas. macOS recorded main-thread excessive recursion / `EXC_BAD_ACCESS` / `SIGSEGV`. Source-orientation baseline comparison is still outstanding; the cause is not attributed to camera rotation. Blender roundtrip success must not be presented as CSP import success.

Disabled-base review can proceed. Enabled user testing/release still needs CSP isolation/acceptance, authenticated live BFF continuity, the remaining model/OS scope and release review. No merge, production access, deployment or paid inference/evaluation was performed.
