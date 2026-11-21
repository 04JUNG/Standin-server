> Material 지원과 최신 runtime 계약은 [반사광 보존 후속 기록](SPECULAR_20261104.md), revision5를 따른다. 이 문서는 revision4 시점의 저장/host 구조와 측정 기록이다.

# Verified registration FBX and retained host — 2026-11-04

This replaces `.blend` as the active personal-model storage. On registration, import the original FBX/CS3C once, retain its byte-identical original and a Blender recovery archive, then export/reimport and validate `character.fbx`. Registration previews use that admitted FBX scene. Subsequent pose jobs only read `character.fbx` through the personal importer, which preserves helper/terminal bones and original transforms. The original solver and final camera contract stay unchanged.

Contract revision **4**, envelope `standin.local-converter/1`, `hello.modelStorageVersion=personal-fbx/1`. Normalization/converter advance to `/4`; importer/rig stay `/2`. `normalizedFormat=fbx`, `normalizedPath/normalizedHash` identify the working FBX; `archivePath/archiveHash` identify the recovery-only `.blend`. Version/fingerprint mismatch blocks legacy caches; original bytes remain available for re-registration. No silent fallback to a `.blend` working file or automatic legacy migration.

The companion native client retains one manifest-verified host per launcher/library root. Capability, import, apply and export reuse it; heavy Blender work remains a cancellable, memory/time-monitored child per job. A fresh host always performs the full package integrity check. Errors/cancelled sessions are closed, dead processes and changed roots/launchers are restarted, request-count bounds trigger reconnection, and failed mutating requests are never silently replayed. App exit closes the host after cancelling active work.

## Evidence

- [Runtime/native/test inventory and geometry](verification-fbx-runtime-20261104.json): Rust personal tests 10, local converter 13, decoder 6, frontend 1,302/1 skip; typecheck/lint/build/clippy and packaging checks pass. The final debug app contains the pinned runtime (6,579 manifest files, changed0). Native-picker UI smoke of this build is unverified because the macOS accessibility tool returned AXError.cannotComplete after a long response delay.
- Same native retained process: initial startup/integrity **5.997s**, warm hello **6.69–6.91ms**, CS3C registration **2.207s**, FBX pose apply+validation **1.679s**, cache apply **18.82ms**, worker export **17.49ms**. These are one machine/sample and exclude network/UI/user-folder final save.
- [Actual 9-output regression](verification-fbx-camera-20261104.json): two approved original CS3C and one permitted FBX × three cameras. `character.blend` was held outside its expected path during all apply operations, proving independence from the recovery file. Original hashes, admitted FBX and archive hashes, cache/restart/rejection/delete preservation are checked. Independent reimport confirms the requested single transform and exact weights/scales across camera variants; maximum world vertex error `5.72e-7`, bone error `8.29e-7`.
- The 9-output package and final native-test package differ only in explanatory version strings in `converter-client-contract.v1.json`; all converter code, binaries and converter fingerprints are identical. The final package itself is tested by native import/apply/export/reuse and decoder-negative tests. Pins are recorded explicitly rather than rewriting an old package manifest.

## Future `.clip` output and release scope

Preserve the original CS3C when supplied. A `.blend` is a recovery representation of supported parsed data, not a replacement for all original CSP metadata. `.clip` is a CSP document; `.cs3c` is a 3D character file. Retaining either model format does not itself implement `.clip` document writing or certify editable CSP layers/cameras. No `.clip` writer, Modeler execution, CSP automation or new decoder payload is added.

Primary references: [CSP document formats](https://help.clip-studio.com/en-us/manual_en/210_file/Open_file.htm), [Modeler character save](https://www.clip-studio.com/site/gd_en/csm/userguide/csm_userguide/310_character_op/310_character_op_h2_0030.htm).

Default installers remain disabled without an explicitly pinned runtime. `releaseReady:false` remains debug-only. Known CSP import crash, authenticated live BFF continuity, other native OS targets and signing/redistribution qualification remain activation/release gates. No merge, deployment, production access or paid inference/evaluation.
