# RC4 on-demand library original validation — 2026-11-11

The request path no longer reads the 45-pose registry or prepared experiment assets. The main API resolves the selected ID in its own current library and transfers its unchanged BVH bytes/hash to the private worker. Frame 0 is retargeted with the pinned original converter and unchanged source body; rig matrices and evaluated skin are prepared automatically. No rough or target is consumed during source preparation.

The 60 frozen RC4 solver/prompt/selector/gate files are unchanged. Preparation preserves the previous 004 B0 exactly: every bone matrix, evaluated vertex, triangle and skin weight has maximum difference 0. This is source preparation parity, not a new pose tuning result.

## Validation scope

- 59 release/adapter tests and 54 smoke tests passed; BFF release tests and build passed.
- Linux worker starts without the registry file or registered asset directory.
- Current staging bundle: 3,503 BVHs passed size, recognized rig-profile and required-bone inventory checks (Mixamo 645, unprefixed Mixamo 2,504, CMU 273, 100STYLE 81). This is input compatibility, not 3,503 completed refinements or quality approvals.
- Actual additional source preparation passed for `cmu_01_03_01932`, `style100_Balance_FR_f002874_26c31151` and `ob_007` (6.09s / 4.60s / 2.81s locally). These are preparation-only tests.
- Full staging tests use the approved original 004 rough and previously unused original poses, not adjusted poses. The known-history exclusion scanned 72 manifests and 91 prior selected pose IDs; it does not claim to enumerate every experiment ever run.

## Runtime and release boundary

Source cache identity binds BVH, source body, converter/preparation source and Blender version. Every cached file is hash-checked; corrupted cache is rebuilt. Each request binds a fresh rough outside the source cache. BFF requires fresh accepted receipts to match the selected pose and original BVH hash. Previously stored valid results remain readable.

Staging only: inference revision 139 and BFF application image from revision 137, RC4 on. A concurrent BFF revision 138 retains the same application image and RC4 flag; it was not overwritten. Production task definitions and desired counts remain equal to the pre-change snapshot. No merge or production deployment was performed.

Single-target extraction, verified fixed source body and the existing safety gate still apply. Missing/invalid originals or incompatible conversions fail closed. Later body/hand/crop choices are not applied to RC4 outputs. Returned camera is applied by the app; it is not embedded in FBX. CSP import is outside this validation.

## Completed native app calls and downloads

| Original library pose | Selected | Request time | Download |
| --- | --- | --- | --- |
| `sd_cmu_13_04_f02064` | Shadow | 180.528s | 1,832,972 bytes; SHA matches accepted receipt |
| `rokoko_Judges_mixamo_00711_mirror` | Preserve | 192.325s | 1,835,548 bytes; SHA matches accepted receipt |

Both ran through the normal search selection and real staging RC4 call, with fresh source preparation (`cache_hit=false`, `zero_manual_edits=true`). Their source BVH hashes match the deployed library bundle. Both accepted previews were visually inspected in the isolated native app with the returned camera, then saved using the app download flow. The second file received a `_v2` name; the first file and previous results remain intact.

Both downloaded files were independently reimported in Blender: one armature, 65 bones, 24,988 evaluated vertices, finite skin coordinates. Full matrix roundtrip remains part of the unchanged server release gate. This additional download reimport checks file integrity; it is not an independent naturalness or collision certification. The second pose remains less similar to the rough than the first; no tuning or gate threshold was changed to make either pass.

[Machine-readable verification](refine-rc4-on-demand-20261111.json) records source provenance, download hashes and returned cameras. Only these two originals completed the full native end-to-end path in this change. All-library input support does not guarantee that every pose will be released: the existing gate can withhold unsuitable corrections.
