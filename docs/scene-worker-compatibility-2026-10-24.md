# Scene worker / BFF130 / client155 compatibility (Draft)

Base: server develop `7a3713f7cbd1e78925f6e78704f5ff706b96bfa2`.
BFF PR130: `131da9bc11189f18d4225dfda767f4295a94dbdc`.
Client PR155: `b6152161f05f43c4c0290cf607cbbf85b8037de3`.
Only Standin-server product code is changed. BFF/client were inspected and tested
in separate snapshots, without editing their product code or dependency trees.

## Changes

1. Restore horizontal camera sensor fit when rendering an exported scene FBX.
   The exporter calibrates horizontal fit, but FBX import resets to AUTO; portrait
   panels then incorrectly enlarge/crop actors in the separate final preview.
   The FBX itself, actor transforms and poses remain unchanged.
2. Port the previously paused isolated worker image/readiness/deployment work onto
   current develop. Keep develop's `artifact_only` engine/stage contract and final
   preview endpoint/packaging unchanged. The old worker branch by itself lacked
   that contract and was rejected by the BFF as SCENE_INTEGRITY.
3. Preserve exact `503 {"detail":"scene worker busy"}` plus Retry-After5, which
   PR130's existing transport identifies as SCENE_BUSY. Timeout504 remains
   recognizable as SCENE_TIMEOUT. No public API/client changes are required.

## Same-FBX before/after comparison

Executed the actual develop final-preview worker and the corrected worker on
identical existing p060/p220/p0718 `app-saved.fbx` inputs using local Blender5.2.0
LTS/buildfbe6228777e7. Original images were reused for reference. These are fresh
renders of saved FBXs, **not fresh image analysis or newly generated poses**.
No per-cut coordinate/yaw/pose overrides were used.

| Cut | Persons | Preview size | Result |
| --- | --- | --- | --- |
| p060 | 2 | 302×512 | portrait enlargement/cropping fixed |
| p220 | 2 | 183×512 | portrait enlargement/cropping fixed |
| p0718 | 3 | 512×295 | before/after PNG bytes identical |

All three source/before/after FBX SHAs match exactly. PNG/report SHA checks pass.
The local standalone HTML embeds original/before/after images; it is delivered as
an artifact, not uploaded with customer roughs, model weights or FBX files.
[Machine-readable render evidence](qa/scene-worker-compat-20261024.json).

A new real Blender camera FBX roundtrip regression covers both portrait ratios,
landscape and square. Maximum normalized projection error was <5e-7 locally,
under the1e-5 gate. CI runs the same test inside the Linux worker image.

## Contract validation

The actual Python Engine/API output was fed to PR130's actual compiled
`prepareRefinedScene` and `sceneRefinementRequest`, with only model/Blender and
owned-source boundaries replaced by synthetic fixtures:

- artifact_only=true → verified-scene-fbx-v1 + empty preview: accepted.
- busy503 → SCENE_BUSY.
- timeout504 → SCENE_TIMEOUT, serverOperationMayContinue=true.

[Machine-readable contract results](qa/scene-worker-bff-20261024.json).
This checks the cross-language boundary; it is not a real model/S3/installed-app
end-to-end run. The latest develop Engine/scenes files are preserved byte-for-byte.
The frozen retarget/bone-map files are also unchanged.

- Server targeted runtime/asset/deploy/pose/quarantine/scene tests:119 passed.
- BFF PR130 scene route/refinement/transport/composition/joined handler tests:64 passed.
- Client PR155 scene API, app/bar hook, wire replay and review panel:77 passed.
- BFF TypeScript build and client TypeScript noEmit passed during the pinned audit.
- Local render6/6 and four real camera roundtrips passed.

## Remaining verification / rollout

No AWS resources were provisioned or deployed, no feature was enabled, and CSP
was not operated. Docker is unavailable locally; Linux image build/import/roundtrip
checks are added to CI, not claimed as locally completed. Approved model/release
assets, credentials and private worker address still need operator provisioning.
The experimental local HumanArt bundle is not relabeled or used as an approved
staging release.

Actual staging plan/build → BFF review → installed app native save → CSP framing
must still be tested with p060/p220/p0718. Only the calibrated master rig is covered.
A shared BFF receipt does not make the worker's in-memory pending plans replica-safe;
initial deployment remains one task/process. Generic422 causes remain combined.
[Runtime and deployment handoff](scene-worker-staging.md).

Keep this PR Draft and BFF refinement disabled until those rollout checks pass.
