# 파츠 v4 HTTP execution verification / 2026-10-30

Authenticated local HTTP handlers executed the real Blender subprocess and the
shared native arm/leg transaction. The four accepted cases saved changed FBX files
and reimported them for joint matrices, surface positions and final skin/contact
checks. One negative fixture returned the exact current FBX bytes.

| Regression | Result | Changed limbs |
|---|---|---|
| T043 | applied | both legs, native rest geometry |
| T050 | applied | left leg, foot +5cm |
| T002 | applied | both arms |
| T018 | applied | both arms |
| T041rollback | retained | unsupported ankle family; input SHA = output SHA |

[Machine evidence](http-native-verification.json) contains request/input/output
hashes, actual engine version, catalog binding and saved verification errors.
Models, donor FBXs and rough images are not included in Git.

This is server integration evidence, not a staging-app result or a fresh quality
score. The model files are the previous accepted native regression inputs.
There is no new VLM inference or 50-cut evaluation here. BFF integration, reviewed
catalog deployment and worker activation remain separate rollout steps. Exact
request/response and rollout instructions: [BFF contract](../../../docs/REFINE_V4_PARTS_HTTP.md).
