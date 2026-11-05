> Historical v1 draft. The normal correspondence issue is resolved by the current file-canonical v2 adapter; see `docs/NEXT_SPRINT/INTERACTIVE_REAL_MODEL_V2_2026-11-04.md` and `real-model-v2-correspondence.json`.

Frozen V3.2.5 and shaped conversion currently produce intermediate FBXs before a display state can be prepared. This draft adds an opt-in, separately versioned post-frozen snapshot boundary: existing hand rules solve on a private scene, approved body/shape snapshots can be reused, and an immutable state feeds independent display and final-file consumers. No new HTTP route or feature activation is included.

The initial/intermediate FBX dependency remains. Approved actual male/female shape assets fail strict split-normal correspondence, so this is not ready for integration or rollout. The proposed diagnostic-only normal exception was reverted; the strict gate rejects the actual export and publishes no ready receipt.

Changes:
- Separate private-scene hand solve from the legacy file wrapper without changing frozen solver files or SHA manifest.
- Add bounded rig/rest/bind/all-weights/split-normal packets, canonical immutable state/hash lineage, latest-revision/cancellation/restart fencing, and fresh-process FBX export/reimport validation.
- Add exact precomputed hand-combination tables plus a shared JS consumer; registered asset and matching file verification are required. No actual atlas is released, and personal FBX/CS3C gain no capability.
- Include the exact design, draft contract/schema guide, sanitized source/hash/metric evidence and integration blockers. Source/generated assets, runtime/controller files and credentials are excluded.

Validation:
- 201 focused tests, including 26 new Python/JS boundary tests; existing smoke 54/54; compileall src/api/scripts/converter; git diff --check.
- Blender 5.2.0 LTS build fbe6228777e7: synthetic real FBX fresh export + fresh reimport passes full/half/bust/shoulder_head/head. Max vertex error 3.48e-7 m, matrix error 2.17e-6, weight error 0.
- Approved master-v2 male and female shape 0.3.4: all five scope snapshots are produced; legacy correspondence fails split normals. Actual male fresh-process final validation rejects and publishes no ready receipt. This is separate from synthetic success.
- One isolated local legacy timing: initial retarget/FBX 676.9 ms, hands import/apply/export 128.5 ms, framing import 66.7 ms, final reimport 66.6 ms, PNG 142.6 ms. No service percentile or speedup claim.
- Local GLB inventory does not prove current Top-5 source/body/scope coverage; no missing-card conversion is introduced.

Before integration: resolve actual split-normal preservation under the strict gate; introduce a new initial-FBX-free solver boundary with frozen-equivalence evidence; implement Converter HTTP operation transport and exact search projection/frame bridge; cross-validate with BFF/Client; expand shapes/presets/mirror/cameras and native/CSP/performance tests. No deployment, AWS changes, paid inference, merge or rollout was performed.

AI-assisted implementation; validated as listed. Requested follow-up design work; no existing issue number supplied.
