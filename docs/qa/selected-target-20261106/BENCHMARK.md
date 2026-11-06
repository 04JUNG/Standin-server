# Source preselection / selected target benchmark — 2026-11-06 KST

User-corrected flow: source refinement before Top5, body/shape/hands only after explicit pose selection. No target5N benchmark or preselection FBX generation. No deployment or paid provider call.

## Resource and fixture limits

- Local macOS 26.6.2 arm64, 15 logical CPUs, no CPU quota. Blender5.2.0 / fbe6228777e7, Blender/OMP/OpenBLAS/MKL threads1 in target children; C1/C2 share the same host. This is **not a production 1vCPU measurement or SLA**. Default C1 stays unchanged. No p95 claim from these small samples.
- Source: synthetic BVH/2D observations, 2 people × 5 candidates, standing/arm bend/leg bend/occlusion/unowned partial. Cold excludes process imports and fixture creation; warm clears result cache but retains process; cache-hit retains exact source result. Real sitting/contact/rough-image accuracy is not certified.
- Target: 4 independent post-selection diagnostic cases per run: approved male/female Finding_01 + relaxed hands, male stored sitting BVH diagnostic, female approved shape recipe + hands. Each target conversion and inspection uses a fresh isolated Blender process. “Warm” means runner-version/OS cache only, not a resident solved-target scene. Stored source is real; selection bindings are local fixtures. Sitting is outside the catalog supported-pose set and is **not** a publication eligibility expansion.
- Target child RSS is a process-lifetime high-water from RUSAGE_CHILDREN, not aggregate simultaneous RSS. Parent CPU/import time is not included in child CPU. Queue wait is included in wall latency. All bytes/model/source hashes and safety findings are in raw JSON; no model/BVH/user image binaries are committed.

## Preselection source (10 candidates, synthetic)

| C | Cold ms | Warm recompute ms | Cache hit ms | Selected reuse10 ms | Peak actual adapter calls | Cold/warm source calls | Cache/selected calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 199.47 | 192.63 | 2.97 | 2.17 | 1 | 10/10 | 0/0 |
| 2 | 215.73 | 210.62 | 3.41 | 2.10 | 2 | 10/10 | 0/0 |

Both runs retained 8 ready_base / 2 ready_refined, exact original bytes on base retention and identical cold/warm source bytes. Every result is source_only; target mesh/contact certification=false. C2 was slower here, so there is no basis to increase default source concurrency. Source adapter is engine-owner code, measured sequentially by target/performance owner after target measurement; raw files include exact code/config/runtime identity.

## Selected target (4 cases per run, actual approved assets)

| C | Mode | Wall s | Child CPU s | Child peak RSS MiB | Structure/GLB/reset | Strict geometry outcomes |
|---|---|---:|---:|---:|---|---|
| 1 | cold | 24.317 | 23.737 | 449.5 | 4/4 passed | 2 blocked, 2 structural-only |
| 1 | warm | 23.043 | 22.937 | 462.7 | 4/4 passed | 2 blocked, 2 structural-only |
| 2 | cold | 21.870 | 27.287 | 462.7 | 4/4 passed | 2 blocked, 2 structural-only |
| 2 | warm | 17.785 | 27.545 | 462.7 | 4/4 passed | 2 blocked, 2 structural-only |

All 16 final-file inspections retain52 bones and exact float32 GLB surface; maximum coordinate error0 in these samples, within existing0.1mm tolerance. No PNG, source solver or VLM call. C2 reduced batch wall time on this multi-core host, but increased shaped-case latency and CPU use; it is an alternative measurement, not a new default or a1vCPU promise.

| C1 cold case | Convert s | Independent inspection s | Outcome |
|---|---:|---:|---|
| male standing | 2.442 | 4.220 | blocked_not_cached |
| female standing | 1.577 | 1.319 | structural_checked_not_skin_safe |
| male sitting-diagnostic-only | 1.725 | 3.760 | blocked_not_cached |
| female standing-shaped | 7.983 | 1.289 | structural_checked_not_skin_safe |

**Safety limits:** male standing/sitting full mesh diagnostics detect skin-triangle intersections; sitting also detects opposite-hand intersection. They are blocked and never enter the success cache. Female assets exceed the unchanged50,000-triangle full audit budget (50,010 triangles), and all samples lack accepted topology policy/fold reference: female structural results are mesh_unverified, **not skin-safe**. Opposite-hand clear_screen alone is not full collision certification. No threshold/policy/solver gate was weakened.

## Reuse and hand-only revision (actual female target)

- Initial selected-target prepare: 3539.817ms; full-result cache hit: 4.601ms; selected reuse: 4.171ms. Exact object/FBX/GLB bytes retained=True. No source/target solver/exporter runs on either reuse call. Structural evidence only; skin_safe=false, export_authorized=false.
- Hand-only revision: 2706.933ms, verified pre-hand body mode=reuse; source solver calls0 and body retarget skipped. Hands/final bytes are newly checked. Body cache is privately partitioned by owner/job/person/candidate-set and immutable source/model/runtime; shaped targets do not use this unshaped FBX cache.
- FBX remains an explicit dependency. For C1 cold unshaped cases, the frozen/body and hand exporter calls total roughly0.11–0.14s per selected result; raw records show each call/size. The shaped retarget_initial_export stage includes model load, neutral creation, retarget, shape/contact correctives and authoring exports; it is not pure serialization. Instrumented rt.export_fbx calls do not include all authoring-controller export_fixed work. No in-memory equivalence or native-v4 performance claim is made.

## Independent restoration and local integration

- Same-process inspection A/B/A deliberately polluted previous mesh, shape key, weights, transform and material. Independent factory-empty restoration and A-repeat artifact/joints/geometry/GLB hashes matched exactly. Audit proves isolated sequential inspection reset; it does not enable warm solved-target mutation. See restore-raw.json.
- Source engine function-level prepare/reuse → selected provider → real female FBX/GLB passed. Input was actual stored BVH with synthetic invalid observations: ready_base/source_only, one source adapter call before selection, zero after selection,52bones, exact surface and byte reuse. This is not a live BFF/UI/native CSP test. See source-target-integration.json.
- Local validation: converter host contracts727 passed/1 optional skip; Blender-only pytest modules are excluded from host collection and actual Blender evidence is above. Smoke54/54, compileall, frozen SHA manifest and git diff check pass. New selected-only contracts29 passed (included in727).

## Files and reproduction

- `target-raw.json`: selected cases/resources/timings/SHA/safety, measured runtime revision. `source-c1-raw.json` and `source-c2-raw.json`: synthetic source per-result metadata/timing and source runtime identity (generated BVH bodies omitted; exact SHA/byte count retained).
- `restore-raw.json`: independent warm inspection restoration. `source-target-integration.json`: selected-only local function integration evidence.
- `tests/converter/measure_selected_target.py` takes local model manifest/shape registry/stored standing and sitting BVH paths plus output directory. `tests/converter/verify_selected_restore.py` runs inside pinned Blender.
- Remaining acceptance: production-equivalent1vCPU/6GiB target/source runs; actual rough/contact accuracy holdouts; accepted full mesh policy and complete certificate; live selected BFF/current-revision wire; non-default appearance adapter integration; selected-scene relationships. These are not asserted by this PR.

Measurement revision and final revision are retained in target-raw.json. A subsequent status-only guard labels identical original/source bytes ready_base; it does not change the measured conversion/inspection/cache path.
