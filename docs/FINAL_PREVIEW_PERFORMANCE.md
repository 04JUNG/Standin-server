# Final preview performance experiment

## Scope and defaults

This opt-in Converter change reduces repeated body retargeting when only hands
change, and repeated factory resets during the first preview conversion. Both
optimizations are disabled by default. It does not change the retarget solver,
hand presets, body shape pipeline, HTTP request contract, or deployment settings.

The eligible path has hands, no shape recipe, a front PNG preview, and no model
preview, thumbnail request, hand-contact screening, or exact V3.2.4 mode. Each
conversion still starts a fresh Blender process. Other paths use the existing
pipeline.

| Setting | Default | Meaning |
| --- | --- | --- |
| `CONVERTER_RETARGET_CACHE_BYTES` | `0` | Per-runner process-memory LRU capacity in bytes; zero disables reuse |
| `CONVERTER_RETARGET_CACHE_TTL_SECONDS` | `600` | Positive, finite lifetime measured with a monotonic clock |
| `CONVERTER_FAST_SCENE_RESET` | `false` | Enable verified empty-scene loads for eligible conversions |
| `CONVERTER_EMPTY_SCENE_TEMPLATE` | Adjacent `empty_factory.blend` | Explicit local template path when fast resets are enabled |
| `STANDIN_LAB_PREVIEW_TRACE` | unset | `1` adds stage timing fields to the worker report |
| `STANDIN_LAB_FIRST_ENTRY_PROFILE` | unset | `1` adds deeper call profiling for dedicated diagnostic runs |

The empty template fixture is `tests/converter/fixtures/empty_factory.blend`:
Blender 5.2.0, build `fbe6228777e7`, SHA-256
`5c51c07d9b0263e0997ab27f49d65f8bcc5cab6211fed8935f7383132904654c`.
It contains no character, motion, image, world, collection, or object. The helper
checks its hash, rejects symlinks, opens it with `load_ui=False` and
`use_scripts=False`, and verifies the empty scene fingerprint after every load.
A missing, altered, or incompatible template fails the conversion closed.

The runtime Docker image deliberately excludes all `.blend` files and tests.
The fixture is **not** copied into that image. Enabling fast resets in a separate
environment requires explicitly providing the verified template and setting its
path; this PR does not enable or deploy it. Retarget reuse can be enabled
independently without a template.

## Artifact integrity

The cache contains only the immutable full-body FBX and body report captured
before hand baking, cropping, and camera rotation. Its identity binds source and
character content hashes, character ID, mirror, frame, output policy, frozen
solver/manifest, Blender build, and runtime code revision. Each consumer gets a
private file copy. Capacity and TTL are per runner, with no shared service or
disk cache. Concurrent cold misses can compute twice; pending work is not
coalesced.

The child independently verifies the base and report. The parent publishes it
only after the full conversion succeeds and the final artifact, preview, hand,
and base receipts pass validation. Failed or corrupt work cannot publish a base.
The final PNG still comes from reimporting the actual final FBX. Existing final
FBX/preview hashes and identity checks remain in force. Cache corruption is an
integrity error; expiration or eviction causes normal recomputation.

Timing instrumentation is local and opt-in. It adds no network telemetry.
Request IDs correlate existing queue controls; entry/reentry classification still
belongs to the caller, while Converter records the stages it actually executes.

## Local evidence

These are local Blender measurements, not production or browser latency. Six
paired trials used the same local character and BVH, with alternating order.
The two measurements below belong to separate sessions and must not be combined
into an end-to-end speedup.

| Experiment | Before median | After median | Change |
| --- | ---: | ---: | ---: |
| Hand change: pre-hand retarget reuse | 1530.4 ms | 904.2 ms | 40.9% faster |
| First entry: fast empty-scene resets, cache enabled in both variants | 1646.8 ms | 1562.6 ms | 5.1% faster |

First-entry improvement was positive in all six pairs. A separate call profile
measured six scene resets totaling 124.8 ms before versus 25.6 ms after. Fresh
Blender launch and body retargeting remain significant costs. Reentry that returns
already retained results avoids conversion; this PR does not implement a new
caller cache or claim browser timing improvements.

Expanded local regression covered two characters, three BVHs, three hand pairs,
half/full scopes, camera rotation, concurrent requests, failures, cache corruption,
and TTL expiry: 71 requests, 63 successes and eight expected failures, with 78
checks. Together with the first-entry benchmark, 87 final FBXs were independently
reimported and rerendered with exact preview pixel equality; 55 optimized/reference
geometry pairs had zero vertex and bone-matrix delta. Existing hand self-collision
diagnostics were preserved; these checks do not establish improved hand quality.
Private character/BVH assets and raw logs are intentionally excluded from the PR.

## Verification

The Converter CI contract job includes the new timing, cache, and reset tests.
They verify opt-in guards, failure propagation, receipt rejection, base immutability,
capacity/TTL, source binding, and final-FBX preview ordering. Run:

```sh
python -m pytest -q tests/converter/test_retarget_cache.py tests/converter/test_fast_scene_reset.py tests/converter/test_preview_instrumentation.py
python tests/test_smoke.py
python -m compileall src api scripts converter converter_api tests/converter
```

The real Blender smoke procedure is a baseline first preview and hand change,
followed by the same two requests with both optimizations enabled and the fixture
path supplied. Assert capture then reuse, matching final artifact/PNG receipts,
independent final-FBX rerenders, and unchanged geometry against the baseline.
Use only existing local assets in a separate output directory. Container/Linux
checks run in CI; the local macOS evidence alone does not confirm those checks.

The final PR checkout passed 343 contract tests (one skipped), 54 smoke checks,
and source compilation. A further four real Blender conversions passed capture/
reuse receipt checks and independent pixel/geometry comparisons. Local Python
dependencies came from the existing offline cache (FastAPI 0.143.3, Starlette
0.49.3, HTTPX 0.28.1), so CI must still verify the repository's pinned dependencies
and Linux container. Requirements were not changed to match the local cache.
