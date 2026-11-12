# Interactive FBX export: single verifier evidence (2026-11-12)

## Change and retained gates

`export_bundle` copies the solved bundle's immutable `staged.fbx` on the host, checks the copied bytes, and runs the existing independent Blender reimport verifier. Previously a separate Blender export process opened `solved.blend`, captured/compared the scene twice, and copied that same staged FBX before the verifier process. The redundant export process is removed from the host path; the legacy CLI stage remains available.

`load_solved` still verifies runtime/state/snapshot/staged-FBX integrity. The fresh verifier still enforces frozen Blender lineage, full geometry/topology/joint/skin correspondence and immutable artifact bytes. Final receipt binding and current-selection checkpoints still gate publication. The BFF retains owner, review, quarantine, expiry and download authorization checks.

## Local benchmark

One before/after run on macOS, Blender 5.2.0 build `fbe6228777e7`, same 9,055,484-byte staged FBX and 45 MB canonical model. Baseline code: develop `45e42e9`. Timing includes `export_bundle`'s own `load_solved`, excludes the separately measured initial fixture load, and excludes network transfer.

| Measurement | Before | After |
|---|---:|---:|
| Initial fixture load | 6.485 s | 7.317 s |
| Export and fresh reimport verification | 38.142 s | 23.324 s |

Local export reduction: **14.819 s (38.9%)**. This is a single local sample, not an operating-server latency promise. The observed production export wait (~113 s), preview wait (~163 s), and actual download (~4.6 s) describe separate stages; this change addresses export preparation only. No production deployment or measurement of this change is included.

Both runs returned `verified=true`, exact FBX SHA-256 `632053c8c39cb14d2531137263492c0e4419dba45713a8fa541472ee8192aa23`, and zero maximum position, matrix, normal and weight error. Topology, joint identity and skin identity were true.

The benchmark copied an existing fixture and rebound only its solver runtime digest plus canonical state hash/snapshot metadata for each runtime. Geometry, FBX and BLEND bytes stayed identical. Its model had previously been recaptured by the local importer to account for macOS/server split-normal differences. This is benchmark-only setup: production `load_solved` validation is unchanged. No user packet, source mesh, credentials or runtime bundle is committed.

## Verification

- `tests/converter/test_interactive*.py`: 133 passed (including state suite: 39 passed).
- `tests/test_smoke.py`: 54/54 passed.
- `python3 -m compileall -q src api scripts converter`: passed.
- `git diff --check`: passed.

The positive regression test requires the exact staged bytes and only the verify child; tampered copies cannot reach the verifier or publish a ready receipt. Existing stale revision, cancellation, child crash and integrity tests remain passing. Merge requires at least one review under `docs/COLLABORATION.md`; deployment remains a separate operation.

## Handoff review and production follow-up

Reviewed against fetched develop `45e42e9`; no merge conflicts or blocking code/safety findings. The final publication fence now has explicit tests where a verifier finishes successfully but the real latest-selection gate receives a newer hand revision, cancellation or restart before publication. In all three cases the private verification proof exists and no public `receipt.json` is written. Product code and the measured runtime remain unchanged from `5edd7a0`.

At `5edd7a0`, GitHub CI, Converter CI and Scene worker CI all passed, including the pinned Linux Blender image and real-conversion job. Those jobs do not replace the exact local interactive-export benchmark above. Review approval was still pending at handoff; CI and this code review do not satisfy or bypass repository protection rules.

After a separately authorized deployment, repeat the same production selection before/after and record preview preparation, export request-to-ready and actual transfer separately. Include repeated cold/warm runs, artifact byte count/SHA, verified receipt and correspondence errors, then exercise a changed hand/shape selection and cancellation during export to ensure stale files stay unavailable. Compare export p50/p95 and error rates; do not extrapolate the single local 38.9% result to production. No production measurements, deployment or infrastructure changes were performed in this review.
