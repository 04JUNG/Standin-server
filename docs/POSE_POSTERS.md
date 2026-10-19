Current full-library/reference-skinning follow-up: [HUMAN_REFERENCE_POSTERS.md](HUMAN_REFERENCE_POSTERS.md). The initial frozen-bake smoke results below cover the earlier implementation only.

# Default-body pose posters

Search cards can display a verified, canonical front orthographic image without starting Blender. This reference uses the pinned `standin-master-v2` default mesh, not the selected body, shape, hands, or search camera. The source BVH is exact, including actual mirrored library entries. No image reflection is used.

`GET /pose/{pose_id}/poster?source_sha256=<sha256>&scope=full` first applies the existing registration/quarantine checks, hashes the current BVH, then validates the poster manifest and PNG digest. Supported anatomical scopes are `full`, `half`, `bust`, `shoulder_head`, and `head`. Responses include source/character/image/render revision hashes, scope, projection, `X-Standin-Poster-Version`, `X-Standin-Poster-Kind`, and `Preview-Only=true`. They never authorize export.

Missing or stale posters return a bounded joint-based reference (`Poster-Kind=pose-only`, character `none`). Its anatomical joint mask is approximate, not an exact mesh cut. Existing warm 2D thumbnails are not accepted as default-body mesh posters. No live Converter fallback runs on this endpoint.

## Offline build and packaging

Resolve the approved default FBX through the character registry (`STANDIN_MASTER_V2_URI`); its SHA must match `config/characters.example.json`. Use the deployment's pinned Blender runtime, then run:

```sh
python scripts/precompute_pose_posters.py --library /path/to/library --out /path/to/library/pose-posters
```

The builder processes one conversion at a time, resumes verified assets, and writes `<source-sha>__<scope>.png` plus a lineage manifest. It records the actual worker, framing, retarget, camera and renderer source revision. Changing those sources invalidates old assets. The library manifest reports complete only after every registered pose has all five scopes. `--pose`/`--scope` support local experiments, but partial libraries cannot pass bundle validation.

`deploy_pose_library.py` validates complete coverage against actual DB/BVH bytes and packages only expected poster PNG/JSON files and `library.json`. Progress logs, temporary files and unrelated images are excluded. Old bundles without a poster directory remain compatible and report explicit pose-only fallback. A present partial/corrupt directory fails packaging.

The full production library must be baked and packaged before claiming default mesh coverage. This change does not bake/upload/deploy that library or migrate any selected-body GLB caches. Runtime dependencies are the matching BFF poster proxy and client; the protected final-export alias is tracked separately in server PR #108.

## Local verification (2026-10-19)

On the same local Mac and Blender 5.2.0, `Finding_01` and its actual mirrored BVH produced ten real PNGs across all five scopes. Source, approved character, scope, renderer revision and PNG hashes were revalidated; the contact sheet was inspected for mesh quality, mirror direction, and bust versus shoulder/head framing.

Offline live bake: median 1.503 seconds/image (ten samples, 1.444–2.453 seconds). Reading those exact baked assets through in-process FastAPI TestClient: first process reads 1.43–6.50 ms, warm 1.34–1.45 ms; five concurrent scope references all ready in 9.43 ms. Missing-asset joint fallback: 7.62 ms first call. These are local serving measurements with warm OS file caches, not network latency, staging/native E2E, production cold-disk timings, or selected-body GLB speedups. The fixture covers two public library poses only.

Tests cover lineage tampering, corrupted PNGs, raised legs/lowered hands under anatomical scopes, exact packaging coverage, absent-library fallback, quarantine and source mismatch. Client tests separately cover independent cards, zero automatic per-card body/model conversion, scope races, opt-in detail cancellation, and BVH/final FBX review separation.
