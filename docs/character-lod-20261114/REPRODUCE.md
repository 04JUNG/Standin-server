# Reproduction

Use a fresh output directory at every authoring/package/preparation step. Inputs
are never overwritten. See `evidence/provenance.json` for input hashes. Core
scripts live in `scripts/character_lod/`; they write no deployment configuration.
Only the separately named local registry is used by these commands.

## 1. Audit and build

Use pinned Blender 5.2.0 / `fbe6228777e7`. Sources must be the exact approved
latest blend/controller. `build.py` verifies both pins and the pinned skin UV
file before creating output. Example, replace paths with your own verified copy:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/character_lod/audit.py -- \
  --blend "$INPUT/male.blend" --out "$NEW/audit-male.json"

blender --background --factory-startup --python-exit-code 1 \
  --python scripts/character_lod/build.py -- \
  --blend "$INPUT/male.blend" --controller "$INPUT/male.py" \
  --gender male --hand-policy articulated --method local \
  --skin-uv "$SKINS/male-base-v0-8-0-uv.json.gz" \
  --extra-protected "$EXTRA/extra-male-1.npy" --out "$NEW/male"
```

Run female with her source/UV, `--gender female`, without `--extra-protected`.
The extra male mask was generated from actual failed morph triangles in the
rejected planar experiment. To reproduce that lineage, build with
`--method planar`, run `repair.py --help`, then use its union of failed-source
neighbourhoods. The exact original mask is hashed in the local delivery receipt.
Planar trials are experimental outputs, never automatically selected.

`build.py` emits separate one-round/three-round local candidates (or 1/7-degree
planar candidates), along with full trial counts. Vertex order is fixed by the
retained source map. A new preview mesh/mapping and UV corners are created.

Make a clean ordinary authoring file with the dependency closure only:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/character_lod/compact.py -- \
  --candidate "$NEW/male/male-protected-3-articulated-local" \
  --out "$NEW/compact/male-protected-3-articulated-local"
```

The reload semantic digest must match. No input file is edited. Blender binary
serialization is not claimed byte-deterministic; each final file receives a new
SHA256. Geometry/topology/correspondence invariants are the reproducibility gate.

## 2. Quality and all presets

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/character_lod/quality.py -- \
  --originals "$ORIGINALS" --registry "$APPROVED_REGISTRY" --gender male \
  --candidate "$NEW/compact/male-protected-3-articulated-local" \
  --out "$NEW/quality-male"
```

Repeat female. Multiple `--candidate` flags compare multiple levels. An optional
`--reference` reuses unchanged raw original reference surfaces; it does not reuse
candidate results or skip candidate exports. `region_metrics.py` computes named
region correspondence errors from these arrays. Every failure and the stricter
comparison-screen result remains in JSON.

`package.py` builds an isolated local registry from the approved recipes and
quality receipts. It creates new character IDs and copied binary paths. Run
`export_delivery.py` on it to export the exact delivered source with actual
recipe/default grid values. Run `finalize_skins.py` to reload those files and
regenerate candidate UV/index/weight manifests. It does not edit the repository's
production skin manifest. Skin compatibility pins belong to this local bundle.
All local registries/skin manifests and final FBXs are under `delivery-package/`.

For the evidence here, body/grid conditions are recorded in `jobs.json` and each
benchmark row. Large custom height is snapped to the published grid (male
189.684899cm, female 189.893087cm) on both sides. The more general authoring QA
also tests 190cm; those are distinct recorded checks.

## 3. Neutral preparation and repeated first conversion

Run the existing `scripts/prepare_preset_bodies.py` inside the fixed Linux
container, once for original and candidate sources. Do not reuse host-generated
prepared receipts: their runtime identity must match Linux amd64 and the current
code/source hashes. Merge only verified `prepared_neutrals.entries` from the two
new local bundles into the benchmark registry. Never alter an approved bundle.

The local run mounts the dedicated skin manifest over the container's read-only
code copy. This is local experimental compatibility, not certification or a repo
pin change. Example using the exact final directory from this run:

```bash
docker run --rm --platform linux/amd64 --cpus 2 --memory 4g --network none \
  -v /private/tmp/standin-mesh-light-20261114:/code:ro \
  -v /private/tmp/standin-mesh-light-assets-20261114:/work \
  -v /private/tmp/standin-final-preparation-qa-20261112/ob_015.bvh:/input.bvh:ro \
  -v /private/tmp/standin-mesh-light-assets-20261114/delivery-package/skin-assets.json:/code/converter/skin_assets.v1.json:ro \
  -w /code -e PYTHONPATH=/code a4d731fe3568 \
  python scripts/character_lod/benchmark.py \
  --package /work/delivery-package \
  --registry /work/delivery-package/shape-bakes.benchmark.json \
  --bvh /input.bvh --out /work/benchmark-new
```

`--smoke` runs one default male pair only, and is not a three-run conclusion.
`--resume` skips only previously persisted rows after an interrupted batch;
each missing conversion still gets a new worker with every completed-result
cache off. Keep the interruption log and report it. `bench_worker.py` delegates
to the real worker and adds timing wrappers; frozen solver files are not edited.

## 4. Preview and actual-FBX visual review

`runtime.py` resamples all published presets using the new candidate preview
mesh, current controller and current source hash. It calls the existing SSR2
builder and records errors. Run the existing independent
`check_sampled_runtime.py` with the generated file and a different seed, then
read its `passed` flag; generation by itself is not a preview quality certificate.
No original SSR2, source vertex indices or skin certification is repinned to a
new topology. In this run a dependency-closure cleanup after sampling can rebind
only the byte-container source hash after proving semantic equivalence; candidate
geometry, correspondence and controller hashes must match exactly.

`render_review.py` exports actual posed FBXs, independently reloads them and
checks all six skins with the unchanged typed skin writer/verifier.
`render_shapes.py` does the same for full-body comparisons of every preset and
large custom. `build_review.py --assets <root>` generates the local browser page.
Render defaults use workbench studio lighting; no synthetic image generation is
used. The local browser page includes actual FBX links and numeric holds.

## 5. Required checks

```bash
python -m unittest tests.test_character_lod_contract -v
python tests/test_smoke.py
python -m compileall -q src api scripts
git diff --check
```

The contract tests reject substituted inputs, invalid/duplicate correspondence,
lost protected connectivity, incomplete paired results, mismatched body/BVH
conditions, and mixed prepared/generated paths. The existing core smoke suite
passed 54/54 in the repository's Python 3.12 environment. Host Python without
Pydantic could not run that suite; no environment dependencies were changed.
