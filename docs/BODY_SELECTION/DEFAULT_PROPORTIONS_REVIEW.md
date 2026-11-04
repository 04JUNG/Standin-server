# Default body proportions review — 2026-11-04

This change creates local review assets for male 8.3 heads and female 8.0 heads.
It does not activate a product default, alter a live catalog, merge another
session, deploy, publish model binaries, or change the frozen Converter solver.
The implementation uses the existing data-only `shape-bake.v2` contract.

## Which default is being preserved

Fresh `origin/develop` was `592ab0f059089e64a5c88a3ad607539124e8e6c6`.
The tracked catalog is empty; it is not evidence of the deployed model.
Read-only live staging inspection found BFF task revision 106 and Converter
revision 31. Their pinned asset manifests were downloaded and matched the local
October 30 package byte for byte:

| Manifest | SHA-256 |
| --- | --- |
| BFF | `45ee1c64fa312bf5d301824f3b67880c07e5644b2ea0fabd05bd192d042fffc3` |
| Converter | `170c4df72c410a4fb7c5b4696e9e387848cc471da0c64d03cb7dd46cb4b85015` |

The common catalog default is `body-male-base`; the Converter fallback is
`standin-master-v2`. Both retain SHA
`7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d`.
The old catalog's first editable BASE choices are `male-base-v0-7-2` and
`female-base-v0-3-4`. The later `male-base-v0-7-6` and `female-base-v0-3-9`
are additional staged CSP review assets, not proof that first-use selection
already switched. Retired fixed adult cards remain subject to the existing
client/BFF retired-model filters.

The local `CONFIRMED_MODELS.json` records formal October 16 confirmation of
0.7.2/0.3.4. The October 30 staging records and draft [Converter PR #155](https://github.com/04JUNG/Standin-server/pull/155)
document the newer 0.7.6/0.3.9 source lineage and pending CSP acceptance. The
separate [breast candidate](FEMALE_BREAST_CANDIDATE.md) is based on 0.3.4 and
is not silently ported into 0.3.9. None of those records is overwritten.
Read-only production inspection also found BFF revision 51 and Converter
revision 3, both pinned to manifest
`c481055df5cd980c6918d32c9620795faaa8d3f5d764ea8ce5cbf52e5dc4745f`.
Its remote bytes match the preserved old package; the catalogs, registries and
four relevant old/default FBXs were hash-checked against the manifest. Production
retains the same common fallback and editable 0.7.2/0.3.4 BASE choices. Thus the
new 0.7.6/0.3.9 authoring lineage is explicitly a newer staged review source.
No production or staging state was changed.

## New immutable identities

| Sex | New character ID | Latest source BASE | Existing editable default retained |
| --- | --- | --- | --- |
| Male | `male-default83-v0-7-6-p1` | `male-base-v0-7-6` | `male-base-v0-7-2` |
| Female | `female-default80-v0-3-9-p1` | `female-base-v0-3-9` | `female-base-v0-3-4` |

Only `heads` changes from the latest source BASE: male 7.201584 → 8.3,
female 7.37018 → 8.0. Height stays 178/175 cm. Existing IDEAL83 is unsuitable
for this request because it also changes physique, muscle, shoulders, waist,
limb lengths and, for male, a calf detail. It is preserved unchanged.

The existing authoring warp changes the head/body relationship, transports the
skeleton through the same warp, and preserves requested stature. This is not
a uniform whole-body scale. No breast, pelvis, shoulder, fat or muscle control
is added to the requested change. Frozen authoring rest, Basis, authored body
keys, topology, weights, material slots and UV state remain byte/hash identical
within each chosen source. Output rest coordinates intentionally follow the
new proportions. Finger calibration consequently has a new matrix hash;
maximum numeric limit change from float calibration was below 0.000407°.
The same policy and its original axis/alignment/neutral gates are retained.

Full source, FBX, rig, weights and shape hashes are in
[the small validation record](DEFAULT_PROPORTIONS_VALIDATION.json).
Every old character entry and published shape recipe is retained exactly.
Preliminary registration exports have different FBX hashes and control ordering
from the published package; the published identities take precedence.

## Measurements and checks

Stature is world-Z floor-to-actual-crown height in neutral T pose. Head height
is actual crown Z minus the actual facial vertex on the authored chin plane.
The 0.3.9 hand topology rebuild left old stored vertex indices; the measurement
verifies them and remaps the chin to the matching central front-face plane.
The new 0.3.9 chin/crown indices are 12003/11997; male uses 5305/11463.

| Model | Stature mm | Head height mm | Measured heads |
| --- | ---: | ---: | ---: |
| Existing male editor BASE | 1780.000 | 247.16784 | 7.20158446 |
| New male | 1780.000 | 214.457831 | 8.30000000 |
| Existing female editor BASE | 1750.000 | 237.44333 | 7.37017972 |
| New female | 1750.000 | 218.750000 | 8.00000000 |

Both target errors are below 1e-12 heads before float32 export. Actual FBX
round trips use the source exporter’s unchanged coordinate/joint/weight gates.
There were 24 surface checks (two sexes × BASE/candidate × T, arms down,
raised arms, combined bent limbs, twist and constrained fist), 12 candidate
posed FBX round trips, and checks of all 30 finger bones per surface. No
nonfinite or collapsed faces or finger-limit violations were found. Independent
triangle-overlap counts increased in none of the 12 comparisons. Existing
overlaps remain, including the constrained fist and female lowered arms;
this is not a claim of collision-free geometry.

The unchanged production `BlenderRunner` also produced eight final FBXs:
two representative local BVHs × male/female × default/slider. Four independent
same-pose pairs passed, with maximum surface error 0.000470465 mm, maximum
joint error 0.002430319 mm and delta correlation above 0.9999999999. The
existing 0.01 mm and correlation gates were not changed. The local inspector
adapter preserves the already-verified shoulder signature during surface
replay and rejects any bone drift, following the reviewed PR #155 precedent.

Existing exact SSR2 bytes can be reused because the source/rules do not change.
The actual client TypeScript parser/evaluator was bundled from the isolated
integration checkout and compared with Blender at the new controls:

| Approximate live preview | Default max mm | Slider max mm |
| --- | ---: | ---: |
| Male | 0.024222 | 0.066557 |
| Female | 0.158234 | 0.220014 |

These are measured sampled-preview errors, separate from exact FBX tolerances.
`BASE` stays the evaluator anchor; explicit catalog controls override `heads`.
No runtime header or binary is relabelled as a new source. New neutral PNGs use
the existing `body-comparison.v1` render script and bind the new FBX SHA.

Related registry/shape/inspection tests: 65 passed. Required smoke: 54/54.
Required compileall passed. Blender was 5.2.0 LTS, build `fbe6228777e7`.

## App and cache contract

The companion app work consumes local `model-manifest.json`. New catalog
entries carry `gender`, `role=default`, and `order`; retained editor defaults
carry `role=previous_default` and a proposed “기존 기본형” label. The review
catalog's common `defaultCharacterId` names the new male. It remains a local
review file, not a deployment instruction. Explicit selection, personal cards,
last-used state, auto recommendations and historical snapshots keep their
original identity and controls. The model work does not change those policies.

New rows inherit the latest CSP BASE `supportedPoseIds=[]`. They must not
silently grant ordinary body-selection fallback eligibility or rewrite search
recommendations. The app session owns first-use/editor default wiring; general
pose eligibility remains a separate existing contract.

New character/asset/shape hashes provide new cache identities. Shaped jobs
already bypass the plain retarget cache; the plain cache binds character ID and
SHA. Old entries are preserved. Neutral PNGs were regenerated; SSR2 is reused
with proven source/controls binding. Final FBX and preview GLB must always be
generated from the selected new recipe and carry its lineage. Canonical pose
posters/templates still name `standin-master-v2`; promoting a new canonical
renderer later requires separately versioned poster/template/GLB generation,
not reusing old SHA-labelled media. Search features/ranking/indexes are unchanged.

Before any later activation, assemble an immutable additive package containing
all existing files plus the two FBXs, full registries/catalog, new neutral PNGs,
source copies and the exact referenced runtimes. User model appearance and
integrated UI approval remain pending. CSP native import/locks and combining
breast7 are outside this change and are not claimed verified.

## Local reproduction and review

Use a fresh output path and the pinned Blender binary. Model sources, licensed
BVHs, generated FBXs, SSR2 files, mesh arrays and screenshots stay local.

1. `scripts/body_default_proportions/build.py` takes `--source-release`,
   `--legacy-registry`, `--catalog-package`, and `--out`. It verifies source
   bytes and published recipes, copies sources and creates two new IDs.
2. `verify_poses.py --release ...` runs authoring/hand/posed FBX checks.
   `scripts/verify_shape_bake.py` runs the actual worker on local BVHs;
   `inspect_bake.py --runs ... --out ...` performs independent surface checks.
3. `render.py --release ... --old-package ...` reads exact old FBXs and renders
   old/new front, side, 3/4 and back with fixed pose/camera/light/material/scale.
   No per-state height normalization or camera exaggeration is used.
4. Bundle `evaluate_runtime.mjs` with esbuild aliases pointing to the reviewed
   client's `shapeRuntime/format.ts` and `evaluate.ts`; it accepts runtime,
   controls JSON and float32 output paths. `verify_runtime.py` checks those
   exact TS outputs against Blender.
5. Run the existing `render_shape_catalog.py` on the two candidate rows.
   `finalize.py` binds checked runtime/PNG files, preserves original models and
   registries, and writes the review manifest with `activationAllowed=false`.
6. Bundle `viewer.mjs` against the client's locked Three.js, then run
   `package_review.py --release ... --viewer-js ...`. Serve `release/review`
   on loopback. It supports old editor/latest BASE/common fallback choices,
   six poses, synchronized rotation and fixed metric scale.

The live viewer was opened and checked directly. Initial camera clipping was
fixed; subsequent screenshots show the full models, including female lowered
arms and male constrained hands. Local `preservation.json` records byte-identical
copies of old defaults, original catalogs/registries, confirmation record and
source hash rechecks. Local `model-status.md` records remaining integration work.
