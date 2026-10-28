# Female breast authoring candidate

This is a draft improvement based on the confirmed female 0.3.4 source. It is
not a production asset replacement. Main merge, S3 publication and deployment
are outside this change. Artist review and actual CSP import remain pending.

## Source and ownership

Server work starts at develop `612be944d387c3dd2543ee2eed6ae95183c101bb`.
The parent working checkout and the parallel hand-limit 0.3.5 candidate were not
used as authoring inputs or modified. The candidate rules are
`standin-female-body-0.3.4.breast6`, independent of that hand-limit version.

Pinned official inputs:

| Input | SHA-256 |
| --- | --- |
| Female 0.3.4 blend | `a3622a9d4ae7fa2e88aa3903db102a3f9c4660b3fb34ce376950e8cd2f2bae32` |
| Female 0.3.4 controller | `df981aedc3784d38e008f6ddb7c7108ce26d8a528ef31539facdabfe2937eb1f` |
| Registration shape-bakes.local.json | `12097a17190236f52c7c4871800c338cd23e3e69cb3b2e1e9eb6df720af7b618` |
| Registration body-models.local.json | `9a1233d6f635b3b982fd8ecd7291d9f5bfc9faff8efc925b2d2b093957380470` |
| Confirmed published asset manifest | `04778b9f947914315f0aa4d3f036dff1a017019e31baf5e9c8803a6d4ef2e2d3` |

The registration source blend and confirmed publication package are byte
identical. Confirmation/publication records establish the official authoring
baseline, not the currently active ECS/BFF catalog. Live activation was not
queried or changed. The existing blend's inherited lock contains older lineage
labels; validation uses the actual pinned source bytes and geometry hashes.

Only breast keys and their local composition/pose reference policy are owned
here. Basis, vertex order, 50,000 triangles, 24,994 vertices, UV state, 52-bone
rig, frozen rest, weights, materials, male model and hand limits are preserved.
Binary authoring assets and runtime packages remain outside Git; source
generation, source pins, small derived test samples and validation reports are
reviewable in the two draft PRs.

## Diagnosis and resulting behavior

The old controller already separates Small and Large. Small changes only Y,
up to 34.086 mm inward; it changes neither height nor width. The screenshot's
sunken appearance cannot therefore be diagnosed as a downward Z translation or
negative use of Large. The depth-only target leaves its lower footprint and
medial chest relationship while flattening the mound.

Small now contracts the mound's footprint and retains convex projection, with
compact front support and pinned axillary attachment. This intentionally
changes saved negative-size silhouettes. Basic zero and all old positive sizes
through +1 keep their meaning. Intermediate reductions are directly inspected;
no extra midpoint key is added unless a measured/visible interpolation failure
requires one.

The Extended target inflates the existing +1 envelope around its attachment
and rotates the added mass down by 16 degrees for lower-pole fullness. A broad
upper-pole fill gives a rounded transition from the chest, while a localized
cap reduction softens the anterior tip. The upper attachment and lateral root
are tapered. This is an authored target, not extrapolation of the old Large key.

| Key | Normal mapping |
| --- | --- |
| BODY_BreastSmall | max(0, -size) |
| BODY_BreastLarge | min(1, max(0, size)) |
| BODY_BreastExtended | max(0, size - 1) |
| BODY_BreastSmallReference | 0; legacy negative shape only during reference evaluation |

Size range is `[-1, 2]`, default `0`, catalog grid `.01`. Height, width,
projection and spacing retain their independent controls. Reference evaluation
first runs the inherited fat/muscle pipeline using the legacy negative target
or old maximum. The breast residual is then warped and locally guarded. Pose
surface correction uses the same reference before restoring the actual breast.
This avoids switching arm/back correction because breast geometry changed.
The reference is an explicitly versioned key; source hash/lock checks remain on.

## Reproduce and inspect

Use Blender 5.2.0, build `fbe6228777e7`, with auto-execution disabled. All output
destinations must be new. The builder checks input byte hashes before loading.

```sh
blender -b --disable-autoexec --python scripts/female_breast/build.py -- \
  --blend "$BREAST_PARENT_BLEND" --controller "$BREAST_PARENT_CONTROLLER" --out "$BREAST_CANDIDATE"
blender -b --disable-autoexec --python scripts/female_breast/verify.py -- \
  --candidate "$BREAST_CANDIDATE" --parent-blend "$BREAST_PARENT_BLEND" \
  --parent-controller "$BREAST_PARENT_CONTROLLER"
blender -b --disable-autoexec --python scripts/build_shape_release.py -- \
  --source "$BREAST_CANDIDATE/model" --out "$BREAST_CANDIDATE/release"
blender -b --disable-autoexec --python scripts/female_breast/render.py -- --candidate "$BREAST_CANDIDATE"
python scripts/female_breast/sheets.py --candidate "$BREAST_CANDIDATE"
```

Build the offline viewer JS using the client lockfile's Three.js and esbuild,
then run `package_preview.py --candidate ... --js ...`. It packages exact
evaluated float32 surfaces with lossless XOR/gzip, no generated imagery. Its
.25 sample selector is for inspection; the product runtime supports the .01
catalog grid. Source hashes are embedded in the viewer.

`qa.json` records a .01 path sweep, representative fat/muscle/thorax and breast
control combinations, T/arms-down/raised/bent surfaces, model invariants, and
fixed FBX reimport errors. Existing unrelated overlaps are recorded, not hidden
or relabelled as a collision-free body. `pose-stress.json` adds coupled extreme
body/pose cases. A broad-phase triangle overlap alone is not an artist verdict.

## Integration and handoff

The companion client change registers the new rule and composes the same local
residual using existing warp/fold code. Its generated runtime binds the exact
blend/controller SHA. The existing BFF and Converter read the range from the
catalog/registry; no hard-coded BFF maximum or API envelope change is needed.

Publish/activate neither asset nor runtime separately. A future reviewed release
must carry the new sources, all preset FBXs, shape-bakes registry, characters,
catalog controls and matching shapeRuntime/source metadata together. Old
character IDs remain bound to their old source and `[-1,1]` grid. Do not change
old immutable files or force old hashes onto the new source.

`migrate_breast_snapshot()` is explicit opt-in and checks the exact old
asset/rules/rig identity. It preserves numeric controls, including negative
values whose appearance intentionally improves. The normal loader continues to
reject old snapshots without this migration.

When combining hand or skin work later, start from these source hashes, require
the same topology/Basis/weights/rig, and port only the named breast keys and
controller mapping/composition policy. Rebuild a new combined version and
repeat source/runtime/FBX checks. Do not overwrite the other session's assets.
