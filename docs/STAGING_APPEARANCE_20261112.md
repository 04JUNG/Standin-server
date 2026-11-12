# Current staging flesh skin and outline

The anatomy release now accepts all four nondefault skins (gray guide, flesh,
flesh guide, muscle lines) and full-body outline on all20 current
male0.8.0/female0.4.0 models. The exact model SHA, triangle indices,
weights, UV corners and texture bytes are pinned independently in Converter,
BFF and client. Flesh supports outline widths0–2; guides/muscle support0or2.
The requested maximum is2. Default skin/outline0 retain their original path.

Manual shape changes retain the registered base character identity and bind the
actual controls, controller/blend hashes, body revision and final source FBX into
the operation. Authored anatomical lines are freshly projected onto current base
rest surfaces and baked into exact UV corners. Presets and slider edits share
proven identical vertex/index/weight correspondence within each gender; UVs are
transferred by that identity rather than nearest-position guesses. Lines follow
the deformed surface; topology and weights are
checked on every postprocess. No body points, bone transforms or weights are
changed by appearance processing.

Fresh Blender writer/verifier processes checked the new flesh-guide UVs and
outline2 on all20 current neutral bodies, all four styles on both base bodies,
and all four styles on both actual slider-edited/bent bodies (36case pairs).
All measured base body errors are0. Actual CSP imports additionally checked a
male neutral flesh-guide, male custom muscle lines and female custom gray-guide,
each with outline2 in its own QA canvas. Lines and black silhouette remained
visible without large black fills. Earlier solid-flesh observations are retained
separately. These observations do not claim visual inspection of every
preset/pose/view. Exact hashes and
controls: [validation](qa/staging-appearance-20261112/validation.json).

New models have54bones and105430/139474vertices; the old52bone/200000triangle
client limits rejected them before display. Bounds now cover the exact pinned
models, up to16MiB compressed assets and64MiB UV inflation, while still rejecting
a mismatched bone count/topology. The198-entry live skin capability requires a
bounded128KiB contract rather than the previous64KiB (actual66066bytes).

Current support is all original skin choices and full individual FBX. Old reviewed
assets retain their existing availability. Cropped skin and new-model cropped outline remain
unadvertised. Arbitrary personal models do not inherit these pins.

Skin assets are installed from a new immutable staging body bundle under
`/app/body-staging/skin`. Staging deployment helpers explicitly set
`CONVERTER_SKIN_ENABLED=true`, `CONVERTER_OUTLINE_ENABLED=true`,
`SKIN_EXPORT_ENABLED=true`, and `OUTLINE_EXPORT_ENABLED=true`; their guards check
these settings and the bundle digest. Production defaults stay off.

Reproduction tools: `package_staging_skin_assets.py`,
`build_staging_appearance_shape_cases.py`, `verify_staging_appearance.py`,
`sync_staging_appearance_pins.py`. Private model files and generated FBX/UV/PNG
assets are distributed through the pinned private release rather than Git.
