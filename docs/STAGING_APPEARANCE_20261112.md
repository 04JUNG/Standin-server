# Current staging flesh skin and outline

The anatomy release now accepts solid flesh skin and full-body outline on all
20 current male0.8.0/female0.4.0 models. The exact model SHA, triangle indices,
weights, UV corners and texture bytes are pinned independently in Converter,
BFF and client. Skin-only and combined outline widths0–2 share the same binding;
the requested maximum is2. Default skin/outline0 retain their original path.

Manual shape changes retain the registered base character identity and bind the
actual controls, controller/blend hashes, body revision and final source FBX into
the operation. The solid flesh texture uses exact corner-center UVs. It follows
the deformed surface without texture stretching seams; topology and weights are
checked on every postprocess. No body points, bone transforms or weights are
changed by appearance processing.

Fresh Blender writer/verifier processes checked skin→outline on20 current
neutral bodies and2 actual slider-edited/bent bodies. All measured base body
errors are0. Actual CSP imports checked male/female neutral width2, a male custom
shape width1.2 and a female custom shape width2. Flesh and black silhouette
remained visible. No large black fills were observed. These four observations
do not claim visual inspection of every preset/pose/view. Exact hashes and
controls: [validation](qa/staging-appearance-20261112/validation.json).

New models have54bones and105430/139474vertices; the old52bone/200000triangle
client limits rejected them before display. Bounds now cover the exact pinned
models while still rejecting a mismatched bone count or topology.

Current support is solid flesh and full individual FBX. New anatomy guide or
muscle artwork needs a separate authoring review; old reviewed assets retain
their existing availability. Cropped skin and new-model cropped outline remain
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
