# Flesh base + mutually exclusive line layers

`flesh-muscle` extends the existing strict skin wire IDs. Existing default/gray-guide/
flesh/flesh-guide/muscle semantics are unchanged. The client composes a base color and
one line layer into an exact ID; it never asks the converter to guess a texture.

Current 20 anatomy bodies include both final and 2048px preview assets for the new
combination, sharing their already verified per-body UVs. Legacy bodies retain their
old supported list. Assets are generated from the existing authored muscle mask,
using the same coverage/ink color over the flesh base. Writer, geometry and bones are
unchanged. CSP pin `f4d37bf100dcdec1de5c115d06fa68b30d3213d864d95e540c2d6bbc3676d265`.

Regenerate the existing guide assets, then:

```bash
python scripts/skin_fbx/compose_flesh_muscle.py --assets "$ARTWORK" --preview
python scripts/package_composed_skin_assets.py --assets "$SKIN_ROOT" --artwork "$ARTWORK"
python scripts/sync_staging_appearance_pins.py --bff "$BFF" --client "$CLIENT"
```

Validation: 832 Python converter tests passed, 1 opt-in skipped (three Blender-only
modules were excluded from the plain-Python runner). Fresh Blender processes exported
and independently verified four skin + outline pairs: male/female bases and actual
slider-adjusted bent male/female bodies. Every reported protected geometry, weight,
normal and matrix error was zero. Both adjusted bodies were imported separately into
CSP; flesh, muscle lines and maximum outline were visibly present. This does not claim
joint editing, .clip roundtrip, or a staging deployment.

See [validation.json](validation.json) and [CSP observation](csp-flesh-muscle.png).

Follow-up staging asset bundle (uploaded, not activated):
`body-models/releases/anatomy-composed-skins-20261112-8149c1568539-converter/manifest.json`
SHA256 `8149c15685398ba279550a9887fa8d05b5208e26b60264e6fa55a6cc20a09771`.
