# Scene camera handoff (04J-289)

## Status

This is server-side camera metadata preparation, **not a completed CSP automatic
framing fix**. A multi-person FBX preserves relative geometry, but importing it
into CSP does not reproduce our render camera automatically in the two tested
cuts. The BFF and desktop client are unchanged by this change.

`scene-camera-v1` is generated from the same camera used for automatic rough
layout. It does not adjust individual characters, add framing geometry, or
change poses. It provides the source aspect ratio, optical FOV, and a proposed
CSP camera recipe. The recipe follows CELSYS's documented formula for the newer
3D format (v1.6.0 onward); applying the complete recipe in CSP is still unverified.

## Where consumers can read it

- Converter `/compose-scene`, ZIP response: `manifest.json` →
  `composition.framing`, only when `rough_layout` is supplied. Existing explicit
  placement requests without a camera retain their previous output.
- v4 `/scene/build`: `verification.results.final.verification.framing`
  (also present for the raw variant). This is an additive nested field; request
  contracts and existing response fields are unchanged.
- Embedded FBX camera object: JSON custom property `standin_framing_recipe` and
  `source_image_width` / `source_image_height`. The export path checks that these
  survive FBX reimport and agree with serialized horizontal optics.
- Raw FBX responses have no JSON sidecar. A consumer needing the recipe without
  parsing FBX should retain the ZIP manifest or v4 build response. Adding metadata
  does not cause CSP to select the embedded camera.

The three status fields intentionally distinguish data availability from UI
integration:

```json
{
  "requires_camera_application": true,
  "automatic_import_framing_verified": false,
  "csp_recipe_application_verified": false
}
```

Do not interpret `roundtrip_verified`, `embedded_camera`, a preview image, or a
successful download as confirmation of CSP canvas alignment.

## Coordinates and application prerequisites

The current layout camera is centered, perspective, front-facing in Blender
(+Y look direction, +Z up). Unsupported rotations, orthographic cameras, and
off-center principal points are rejected instead of emitting a misleading
recipe. The existing layout paths use this supported camera.

Blender metres `(x, y, z)` map to CSP centimetres `(100x, 100z, -100y)`.
The focal point is 100 cm ahead of the camera. Its distance is arbitrary for
this centered projection; its direction must be preserved.

For width `W`, height `H`, and horizontal FOV `h`:

```
tan(v / 2) = tan(h / 2) * H / W
CSP perspective = 100 * tan(v / 2) / tan(80 degrees)
```

Use full precision when possible; `perspective_ui_value` is a rounded convenience
value, not the basis of the mathematical verification. Consumers must validate
the destination version and its accepted numeric ranges before application.

A future CSP adapter must apply camera position, focal point, roll, perspective,
and source canvas aspect ratio together. Disable linked camera advance/retreat
and adapting the view to the selected object while applying the recipe. Ensure
the imported scene transform remains unchanged, then re-read the resulting
camera values and verify projected landmarks against the rough. This document
does not establish a supported CSP automation API or implement that adapter.

Sources: [CELSYS FOV formula](https://www.clip-studio.com/clip_site/support/help/detail/svc/53/tid/88435),
[CSP camera settings](https://help.clip-studio.com/en-us/manual_en/810_subtools/C.htm).

## Local evidence, 2026-10-28

The inputs were existing v4 automatically generated integrated FBXs, not new
staging inference. They were re-exported with this code and imported into fresh
rough image copies using CSP **File → Import → 3D Data**, default realistic
rendering. No character or camera fitting was performed by hand.

| Check | p060 (690×1168) | p220 (690×1926) |
|---|---:|---:|
| People retained in one FBX / visible in CSP | 2 / 2 | 2 / 2 |
| Maximum geometry error after re-export | 0.00494 mm | 0.00486 mm |
| Server camera projection error | 0.000954 px | 0.000304 px |
| Computed CSP perspective | 10.86372 | 17.91397 |
| CSP default-import rough alignment | Failed | Failed |

Both recipes use camera `[0,150,1000]` and focal point `[0,150,900]` cm.
The CSP default import still showed perspective `4.72`; p060's foreground
character was not large enough, and p220's background character was too large.
The UI input attempt did not successfully apply the entire recipe, so no
post-application success is claimed. Literal Finder drag-and-drop, a deployed
app save, and Linux container execution were not validated locally.

The local comparison is `outputs/04j-289-csp-camera-recipe-20261028/index.html`
in the experiment workspace (not a repository artifact or a public URL).

## Reproducible checks

```sh
python -m pytest -q tests/converter/test_scene_camera.py \
  tests/converter/test_scene_contract.py tests/converter/test_scene_runner.py \
  tests/converter/test_rough_layout.py tests/test_scene_worker_artifact_contract.py

blender --background --factory-startup --python-exit-code 1 \
  --python tests/converter/verify_scene_preview_camera.py

# After generating the converter fixtures; CI uses those synthetic fixtures.
blender --background --python-exit-code 1 \
  --python tests/converter/verify_scene_roundtrip.py -- --out /tmp/scene-check
```

The projection check covers portrait, tall portrait, landscape and square
canvases. The scene roundtrip check covers one, two and three people plus rough
layout with camera metadata. These tests establish serialization and mathematical
projection, not CSP UI behavior. Keep this work in Draft until the CSP application
path and end-to-end import validation are agreed and completed.
