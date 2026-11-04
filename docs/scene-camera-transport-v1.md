# Final FBX camera transport

The scene compiler already preserves a `scene-camera-v1` recipe in the FBX. This
additive transport makes the verified final camera available without parsing FBX
in the BFF or inventing optics there.

- `/compose-scene` returns `X-Standin-Scene-Camera`, standard base64 of compact UTF-8
  JSON (maximum 8192 characters). Binary/ZIP bodies and existing hashes are unchanged.
- The v4 `/scene/build` JSON returns the same object under `camera`.
- Object: `{ version: "scene-camera-artifact-v1", artifact_sha256, recipe }`.
- `artifact_sha256` hashes the final **FBX**, including for ZIP responses. The
  recipe comes from the compiler's final round-trip verification, never raw layout.
- `recipe.source_dimensions`, projection, camera/focal point, perspective and
  import preconditions remain together. Coordinate units are y-up centimeters.

See `tests/fixtures/scene-camera-artifact-v1.json` for the same generated fixture
consumed by BFF/client contract tests. Real Blender round-trip verification checks
portrait, landscape and square camera projections. Synthetic transport tests do
not establish CSP import accuracy.

The BFF/client negotiate this additive metadata separately from basic scene save.
Old consumers must continue to receive their supported unavailable-camera shape.
Deploy the compatible BFF/client before relying on this metadata. A missing field
from an old worker/Converter remains explicitly unavailable; invalid present data
must fail integrity checks rather than disappear into that fallback.

## Remaining CSP boundary

FBX camera inclusion is not automatic CSP camera application. No verified native
CSP adapter is introduced here. Screen capture cropping also does not identify
the CSP document canvas, its zoom or viewport translation. Those remain explicit
gaps; do not mark automatic framing or CSP visual acceptance as verified.

The published CSP SDK describes filter plugins, not a verified 3D camera control
interface: https://www.clip-studio.com/clip_site/download/clipstudiopaint/cspplugin
The glTF importer explicitly does not apply cameras, so changing the export to
glTF is not a framing solution:
https://support.clip-studio.com/en-us/faq/articles/20220049

## Asset status correction (2026-10-29)

The earlier `scene-worker-staging.md` candidate-only description was an initial
local inventory, not the latest release status. Server PR #146 records the full
`scene-v4-20261029-staging-002` S3 upload and Linux worker startup checks. Obtain
its exact URI, manifest/archive hashes and review reference from the release
operator; this camera change neither publishes assets nor enables AWS flags.
