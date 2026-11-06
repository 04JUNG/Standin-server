# Actual CSP skin / outline observations — 2026-11-06

The original three comparison sheets and `validation.json` below document the **initial glossy material**. The user requested removing its metallic highlights. The explicit matte follow-up has separate photos/hashes; see [before/after](csp-matte-before-after.png) and `matte-photo-evidence.json`. The corrected material changes only the new skin material, and leaves the body, UV, texture and outline intact.

These are crops of actual screenshots taken by the sole CSP operator, with labels and layout added. Model pixels were not repainted. `photo-evidence.json` records raw photo SHA256, crop rectangle and FBX SHA256; `validation.json` records the exact tested file hashes. Raw photos, models, UV data and original texture images remain in ignored local artifacts.

CSP EX 5.1.1 build 202607291835, fresh 1000×768 / 72dpi / 100% test canvases, automatic 3D import, no manual texture locator or companion PNG, no changes to CSP outline/culling/render settings. Exact Basic/Realistic mode was not read. Each imported source directory still contained only its FBX after import. This establishes single-file import; it does not locate CSP's internal extraction cache.

- `csp-skins.png`: male/female × four nondefault styles, skin-only. All eight combined variants were also imported separately: 16/16 neutral imports passed.
- `csp-outline.png`: the same flesh-guide skin with logical outline choice 0 versus 2. CSP can still draw its own contour at choice 0. Choice 2 preserves the warm body and black outline; the opaque black-body failure is absent.
- `csp-posed.png`: existing approved crouch source (left) and flesh-guide + outline 2 (right), male and female. Bent arms/knees and body posture match the sources. Only these two representative posed styles were visually imported; all eight posed styles passed file verification.

Existing CSP lighting produces strong glossy highlights; guide/muscle lines are faint in highlighted areas. These screenshots are evidence of the observed result, not a claim of pixel identity with the product viewer or all camera angles. A separate material diagnostic is not part of these accepted file hashes. No `.clip` export or CSP joint editing was tested or required.

## Current explicit matte acceptance

All18 new exact files passed actual CSP import: skin-only8 + combined8 + representative crouch flesh-guide combined2. Every row keeps visible skin color and matte shading, with guide/muscle lines where applicable, body not black, pose preserved and combined black contours. Male neutral combined and female posed combined also passed actual camera orbits. No manual sidecar, texture locator or settings change. Initial source folders remain FBX-only. The new fourteen-source actual converter HTTP run produced28 outputs and matched all18 observed CSP hashes.

- `csp-matte-before-after.png`: same male flesh-guide skin-only, original glossy versus explicit matte. The generated product matte bytes exactly match this observed diagnostic file.
- `csp-matte-skins.png`: current male/female ×4 nondefault styles.
- `csp-matte-posed.png`: original approved crouch sources versus current matte flesh-guide + outline2.
- `csp-matte-orbit.png`: representative actual camera changes without mirror-like bands.
- `matte-validation.json`: exact new material writer/runtime/source/output/UV/texture hashes,20 typed before/after comparisons (only new skin material changed),14 actual HTTP source cases and13 rejected synthetic binary mutations. `matte-photo-evidence.json` binds raw screenshot hashes/crop rectangles to final FBX bytes.

Original glossy16neutral/2posed/native evidence remains historical. Existing CSP lighting still shades the matte models; this is not an unlit rendering or pixel equivalence guarantee for all renderer modes, poses and camera angles. Full current approved models only. Product native save refresh is separately recorded by the sole client/CSP owner.
