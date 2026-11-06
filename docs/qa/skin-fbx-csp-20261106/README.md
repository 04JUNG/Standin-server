# Actual CSP skin / outline observations — 2026-11-06

These are crops of actual screenshots taken by the sole CSP operator, with labels and layout added. Model pixels were not repainted. `photo-evidence.json` records raw photo SHA256, crop rectangle and FBX SHA256; `validation.json` records the exact tested file hashes. Raw photos, models, UV data and original texture images remain in ignored local artifacts.

CSP EX 5.1.1 build 202607291835, fresh 1000×768 / 72dpi / 100% test canvases, automatic 3D import, no manual texture locator or companion PNG, no changes to CSP outline/culling/render settings. Exact Basic/Realistic mode was not read. Each imported source directory still contained only its FBX after import. This establishes single-file import; it does not locate CSP's internal extraction cache.

- `csp-skins.png`: male/female × four nondefault styles, skin-only. All eight combined variants were also imported separately: 16/16 neutral imports passed.
- `csp-outline.png`: the same flesh-guide skin with logical outline choice 0 versus 2. CSP can still draw its own contour at choice 0. Choice 2 preserves the warm body and black outline; the opaque black-body failure is absent.
- `csp-posed.png`: existing approved crouch source (left) and flesh-guide + outline 2 (right), male and female. Bent arms/knees and body posture match the sources. Only these two representative posed styles were visually imported; all eight posed styles passed file verification.

Existing CSP lighting produces strong glossy highlights; guide/muscle lines are faint in highlighted areas. These screenshots are evidence of the observed result, not a claim of pixel identity with the product viewer or all camera angles. A separate material diagnostic is not part of these accepted file hashes. No `.clip` export or CSP joint editing was tested or required.
