# Staging anatomy release — 2026-11-11

Approved source geometry: female v5.1 (`standin-female-v5.1-raised-shoulders.fbx`)
and male v5.4 (`standin-male-v5.4-soft-shoulders.fbx`). These precede the rejected
face revision. No face sculpting is included in this release.

## Identities and compatibility

| | Male | Female |
|---|---|---|
| New character | `male-base-v0-8-0` | `female-base-v0-4-0` |
| Controller rules | `standin-body-0.8.0` | `standin-female-body-0.4.0` |
| Full mesh vertices | 139,474 | 105,430 |
| Body controls | 33 | 39 |
| Preview vertices | 56,001 | 56,000 |

The approved 54-bone skeleton, skin weights and anatomical PSDs are retained.
Neutral body displacement fields are transferred from the prior shape authoring
sources while protecting the approved head, hands and feet. New defaults match
the selected geometry to less than 0.01 mm after floor alignment.

Old character IDs, hashes and shape sources remain available for saved bodies.
The new defaults have separate immutable identities. Previously selected bodies
are not silently rewritten. The old fixed defaults are labelled as previous
defaults when the new catalog is active.

`converter/shape_bake.py` accommodates the two driven shoulder helpers without
changing the frozen retarget solver: it transports the solved pose, restores
drivers, and solves upper-arm local rotations while keeping the requested world
directions. The approved shoulder/hip PSDs are then evaluated.

## Thumbnails and body preview

- All 158 hand thumbnails (79 presets, both sides) were rendered from the approved
  male v5.4 hand mesh. Preset angles and hand-library semantic identity remain intact.
- The thumbnail manifest revision changes the catalog ETag. Existing client URL
  contracts remain unchanged; per-image validators refresh changed PNGs.
- Both new body cards use newly rendered neutral PNGs. The BFF supplies verified
  PNG data URLs for the two defaults because the installed client's model-card
  `<img>` does not attach installation authentication headers.
- Both new bodies include SSR2 previews under the existing 32 MiB limit, bound
  to the exact controller and blend hashes used by FBX conversion.
- Static pose-library reference posters retain their existing frozen reference
  identity. They are distinct from the selected-body preview and exported model.

New-body slider ranges are deliberately narrower for nonlinear extremes:
`heads` 7.2–8.7, `physique` −0.6–0.6, and `muscle` 0–1.5. Male `muscle_bulk`
is 0–0.5; female `fat_gain` is 0–1. The existing older bodies retain their ranges.
Height/limb interactions were added to the sampled preview explicitly.

## Verification

Independent preview samples (seed 31337, 12 few-control edits, 8 basic-control
combinations, 4 combinations of all controls; maximum vertex error):

| | Default | Few controls | Basic controls | All controls, stress |
|---|---:|---:|---:|---:|
| Male | <0.001 mm | 0.20 mm | 0.95 mm | 12.50 mm |
| Female | <0.001 mm | 0.49 mm | 0.98 mm | 4.54 mm |

The preview is an approximation; FBX export evaluates the full controller.
The current client's actual SSR2 parser accepts both files.

Source QA covers six shape variants, overhead/seated poses and posed-FBX
round trips. Linux conversion QA additionally covers actual library poses,
explicit overhead/seated probes, hand application and all four source-rig
profiles (100STYLE, CMU, prefixed and unprefixed Mixamo). All 3,503 current
staging BVHs pass the frozen solver's required bone mapping checks. This is
structural coverage and representative conversion QA, not visual review of
every library pose.

Server contract checks: 42 passed. BFF targeted checks: 31 passed under the
actual ARM64 Linux runtime, plus body-selection tests under Windows. Existing
Linux-only absolute-path fixtures were verified under Linux.

## Deployment and reproduction

Runtime evidence and exact deployment/rollback snapshots live in
`data/diagnostics/staging-character-upgrade-20261111` in the main server checkout.
Do not commit these task definitions: they contain deployment environment values.

- Converter manifest: `body-models/releases/anatomy-20261111-bf9439491bef-converter/manifest.json`
- BFF manifest: `body-models/releases/anatomy-20261111-5ab242a0112a-bff/manifest.json`
- Inference image: `sha256:847da720c2c84e105773b18719fdc7225e0ecf09b0d005d242dafb76450fbcb3`
- Converter image: `sha256:1c4a508f4518a5108f35b762184ad5524a06c6d48f44aaa42ec3ddc3459b389b`
- BFF image: `sha256:7f3f848990c5a15db4aa940046a8fa5961f010dfb82485b542aa56c570d5b078`

The converter initializer's aggregate limit is 512 MiB for the 425 MiB release;
its 300 MiB per-file limit, path allowlist and per-file hash checks are unchanged.
The BFF initializer retains its 300 MiB limit. Asset filenames use hyphens,
including version separators, to match the existing initializer path rules.
The concurrent BFF revision 120 quota-exemption change was preserved.

The BFF task role required an additional `s3:GetObject` grant for the single new
BFF release prefix. The user explicitly approved this scope on 2026-11-11.
Policy `StagingAnatomyAssets20261111` grants no write/delete access. Earlier BFF
121 startup attempts generated `body-assets-init=1` alerts due to AccessDenied;
the prior healthy BFF 120 remained running during those failed attempts.

Final rollout: inference 127, converter 42, BFF 121. BFF's retry after the approved
grant completed with zero failed tasks. Public `/healthz` returned 200.
An isolated task using the deployed BFF image, environment and asset initializer
verified the new catalog, both SSR2 files and hand revision, then called the live
converter with `shoulders=0.2`, `muscle=0.5`, left fist and right relaxed hand:

| Live export | FBX bytes | Preview bytes | Time |
|---|---:|---:|---:|
| Male | 9,051,084 | 161,372 | 79.653 s |
| Female | 6,707,644 | 161,707 | 50.718 s |

Both artifact integrity checks passed. The QA task emitted `LIVE_QA_PASS` and
both containers exited 0. Its log is `live-qa-bff.log` in the diagnostic folder.
Temporary registry credentials were removed. Completed sample caches and failed
local release attempts were removed, recovering approximately 1.4 GiB.

Reproduction scripts are in `scripts/character_upgrade`. Sampling runs inside
the pinned Blender. Export metadata with `--export-metadata`, then compress
completed samples with host NumPy using `--metadata`; Blender's bundled Windows
BLAS makes large matrix compression substantially slower. Run independent
checks and the installed client parser before publishing.
