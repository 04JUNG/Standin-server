# Approved anatomy presets in staging

All nine male and eleven female catalog templates now derive from the approved
pre-face-change sources (`standin-body-0.8.0`, `standin-female-body-0.4.0`). This
includes the 8.3-, 7- and 6.4-head styles, muscular/bulk styles, and soft/fat styles.
The named head-count recipes use head_scale=1 so an additional shrink multiplier
does not contradict their labels. Existing default FBX identities are retained.

Each new neutral FBX is exported and reimported, checking vertices, all 54 bones,
joint positions and skin weights. Both raised-arm and seated poses exercise the
approved scapular helpers and transported corrective shape keys for every preset.
Thumbnails are rendered from those same sources and bind to the preset FBX hash.
The frozen retarget solver is unchanged.

Live preview uses one shared SSR2 runtime per gender, below the client's 32 MiB
limit. Its header names the exact Blender/controller source hashes used for export.
The shared basis is extended with measured residual directions and control-pair
corrections, without simplifying the approved export mesh.

## Validation and limits

Run the scripts under `scripts/character_upgrade` in this order:

1. `probe_presets.py` and `build_presets.py` for each gender using the immutable approved source directory.
2. `sample_runtime_pairs.py`, then `extend_preset_runtime.py` to build shared previews.
3. `sample_preset_controls.py` with calibration seed 271811 (12/8/4 samples), then independent seed 650913 (6/4/2 samples, `--suffix independent`).
4. `verify_preset_runtime.py --independent`, the installed client's actual SSR2 parser, and `render_preset_poses.py` for visual inspection.
5. `package_presets.py`, which refuses unverified source/preview identities or failed independent checks.

The neutral preview maximum error must be below 0.5 mm. Interactive preview is an
approximation: few/basic/all-control tests respectively require maximum errors
below 10/15/20 mm and 99.9th-percentile errors below 5/10/15 mm. Both measurements
are retained in the report. The percentile criterion avoids treating a few local
fold-guard vertices as an error across the whole body; the separate maximum cap
still limits those vertices. FBX exports use the full authoring geometry, not this
approximation. The large-fat male template is published up to 185 cm and 7.7 heads;
the wider simultaneous stature/proportion range failed the preview check.

## Deployment contract

`config/staging-anatomy-release.json` pins the converter manifest and all 20 preset
paths. Every staging converter deployment prepares those settings and verifies
them before ECS activation. The initializer verifies the manifest and every file;
only the aggregate converter budget increases to 768 MiB (300 MiB per file stays).
Normal ECS health rollback remains enabled. No production configuration changes.

The BFF has the matching release pin. Its analysis worker receives the identical
catalog in a catalog-only manifest under its existing read-only S3 prefix; it
does not need model binaries or additional IAM permissions. Thus newly captured
jobs use the same presets as the picker and export service.

Published historical asset IDs and recipes remain immutable. New pickers hide
superseded templates when their exact replacement exists. Bare inherited last-used
presets migrate with a selection revision increment. Explicit historical choices
and saved custom controls remain available, including the metadata needed to edit
saved cards. No user-authored measurements are silently reinterpreted.
