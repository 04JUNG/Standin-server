# V5 P0 data audit

Pinned dataset: Standin-master-docs@61eed393408552e1a0b28645f2e409be217bfcfb.

Use `python -m experiments.refine_v5.dataset --root PRIVATE_DIR --download`.
Native audit: Blender `--background --factory-startup --python experiments/refine_v5/audit_native.py -- PRIVATE_DIR`.

The downloader verifies Git blob hashes and records SHA256. Native importer reads
baked rest geometry as world-space joints/matrices, retaining import errors.
Neither file names nor model scores establish person identity, visibility, artist,
scene grouping, camera calibration or mesh safety. Missing seeds stay missing.
Splits remain unassigned until prior exposure/artist/scene review is completed;
known inspected groups 001/018/026/041/074/087 are development only. All persons
from one cut must stay together. A completed machine audit is not a certified
60/20/20 split. Raw images/FBX/checkpoints belong in private output, not Git.

Follow-up: use `curation` and `visual_review_v2.json` for the versioned agent review
and exploratory 60/20/20 cut split. Initial inventory splits remain historical.
Direct human coordinate labels and artist-disjoint certification remain absent.
