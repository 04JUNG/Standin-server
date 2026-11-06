# V5 P0 machine audit

Base develop: 160f33d57dfcb9720f858066cdad587c0e76161b.
Worktree: /private/tmp/standin-refine-v5-20261106/p0.
Private data: /private/tmp/standin-refine-v5-20261106/data.

114 rough/GT pairs (100 cut IDs) downloaded from the pinned dataset and verified
by Git blob hash and SHA256. All 114 native FBXs imported and yielded finite,
nondegenerate native world geometry and torso-normalized labels. All have baked
rest poses and no embedded camera. 112 have 52 bones; 047.1 and 059 have 65.

The user confirmed original B0 records are unavailable. Rough-only real RTMPose
and geometric library Top1 replay reconstructed 95 seeds. 19 ambiguous/no-person
inputs remain unsupported; no arbitrary largest person was substituted. These
are diagnostic seeds, not production eligibility/ownership certificates. GT was
not opened by the seed selection program. Every input remains in pairing.json.

Files: inventory.json, native_audit.json, native_labels.json, pairing.json,
review_template.json under private data; all-seeds/seeds.json alongside it.
Content digests and counts are in evaluation/refine_v5/p0-summary.json.
No paid provider calls, deployments, or raw datasets committed.

Human artist/scene/exposure grouping, anatomical visibility and camera review
remain pending. Known exposed groups are development; all others are unassigned.
A certified 60/20/20 split and blind evaluation are not claimed. Those reviews
cannot be inferred from filenames or model confidence. Different numbering in
other master-docs datasets must not be joined without image hash equality.

P0 tests cover tampered/missing files, cut grouping, unassigned split, finite
geometry, ambiguous suffix mappings and native baked-position labels: 4 passed.
