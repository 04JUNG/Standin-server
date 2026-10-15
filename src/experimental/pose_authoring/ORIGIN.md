# Solver provenance

Local reuse of `Standin-server/out/pose-library-generated-20261015/tools/ext/`
from the user's 2026-10-15 authoring batch, not a third-party package download.
Source files were copied mechanically with package imports relocated. Removed
the original `sys.path` mutation; `rig.py` reads `POSE_GENERATION_RIG` instead of
the batch-relative `tools/lib` path, which did not exist in the supplied checkout.
No ROM, collision, balance, IK, or audit threshold was changed.

Original file SHA-256 (before relocation):

- rig.py: cbb5c79758a299f8a4e212cb189b21f64e22670127a2e6744eec1d120490418c
- author.py: bb34c31bf2551602324cd90b9ef66918cff06e741205b2be6e383d5918c395e9
- pipeline.py: 3231adabad6e3a00d32f300fe2547ba39fe75203d02741a3525f5d9c64e6acd3
- anatomy.py: 7a13a2b8151b102b799dd9d7a86829bfdd7b2ac387668102b78f1f740d57120c
- audit.py: 099bb41b433d77963b4fa396ddba325b1e1b2db00b5c43f3f108f4763fa3ae36
- presets.py: 9a73460db5cb7075b68cf4f1c83723dfcf9894090dad82e5b1eee4ddc81c0809
- gen_bases.py: 00c1eae13aeb1cf3ab1d6dfcd3ce82d9584acf4880f4e47352d4eaee12640212
- gen_upper.py: eb650d85cf1308b3ce0c86f17a62933b0592d2fbdb227129f53b5933502a4662

The neutral skeleton/finger-preset asset is NOT embedded in this package. It must
be provisioned from the validated library and match SHA-256
`38eb05b7432893fefd4cf31cb6f5fb07e21009946541d5a9d48d19c949991ee1`.
Original batch attribution: Quaternius / Gonzalo Furnier, CC0-1.0,
https://quaternius.com/packs/universalanimationlibrary.html . Fixed bone lengths
and the captured fist are reused; source body motion is not copied.

The supplied catalog actually contains 38 bases and 138 upper actions. The
attached design's 37/125 counts describe an earlier batch. Some incompatible or
externally supported presets may be rejected by the single-person service.

`audit.py` only measures; the service enforces its numerical gates separately
in `src/natural_pose_worker.py::independent_reasons`. All saved-file gates must
pass. These are geometric proxies, not skin-mesh collision certification or
semantic accuracy. Human review remains mandatory.
