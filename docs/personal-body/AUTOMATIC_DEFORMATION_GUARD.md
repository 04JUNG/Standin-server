# Automatic deformation guard — experimental implementation

Dedicated worktree /private/tmp/automatic-deformation-guard, branch codex/automatic-deformation-guard, base 6a3296f55.

The numpy guard measures per-triangle singular strain, edge stretch/compression, relative area, signed hinge angles, authored geometric crease deviations, fixed positions and explicitly supplied seam pairs against the original posed baseline. It backtracks proposed movement and checks a quadratic signed projected-area polynomial along the whole linear step. This conservative trajectory check prevents triangle collapse during accepted steps; it does not establish surface injectivity or absence of self-intersection. Seam identification is not implemented. Caller-supplied pairs require prior semantic validation.

The independent diagnostic runner replays the 12 previous female CMU candidates. All 12 are rejected. No model weights, source geometry, topology or production entry points are edited by these experiments.

A second experiment constructs sparse numerical constraint derivatives and solves a bounded linear program for a constrained direction. It performs 12 steps on the same 697-vertex patch, then revalidates on the complete mesh. Max edge stretch changes from 198.628788 to 198.627565. The target edge changes from 94.660080 mm to 94.659498 mm. This is negligible improvement. Global sharp/folded/compressed counts remain 783/482/180. These accepted microscopic steps are diagnostic stagnation, not repair success.

Raw evidence:
- /private/tmp/automatic-deformation-guard-results/report.json
- /private/tmp/automatic-direction-results/report.json
- /private/tmp/automatic-direction-results/candidate.npz

Five unit checks pass: scale-invariant rigid motion, stretching rejection/backtracking, mid-step triangle collapse, explicit seam separation, input immutability. Smoke 54/54 passes.

Remaining work before production integration:
- repair direction and adaptive boundary must make substantive progress under constraints;
- identify safe seam correspondences and classify ambiguous boundaries;
- inspect nonmanifold and degenerate source topology with structured outcomes;
- surface crossing checks, authored detail/silhouette render validation;
- multiple models, pose continuity and export/reimport verification;
- runtime dependency/budget integration. Sparse optimizer is experimental and uses scipy's private numerical differentiation helper, not suitable as a shipped dependency contract.

Current thresholds (2x stretch, 0.2x compression, 60 degree bend and 10 degree crease tolerance) are initial diagnostic policy, not validated universal deformation limits. No general automatic-repair quality claim is supported by this experiment.

## Adaptive boundary investigation — 2026-11-06 KST

The initial guard incorrectly capped an authored deep crease using only the posed hinge angle and a generic 60 degree bound. Returning toward a deeper rest crease could conflict with that cap. The cap now includes the rest hinge magnitude. A synthetic 100 degree rest / 70 degree pose hinge regression verifies that recovery to 90 degrees is allowed and overshoot to 120 degrees is rejected.

The previous linear program often proposed a direction tangent to an active constraint. Nonlinear curvature then violated that constraint, forcing nearly zero backtracking steps. The experiment now requests a small inward margin proportional to the Jacobian row norm and trust region. An elastic linear program diagnoses incompatible inward constraints. Such infeasibility is specific to this linearization and margin; it is not proof that no nonlinear repair exists.

Automatic boundary adjustment expands two adjacency rings around diagnosed blocking vertices, respecting source anchors. Before correcting the crease rule, 697 → 732 → 737 free vertices left crease conflicts. After correction, 697 → 717 removed the initial stretch boundary conflict. With stricter strain bounds, 717 → 782 → 787 → 806 opened a usable direction. Initial seed is the previous female CMU diagnostic patch; global automatic seed detection for arbitrary models is not implemented by this experiment.

| Policy | Worst edge ratio | Target edge mm | New >2x edges | New sharp/folded | Maximum additional rest-crease error |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 | 198.628788 | 94.660080 | — | — | — |
| No new threshold violations, 717 vertices, 200 steps | 190.877340 | 90.965990 | 0 | 0 / 0 | Initial 10 degree tolerance applies |
| No new stretching beyond rest and no worsening existing stretch, 806 vertices, 200 steps | 194.594896 | 92.737657 | 0 | 0 / 0 | 9.188917 degrees |
| Same plus no additional rest-crease error | 198.615621 | 94.653805 | 0 | 0 / 0 | 0.00000493 degrees (numerical tolerance) |

Strict stretching bounds use max(1, original posed strain) for each edge and principal triangle stretch. Compressed edges may recover toward rest. Counts of increased lengths therefore do not equal counts of newly stretched edges. The 806-vertex result has zero newly stretched edges and zero already stretched edges worsened at tolerance 1e-7. It has 500 increased lengths in initially compressed edges.

The strict-stretch candidate reduces global sharp/folded counts 783/482 → 767/470 and >2x edges 1785 → 1779. Compression180 and collapsed1 persist. Max movement is 1.168125 mm and outside-support movement is exactly zero. Its 9.19 degree extra crease error prevents claiming exact shape preservation.

The strict-shape candidate stagnates after nine attempts. Its target length improvement is only 0.006275 mm and sharp/folded/compressed counts do not improve. The accepted position change is at most 0.003874 mm. This is not an adequate repair.

Strict-stretch output was exported and reimported in pinned Blender. Original weights, polygon topology and existing bone hierarchy compare exactly. Max position roundtrip error is 5.794234e-7 m. Matched front/back diagnostic renders use recomputed geometric normals for both columns. Visual differences are small; original appearance restoration and CSP acceptance are not established. Strict-shape candidate has array-level validation only, not a separate FBX roundtrip.

Six guard regression tests and smoke54/54 pass. Experimental sparse derivatives still use scipy private APIs; no runtime packaging claim is made. Full data are in adaptive-boundary-evidence-20261106.json. Raw geometry/FBX/renders remain under /private/tmp/adaptive-boundary-* outside Git.

The experiment demonstrates controlled, modest improvement with bounded shape tolerance. Simultaneous substantial repair and zero added crease error remains unresolved. Next solver work must address nonlinear constraint curvature and crease transport within the feasible set; expanding the patch alone has not established that outcome.
