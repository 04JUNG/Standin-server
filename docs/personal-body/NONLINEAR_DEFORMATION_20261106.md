# Nonlinear constrained deformation experiment

Linear proposals were tangent to curved constraints. New iterations minimize the L1 correction needed to restore nonlinear feasibility before accepting a step. The original trajectory and full-mesh checks still apply.

A numerical clearance below 60 degrees now prevents solver tolerance from creating new sharp classifications. Earlier candidates crossed this threshold by up to 0.00000312 degrees and were not promoted. The final candidate was recomputed and independently audited.

Following the initial 1.59x candidate's direction on an automatically expanded 824-vertex patch gives max stretch 198.628788 → 185.267540 and target edge 94.660080 → 88.292540 mm. Max vertex motion is 3.976265 mm and outside-support movement is zero. The nonlinear iteration stalls at iteration35.

| Metric | Before | After | New |
| --- | ---: | ---: | ---: |
| Edges above2x | 1785 | 1768 | 0 |
| Sharp hinges | 783 | 744 | 0 |
| Folded hinges | 482 | 453 | 0 |
| Compressed edges | 180 | 180 | 0 |
| Collapsed faces | 1 | 1 | 0 |

Newly stretched edges and already stretched edges worsened both equal0. There are480 increased lengths in initially compressed edges. Additional original crease deviation is at most0.000000208 degrees. However,214 original creases remain more than10 degrees away from rest. Full authored shape restoration is not established.

The callable experimental core automatic_deformation.correct provides model-independent worst-stretch seeding, scale-relative geodesic patches, generated strain-reduction targets, boundary expansion, nonlinear correction and unresolved statuses. It uses no female-specific names or vertex IDs. Diagnostic scripts report the known female edge separately.

The same automatic policy with12 iterations and two boundary attempts was run on all six approved staged outputs. Only female CMU improved198.628788 →194.570540. Male seated had no severe-stretch trigger. Male CMU, female seated and both child poses had no accepted improvement. A longer automatic run before threshold-clearance correction reached187.620220 but produced numerical sharp threshold crossings and was not promoted.

This is an experimental callable core. Generic production completion remains unresolved. Severe-stretch seeding does not cover every deformation, semantic seam classification and self-intersections remain unchecked, and cross-model improvement is insufficient. Runtime packaging and performance validation remain pending. The numerical derivative uses a private scipy helper. Existing production entry points are not wired to this module.

Ten focused tests and smoke54/54 pass. Scalar evidence is in nonlinear-deformation-evidence-20261106.json. Private FBX and renders are under /private/tmp/nonlinear-verified-visuals. Matched renders recompute geometric normals in both columns and do not certify CSP appearance.

Actual FBX reimport preserves weights, polygon connectivity and existing bone hierarchy exactly. Maximum position roundtrip error is0.000579424 mm. Front/back closeups were generated; the front view shows a small local change, with substantial existing groin deformation still present.
