# Female CMU pose-path and surface audit

For the same registered female FBX and CMU target, we interpolated local bone rotations from neutral by quaternion slerp and interpolated translation and scale linearly. This constructs a diagnostic path. BVH playback timing and camera observations are not available for these samples. Endpoints match the original neutral and independently recorded raw retarget result within 20 micrometres.

| Fraction of target pose | Original worst edge ratio | Edges above 2x |
| ---: | ---: | ---: |
| 0 | 1.00 | 0 |
| 0.001 | 1.19 | 0 |
| 0.005 | 2.03 | 1 |
| 0.05 | 12.43 | 27 |
| 0.5 | 116.87 | 566 |
| 1 | 219.90 | 1853 |

At neutral, raw solver, arm and baked stages, disjoint faces 17862 and 154904 are separated. After v10 surface correction both triangles have edge hits strictly inside the opposite face, with a 0.098409 mm intersection segment. This is a confirmed new crossing at that location. The chosen pairs did not overlap at the sampled intermediate skeletal poses; between-sample contact elsewhere is unverified.

In Blender BVH endpoint checks without shared-vertex neighbors, the baked pose has 1754 intersecting face pairs. At full v10 correction there are 616 added and 770 removed relative to baked. Even 1/16 of that displacement introduces 82 pairs and removes104. Pair counts refer to this concrete single mesh. They are not counts of visually distinct holes or a swept-motion guarantee.

A geometric diagnostic clipped correction at the vertices of all introduced pairs, rechecking all faces each iteration. After five clipping updates there were zero introduced endpoint pairs and 103 resolved pairs, with7192 vertices still moved. However, relative to baked it introduced190 sharp bends,121 folded hinges,132 edges above2x and23 compressed edges. Max stretch remained219.90x. This candidate fails qualification.

These tests separate two mechanisms: severe stretch already starts near the beginning of the constructed skeletal path, while at least one crossing originates later during surface correction. Diagnostic face indices are observations, not solver rules. No new product runtime was built.

At the first sampled 2x crossing (0.5% of the constructed pose), the automatic solver proposed2.03294 →1.99270x but rejected that proposal. Its first unresolved pair, faces17861 and86120, already crosses in the registered neutral mesh with a 0.04635 mm segment, and at the early pose with a 0.04669 mm segment. The search returned its original within the 30-second experimental budget. This is evidence that original self-contact can block even an otherwise useful early proposal. It does not justify allowing arbitrary growth of existing intersections.
