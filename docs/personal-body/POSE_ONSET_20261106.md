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

## Existing contacts and skinning transport

Before solving a region, the experimental code now pins the vertices of both faces in every detected existing non-neighbor contact touching that region. If the scan exceeds its pair budget the region is frozen. The final swept tests still run. This is conservative and leaves inter-object seams and shared-vertex triangle pairs outside verified coverage.

On the first 2x sample it found89 pairs and fixed79 previously movable vertices. Peak ratio fell from2.03294 to2.00544. No new sharp hinges, folds, compressed edges or newly stretched edges were detected. This is a small, incomplete repair. Applying the accepted offset at later poses using the original weighted bone matrices was explicitly tested. Forward skinning reproduced the observed positions of affected vertices to within20 micrometres. At the next sampled fraction (1%) the transported offset reduced3.15787 to3.12972 but violated local strain, crease and contact constraints. Every later sampled transport also failed. The offset was not adopted.

At the final female CMU pose, the worst edge itself has both endpoints in fixed contact triangles. Boundary radii of 1.2%, 2.4% and 4.8% of the mesh extent all give the same exact lower bound of198.63x under this fixed-contact policy. A full-mesh endpoint BVH diagnostic puts both endpoints in the same connected contact set of75 vertices, so even a common rigid transform of that set would preserve the edge length. This bound applies to the tested policy. It says nothing about the feasibility of other intersection-preserving deformations.

Four of six prior poses passed the prototype's partial-improvement criterion. The child CMU peak fell4.789 to4.674. Female CMU stayed at198.629 while >2x edges fell1785 to1783 and folded hinges482 to479. No new violations of the audited geometry thresholds occurred in the four accepted cases. The female CMU FBX roundtrip preserved weights, polygons and existing bone hierarchy, with max position error0.00058mm. The held-out jump and independent local-bump rig remained unresolved. No qualified geometry runtime is available.

## Separate CSP investigation

We read the existing six crash logs and parsed the original-rig and diagnostic FBXs offline. No new CSP imports were performed. All six logs show the same repeated CSP executable offset in the crashing main-thread stack. There are no symbols identifying the recursive function. In the source-orientation male and female FBXs, cluster-to-bone links, finite nonsingular bind matrices, uniqueness of normalized bone names and full vertex weight coverage passed the checks. Both contain empty clusters. The original-rig male file has a maximum of nine positive influences per vertex, versus eight for the female control. The two current male app outputs each have4690 vertices with more than eight influences; both current female outputs have zero. This is a discriminator for a future controlled test, not a confirmed importer limit or justification to truncate weights. CSP import compatibility and posed geometry remain separate acceptance gates.
