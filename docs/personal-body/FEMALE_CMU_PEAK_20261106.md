# Female CMU pelvic edge

The registered female CMU pose has a 198.62885× peak after FBX reimport. The next most stretched staged edge is 154.96×. The reference edge measures 0.477 mm at rest. The acceptance gate requires relief on that same edge together with the whole-mesh shape and swept-contact checks.

| Experiment on the same pose | Result on the target | Reason for retaining the input |
|---|---:|---|
| Select serialized vertex moves from the previous 185.27× proposal | No feasible subset for a 1% peak reduction | The original or trial position at each of 824 variables could not satisfy the feature constraints and target together. |
| Select moves from the bone-gradient proposal | No feasible subset for a 1% peak reduction | The same test exhausted 705 binary variables. |
| Compensated twist of both thighs about the hip-to-knee axes | 219.90× raw skinning to 192.46× | The full two-angle domain has a lower bound of 191.92×, and the best sample introduced 167 severe edges. This is a raw-stage comparison. |
| Transport an intact early-pose patch using a rigid fit to its boundary | 2.03× | At the widest tested boundary the global peak was 2.72×, with 1,757 newly severe edges. |
| Local nonlinear shape fit with existing-contact witnesses | 193.00006× after FBX reimport | One new sharp hinge, 147 previously stretched edges worsened, and a staged endpoint audit found 22 newly crossing triangle pairs. The selected FBX was the original input. |
| Add separating planes for pairs measured on the preceding candidate | 193.00× before FBX export | The endpoint audit found 14 new proper intersections after the boundary and path constraints changed the displacement. |

The new intersection counts cover disjoint triangle pairs over the whole mesh. Each reported endpoint intersection was confirmed by two strict interior edge/face hits. Independently sampled intermediate states also showed new proper crossings. Static chance of contact alone was insufficient for adoption.

## Where the boundary constrains repair

The edge and face stretch bounds are convex. An edge contributes a ball and a triangular face contributes a bound on the largest singular value of its deformation matrix. A conic solver minimized the target ratio while keeping the exterior fixed, allowing compression and ignoring contact and hinge limits in this lower-bound experiment.

| Free vertices | Computed minimum ratio | Dual objective |
|---:|---:|---:|
| 369 | 112.18464 | 112.18464 |
| 1,342 | 83.79042 | 83.79041 |
| 7,548 | 59.09849 | 59.09820 |

These are numerical bounds for those specific boundaries on this pose. They explain why a proposal can shorten the selected edge in a small patch only by exceeding another local upper bound. Releasing exterior vertices implicated by the infeasibility dual expanded the initial 369-vertex patch to 2,242 vertices in 25 steps. A 2× target still returned primal infeasible for that support.

A wider 15,255-vertex patch achieved 2.00× on the target under edge and face stretch bounds and conservative compression bounds. The candidate's global peak was 38.71×. It passed the compression and collapse checks, but introduced 148 new folds above 90° and 384 new sharp hinges above 60°. The maximum movement was 55.08 mm. A linearized hinge energy reduced bend-limit violations from 643 to 373, with 169 crease violations remaining. Neither trial passed the final gate.

## Contact and pose alternatives

The interior of the peak contact component has 75 vertices and 149 proper intersecting triangle pairs. Free sliding of the intersection endpoints permitted first-order shortening at two derivative step sizes. Including adjacent faces and testing finite steps exposed bend, crease and swept intersection constraints. In the 193× trial the initial path increased one point-to-plane depth by about 0.00034 mm halfway through despite reducing that depth by the endpoint. Enforcing the initial rate and intermediate clearances certified that pair. Other pairs still failed.

Two rotation-center probes used original weights and bone matrices. A triangle-centroid quadrature approximation to rest-weight similarity produced a 226.03× peak. A separate pose-dependent center that sought agreement between bone transforms produced 249.23×. These pose-specific results ruled out these two proposals. The rest-weight experiment was motivated by [Le and Hodgins, Real-time Skeletal Skinning with Optimized Centers of Rotation](https://studios.disneyresearch.com/2016/07/11/real-time-skeletal-skinning-with-optimized-centers-of-rotation__trashed/).

All experiments run outside the packaged converter. The conic experiments used Clarabel 0.11.1 installed in a separate temporary Python path. The geometry runtime used for these captures was independently audited against all 6,591 listed files, with matching hashes and no unlisted files. Its manifest SHA-256 is eaac3841c8b9e693ff88c959eabaceaf1ecc298e547eede748e83ad4a0de1d70. No female CMU candidate qualifies for adoption.

## Final FBX decision

Re-evaluating hinge angles and restoring the patch reduced newly introduced 90° folds from 148 to 33, then 16 and finally one. The last staged candidate had 48 new sharp hinges above 60° and no newly severe edges. On the straight correction path, 485 proper triangle intersections were new at the endpoint and 1,095 distinct pairs appeared across four sampled fractions.

Reimport of this candidate measured **198.62885× to 1.99996×** on the original peak edge. The largest remaining edge was 38.30403×. The authored edge belongs to two original triangles. Polygon records and weights were preserved and vertices outside the patch had zero drift. However, the renderer replaced ten triangles with ten different triangles elsewhere. The final geometry gate selected the input FBX. This candidate is diagnostic only.

A separate path-polynomial experiment certified all 12 targeted existing contact pairs on one smaller correction, compared with 11 before. Whole-mesh validation still found another unresolved pair. Compensating thigh twist and knee swivel while fixing both hips and ankles also left a numerical lower bound above 170× on raw skinning with 30 mm maximum knee drift. These experiments support continued work on coupled shape and contact constraints, but do not establish a general repair. No geometry runtime is qualified.
