# Pose-derived edge targets on the registered female rig

The experiment reconstructs each posed vertex from the authored bone weights and pose matrices. Reconstruction differs from the recorded skinning stage by at most 0.604 µm in CMU and 0.634 µm in the independent jump pose. Across the worst CMU edge, the mean linear transform applied to the rest edge contributes a 0.401 mm vector. The weight-gradient term contributes a 104.547 mm vector. These vectors describe raw skinning. The v10 output stage subsequently measures 198.63 times the rest length.

The probe selects the peak edge by measurement. It fits a displacement field with fixed patch boundaries, varying the geodesic radius, gradient cap and step strength. It consumes the original weights read-only. The following rows show the lowest target-edge ratio among 12 proposals per case, before checking the safety constraints.

| Pose | Original peak | Target edge after | Mesh peak after | Newly above 2× | Eligible proposals |
|---|---:|---:|---:|---:|---:|
| CMU | 198.63 | 1.25 | 38.72 | 368 | 0/12 |
| Held-out jump, same rig | 200.96 | 1.15 | 30.11 | 272 | 0/12 |
| Constructed 0.5% CMU pose | 2.03294 | 1.27537 | 1.91823 | 0 | 0/12 |

The early proposal introduced local strain and bend violations even though the global maximum fell below 2. Each failed proposal retains the original output.

## Controlled comparison at onset

At a radius of 0.012 body extents, the contact guard left 173 movable vertices. Each direction received up to 12 iterations and a 60-second solver budget under the same geometry and motion guards. The length-based direction reached 2.00544 from 2.03294 in 12 iterations. The bone-derived direction reached 2.02431 before its deadline and missed the minimum improvement threshold. The independent audit of the accepted length-based candidate found zero newly stretched edges and zero new threshold crossings for compression, sharpness, folding and collapse.

With the automatic boundary and a 90-second solver budget, the length-based method reached 1.99985 at onset. The single edge above 2× left that class. Maximum displacement was 0.02578 mm, with zero displacement outside the patch. The swept-contact and independent geometry audits passed at their configured tolerances. The elapsed-time limit was reached. The constructed onset is an interpolation of the original rig’s bone transforms. Actual final-pose repair still requires a different feasible correction. All scripts in this experiment are offline probes.
