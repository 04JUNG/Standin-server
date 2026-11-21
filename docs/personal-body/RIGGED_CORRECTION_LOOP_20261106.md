# Rigged-file correction experiment

The solver now tries spatially separate severe edges, adapts its patch radius to local mesh spacing, and can restore feasibility after a tangent step. It compares accumulated damage above the 2x threshold as well as maximum stretch. Final file changes require a severe edge to leave the damage class or at least 1% peak relief. These are experimental selection criteria. Crossing them does not certify visual quality.

The default 30-second per-file search budget is soft. Finishing an in-flight operation, running the geometry checks and serializing results can extend elapsed time. The unbudgeted tangent control was stopped before completing its full set. A held-out pose took approximately153 seconds in that control.

The swept contact test can now certify shorter portions of a straight displacement separately. Each portion must have a separating axis over both endpoint vertex sets. Subdivision exhaustion rejects the pair. The first unresolved pair for the earlier 185.27x candidate already overlaps or touches within numerical tolerance at both endpoints. This observation does not quantify penetration severity. The 1.59x candidate still has a pair with separated endpoints and unproven clearance throughout motion. Both continue to be excluded.

The seated female pose showed a small improvement. The count of edges above2x fell from403 to399, new sharp and folded hinges were zero, and existing folded hinges fell from193 to190. Peak stretch remained26.64x. Maximum vertex movement was0.32mm. FBX export and reimport preserved vertex weights, polygon connectivity and existing bone hierarchy. Maximum reimport position error was0.000412mm. This diagnostic uses recomputed geometric normals. The change is too small to establish visual restoration.

The female CMU pose and held-out jump-split pose remained at198.63x and200.96x. Both failed to produce a materially improved accepted output within budget. Two independent synthetic two-bone FBXs were exported and reimported before applying bone motion. One has a stretch across a full ring, the other a localized displaced patch. Both retained the original output under final materiality checks. Their peaks were4.06x and3.06x. These controls exercise skinning, but do not stand in for independent production characters.

No geometry runtime is qualified. Near-contact and actual pre-existing intersections must be distinguished in subsequent experiments. Preserving the intended rig pose as well as skin shape is necessary when evaluating a candidate that reduces damage.
