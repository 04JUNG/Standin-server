# Split boundary checks

The experimental solver inventories outer edges on the triangulated rest mesh. It records a structural correspondence only for oppositely oriented coincident edges with distinct vertex IDs, identical weighted bone IDs on both sides, opposing interior directions, unique partners and at least two consecutive matched edges. Different weights, one-off coincidence, same-side sheets and multiply matched positions do not authorize joining geometry. Exact coordinates and exact influence signatures make the check conservative. Near seams and independently transformed mesh objects are not covered.

For each matched vertex pair, the audit reports the input and output distance. If a candidate opens a pair relative to the same pre-step pose, the gate returns a copy of the pre-step coordinates. Previously open gaps are reported without inferring author intent or generating new topology. Matching identifies candidates for inspection; it does not weld boundaries, transfer weights or make disconnected components interchangeable for isolated-influence detection.

All six approved staged outputs and the additional jump pose had zero outer edges on their supplied triangulations. These seven results cannot validate gap repair on real split assets. They also do not assess internal nonmanifold edges, missing faces or self-intersection.

A new synthetic three-bone FBX includes a duplicated polyline across its sheet. After actual export/reimport its topology contains104 boundary edges and the check matches13 vertex pairs. The pairs have zero rest and posed gap. The solver returns the original pose, with the split preserved. In a separate injected positional trial, a displacement of2mm on one side opens all13 pairs. The gate detects all13 and restores the exact input coordinates. A mismatched-weight control and stacked same-side triangles are excluded. This verifies detection and rollback of a postprocessing regression with a known reference; it does not establish safe reconstruction of a gap already present in a user's input.

No production entry point, runtime, release flag or deployment is changed. This detector does not resolve the female CMU peak stretch.
