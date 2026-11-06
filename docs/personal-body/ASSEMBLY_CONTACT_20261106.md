# Contact checks across a rigged file

The experimental single-mesh guard now also checks moving faces that share one vertex or an edge. For a vertex, it looks for a plane through the moving common point that keeps the remaining vertices strictly on opposite sides over the full linear step. For an edge it bounds signed quadratic cross products over the interval. Inconclusive intervals can be subdivided; exhausting subdivision or pair budgets rejects the step. Unchanged pairs are skipped. This is a conservative floating-point test with a separation margin, not an exact predicate for every mesh degeneracy.

The previously accepted female seated/CMU and child seated/CMU candidates passed this additional neighbor check. The female CMU candidate passed6801 vertex-only and1991 edge-neighbor pairs. Its peak remains198.63x. A synthetic crossing that appears only in the middle of a displacement is rejected.

An experimental assembly wrapper accepts corresponding rest and posed arrays in one world frame, pins coincident source coordinates across objects, and evaluates each mesh independently before checking cross-object swept contact. If the composed displacement fails, smaller fractions of the same original displacement may be certified. Every partial step repeats the local shape, collapse and contact checks. No safe meaningful step returns the input coordinates for the entire assembly. Shared geometric features can be recognized for contact tests without welding vertices or changing object membership. General semantic part matching and arbitrary animated playback remain unverified.

Four synthetic two-mesh FBXs use the same three-bone body and an accessory attached to the rig. Both objects were exported, reimported, posed and evaluated. Their body weights and target pose are identical. The source spike error against the independent clean reference was270.34mm.

| Accessory placement | Accepted body proposal fraction | Remaining spike error | Peak stretch |
| --- | ---: | ---: | ---: |
| Clear | 1 | 3.17mm | 1.345x |
| Across the spike's path | 1/8 | 236.94mm | 5.906x |
| At the nominal tip | 1 | 3.17mm | 1.345x |
| Already embedded in the spike | 0 | 270.34mm | 6.694x |

For the obstacle case, endpoints of the full body-only move each have zero detected strict interior intersections with the accessory. At fractions0.25 and0.5 there are respectively one and six. Continuous separation rejects this full move. The accepted shorter step has zero strict intersections at all five sampled positions and passes the continuous test. For the embedded case the baseline has four strict crossings. The whole file returns to the input pose; these pre-existing crossings remain. The nominal-tip case is interpreted using the actual reimported coordinates and is not proof about arbitrarily coincident surfaces.

After FBX export/reimport, all four diagnostic outputs contain exactly the original two mesh objects and three bones. Vertex weights, polygon connectivity, material assignments and full bone hierarchy match the actual input files. Both solid material property sets pass the existing comparison. These controls do not qualify a new runtime. Independent personal multi-object assets, inter-object semantic seams and end-to-end CSP compatibility still require verification.
