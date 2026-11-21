# Isolated influence repair

This experimental branch handles a narrow source of skin damage. A severely stretched edge nominates its endpoints. A candidate vertex must have a bone weight at least five percentage points above that bone's weight on every one-ring neighbor, and that bone must also drive a connected edge on the same surface component. The ring must determine a non-collinear rigid transform from rest to posed geometry with maximum residual below 0.0001 of its radius. The rest vertex transformed through that fit supplies a positional target. The algorithm never reads the expected shape used for scoring, assigns model-specific vertex IDs, or changes the weights.

The original pose is used for every position, strain, crease, trajectory and swept-contact check. After a full step fails, shorter fractions are independently checked from that same original pose. A valid partial step can be refined with seven interval splits. Authored anchors, open edges, nonmanifold edges and coincident source vertices are protected. Missing a matching neighborhood or failing the checks returns the original or leaves the existing experimental fallback available. A ring fit alone cannot establish the artist's intent.

## Rigged positive and negative controls

Two independent rigs were written to FBX and reimported before pose sampling. A two-bone curved cylinder tests a single misplaced influence with substantial intentional distal motion. The second rig has three differently rotating bones, a relief surface with smooth weight transitions, partial influence at the damaged vertex, two solid materials and authored shading. Separate clean and damaged three-bone FBXs are evaluated at the same pose. Only the evaluator reads the clean reference. These synthetic inputs do not establish production coverage.

| Control | Largest positional error against the declared pose | Peak stretch | Outcome |
| --- | ---: | ---: | --- |
| Smooth cylinder | 180 mm → 0.000030 mm | 3.339x → 3.333x | One vertex repaired; legitimate extension elsewhere remains |
| Same cylinder, imported sharp anchors | 180 mm → 180 mm | 3.339x → 3.339x | Return original |
| Three-bone relief | 270.338 mm → 3.168 mm | 6.694x → 1.345x | One vertex repaired; 98.828125% of proposed displacement passed |
| Clean relief reference | zero applied movement | below2x | Return original |

The first cylinder run omitted imported sharp flags. The final flat/smooth controls use the actual imported sharp flags and mesh boundaries. The near-exact recovery applies only to the smooth case. The flat case has288 protected source vertices and declines the change.

For the relief rig, a full step failed the projected-area trajectory test. The accepted fraction passed from the original pose. New stretched or worsened stretched edges, sharp/folded hinges, severe compression and collapsed faces are all zero. Eight sharp hinges and both folded hinges were removed. The source rest mesh, original weights and intended motion of every other vertex were preserved. The diagnostic FBX is a posed rigged-rest output, consistent with the converter's output contract.

The final FBX was compared directly with the input FBX after reimport. Full bone names and parent relationships, original weights, polygon topology, two solid material property sets and per-polygon assignments matched. No bones were added. Position roundtrip error was at most0.000137mm. Source normals were updated locally;1974 corners outside the moved vertex's incident neighborhood changed by at most0.00550 degrees after re-encoding. The all-corner maximum, including the repaired area, was4.407 degrees. CSP was not tested.

## Approved real inputs

All three approved registered models supplied vertex weights by original group index. Separate weight files must match both the registered rest-position and triangle hashes. Across the original six outputs and the held-out jump pose there were zero eligible weight spikes among unprotected severe-edge endpoints under this rule. It does not address the female CMU 198.63x edge. The independent rig tests prove that a distinct subset can improve while keeping intended motion; they do not justify adding the solver to the production runtime.

The new selection thresholds are experiment choices. A deliberately animated isolated vertex may resemble an error. Larger connected weight-gradient regions, author-intent ambiguity and additional independent real assets remain unqualified. The production converter and immutable v10 runtime are not modified by this experiment.
