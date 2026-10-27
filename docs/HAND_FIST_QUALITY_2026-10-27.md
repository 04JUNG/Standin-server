# Closed-fist quality verification — 2026-10-27

`fist` and `tremble_fist` previously folded and pinched on the registered default
male/female meshes. Their optional `closed_fist_v2` articulation profile now uses
coherent cumulative bending, explicit thumb opposition, volume-preserving skin
deformation and eight local smoothing iterations. Smoothing retains topology and
skin groups. Thumb opposition is tightened so the distal thumb lies alongside
the curled index rather than projecting away from the fist. The female rig also needs guarded middle/ring root correction: it
only runs when a root is more than 1.18 times the outer fingers' wrist distance.
Reconstructed chains retain their tips; local skin weights retain total mass.

The other 77 presets use the existing rotation path. When a fist is selected,
the mesh's armature bake uses volume preservation, including any other selected
hand. The audit below checks the other presets individually, without a fist on
the opposite side.

The library content version is `hand-library-v2`; schema version remains 1.
`hand-bake-v1` remains the response protocol for BFF compatibility. Deploy the
matching library, converter and thumbnail manifest together; the library SHA
invalidates older geometry/thumbnail caches. Only four fist-family PNGs change;
the other 154 thumbnails retain their original bytes.

## Verification

- Python suite: 460 passed, 1 skipped. Three modules needing Blender/mathutils
  are excluded from standalone Python; Blender integration is checked separately.
- Smoke suite: 54/54 passed; compileall and diff whitespace checks passed.
- Blender 5.2: six export/reimport cases across the two registered characters
  (both fists, both tremble fists, right fist with left unselected).
- Eight saved-FBX hand configurations: no capsule contacts and no noncoplanar
  nonadjacent triangle crossings detected within the same hand.
- Maximum non-hand vertex movement: 0.0066 mm. Maximum unselected-hand movement:
  0.00045 mm. Body-bone matrix error stays below 1e-5. Mesh vertex/face counts,
  skin-group names, total skin weights and input file bytes are preserved.
- Fist and tremble-fist exported meshes match to a micrometre on each character.
- Remaining 77 presets, male/female × left/right = 308 configurations: capsule
  and surface screening results match the previous implementation. Capsule
  contact candidates occur in 26 presets (25 exceed 3 mm of approximate capsule
  overlap); surface crossing candidates occur in 49. These include intended
  contact and fold creases, so they are review candidates rather than confirmed
  defects. Finger-heart, V sign, thumbs-up and call-me warrant visual follow-up.

Capsules approximate skin thickness. Triangle screening does not certify
containment or collisions with the body, clothing, props or another person.
These results cover the two registered character rigs in the tested posed-rest
input, not arbitrary rigs or poses. Diagnostic hand photos use a neutral spatial
crop to retain complete fingertips; the wrist cut is a display boundary.
Character/BVH binaries and generated FBX reviews are kept outside the repository.

## Reproduce with local assets

```sh
# Run once per character, using a body-only posed-rest FBX.
blender -b --factory-startup --python-exit-code 1 \
  --python tests/converter/verify_hand_fists.py -- \
  --base /path/posed-rest.fbx --out /tmp/fist-check --label female

# Audit the delivered geometry without applying another preset.
blender -b --factory-startup --python-exit-code 1 \
  --python scripts/audit_hand_library.py -- \
  --character /path/character.fbx --base /tmp/fist-check/female/fist.fbx \
  --out /tmp/fist-review --presets fist --as-is --render

# Independently screen the remaining 77 presets.
blender -b --factory-startup --python-exit-code 1 \
  --python scripts/audit_hand_library.py -- \
  --character /path/character.fbx --out /tmp/remaining-hands \
  --exclude-articulated
```

This change is prepared for review; it does not merge or deploy itself.
