# Geometry acceptance on 2026-11-06

The personal-body runtime remains v10, with manifest eaac3841c8b9e693ff88c959eabaceaf1ecc298e547eede748e83ad4a0de1d70. The experimental solver and its scipy dependency live under scripts/local_converter/experimental, outside the converter tree copied by the runtime packager.

The experimental acceptance check examines pairs of nonadjacent triangles in a single mesh. To accept a displacement it must find a separating axis for all six endpoint positions of each potentially overlapping pair over the straight path from the original pose. Uncertain contact and exhausted pair budgets cause fallback. It can conservatively reject existing contact or close motion. Shared-vertex pairs, other objects, and curved playback between poses remain outside its coverage.

Open and nonmanifold boundaries, authored anchors and vertices with identical rest positions are pinned. This avoids guessing whether a garment, lip or ornament should be joined. Intentional seam correspondence and gap repair still require separate evidence.

The female CMU candidate that previously reached 1.59x on one edge failed separation at faces 17674 and 85659. The 185.27x constrained diagnostic failed at faces 13040 and 79976. These observations identify unresolved pairs, not confirmed new penetrations.

The revised policy accepted zero corrections on the six existing poses. It left the original v10 geometry intact in all six. An additional jump-split BVH on the approved female rig produced a maximum of 200.96x and also fell back. An independent synthetic curved grid with an imposed asymmetric stretch remained at 3.8x at both tested scales. A new independent real character rig remains unverified. The extra BVH has no artist-authored target or calibrated reference camera, so it checks geometric behavior only.

Source registration, retarget output, pose-specific postprocessing and final export have distinct identifiers in the shared handoff. These solver tests address post-retarget geometry. The previously reported CSP exit on importing a v10 male FBX belongs to native compatibility investigation.

Quality qualification failed. No new runtime was built. All measured inputs returned their prior geometry, so the experiment offers conservative rejection rather than a qualified repair.
