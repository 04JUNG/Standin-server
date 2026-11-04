# CSP shaped neutral export

`POST /csp-shape` is internal-only and disabled unless `CSP_SHAPE_ENABLED=true`.
It accepts exactly `character_id`, `character_sha256`, `shape` (`body-shape.v1`)
and `expected_shape_source` (rules, blend/controller SHA-256). The registered
character and shape-bake.v2 source must match; explicit shape/source binding is
required even for default controls. No request-selected paths or source code
are accepted. Approved files are checked before execution and inside Blender.

A fresh Blender process shares the converter's bounded priority slots and
request cancellation/deadline. It opens the approved source without automatic
script execution, applies requested controls through the pinned controller,
exports the neutral FBX, then builds a normalized VRM0 with all 52 humanoid
bones, four normalized skin influences and local above-hip weight support.
The same source/controls are independently recorded in the returned report.
Neutral glTF positions are checked against the actual source FBX within
0.015 mm. This does not simulate CSP angle limits or prove pose aesthetics.

Returns `{vrm_base64, report}`. The BFF verifies the artifact, character, shape,
source and each requested control before packaging it with the selected BVH.
The VRM is neutral; the user imports it and applies the separate BVH in CSP.
CSP joint locks are not embedded. Hands, camera and merged scene behavior are
outside this neutral-body endpoint. Existing frozen retarget code and ordinary
FBX output are unchanged.

The independent FBX inspector restores and verifies native bone frames before
recomputing the source surface. For controllers with coupled shoulder limits it
rebuilds the current pose signature, avoiding a second projection of the already
projected pose. It also rejects any bone-frame drift during surface correction;
the existing 0.01 mm surface bound is unchanged.
