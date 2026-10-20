"""Calibrated Mixamo core, preserving optional unweighted terminal nodes."""

CORE = {"Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"}
for side in ("Left", "Right"):
    CORE.update(
        side + s
        for s in (
            "Shoulder",
            "Arm",
            "ForeArm",
            "Hand",
            "UpLeg",
            "Leg",
            "Foot",
            "ToeBase",
        )
    )
    CORE.update(
        side + "Hand" + finger + str(i)
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
        for i in (1, 2, 3)
    )
TERMINAL = {
    "Head",
    *(side + "ToeBase" for side in ("Left", "Right")),
    *(
        side + "Hand" + finger + "3"
        for side in ("Left", "Right")
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
    ),
}


def validate_rig(arm, meshes):
    names = {b.name for b in arm.data.bones}
    required = {"mixamorig:" + n for n in CORE}
    if not required <= names:
        raise ValueError("uncalibrated scene rig: missing core")
    for name in names - required:
        bone = arm.data.bones[name]
        if (
            not name.endswith("_end")
            or name[:-4] not in {"mixamorig:" + n for n in TERMINAL}
            or bone.parent is None
            or bone.parent.name != name[:-4]
            or bone.children
        ):
            raise ValueError("uncalibrated scene rig: unexpected terminal")
        for mesh in meshes:
            group = mesh.vertex_groups.get(name)
            if group and any(
                g.group == group.index and g.weight > 1e-6
                for v in mesh.data.vertices
                for g in v.groups
            ):
                raise ValueError("weighted terminal cannot use core-only calibration")
    return names
