"""Explicit mappings for the two public CC0 Quaternius standard rigs."""

BODY = {
    "Hips": ("DEF-hips", "pelvis"), "Spine": ("DEF-spine.001", "spine_01"),
    "Spine1": ("DEF-spine.002", "spine_02"), "Spine2": ("DEF-spine.003", "spine_03"),
    "Neck": ("DEF-neck", "neck_01"), "Head": ("DEF-head", "Head"),
}
for side, suffix in (("Left", "l"), ("Right", "r")):
    for target, first, second in (
        ("Shoulder", "shoulder", "clavicle"), ("Arm", "upper_arm", "upperarm"),
        ("ForeArm", "forearm", "lowerarm"), ("Hand", "hand", "hand"),
        ("UpLeg", "thigh", "thigh"), ("Leg", "shin", "calf"),
        ("Foot", "foot", "foot"), ("ToeBase", "toe", "ball"),
    ):
        BODY[side + target] = (f"DEF-{first}.{suffix.upper()}", f"{second}_{suffix}")
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for joint in (1, 2, 3):
            first = "thumb" if finger == "Thumb" else "f_" + finger.lower()
            BODY[f"{side}Hand{finger}{joint}"] = (
                f"DEF-{first}.{joint:02d}.{suffix.upper()}", f"{finger.lower()}_{joint:02d}_{suffix}")


def mapping(version: int) -> dict[str, str]:
    if version not in (1, 2):
        raise ValueError("unsupported Quaternius rig version")
    return {names[version - 1]: target for target, names in BODY.items()}
