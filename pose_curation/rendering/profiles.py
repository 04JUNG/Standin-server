"""Offline rig adapters; production converter profiles stay unchanged."""

# Chest3 remains in the imported hierarchy, so its transform contributes to
# Chest4. Map the top spine to Chest4, where the neck and clavicles branch.
STYLE100 = {
    "hips": "Hips", "spine": "Chest", "spine1": "Chest2", "spine2": "Chest4",
    "neck": "Neck", "head": "Head",
}
for side, prefix in (("L", "Left"), ("R", "Right")):
    for canonical, source in (
        ("shoulder", "Collar"), ("upperarm", "Shoulder"), ("forearm", "Elbow"),
        ("hand", "Wrist"), ("upleg", "Hip"), ("leg", "Knee"),
        ("foot", "Ankle"), ("toe", "Toe"),
    ):
        STYLE100[f"{canonical}.{side}"] = prefix + source


def register() -> None:
    """Register only inside the dedicated Blender preview process."""
    from converter.bone_map import PROFILES
    PROFILES["100style"] = dict(STYLE100)
