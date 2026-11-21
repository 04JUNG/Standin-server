"""Local-only known CS3C humanoid mapping; no server/frozen solver edits."""
from converter import bone_map
from converter.local.contract import LocalError


def register():
    bone_map.PROFILES["cs3c_humanoid_v2"] = {
        "hips": "Hips", "spine": "Spine", "spine1": "Chest", "spine2": "UpperChest",
        "neck": "Neck", "head": "Head",
        **{role + "." + letter: side + source
           for side, letter in (("Left", "L"), ("Right", "R"))
           for role, source in (("shoulder", "Shoulder"), ("upperarm", "UpperArm"),
                                ("forearm", "LowerArm"), ("hand", "Hand"),
                                ("upleg", "UpperLeg"), ("leg", "LowerLeg"), ("foot", "Foot"))}
    }


def skin_probe_names(bone_names):
    """Resolve the same five independent FK probes for every known local rig."""
    register()
    try:
        table = bone_map.PROFILES[bone_map.resolve_profile(list(bone_names))]
    except ValueError:
        raise LocalError("UNSUPPORTED_RIG", "Unsupported CS3C humanoid joint mapping") from None
    roles = ("forearm.L", "forearm.R", "leg.L", "leg.R", "spine")
    names = set(bone_names)
    if any(table.get(role) not in names for role in roles):
        raise LocalError("UNSUPPORTED_RIG", "Required CS3C skin probe joints are missing")
    return tuple(table[role] for role in roles)
