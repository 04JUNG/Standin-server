"""Local-only known CS3C humanoid mapping; no server/frozen solver edits."""
from converter import bone_map


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
