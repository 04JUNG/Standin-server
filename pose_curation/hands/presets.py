"""Hand proportions and poses in a Y-up, palm-down T-rest coordinate system.

Lengths are fractions of the original wrist-to-hand-tip End Site. These are
anatomical defaults, not recovered motion-capture measurements. Each finger
has MCP/PIP/DIP joints; the thumb has CMC/MCP/IP joints.
"""
from __future__ import annotations

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
# Base position and three segment lengths, scaled by the source hand length.
GEOMETRY = {
    "Index": ((.43, 0, .12), (.19, .12, .10)),
    "Middle": ((.45, 0, .02), (.22, .16, .12)),
    "Ring": ((.43, 0, -.08), (.205, .14, .105)),
    "Pinky": ((.39, 0, -.18), (.145, .095, .075)),
}
FLEXION = {
    "open": {finger: (0, 0, 0) for finger in FINGERS},
    "relaxed": {"Thumb": (8, 12, 8), "Index": (12, 18, 10),
                "Middle": (15, 22, 12), "Ring": (18, 25, 15), "Pinky": (22, 28, 18)},
    "fist": {"Thumb": (25, 35, 25), "Index": (65, 85, 55),
             "Middle": (70, 85, 55), "Ring": (75, 85, 55), "Pinky": (80, 85, 55)},
}
THUMB_ADDUCTION = {"open": 0, "relaxed": 12, "fist": 35}


def finger_offsets(finger: str, side: str, length: float) -> list[list[float]]:
    """Four vectors: base, middle joint, distal joint, fingertip."""
    sign = 1 if side == "Left" else -1
    if finger == "Thumb":
        offsets = ((.14, -.025, .13), (.145, -.025, .13), (.12, 0, .09), (.10, 0, .05))
    else:
        base, segments = GEOMETRY[finger]
        offsets = (base, *((size, 0, 0) for size in segments))
    return [[sign * x * length, y * length, z * length] for x, y, z in offsets]


def finger_angles(preset: str, finger: str, side: str, joint: int) -> list[float]:
    """XYZ channel angles. Both hands curl toward -Y, with mirrored handedness."""
    sign = 1 if side == "Left" else -1
    adduction = sign * THUMB_ADDUCTION[preset] if finger == "Thumb" and joint == 0 else 0
    return [0, adduction, -sign * FLEXION[preset][finger][joint]]
