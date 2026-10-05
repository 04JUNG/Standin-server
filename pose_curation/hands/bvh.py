"""Add finger branches while preserving every original body transform."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from src.bvh import parse_bvh, channel_starts, fk, load_coco17
from .presets import FINGERS, FLEXION, finger_angles, finger_offsets


@dataclass
class Joint:
    name: str
    offset: list[float]
    channels: list[str]
    values: list[float]
    end: bool = False
    children: list["Joint"] = field(default_factory=list)


def _finger(side: str, name: str, length: float, preset: str) -> Joint:
    offsets = finger_offsets(name, side, length)
    joints = [Joint(f"{side}Hand{name}{n + 1}", offsets[n],
                    ["Xrotation", "Yrotation", "Zrotation"], finger_angles(preset, name, side, n))
              for n in range(3)]
    for parent, child in zip(joints, joints[1:]):
        parent.children.append(child)
    joints[-1].children.append(Joint(joints[-1].name + "_End", offsets[3], [], [], end=True))
    return joints[0]


def _serialize(root: Joint, frame_time: float) -> str:
    lines, values = ["HIERARCHY"], []

    def visit(joint: Joint, depth: int) -> None:
        indent = "  " * depth
        kind = "End Site" if joint.end else f"{'ROOT' if depth == 0 else 'JOINT'} {joint.name}"
        lines.extend([indent + kind, indent + "{", indent + "  OFFSET " + " ".join(f"{v:.9f}" for v in joint.offset)])
        if not joint.end:
            lines.append(indent + f"  CHANNELS {len(joint.channels)} " + " ".join(joint.channels))
            values.extend(joint.values)
        for child in joint.children:
            visit(child, depth + 1)
        lines.append(indent + "}")

    visit(root, 0)
    lines.extend(["MOTION", "Frames: 1", f"Frame Time: {frame_time:.9f}", " ".join(f"{v:.9f}" for v in values)])
    return "\n".join(lines) + "\n"


def augment(source: Path, destination: Path, *, left: str = "relaxed", right: str = "relaxed") -> dict:
    if left not in FLEXION or right not in FLEXION:
        raise ValueError("unknown hand preset")
    joints, frames = parse_bvh(str(source))
    if len(frames) != 1 or not np.isfinite(frames).all():
        raise ValueError("hand augmentation requires one finite BVH frame")
    starts = channel_starts(joints)
    nodes = [Joint(j[0], j[2].tolist(), list(j[3]), frames[0, start:start + len(j[3])].tolist(), bool(j[4]))
             for j, start in zip(joints, starts)]
    for node, joint in zip(nodes, joints):
        if joint[1] >= 0:
            nodes[joint[1]].children.append(node)
    by_name = {node.name: node for node in nodes}
    lengths = {}
    for side, preset in (("Left", left), ("Right", right)):
        wrist = by_name.get(side + "Wrist") or by_name.get(side + "Hand")
        if wrist is None or len(wrist.children) != 1 or not wrist.children[0].end:
            raise ValueError(f"expected unaugmented {side}Wrist/Hand with one End Site")
        tip = np.array(wrist.children[0].offset)
        length = float(np.linalg.norm(tip))
        if length <= 1e-6 or abs(tip[0]) / length < .99 or (tip[0] > 0) != (side == "Left"):
            raise ValueError("unsupported wrist rest direction or length")
        wrist.children = [_finger(side, finger, length, preset) for finger in FINGERS]
        lengths[side] = length
    match = re.search(r"Frame\s+Time:\s*([\d.eE+-]+)", source.read_text(encoding="utf-8-sig"))
    if match is None:
        raise ValueError("missing frame time")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_serialize(nodes[0], float(match.group(1))), encoding="utf-8", newline="\n")
    output, output_frames = parse_bvh(str(destination))
    if output_frames.shape != (1, frames.shape[1] + 90):
        raise ValueError("expected 30 added joints and 90 rotation channels")
    before, after = fk(joints, frames[0]), fk(output, output_frames[0])
    output_by_name = {j[0]: after[i] for i, j in enumerate(output)}
    for index, joint in enumerate(joints):
        if not joint[4] and not np.allclose(before[index], output_by_name[joint[0]], atol=1e-5, rtol=0):
            raise ValueError(f"hand augmentation moved body joint {joint[0]}")
    if not np.allclose(load_coco17(str(source))[0], load_coco17(str(destination))[0], atol=1e-5, rtol=0):
        raise ValueError("hand augmentation changed search keypoints")
    return {"added_joints": 30, "added_rotation_channels": 90, "left": left, "right": right,
            "hand_lengths": lengths, "body_fk_preserved": True, "source": "procedural_preset"}
