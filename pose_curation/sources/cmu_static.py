"""Captured CMU body landmarks -> a static humanoid with anatomical rest axes.

Internal spine gauges are distributed; elbow roll follows the captured bend
plane. External landmarks are preserved and checked after BVH serialization.
CMU finger channels are not measured: they are replaced by explicit presets.
"""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .acclaim import Acclaim
from ..authored import rotation_between
from ..kinematics import anatomical_arm_frames
from ..hands.bvh import Joint, _serialize, augment
from src.bvh import parse_bvh, fk


def landmarks(motion, index):
    ends, rotations = motion.forward(index)
    points = {
        "Hips": ends["root"],
        "Spine1": ends["lowerback"],
        "Spine2": ends["upperback"],
        "Neck": ends["thorax"],
        "Head": ends["upperneck"],
        "Head_End": ends["head"],
    }
    points["Spine"] = points["Hips"] * 0.7 + points["Spine1"] * 0.3
    for side, p in [("Left", "l"), ("Right", "r")]:
        for suffix, source in [
            ("Shoulder", "thorax"),
            ("Arm", p + "clavicle"),
            ("ForeArm", p + "humerus"),
            ("Hand", p + "radius"),
            ("UpLeg", p + "hipjoint"),
            ("Leg", p + "femur"),
            ("Foot", p + "tibia"),
            ("ToeBase", p + "foot"),
            ("ToeBase_End", p + "toes"),
        ]:
            points[side + suffix] = ends[source]
    return points, rotations


def export(
    motion: Acclaim,
    index: int,
    destination: Path,
    *,
    yaw=0.0,
    floor=0.0,
    hands=("relaxed", "relaxed"),
):
    points, captured = landmarks(motion, index)
    root_rotation = Rotation.from_euler("Y", yaw, degrees=True).as_matrix()
    translation = np.array([-points["Hips"][0], -floor, -points["Hips"][2]])
    points = {name: root_rotation @ (p + translation) for name, p in points.items()}
    pelvis = root_rotation @ captured["root"]
    chest = root_rotation @ captured["thorax"]
    delta = Rotation.from_matrix(pelvis.T @ chest).as_rotvec()
    worlds = {
        "Hips": pelvis,
        "Neck": root_rotation @ captured["lowerneck"],
        "Head": root_rotation @ captured["upperneck"],
    }
    for name, fraction in [("Spine", 1 / 3), ("Spine1", 2 / 3), ("Spine2", 1.0)]:
        worlds[name] = pelvis @ Rotation.from_rotvec(delta * fraction).as_matrix()
    parents = {
        "Hips": None,
        "Spine": "Hips",
        "Spine1": "Spine",
        "Spine2": "Spine1",
        "Neck": "Spine2",
        "Head": "Neck",
        "Head_End": "Head",
    }
    for side, sign in [("Left", 1), ("Right", -1)]:
        p = lambda suffix: points[side + suffix]
        upper, lower = anatomical_arm_frames(
            p("ForeArm") - p("Arm"),
            p("Hand") - p("ForeArm"),
            side,
            rotation_between,
            rotation_between([sign, 0, 0], p("ForeArm") - p("Arm")),
        )
        worlds[side + "Shoulder"] = (
            rotation_between(chest @ np.array([sign, 0, 0]), p("Arm") - p("Shoulder"))
            @ chest
        )
        worlds[side + "Arm"] = upper
        worlds[side + "ForeArm"] = lower
        worlds[side + "Hand"] = lower
        points[side + "Hand_End"] = p("Hand") + lower @ np.array([sign * 16.0, 0, 0])
        # Femur/tibia axes transport the capture's rotation, rebased from its
        # non-vertical anatomical zero direction into canonical -Y.
        prefix = "l" if sign == 1 else "r"
        bones = {b.name: b for b in motion.bones}
        for suffix, bone_name in [("UpLeg", "femur"), ("Leg", "tibia")]:
            bone = bones[prefix + bone_name]
            worlds[side + suffix] = (
                root_rotation
                @ captured[prefix + bone_name]
                @ rotation_between([0, -1, 0], bone.vector)
            )
        toe = p("ToeBase") - p("Foot")
        toe /= np.linalg.norm(toe)
        side_axis = worlds[side + "Leg"][:, 0].copy()
        side_axis -= toe * (toe @ side_axis)
        side_axis /= np.linalg.norm(side_axis)
        worlds[side + "Foot"] = np.column_stack(
            (side_axis, np.cross(toe, side_axis), toe)
        )
        worlds[side + "ToeBase"] = worlds[side + "Foot"]
        for child, parent in [
            ("Shoulder", "Spine2"),
            ("Arm", side + "Shoulder"),
            ("ForeArm", side + "Arm"),
            ("Hand", side + "ForeArm"),
            ("Hand_End", side + "Hand"),
            ("UpLeg", "Hips"),
            ("Leg", side + "UpLeg"),
            ("Foot", side + "Leg"),
            ("ToeBase", side + "Foot"),
            ("ToeBase_End", side + "ToeBase"),
        ]:
            parents[side + child] = parent
    nodes = {}
    if any(
        not np.allclose(w.T @ w, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(w), 1.0)
        for w in worlds.values()
    ):
        raise ValueError("invalid anatomical rotation frame")
    for name, parent in parents.items():
        end = name.endswith("_End")
        world = worlds.get(name, worlds.get(parent))
        offset = (
            worlds[parent].T @ (points[name] - points[parent])
            if parent
            else np.zeros(3)
        )
        local = worlds[parent].T @ world if parent else world
        channels = (
            []
            if end
            else (["Xposition", "Yposition", "Zposition"] if not parent else [])
            + ["Zrotation", "Xrotation", "Yrotation"]
        )
        values = (
            []
            if end
            else (points[name].tolist() if not parent else [])
            + Rotation.from_matrix(local).as_euler("ZXY", degrees=True).tolist()
        )
        nodes[name] = Joint(name, offset.tolist(), channels, values, end)
        if parent:
            nodes[parent].children.append(nodes[name])
    body = destination.with_suffix(".body.bvh")
    body.parent.mkdir(parents=True, exist_ok=True)
    body.write_text(_serialize(nodes["Hips"], 1 / 120), encoding="utf-8")
    joints, frames = parse_bvh(str(body))
    actual = fk(joints, frames[0])
    error = max(np.linalg.norm(actual[i] - points[j[0]]) for i, j in enumerate(joints))
    if error > 1e-5:
        raise ValueError(f"CMU export changed capture landmarks: {error}")
    fingers = augment(body, destination, left=hands[0], right=hands[1])
    return {
        "body_fk_max_error_cm": float(error),
        "elbow_frames_reconstructed_from_captured_bend_plane": True,
        "distributed_spine": True,
        "root_yaw_degrees": yaw,
        "floor_cm": floor,
        "hand_augmentation": {**fingers, "captured_from_source": False},
        "wrist_orientation": "forearm-aligned synthetic neutral; not captured hand orientation",
    }
