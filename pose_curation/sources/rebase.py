"""Change BVH rest axes without changing the measured motion's body FK.

ACCAD's Acclaim-derived BVHs contain non-T-pose offsets. Renaming those bones
alone is not a valid humanoid conversion. For every bone we choose a constant
rest-axis rotation B. Offsets become B_parent @ offset and world rotations
become R_world @ inverse(B); their product, and hence every joint position,
is unchanged. No inverse kinematics or generated query coordinates are used.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from pose_curation.authored import rotation_between
from pose_curation.hands.bvh import Joint, _serialize
from pose_curation.kinematics import anatomical_arm_frames
from src.bvh import channel_starts, fk, parse_bvh

RENAME = {"ToSpine": "Spine", "Spine": "Spine1", "Spine1": "Spine2"}


def rest_axes(joints):
    names = {j[0]: i for i, j in enumerate(joints)}
    matrices = []
    for index, j in enumerate(joints):
        children = [c for c in joints if c[1] == index]
        if not children:
            matrices.append(np.eye(3))
            continue
        name = j[0]
        if name in ("Hips", "Spine1"):
            left, right = (
                ("LeftUpLeg", "RightUpLeg")
                if name == "Hips"
                else ("LeftShoulder", "RightShoulder")
            )
            up = "ToSpine" if name == "Hips" else "Neck"
            x = joints[names[left]][2] - joints[names[right]][2]
            x = x / np.linalg.norm(x)
            y = joints[names[up]][2].copy()
            y -= x * (y @ x)
            y /= np.linalg.norm(y)
            matrices.append(np.column_stack((x, y, np.cross(x, y))).T)
        else:
            if any(part in name for part in ("Shoulder", "Arm", "Hand")):
                target = [1 if name.startswith("Left") else -1, 0, 0]
            elif any(part in name for part in ("Foot", "ToeBase")):
                target = [0, 0, 1]
            elif any(part in name for part in ("UpLeg", "Leg")):
                target = [0, -1, 0]
            else:
                target = [0, 1, 0]
            matrices.append(rotation_between(children[0][2], target))
    return matrices


def export(source, frame_index, destination, *, yaw=0.0, floor=0.0):
    joints, frames = parse_bvh(str(source))
    frame = frames[frame_index]
    starts = channel_starts(joints)
    bind = rest_axes(joints)
    old_world = {}
    new_world = {}
    nodes = []
    root_rotation = Rotation.from_euler("Y", yaw, degrees=True).as_matrix()
    old_positions = fk(joints, frame)
    names = {j[0]: i for i, j in enumerate(joints)}
    arm_frames = {}
    for side in ("Left", "Right"):
        shoulder, elbow, wrist = [
            old_positions[names[side + part]] for part in ("Arm", "ForeArm", "Hand")
        ]
        upper, lower = anatomical_arm_frames(
            root_rotation @ (elbow - shoulder),
            root_rotation @ (wrist - elbow),
            side,
            rotation_between,
            np.eye(3),
        )
        arm_frames.update(
            {side + "Arm": upper, side + "ForeArm": lower, side + "Hand": lower}
        )
    translation = np.array([-old_positions[0][0], -floor, -old_positions[0][2]])
    for index, (name, parent, offset, channels, end) in enumerate(joints):
        values = frame[starts[index] : starts[index] + len(channels)]
        axes = "".join(c[0] for c in channels if c.endswith("rotation"))
        angles = [v for c, v in zip(channels, values) if c.endswith("rotation")]
        local = (
            Rotation.from_euler(axes, angles, degrees=True).as_matrix()
            if axes
            else np.eye(3)
        )
        old_world[index] = (old_world[parent] if parent >= 0 else np.eye(3)) @ local
        new_world[index] = root_rotation @ old_world[index] @ bind[index].T
        if name in arm_frames:
            new_world[index] = arm_frames[name]
        rotation = (new_world[parent].T if parent >= 0 else np.eye(3)) @ new_world[
            index
        ]
        new_offset = bind[parent] @ offset if parent >= 0 else np.zeros(3)
        output_channels = (
            []
            if end
            else (["Xposition", "Yposition", "Zposition"] if parent < 0 else [])
            + ["Zrotation", "Xrotation", "Yrotation"]
        )
        output_values = (
            []
            if end
            else (
                (root_rotation @ (old_positions[0] + translation)).tolist()
                if parent < 0
                else []
            )
            + Rotation.from_matrix(rotation).as_euler("ZXY", degrees=True).tolist()
        )
        renamed = RENAME.get(name, name)
        nodes.append(
            Joint(
                renamed, new_offset.tolist(), output_channels, output_values, bool(end)
            )
        )
        if parent >= 0:
            nodes[parent].children.append(nodes[-1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_serialize(nodes[0], 1 / 30), encoding="utf-8")
    out, of = parse_bvh(str(destination))
    actual = fk(out, of[0])
    # Hand End Sites are virtual tips, replaced by the explicit finger rig.
    body_indices = [
        i for i, j in enumerate(joints) if j[0] not in ("LeftHand_End", "RightHand_End")
    ]
    error = max(
        np.linalg.norm(actual[i] - root_rotation @ (old_positions[i] + translation))
        for i in body_indices
    )
    if error > 1e-5:
        raise ValueError(f"rebased motion changed joint geometry: {error}")
    return {
        "rest_axes_rebased": True,
        "body_fk_max_error_cm": float(error),
        "source_frame_0based": frame_index,
        "elbow_frames_reconstructed_from_captured_bend_plane": True,
        "root_yaw_degrees": yaw,
        "floor_cm": floor,
    }
