"""Transfer added BVH finger channels to the character, after body conversion.

The frozen body converter maps only its 22 core joints. This offline adapter
adds finger rotations without altering that converter or its safety policies.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix

from converter import retarget
from pose_curation.hands.presets import FINGERS
from src.bvh import parse_bvh, channel_starts

FINGER_NAMES = tuple(f"{side}Hand{finger}{joint}" for side in ("Left", "Right")
                     for finger in FINGERS for joint in (1, 2, 3))
# BVH +Y up / +Z forward -> Blender +Z up / -Y forward (same BVH importer).
TO_BLENDER = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))


def rest_bases(character: str) -> dict[str, Matrix]:
    """Capture local finger axes before the body converter rebases the bind pose."""
    retarget.reset_scene()
    arm, _ = retarget.import_character(character)
    bases = {}
    for name in FINGER_NAMES:
        bone = arm.data.bones.get("mixamorig:" + name)
        if bone is None:
            raise ValueError(f"character is missing finger bone {name}")
        bases[name] = (arm.matrix_world @ bone.matrix_local).to_3x3().normalized()
    return bases


def apply_channels(arm, bvh: str, bases: dict[str, Matrix]) -> dict:
    joints, frames = parse_bvh(bvh)
    starts = channel_starts(joints)
    rows = {j[0]: (j, start) for j, start in zip(joints, starts)}
    present = set(FINGER_NAMES) & set(rows)
    if not present:
        return {"applied_joints": 0, "source": "body_only"}
    if len(present) != 30 or len(frames) != 1:
        raise ValueError("expected all 30 finger joints in one frame")
    body_before = {bone.name: bone.matrix.copy() for bone in arm.pose.bones
                   if bone.name.removeprefix("mixamorig:") not in FINGER_NAMES}
    for name in FINGER_NAMES:
        joint, start = rows[name]
        if set(joint[3]) != {"Xrotation", "Yrotation", "Zrotation"} or len(joint[3]) != 3:
            raise ValueError("finger adapter expects three BVH rotation channels")
        rotation = Matrix.Identity(3)
        for channel, value in zip(joint[3], frames[0, start:start + 3]):
            rotation = rotation @ Matrix.Rotation(math.radians(float(value)), 3, channel[0])
        world_delta = TO_BLENDER @ rotation @ TO_BLENDER.transposed()
        basis = bases[name]
        # BVH channels rotate in the parent rest axes; Blender pose channels
        # rotate in each bone's local axes. Conjugation changes only that basis.
        local = basis.transposed() @ world_delta @ basis
        bone = arm.pose.bones.get("mixamorig:" + name)
        if bone is None:
            raise ValueError(f"converted character lost finger bone {name}")
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = local.to_quaternion()
    bpy.context.view_layer.update()
    for name, before in body_before.items():
        after = arm.pose.bones[name].matrix
        if max(abs(before[i][j] - after[i][j]) for i in range(4) for j in range(4)) > 1e-6:
            raise ValueError(f"finger transfer moved body bone {name}")
    for name in FINGER_NAMES:
        if not all(math.isfinite(value) for row in arm.pose.bones["mixamorig:" + name].matrix for value in row):
            raise ValueError(f"non-finite finger transform: {name}")
    return {"applied_joints": 30, "source": "bvh_channels",
            "body_bones_preserved": True, "finite_finger_joints": True}
