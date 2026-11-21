"""Rigid presentation rotation, independent of the frozen retarget solver.

Matrices map original BVH Y-up coordinates into a fixed front camera frame.
They never reflect, scale, articulate or straighten a character.
"""
import math

CAMERA_VERSION = "candidate-camera-v1"


def validate_rotation(value):
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in value)):
        raise ValueError("camera_rotation must be a 3x3 matrix")
    if any(type(x) not in (int, float) or not math.isfinite(x) for row in value for x in row):
        raise ValueError("camera_rotation must contain finite numbers")
    for i in range(3):
        for j in range(3):
            if abs(sum(value[i][k] * value[j][k] for k in range(3)) - (i == j)) > 1e-6:
                raise ValueError("camera_rotation must be orthonormal")
    a, b, c = value
    determinant = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
                   + a[2]*(b[0]*c[1]-b[1]*c[0]))
    if abs(determinant - 1) > 1e-6:
        raise ValueError("camera_rotation must not reflect the character")
    return value


def rotate_scene(arm, matrix):
    """Rotate all character roots once about Hips, preserving hierarchy/skin."""
    import bpy
    from mathutils import Matrix
    from converter.bone_map import PROFILES, resolve_profile

    validate_rotation(matrix)
    basis = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    rotation = (basis @ Matrix(matrix) @ basis.transposed()).to_4x4()
    profile = resolve_profile(list(arm.data.bones.keys()))
    hips = arm.pose.bones[PROFILES[profile]["hips"]]
    pivot = arm.matrix_world @ hips.head
    transform = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot)
    # Snapshot first: changing a parent changes children's matrix_world too.
    roots = [(obj, obj.matrix_world.copy()) for obj in bpy.context.scene.objects
             if obj.parent is None and obj.type in {"ARMATURE", "MESH", "EMPTY"}]
    for obj, world in roots:
        obj.matrix_world = transform @ world
    bpy.context.view_layer.update()
