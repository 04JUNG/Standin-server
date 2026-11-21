"""Explicit, immutable presentation transform of the original Y-up BVH.

This is a rigid model rotation about target Hips, applied once after retarget.
It is not a camera/extrinsic matrix, a skeleton edit, or a source unit change.
"""
from converter.camera import validate_rotation
from converter.local.contract import LocalError, hash_value

CAMERA_VERSION = "personal-camera/1"
SOURCE_CAMERA = {"version": CAMERA_VERSION, "sourceBvhSha256": None, "rotation": None}


def validate_camera(value):
    if not isinstance(value, dict) or set(value) != {"version", "sourceBvhSha256", "rotation"}:
        raise LocalError("CAMERA_UNSUPPORTED", "Explicit personal camera contract required")
    if value["version"] != CAMERA_VERSION:
        raise LocalError("CAMERA_UNSUPPORTED", "Unknown personal camera contract")
    if value["rotation"] is not None:
        try:
            hash_value(value["sourceBvhSha256"])
            validate_rotation(value["rotation"])
        except (ValueError, LocalError):
            raise LocalError("CAMERA_UNSUPPORTED", "Invalid selected-candidate camera") from None
    elif value["sourceBvhSha256"] is not None:
        hash_value(value["sourceBvhSha256"])
    # Copy so later caller mutation cannot change a prepared identity.
    return {"version": CAMERA_VERSION, "sourceBvhSha256": value["sourceBvhSha256"],
            "rotation": None if value["rotation"] is None else [list(row) for row in value["rotation"]]}


def apply_camera(arm, meshes, value):
    from mathutils import Matrix
    from converter.camera import rotate_scene
    from converter.local.inspection import surface, require
    camera = validate_camera(value)
    if camera["rotation"] is None:
        return {"version": CAMERA_VERSION, "applied": False, "maxRigidSurfaceError": 0.0}
    from converter.bone_map import PROFILES, resolve_profile
    mapping = PROFILES[resolve_profile(list(arm.data.bones.keys()))]
    pivot = arm.matrix_world @ arm.pose.bones[mapping["hips"]].head
    basis = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    rotation = basis @ Matrix(camera["rotation"]) @ basis.transposed()
    # Compare each deformed vertex independently against the requested rigid transform.
    before = surface(meshes)
    rotate_scene(arm, camera["rotation"])
    after = surface(meshes)
    # surface() returns the same deterministic ordered world vertices for each named mesh.
    error = 0.0
    for name, points in before.items():
        require(name in after and len(points) == len(after[name]), "INVALID_SKIN", "Camera changed mesh topology")
        for point, actual in zip(points, after[name]):
            from mathutils import Vector
            expected = pivot + rotation @ (Vector(point) - pivot)
            error = max(error, (expected - Vector(actual)).length)
    require(error < 2e-5, "INVALID_SKIN", "Camera changed the posed character surface")
    return {"version": CAMERA_VERSION, "applied": True, "maxRigidSurfaceError": error}
