"""Reviewed, bounded thumb correction for a content-bound authored grip.

This is not source restoration: the receipt records the small CMC adjustment.
Other digits, wrist, and body bones stay unchanged. Explicit hand selections
and scene hand defaults take precedence and never receive this correction.
"""

PROFILE = "authored_cylindrical_grip_v1"
ANGLES = {"left": (25, 15), "right": (20, 25)}


def bake(path, sides, *, selected=None, defaults=None, overrides=None):
    wanted = [
        side
        for side in sides
        if not (selected or {}).get(side) and not (defaults or {}).get(side)
    ]
    if not wanted:
        return dict(applied=False, profile=PROFILE, sides={})
    import math
    import bpy
    from mathutils import Matrix, Vector
    from converter import retarget as rt
    from converter.hand_geometry import hand_frame
    from converter.hands import _side_geometry

    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    before = {
        bone.name: bone.matrix.copy()
        for bone in arm.pose.bones
        if not any(
            bone.name.startswith("mixamorig:" + side.title() + "HandThumb")
            for side in wanted
        )
    }
    records = {}
    for side in wanted:
        if side not in ANGLES:
            raise ValueError("unknown thumb correction side")
        wrist, bases, _ = _side_geometry(arm, side, "mixamorig:")
        frame = hand_frame(side, wrist, bases)
        normal = (arm.matrix_world.to_3x3() @ Vector(frame.palm_normal)).normalized()
        thumb = arm.pose.bones["mixamorig:" + side.title() + "HandThumb1"]
        original = arm.matrix_world @ thumb.matrix
        direction = (arm.matrix_world.to_3x3() @ (thumb.tail - thumb.head)).normalized()
        outward = arm.matrix_world.to_3x3() @ Vector(bases["thumb"] - bases["index"])
        spread_axis = normal if normal.cross(direction).dot(outward) > 0 else -normal
        lift_axis = direction.cross(-normal).normalized()
        spread, lift = (overrides or {}).get(side, ANGLES[side])
        if any(
            type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 40
            for x in (spread, lift)
        ):
            raise ValueError("thumb correction exceeds bounded profile")
        rotation = Matrix.Rotation(
            math.radians(spread), 4, spread_axis
        ) @ Matrix.Rotation(math.radians(lift), 4, lift_axis)
        target = rotation @ original
        target.translation = original.translation
        thumb.matrix = arm.matrix_world.inverted() @ target
        bpy.context.view_layer.update()
        records[side] = dict(spread_deg=spread, lift_deg=lift)
    error = max(
        abs(arm.pose.bones[name].matrix[r][c] - matrix[r][c])
        for name, matrix in before.items()
        for r in range(4)
        for c in range(4)
    )
    if error > 1e-5:
        raise ValueError("thumb correction changed body or other digits")
    rt.apply_output_mode(arm, meshes, "rigged_rest")
    rt.export_fbx(str(path), embed_textures=False)
    return dict(
        applied=True,
        profile=PROFILE,
        sides=records,
        body_and_other_digits_matrix_max_error=error,
        source_bvh_unchanged=True,
        visual_approval=False,
    )
