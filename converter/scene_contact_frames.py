"""Anatomical palm frames and bounded hand rotation for opt-in contact repairs."""

from mathutils import Vector, Matrix
from converter.scene_placement import hand_fit


def rotate_palm(arm, side, goal):
    from converter.scene_alignment import _apply_arm
    from converter.hand_geometry import hand_frame
    from converter.hands import _side_geometry

    fit = hand_fit(arm, side.title())
    wrist, bases, _ = _side_geometry(arm, side, "mixamorig:")
    normal = (
        arm.matrix_world.to_3x3() @ Vector(hand_frame(side, wrist, bases).palm_normal)
    ).normalized()
    along = (fit["along"] - goal * fit["along"].dot(goal)).normalized()
    local = Matrix((fit["along"], normal, fit["along"].cross(normal))).transposed()
    target = Matrix((along, goal, along.cross(goal))).transposed()
    rotation = (target @ local.transposed()).to_quaternion()
    original = (
        arm.matrix_world @ arm.pose.bones["mixamorig:" + side.title() + "Hand"].matrix
    )
    return _apply_arm(arm, side.title(), fit, original, rotation)


def span(fit, axis, scale, padding=0.008):
    return (
        min((v - fit["center"]).dot(axis) for v in fit["surface"]) - padding * scale,
        max((v - fit["center"]).dot(axis) for v in fit["surface"]) + padding * scale,
    )


def palm(arm, side):
    from converter.hand_geometry import hand_frame
    from converter.hands import _side_geometry

    wrist, bases, _ = _side_geometry(arm, side.lower(), "mixamorig:")
    normal = (
        arm.matrix_world.to_3x3()
        @ Vector(hand_frame(side.lower(), wrist, bases).palm_normal)
    ).normalized()
    f = hand_fit(arm, side)
    w = arm.matrix_world @ arm.pose.bones["mixamorig:" + side + "Hand"].head
    c = (
        w
        + f["along"]
        * (
            arm.matrix_world @ arm.pose.bones["mixamorig:" + side + "HandMiddle1"].head
            - w
        ).length
        * 0.65
    )
    points = [
        v
        for v in f["surface"]
        if abs((v - c).dot(f["axis"])) < f["width"] * 0.3
        and abs((v - c).dot(f["along"])) < f["width"] * 0.25
    ]
    height = max((v - c).dot(normal) for v in points)
    return c + normal * height, normal, f
