"""Bounded two-link arm adjustment. Source BVH and non-arm bones are preserved."""

import math
import bpy
from mathutils import Vector, Quaternion
from converter.scene_placement import hand_fit


def move(arm, side, target_wrist, rotation=Quaternion()):
    upper, fore, hand = [
        arm.pose.bones["mixamorig:" + side + n] for n in ("Arm", "ForeArm", "Hand")
    ]
    matrices = [arm.matrix_world @ bone.matrix for bone in (upper, fore, hand)]
    shoulder, elbow, wrist = [m.translation.copy() for m in matrices]
    a, b = (elbow - shoulder).length, (wrist - elbow).length
    if (target_wrist - shoulder).length > a + b - 0.004:
        target_wrist = shoulder + (target_wrist - shoulder).normalized() * (
            a + b - 0.004
        )
    displacement = (target_wrist - wrist).length
    if displacement > 0.18:
        raise ValueError("arm displacement exceeds 18cm")
    direction = (target_wrist - shoulder).normalized()
    distance = (target_wrist - shoulder).length
    if distance > a + b - 0.001 or distance < abs(a - b) + 0.001:
        raise ValueError("unreachable wrist")
    midpoint = shoulder + direction * (
        (a * a - b * b + distance * distance) / (2 * distance)
    )
    radius = math.sqrt(max(0, a * a - (midpoint - shoulder).length_squared))
    radial = (elbow - midpoint) - direction * (elbow - midpoint).dot(direction)
    radial = radial.normalized() * radius
    fit = hand_fit(arm, side)
    along = rotation @ fit["along"]
    choices = []
    for degrees in range(-100, 101, 2):
        candidate = midpoint + Quaternion(direction, math.radians(degrees)) @ radial
        bend = math.degrees(along.angle((target_wrist - candidate).normalized()))
        choices.append(
            (max(0, bend - 35) * 10 + (candidate - elbow).length * 10, candidate, bend)
        )
    _, new_elbow, bend = min(choices, key=lambda row: row[0])
    if bend > 42:
        raise ValueError("wrist bend exceeds 42deg")
    du = (elbow - shoulder).rotation_difference(new_elbow - shoulder)
    df = (wrist - elbow).rotation_difference(target_wrist - new_elbow)
    m = du.to_matrix().to_4x4() @ matrices[0]
    m.translation = shoulder
    upper.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    m = df.to_matrix().to_4x4() @ matrices[1]
    m.translation = new_elbow
    axis = (target_wrist - new_elbow).normalized()
    relative = rotation @ df.inverted()
    projection = axis * Vector(relative[1:]).dot(axis)
    twist = Quaternion((relative.w, *projection)).normalized()
    if min(twist.angle, 2 * math.pi - twist.angle) > math.radians(100):
        raise ValueError("forearm roll exceeds 100deg")
    m = twist.to_matrix().to_4x4() @ m
    m.translation = new_elbow
    fore.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    m = rotation.to_matrix().to_4x4() @ matrices[2]
    m.translation = target_wrist
    hand.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    return dict(displacement=displacement, wrist_bend=bend)
