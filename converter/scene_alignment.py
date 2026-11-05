"""Explicit scene-only grip correction, after the frozen body conversion.

Only registered recipes opt in. Shoulder/wrist positions, limb lengths and all
non-arm bones are preserved. A bounded elbow swivel prevents solving a shared
handle by sharply bending the wrists. This is not an automatic quality approval.
"""

from __future__ import annotations

import math


def _frames(arm):
    from converter.scene_placement import hand_fit

    fits = {side: hand_fit(arm, side) for side in ("Left", "Right")}
    matrices = {
        side: arm.matrix_world @ arm.pose.bones[f"mixamorig:{side}Hand"].matrix
        for side in fits
    }
    return fits, matrices


def _shaft_rotations(fits, matrices):
    from mathutils import Quaternion

    wrists = {side: value.translation for side, value in matrices.items()}
    rotations = {side: Quaternion() for side in fits}
    for _ in range(100):
        centers = {
            side: wrists[side] + rotations[side] @ (fit["center"] - wrists[side])
            for side, fit in fits.items()
        }
        if (centers["Left"] - centers["Right"]).length < 0.035:
            raise ValueError("two-hand shaft requires distinct grip positions")
        line = (centers["Left"] - centers["Right"]).normalized()
        errors = []
        for side, fit in fits.items():
            current = rotations[side] @ fit["axis"]
            goal = line if current.dot(line) > 0 else -line
            step = current.rotation_difference(goal)
            errors.append(step.angle)
            rotations[side] = Quaternion().slerp(step, 0.35) @ rotations[side]
        if max(errors) < 1e-6:
            break
    return rotations


def _rifle_rotations(arm, fits, matrices, *, mirror=False):
    from mathutils import Quaternion, Matrix

    wrists = {side: m.translation for side, m in matrices.items()}
    forearms = {
        side: (
            wrists[side]
            - arm.matrix_world @ arm.pose.bones[f"mixamorig:{side}ForeArm"].head
        ).normalized()
        for side in fits
    }
    up = (
        arm.pose.bones["mixamorig:Spine2"].head - arm.pose.bones["mixamorig:Hips"].head
    ).normalized()
    up = (arm.matrix_world.to_3x3() @ up).normalized()
    rotations = {side: Quaternion() for side in fits}
    primary, secondary = ("Left", "Right") if mirror else ("Right", "Left")
    right = fits[primary]
    sign = 1 if right["axis"].dot(up) >= 0 else -1
    top = (
        max((p - right["center"]).dot(right["axis"] * sign) for p in right["surface"])
        + 0.018
    )
    for _ in range(80):
        centers = {
            side: wrists[side] + rotations[side] @ (fit["center"] - wrists[side])
            for side, fit in fits.items()
        }
        start = centers[primary] + rotations[primary] @ right["axis"] * sign * top
        aim = (centers[secondary] - start).normalized()
        targets = {
            primary: (up - aim * up.dot(aim)).normalized() * sign,
            secondary: aim if fits[secondary]["axis"].dot(aim) > 0 else -aim,
        }
        for side, axis in targets.items():
            along = (forearms[side] - axis * forearms[side].dot(axis)).normalized()
            local = Matrix(
                (
                    fits[side]["axis"],
                    fits[side]["along"],
                    fits[side]["axis"].cross(fits[side]["along"]),
                )
            ).transposed()
            goal = Matrix((axis, along, axis.cross(along))).transposed()
            rotations[side] = rotations[side].slerp(
                (goal @ local.transposed()).to_quaternion(), 0.25
            )
    return rotations


def _apply_arm(arm, side, fit, original_hand, rotation):
    import bpy
    from mathutils import Quaternion, Vector

    upper = arm.pose.bones[f"mixamorig:{side}Arm"]
    fore = arm.pose.bones[f"mixamorig:{side}ForeArm"]
    hand = arm.pose.bones[f"mixamorig:{side}Hand"]
    shoulder = arm.matrix_world @ upper.head
    elbow = arm.matrix_world @ fore.head
    wrist = original_hand.translation
    old_upper, old_fore = (
        arm.matrix_world @ upper.matrix,
        arm.matrix_world @ fore.matrix,
    )
    reach_axis = (wrist - shoulder).normalized()
    center = shoulder + reach_axis * (elbow - shoulder).dot(reach_axis)
    radial = elbow - center
    palm_along = rotation @ fit["along"]
    before_bend = math.degrees(fit["along"].angle((wrist - elbow).normalized()))
    choices = []
    for degrees in range(-90, 91, 2):
        candidate = center + Quaternion(reach_axis, math.radians(degrees)) @ radial
        bend = math.degrees(palm_along.angle((wrist - candidate).normalized()))
        score = max(0, bend - 38) * 10 + abs(degrees) * 0.05
        choices.append((score, degrees, candidate, bend))
    _, swivel, target, bend = min(choices, key=lambda row: row[0])
    if (
        bend > 42
        or (target - elbow).length
        > ((elbow - shoulder).length + (wrist - elbow).length) * 0.6
    ):
        raise ValueError("grip correction exceeds the wrist/elbow review limit")
    du = (elbow - shoulder).rotation_difference(target - shoulder)
    df = (wrist - elbow).rotation_difference(wrist - target)
    m = du.to_matrix().to_4x4() @ old_upper
    m.translation = shoulder
    upper.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    m = df.to_matrix().to_4x4() @ old_fore
    m.translation = target
    fore.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    axis = (wrist - target).normalized()
    relative = rotation @ df.inverted()
    projection = axis * Vector(relative[1:]).dot(axis)
    twist = Quaternion((relative.w, *projection)).normalized()
    if math.degrees(twist.angle) > 100:
        raise ValueError("grip correction exceeds forearm roll limit")
    m = twist.to_matrix().to_4x4() @ m
    m.translation = target
    fore.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    m = rotation.to_matrix().to_4x4() @ original_hand
    m.translation = wrist
    hand.matrix = arm.matrix_world.inverted() @ m
    bpy.context.view_layer.update()
    position_error = max(
        (arm.matrix_world @ upper.head - shoulder).length,
        (arm.matrix_world @ hand.head - wrist).length,
        abs(
            (arm.matrix_world @ fore.head - shoulder).length - (elbow - shoulder).length
        ),
        abs(
            (arm.matrix_world @ hand.head - arm.matrix_world @ fore.head).length
            - (wrist - elbow).length
        ),
    )
    if position_error > 1e-5:
        raise ValueError("grip correction changed shoulder/wrist or arm length")
    return dict(
        wrist_before_deg=before_bend,
        wrist_after_deg=bend,
        elbow_swivel_deg=swivel,
        elbow_movement=(target - elbow).length,
        forearm_roll_deg=math.degrees(twist.angle),
        joint_length_max_error=position_error,
    )


def bake_alignment(
    path, policy, *, mirror=False, selected_hands=None, prepared_hand_defaults=None
):
    if not policy:
        return {"applied": False, "policy": "none"}
    if policy not in {"two_hand_shaft", "rifle_support"}:
        raise ValueError("unknown scene grip correction")
    from converter import retarget as rt

    presets = {}
    support_side = "right" if mirror else "left"
    if policy == "rifle_support" and not (selected_hands or {}).get(support_side):
        presets = {support_side: "hold_cup"}
        if (prepared_hand_defaults or {}).get(support_side) != "hold_cup":
            raise ValueError(
                "rifle support default must be prepared before source finger transfer"
            )
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    fits, matrices = _frames(arm)
    fingers_before = {
        bone.name: arm.pose.bones[f"mixamorig:{side}Hand"].matrix.inverted()
        @ bone.matrix
        for side in ("Left", "Right")
        for bone in arm.pose.bones
        if bone.name.startswith(f"mixamorig:{side}Hand")
        and bone.name != f"mixamorig:{side}Hand"
    }
    untouched = {
        bone.name: bone.matrix.copy()
        for bone in arm.pose.bones
        if not any(
            bone.name.startswith(f"mixamorig:{side}{part}")
            for side in ("Left", "Right")
            for part in ("Arm", "ForeArm", "Hand")
        )
    }
    rotations = (
        _shaft_rotations(fits, matrices)
        if policy == "two_hand_shaft"
        else _rifle_rotations(arm, fits, matrices, mirror=mirror)
    )
    diagnostics = {
        side: _apply_arm(arm, side, fits[side], matrices[side], rotations[side])
        for side in fits
    }
    error = max(
        abs(arm.pose.bones[name].matrix[r][c] - matrix[r][c])
        for name, matrix in untouched.items()
        for r in range(4)
        for c in range(4)
    )
    if error > 1e-6:
        raise ValueError("scene correction changed a torso, head or leg bone")
    finger_error = 0.0
    for name, matrix in fingers_before.items():
        side = "Left" if name.startswith("mixamorig:Left") else "Right"
        actual = (
            arm.pose.bones[f"mixamorig:{side}Hand"].matrix.inverted()
            @ arm.pose.bones[name].matrix
        )
        finger_error = max(
            finger_error,
            max(abs(actual[r][c] - matrix[r][c]) for r in range(4) for c in range(4)),
        )
    if finger_error > 1e-5:
        raise ValueError("arm alignment changed the selected local finger pose")
    rt.apply_output_mode(arm, meshes, "rigged_rest")
    rt.export_fbx(str(path), embed_textures=False)
    return dict(
        applied=True,
        policy=policy,
        hands=diagnostics,
        default_hand_presets=presets,
        local_finger_matrix_max_error=finger_error,
        torso_head_legs_matrix_max_error=error,
        source_bvh_unchanged=True,
        visual_approval=False,
    )
