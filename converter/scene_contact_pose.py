"""Arm-only contact repairs with geometry checks and preservation receipts.

Searches are bounded and deterministic. A successful fit never grants approval.
Original BVH, torso, head, legs and hand-local finger transforms remain intact.
"""

import bpy
from mathutils import Vector
from converter import retarget as rt
from converter.scene_contact_ik import move
from converter.scene_contact_frames import rotate_palm
from converter.scene_contact_profiles import get_profile
from converter.hand_contact import screen_hands
from converter.scene_hand_body_review import screen as screen_body


def _snapshot(arm):
    return {b.name: b.matrix_basis.copy() for b in arm.pose.bones}


def _restore(arm, state):
    for bone in arm.pose.bones:
        bone.matrix_basis = state[bone.name]
    bpy.context.view_layer.update()


def _point(arm, name):
    return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head


def separate_hands(arm, meshes, lateral):
    if screen_hands(meshes)["status"] == "clear_screen":
        return []
    state = _snapshot(arm)
    wrists = {side: _point(arm, side + "Hand") for side in ("Left", "Right")}
    for distance in (0.015, 0.025, 0.035, 0.05, 0.07, 0.09):
        _restore(arm, state)
        try:
            diagnostics = [
                move(arm, side, wrists[side] + lateral * sign * distance)
                for side, sign in (("Left", 1), ("Right", -1))
            ]
        except ValueError:
            continue
        if screen_hands(meshes)["status"] == "clear_screen":
            return diagnostics
    raise ValueError("no bounded opposite-hand correction")


def clear_body(arm, meshes, forward, lateral, original_wrists):
    result = screen_body(meshes)
    if result["status"] == "clear_screen":
        return []
    if result["status"] == "unavailable":
        raise ValueError("hand/body skin screen unavailable")
    affected = [
        s
        for s in ("Left", "Right")
        if any(k.startswith(s + "Hand") for k in result["regions"])
    ]
    state = _snapshot(arm)
    wrists = {s: _point(arm, s + "Hand") for s in affected}
    up = Vector((0, 0, 1))
    for direction in (forward, up, forward + up, lateral, -lateral, "outward"):
        for distance in (0.02, 0.04, 0.06, 0.09, 0.12):
            _restore(arm, state)
            try:
                diagnostics = []
                for side in affected:
                    delta = (
                        lateral * (1 if side == "Left" else -1) + up * 0.5
                        if isinstance(direction, str)
                        else direction
                    )
                    diagnostics.append(
                        move(arm, side, wrists[side] + delta.normalized() * distance)
                    )
            except ValueError:
                continue
            if (
                max(
                    (_point(arm, s + "Hand") - w).length
                    for s, w in original_wrists.items()
                )
                > 0.18
            ):
                continue
            if (
                screen_body(meshes)["status"] == "clear_screen"
                and screen_hands(meshes)["status"] == "clear_screen"
            ):
                return diagnostics
    raise ValueError("no bounded hand/body clearance")


def bake(path, profile_name, spec, *, mirror=False, selected_hands=None):
    if not profile_name:
        return {"applied": False, "profile": "none"}
    profile = get_profile(profile_name)
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    originals = {b.name: b.matrix.copy() for b in arm.pose.bones}
    wrists = {s: _point(arm, s + "Hand") for s in ("Left", "Right")}
    local_fingers = {
        b.name: arm.pose.bones["mixamorig:" + s + "Hand"].matrix.inverted() @ b.matrix
        for s in wrists
        for b in arm.pose.bones
        if b.name.startswith("mixamorig:" + s + "Hand")
        and b.name != "mixamorig:" + s + "Hand"
    }
    lateral = (_point(arm, "LeftArm") - _point(arm, "RightArm")).normalized()
    forward = lateral.cross(
        (_point(arm, "Spine2") - _point(arm, "Hips")).normalized()
    ).normalized()
    diagnostics = []
    if profile.guide == "pistol-cradle":
        from converter.scene_contact_firearm import cradle

        diagnostics.extend(cradle(arm, meshes, mirror=mirror))
    else:
        grip = spec.split(":")[1] if spec else "B"
        if profile.palms_up or profile.palms_down or profile.palms_in:
            for side in ("left", "right"):
                if grip == "B" or side[0].upper() == grip:
                    normal = -lateral if side == "left" else lateral
                    if not profile.palms_in:
                        normal = Vector((0, 0, -1 if profile.palms_down else 1))
                    if profile.palms_down:
                        from converter.scene_contact_frames import palm

                        old_normal = palm(arm, side.title())[1]
                        state = _snapshot(arm)
                        for blend in (1, 0.75, 0.5, 0.25):
                            _restore(arm, state)
                            try:
                                turn = rotate_palm(
                                    arm,
                                    side,
                                    old_normal.lerp(normal, blend).normalized(),
                                )
                            except ValueError:
                                continue
                            diagnostics.append(turn)
                            break
                        else:
                            _restore(arm, state)
                    else:
                        diagnostics.append(rotate_palm(arm, side, normal))
        for side, sign in (("Left", 1), ("Right", -1)):
            delta = (
                forward * profile.forward
                + Vector((0, 0, profile.upward))
                + lateral * sign * profile.outward
            )
            if delta.length:
                diagnostics.append(move(arm, side, _point(arm, side + "Hand") + delta))
        diagnostics.extend(separate_hands(arm, meshes, lateral))
    diagnostics.extend(clear_body(arm, meshes, forward, lateral, wrists))
    if profile_name == "knee-contact":
        # Close the visible gap above the thighs without accepting a skin crossing.
        for side in ("Left", "Right"):
            state = _snapshot(arm)
            wrist = _point(arm, side + "Hand")
            best = state
            best_move = None
            for distance in (
                0.005,
                0.01,
                0.015,
                0.02,
                0.025,
                0.03,
                0.035,
                0.04,
                0.045,
                0.05,
                0.055,
                0.06,
            ):
                _restore(arm, state)
                try:
                    moved = move(arm, side, wrist - Vector((0, 0, distance)))
                except ValueError:
                    break
                if (
                    screen_body(meshes)["status"] != "clear_screen"
                    or screen_hands(meshes)["status"] != "clear_screen"
                ):
                    break
                best, best_move = _snapshot(arm), moved
            _restore(arm, best)
            if best_move is not None:
                diagnostics.append(best_move)
    untouched = {
        n: m
        for n, m in originals.items()
        if not any(
            n.startswith("mixamorig:" + s + p)
            for s in wrists
            for p in ("Arm", "ForeArm", "Hand")
        )
    }
    body_error = max(
        abs(arm.pose.bones[n].matrix[r][c] - m[r][c])
        for n, m in untouched.items()
        for r in range(4)
        for c in range(4)
    )
    finger_error = 0.0
    for name, before in local_fingers.items():
        side = "Left" if name.startswith("mixamorig:Left") else "Right"
        actual = (
            arm.pose.bones["mixamorig:" + side + "Hand"].matrix.inverted()
            @ arm.pose.bones[name].matrix
        )
        finger_error = max(
            finger_error,
            max(abs(actual[r][c] - before[r][c]) for r in range(4) for c in range(4)),
        )
    movement = max(
        (_point(arm, s + "Hand") - wrist).length for s, wrist in wrists.items()
    )
    if body_error > 1e-5 or finger_error > 1e-5 or movement > 0.18:
        raise ValueError("contact correction exceeds preservation or movement bounds")
    rt.apply_output_mode(arm, meshes, "rigged_rest")
    rt.export_fbx(str(path), embed_textures=False)
    return dict(
        applied=True,
        profile=profile_name,
        hands=diagnostics,
        torso_head_legs_matrix_max_error=body_error,
        local_finger_matrix_max_error=finger_error,
        wrist_movement_max=movement,
        source_bvh_unchanged=True,
        visual_approval=False,
    )
