"""Private-scene hand adapter; keep reviewed legacy hand file bytes intact.

The solved body is the develop hand refactor without its FBX I/O. Shared
geometry helpers remain in the approved legacy module; no geometry rules change.
"""
from __future__ import annotations
import math
from typing import Mapping, Optional
from converter.hands import (HAND_BAKE_VERSION, _BODY_TOLERANCE, _prefix,
    _side_geometry, _tip_metrics, _posed_chains, _hand_length)
from converter.hand_collision import contacts
from converter.hand_geometry import finger_rotations, hand_frame, matrix_to_quaternion
from converter.hand_library import (FINGERS, JOINTS_PER_FINGER, SIDES,
    HandLibraryError, bone_name, load_library)

def solve_hands(arm, meshes, hands: Mapping[str, Optional[str]], *, library_path: Optional[str] = None) -> dict:
    """Apply the existing hand rules to a private scene, without file I/O.

    This boundary includes rig repair, skin changes, baking and smoothing. A
    caller must snapshot it before any exporter mutates/resets the scene.
    It does not remove the frozen body's intermediate FBX dependency.
    """
    import bpy
    from mathutils import Quaternion

    from converter import retarget as rt

    library = load_library(library_path)
    wanted = {side: hands.get(side) for side in SIDES if hands.get(side)}
    if not wanted:
        raise HandLibraryError("bake_hands called without any preset")
    presets = {side: library.preset(preset_id) for side, preset_id in wanted.items()}

    prefix = _prefix(arm)
    finger_names = {bone_name(side, finger, joint, prefix)
                    for side in SIDES for finger in FINGERS for joint in range(JOINTS_PER_FINGER)}
    body_before = {pb.name: pb.matrix.copy() for pb in arm.pose.bones if pb.name not in finger_names}

    rig_repairs = {}
    closed_sides = [side for side, preset in presets.items()
                    if preset.articulation is not None]
    if closed_sides:
        from converter.hand_rig import repair_digit_roots
        for side in closed_sides:
            rig_repairs[side] = repair_digit_roots(arm, meshes, side, prefix)
        for mesh in meshes:
            for modifier in mesh.modifiers:
                if modifier.type == "ARMATURE":
                    modifier.use_deform_preserve_volume = True

    report_sides: dict[str, dict] = {}
    applied = 0
    for side, preset in presets.items():
        wrist, bases, rests = _side_geometry(arm, side, prefix)
        frame = hand_frame(side, wrist, bases)
        before = _tip_metrics(arm, side, prefix, frame.palm_normal)
        rotations = finger_rotations(side, preset, frame, rests, bases)
        for (finger, joint), local in rotations.items():
            pose_bone = arm.pose.bones[bone_name(side, finger, joint, prefix)]
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = Quaternion(matrix_to_quaternion(local))
            applied += 1
        bpy.context.view_layer.update()
        after = _tip_metrics(arm, side, prefix, frame.palm_normal)
        # 표준 두께를 이 리그의 손 길이에 맞춰 관통을 잰다. 메시 자체의 교차는 Blender 검증이 본다.
        found = contacts(_posed_chains(arm, side, prefix), palm_normal=frame.palm_normal,
                         pinky_side=frame.pinky_side, length_axis=frame.length_axis, wrist=wrist,
                         hand_length=_hand_length(arm, side, prefix))
        report_sides[side] = {"preset": preset.preset_id, "before": before, "after": after,
                              "self_collisions": [
                                  {"a": c.a, "b": c.b, "depth_m": round(c.depth, 6)} for c in found]}

    for name, before in body_before.items():
        after = arm.pose.bones[name].matrix
        if max(abs(before[r][c] - after[r][c]) for r in range(4) for c in range(4)) > _BODY_TOLERANCE:
            raise RuntimeError(f"hand bake moved body bone {name}")
    for name in finger_names:
        pose_bone = arm.pose.bones.get(name)
        if pose_bone is not None and not all(
                math.isfinite(v) for row in pose_bone.matrix for v in row):
            raise RuntimeError(f"non-finite finger transform: {name}")

    # convert.py와 같은 규약: 포즈를 메시에 굽고 rest로 승격한 뒤 같은 설정으로 내보낸다.
    rt.apply_output_mode(arm, meshes, "rigged_rest")
    if closed_sides:
        from converter.hand_rig import smooth_baked_fingers
        for iterations in sorted({presets[side].articulation.smoothing_iterations for side in closed_sides}):
            sides = [side for side in closed_sides
                     if presets[side].articulation.smoothing_iterations == iterations]
            smooth_baked_fingers(meshes, sides, prefix, iterations)

    return {
        "version": HAND_BAKE_VERSION,
        "library_version": library.library_version,
        "library_sha256": library.sha256,
        "left": hands.get("left"),
        "right": hands.get("right"),
        "applied_bones": applied,
        "body_bones_preserved": True,
        "rig_repairs": rig_repairs,
        "sides": report_sides,
        "self_collision_free": all(not s["self_collisions"] for s in report_sides.values()),
    }
