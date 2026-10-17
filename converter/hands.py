"""손 굽기 단계(04J-290). V3.2.5 몸 변환이 끝난 산출 FBX에 손 프리셋을 적용한다.

framing과 같은 후처리 위치에서 돈다: 산출 FBX를 다시 불러 → 손가락 30본의 로컬 회전만
정하고 → rigged_rest 규약대로 메시에 굽고 포즈를 rest로 승격 → 같은 설정으로 내보낸다.
요청에 손이 없으면 이 모듈은 호출되지 않으며 산출 FBX는 바이트 단위로 기존과 같다.

동결 solver(retarget.py 등, SHA256SUMS.v325)는 수정하지 않는다. 여기서 쓰는
``import_character``/``apply_output_mode``/``export_fbx``는 그 파일의 공개 함수다.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np

from converter.hand_collision import contacts
from converter.hand_geometry import FingerRest, finger_rotations, hand_frame, matrix_to_quaternion
from converter.hand_library import (
    FINGERS, JOINTS_PER_FINGER, SIDES, HandLibraryError,
    bone_name, hand_bone_name, load_library,
)

HAND_BAKE_VERSION = "hand-bake-v1"
_BODY_TOLERANCE = 1e-5


def _prefix(arm) -> str:
    for bone in arm.data.bones:
        if bone.name.endswith("LeftHand"):
            return bone.name[: -len("LeftHand")]
    raise HandLibraryError("character has no LeftHand bone")


def _vec(v) -> np.ndarray:
    return np.array([v[0], v[1], v[2]], dtype=float)


def _basis(matrix) -> np.ndarray:
    m = matrix.to_3x3().normalized()
    return np.array([[m[r][c] for c in range(3)] for r in range(3)], dtype=float)


def _side_geometry(arm, side: str, prefix: str):
    bones = arm.data.bones
    hand = bones.get(hand_bone_name(side, prefix))
    if hand is None:
        raise HandLibraryError(f"character is missing {side} hand bone")
    bases, rests = {}, {}
    for finger in FINGERS:
        for joint in range(JOINTS_PER_FINGER):
            bone = bones.get(bone_name(side, finger, joint, prefix))
            if bone is None:
                raise HandLibraryError(
                    f"character is missing finger bone {bone_name(side, finger, joint, prefix)}")
            head = _vec(bone.head_local)
            tail = _vec(bone.tail_local)
            direction = tail - head
            if np.linalg.norm(direction) < 1e-8:
                raise HandLibraryError(f"zero-length finger bone {bone.name}")
            rests[(finger, joint)] = FingerRest(head, direction / np.linalg.norm(direction),
                                                _basis(bone.matrix_local))
            if joint == 0:
                bases[finger] = head
    return _vec(hand.head_local), bases, rests


def _tip_metrics(arm, side: str, prefix: str, palm_normal: np.ndarray) -> dict:
    """현재 포즈의 손가락 끝 위치로 굽힘 결과를 숫자로 남긴다(사람 검수 보조)."""
    wrist = _vec(arm.pose.bones[hand_bone_name(side, prefix)].head)
    out = {}
    for finger in FINGERS:
        tip = _vec(arm.pose.bones[bone_name(side, finger, JOINTS_PER_FINGER - 1, prefix)].tail)
        out[finger] = {
            "tip_to_wrist": round(float(np.linalg.norm(tip - wrist)), 6),
            "tip_palm_offset": round(float(np.dot(tip - wrist, palm_normal)), 6),
        }
    return out


def _posed_chains(arm, side: str, prefix: str) -> dict:
    out = {}
    for finger in FINGERS:
        bones = [arm.pose.bones[bone_name(side, finger, j, prefix)] for j in range(JOINTS_PER_FINGER)]
        out[finger] = [_vec(b.head) for b in bones] + [_vec(bones[-1].tail)]
    return out


def _hand_length(arm, side: str, prefix: str) -> float:
    wrist = _vec(arm.data.bones[hand_bone_name(side, prefix)].head_local)
    tip = _vec(arm.data.bones[bone_name(side, "middle", JOINTS_PER_FINGER - 1, prefix)].tail_local)
    return float(np.linalg.norm(tip - wrist))


def bake_hands(path, hands: Mapping[str, Optional[str]], *, library_path: Optional[str] = None) -> dict:
    """산출 FBX(``path``)를 제자리에서 손 프리셋이 적용된 FBX로 바꾼다."""
    import bpy
    from mathutils import Quaternion

    from converter import retarget as rt

    library = load_library(library_path)
    wanted = {side: hands.get(side) for side in SIDES if hands.get(side)}
    if not wanted:
        raise HandLibraryError("bake_hands called without any preset")
    presets = {side: library.preset(preset_id) for side, preset_id in wanted.items()}

    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    prefix = _prefix(arm)
    finger_names = {bone_name(side, finger, joint, prefix)
                    for side in SIDES for finger in FINGERS for joint in range(JOINTS_PER_FINGER)}
    body_before = {pb.name: pb.matrix.copy() for pb in arm.pose.bones if pb.name not in finger_names}

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
    rt.export_fbx(str(path), embed_textures=False)

    return {
        "version": HAND_BAKE_VERSION,
        "library_version": library.library_version,
        "library_sha256": library.sha256,
        "left": hands.get("left"),
        "right": hands.get("right"),
        "applied_bones": applied,
        "body_bones_preserved": True,
        "sides": report_sides,
        "self_collision_free": all(not s["self_collisions"] for s in report_sides.values()),
    }
