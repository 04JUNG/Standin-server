"""손 프리셋을 리그에 적용하기 위한 리그 독립 회전 계산(04J-290). numpy만 쓴다.

원칙
----
1. 접는 축은 그 리그의 rest 기하(손목·손가락 뿌리 위치)에서 매번 계산한다. 본 roll·
   이름 규약·좌우 미러와 무관하게 손바닥 쪽으로 접힌다.
2. 손바닥 방향은 손가락 뿌리의 손잡이성(chirality)으로 정한다. 세계 축(T-pose 손바닥=
   아래)을 쓰지 않는다. rigged_rest 출력은 포즈가 rest에 구워져 손 방향이 포즈마다 다르기
   때문이다. 강체 변환은 외적의 방향을 보존하므로 ``cross(검지 뿌리, 새끼 뿌리)``의
   부호는 어떤 포즈에서도 같은 쪽을 가리킨다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from converter.hand_library import FINGERS, JOINTS_PER_FINGER, HandLibraryError, HandPreset


# ---------------------------------------------------------------------------

def _unit(v: np.ndarray, what: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-8:
        raise HandLibraryError(f"degenerate hand geometry: {what}")
    return v / n


def axis_angle_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = _unit(np.asarray(axis, dtype=float), "rotation axis")
    theta = math.radians(degrees)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


@dataclass(frozen=True)
class HandFrame:
    """손 하나의 rest 기하에서 얻은 해부학 축(아마추어 공간, 단위벡터)."""
    palm_normal: np.ndarray      # 손바닥이 향하는 쪽
    length_axis: np.ndarray      # 손목 → 중지 뿌리
    pinky_side: np.ndarray       # 손바닥 평면 안에서 검지 → 새끼 방향


def hand_frame(side: str, wrist: np.ndarray, bases: Mapping[str, np.ndarray]) -> HandFrame:
    """손목과 손가락 첫 마디 뿌리(head) 위치로 손바닥 방향을 정한다.

    Mixamo T-pose 기하(Blender Z-up, 정면 -Y): 왼손은 ``cross(검지, 새끼)``가 손등(+Z), 오른손은 손바닥(-Z)을
    향한다. 손잡이성은 강체 포즈에서 보존되므로 이 부호 규칙은 어떤 rest에도 유효하다.
    엄지가 검지 쪽에 있는지로 다시 확인해, 좌우가 바뀐 리그는 조용히 반대로 접지 않고
    실패시킨다.
    """
    wrist = np.asarray(wrist, dtype=float)
    index = np.asarray(bases["index"], dtype=float) - wrist
    pinky = np.asarray(bases["pinky"], dtype=float) - wrist
    middle = np.asarray(bases["middle"], dtype=float) - wrist
    thumb = np.asarray(bases["thumb"], dtype=float) - wrist
    normal = _unit(np.cross(index, pinky), "index/pinky bases are collinear")
    if side == "left":
        normal = -normal
    elif side != "right":
        raise HandLibraryError(f"unknown side {side!r}")
    length = _unit(middle, "middle base coincides with wrist")
    # 손바닥 평면 안에서 엄지는 검지 쪽에 있어야 한다.
    # 손 길이축 성분을 빼고 손바닥 가로 방향만 비교한다(엄지 뿌리는 손목 가까이 있어서
    # 길이축 성분이 섞이면 판정이 흐려진다).
    across = (index - pinky) - np.dot(index - pinky, length) * length
    thumb_side = (thumb - middle) - np.dot(thumb - middle, length) * length
    if np.dot(thumb_side, across) <= 0:
        raise HandLibraryError(f"{side} thumb is not on the index side")
    return HandFrame(normal, length, _unit(-across, "palm width"))


@dataclass(frozen=True)
class FingerRest:
    """한 손가락 마디의 rest 상태(아마추어 공간)."""
    head: np.ndarray
    direction: np.ndarray        # head → tail 단위벡터(본의 Y축)
    basis: np.ndarray            # 3x3, 열 = 본 로컬 X/Y/Z 축(아마추어 공간)


# 엄지는 손바닥 정면이 아니라 손바닥을 **가로질러** 새끼 쪽으로 접힌다(대립 운동).
# 엄지를 접는 목표 방향 = 손바닥 방향 + 이 비율 × 새끼 쪽. 0이면 다른 손가락처럼 손바닥을
# 뚫고 내려가고, 0.7 근처면 주먹에서 검지·중지 바깥을 덮는다(표준 손 미리보기로 맞춘 값).
THUMB_ACROSS_WEIGHT = 0.7


def finger_rotations(
    side: str,
    preset: HandPreset,
    frame: HandFrame,
    rests: Mapping[tuple[str, int], FingerRest],
    bases: Mapping[str, np.ndarray],
) -> dict[tuple[str, int], np.ndarray]:
    """각 손가락 마디의 **본 로컬** 회전행렬(3x3)을 돌려준다.

    Blender pose bone의 회전은 그 본의 rest 축(``bone.matrix_local``)에서 정의된다.
    아마추어 공간 회전 R을 로컬로 옮기면 ``B^T R B``다. 자식 마디는 부모의 회전을 이미
    물려받으므로, rest 기준 축으로 계산한 로컬 회전을 그대로 쌓으면 축이 손가락을 따라
    함께 움직인다(해부학적 굽힘).
    """
    rotations: dict[tuple[str, int], np.ndarray] = {}
    middle_base = np.asarray(bases["middle"], dtype=float)
    index_base = np.asarray(bases["index"], dtype=float)
    for finger in FINGERS:
        pose = preset.fingers[finger]
        for joint in range(JOINTS_PER_FINGER):
            rest = rests[(finger, joint)]
            direction = _unit(rest.direction, f"{finger}{joint + 1} direction")
            # 손바닥 쪽으로 접는 축: dir × n. 이 축으로 +θ 돌리면 끝이 n 쪽으로 간다.
            target = frame.palm_normal
            if finger == "thumb":
                target = frame.palm_normal + THUMB_ACROSS_WEIGHT * frame.pinky_side
            curl_axis = np.cross(direction, target)
            if np.linalg.norm(curl_axis) < 1e-4:
                raise HandLibraryError(f"{side} {finger}{joint + 1} points along palm normal")
            rotation = axis_angle_matrix(curl_axis, pose.curl[joint])
            if joint == 0 and pose.spread:
                if finger == "middle":
                    lateral = index_base - middle_base
                elif finger == "thumb":
                    lateral = np.asarray(bases["thumb"], dtype=float) - index_base
                else:
                    lateral = np.asarray(bases[finger], dtype=float) - middle_base
                spread_axis = frame.palm_normal.copy()
                toward = np.cross(spread_axis, direction)
                if np.dot(toward, lateral) < 0:
                    spread_axis = -spread_axis
                rotation = rotation @ axis_angle_matrix(spread_axis, pose.spread)
            basis = np.asarray(rest.basis, dtype=float)
            rotations[(finger, joint)] = basis.T @ rotation @ basis
    return rotations


def matrix_to_quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 회전 → (w, x, y, z). Blender 없이 테스트하기 위한 순수 구현."""
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)
