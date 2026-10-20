"""Observable shoulder roll and bounded head/torso separation in shared axes."""

import math
import numpy as np
from ..orientation import Orientation

VERSION = "articulated-bust-v1"
MAX_NECK_DEGREES = 55


def relative_rotation(face: Orientation, body: Orientation):
    rotation = body.matrix().T @ face.matrix()
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    if angle > MAX_NECK_DEGREES + 1e-6:
        raise ValueError(
            f"머리와 몸통의 상대 회전 {angle:.1f}°가 제작 제한 55°를 넘습니다. 몸통 방향을 먼저 맞춰 주세요."
        )
    return rotation, angle


def shoulder_roll(points, size, body: Orientation):
    p = np.asarray(points, float)
    if (
        p.shape != (2, 2)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p >= size).any()
    ):
        raise ValueError("이미지 안에서 양쪽 어깨를 지정해 주세요.")
    line = p[1] - p[0]
    if line[0] <= 0 or np.linalg.norm(line) < 15:
        raise ValueError(
            "화면 왼쪽 어깨부터 오른쪽 어깨 순으로 지정해 주세요. 어깨 간격은 15px 이상이어야 합니다."
        )
    projected = (body.matrix() @ [1.0, 0.0, 0.0])[:2] * [1, -1]
    if np.linalg.norm(projected) < 0.15:
        raise ValueError(
            "몸통이 완전 측면에 가까워 어깨 기울기를 정하기 어렵습니다. 직접 기울기를 조절해 주세요."
        )
    if projected[0] < 0:
        projected *= -1
    delta = math.degrees(
        math.atan2(line[1], line[0]) - math.atan2(projected[1], projected[0])
    )
    roll = (body.roll + delta + 180) % 360 - 180
    return Orientation(body.yaw, body.pitch, roll)


def validate_region(region, size):
    p = np.asarray(region, float)
    if p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError("얼굴 영역의 두 모서리를 지정해 주세요.")
    low, high = p[:2], p[2:]
    if (low < 0).any() or (high > size).any() or ((high - low) < 16).any():
        raise ValueError(
            "이미지 안에서 가로·세로 16px 이상의 얼굴 영역을 지정해 주세요."
        )
    return p.tolist()
