"""손가락 관통 검사(04J-290). numpy만 쓴다.

손가락 마디를 캡슐(둥근 원기둥), 손바닥을 두께 있는 판으로 보고 서로 파고드는 깊이를 잰다.
미리보기 페이지의 마네킹 손과 **같은 두께**를 쓰므로, 여기서 0이면 화면에서도 겹치지 않는다.

두께는 손 길이 18cm 손 기준 실측 근사값이며 손 길이에 비례해 늘거나 준다.
- 손가락 지름: 검지 첫째 마디 약 1.7cm, 끝마디 약 1.35cm
- 손바닥 두께: 약 2.5cm

검사 대상
1. 서로 다른 손가락의 마디끼리 (엄지 포함)
2. 같은 손가락에서 이웃하지 않는 마디끼리 (첫째 ↔ 셋째)
3. 손가락 둘째·셋째 마디(엄지는 둘째 마디 뒤쪽·셋째 마디)와 손바닥 판

제외
- 이웃 손가락의 **뿌리 구슬**끼리: 손바닥 안에서 나란히 나오는 자리라 실제 손도 붙어 있다.
  대신 뿌리를 벗어난 구간(마디 길이의 30% 이후)은 검사한다.
- 엄지 첫째 마디(CMC)는 손바닥 살 안에 있으므로 손바닥 판과 검사하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from converter.hand_library import FINGERS

REFERENCE_HAND_LENGTH = 0.18
# [첫째, 둘째, 셋째 마디 시작, 손끝] 반지름(m, 손 길이 18cm 기준). 마디 캡슐은 양끝 반지름의 평균.
RADII = {
    "thumb": (0.0104, 0.0093, 0.0082, 0.0074),
    "index": (0.0085, 0.0076, 0.0068, 0.0062),
    "middle": (0.0088, 0.0080, 0.0071, 0.0064),
    "ring": (0.0083, 0.0075, 0.0067, 0.0060),
    "pinky": (0.0075, 0.0067, 0.0060, 0.0054),
}
PALM_HALF_THICKNESS = 0.0125
WRIST_HALF_WIDTH = 0.026
# 피부끼리 닿는 것은 허용하고, 이보다 깊게 파고들면 관통으로 본다(m, 18cm 손 기준).
TOLERANCE = 0.0005
ROOT_FRACTION = 0.3


@dataclass(frozen=True)
class Contact:
    a: str          # "index2" 처럼 손가락+마디 번호, 손바닥은 "palm"
    b: str
    depth: float    # m, 양수 = 관통


def _closest_between_segments(p1, q1, p2, q2):
    """두 선분 사이 최단 거리와 각 선분 위 매개변수(s, t)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r)), 0.0, 0.0
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t))), s, t


def _segments(points: Mapping[str, Sequence], scale: float):
    out = []
    for finger in FINGERS:
        chain = [np.asarray(p, dtype=float) for p in points[finger]]
        radii = RADII[finger]
        for joint in range(3):
            radius = 0.5 * (radii[joint] + radii[joint + 1]) * scale
            out.append((finger, joint, chain[joint], chain[joint + 1], radius))
    return out


def _inside_polygon_2d(point, polygon) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > point[1]) != (y2 > point[1]):
            x = x1 + (point[1] - y1) * (x2 - x1) / (y2 - y1)
            if point[0] < x:
                inside = not inside
    return inside


def contacts(points: Mapping[str, Sequence], *, palm_normal, pinky_side, length_axis,
             wrist=(0.0, 0.0, 0.0), hand_length: float = REFERENCE_HAND_LENGTH,
             tolerance: float = TOLERANCE) -> list[Contact]:
    """관통한 접촉 목록(깊이 큰 순). points는 손가락마다 [뿌리, 둘째, 셋째, 끝] 위치."""
    scale = hand_length / REFERENCE_HAND_LENGTH
    tol = tolerance * scale
    segs = _segments(points, scale)
    found: dict[tuple[str, str], float] = {}

    def record(a, b, depth):
        if depth > tol:
            key = tuple(sorted((a, b)))
            found[key] = max(found.get(key, 0.0), depth)

    for i in range(len(segs)):
        fi, ji, pi, qi, ri = segs[i]
        for j in range(i + 1, len(segs)):
            fj, jj, pj, qj, rj = segs[j]
            if fi == fj and abs(ji - jj) < 2:
                continue
            if fi != fj and ji == 0 and jj == 0:
                # 뿌리 구슬끼리는 붙어 있는 것이 정상. 뿌리에서 벗어난 구간만 본다.
                pi2 = pi + (qi - pi) * ROOT_FRACTION
                pj2 = pj + (qj - pj) * ROOT_FRACTION
                dist, _, _ = _closest_between_segments(pi2, qi, pj2, qj)
            else:
                dist, _, _ = _closest_between_segments(pi, qi, pj, qj)
            record(f"{fi}{ji + 1}", f"{fj}{jj + 1}", ri + rj - dist)

    # 손바닥 판: 손목 양끝·엄지 둘째 마디 뿌리·네 손가락 뿌리로 둘러싼 다각형 × 두께
    n = np.asarray(palm_normal, dtype=float)
    u = np.asarray(pinky_side, dtype=float)
    v = np.asarray(length_axis, dtype=float)
    wrist = np.asarray(wrist, dtype=float)
    outline = [wrist - u * WRIST_HALF_WIDTH * scale, np.asarray(points["thumb"][1], dtype=float),
               *[np.asarray(points[f][0], dtype=float) for f in ("index", "middle", "ring", "pinky")],
               wrist + u * WRIST_HALF_WIDTH * scale]
    origin = wrist
    polygon = [(float((p - origin) @ u), float((p - origin) @ v)) for p in outline]
    half = PALM_HALF_THICKNESS * scale
    for finger, joint, p, q, radius in segs:
        if joint == 0:
            continue                      # 첫째 마디는 손바닥 가장자리에서 시작한다
        start = 0.35 if (finger == "thumb" and joint == 1) else 0.0
        for k in np.linspace(start, 1.0, 9):
            point = p + (q - p) * k
            rel = point - origin
            if not _inside_polygon_2d(((rel @ u), (rel @ v)), polygon):
                continue
            depth = radius + half - abs(float(rel @ n))
            record(f"{finger}{joint + 1}", "palm", depth)
    return [Contact(a, b, d) for (a, b), d in sorted(found.items(), key=lambda kv: -kv[1])]
