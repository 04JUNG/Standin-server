"""주먹 계열 손에서 엄지 자리를 전수 탐색한다(04J-290).

다른 손가락은 정의된 그대로 두고, 엄지만 바꿔 '말린 검지·중지의 둘째 마디 바깥을 덮는' 자리를
찾는다. 후보 중 관통이 없고(여유 1mm) 목표 자리에 가장 가까운 각도를 고른다.

    python scripts/fit_wrapped_thumb.py fist pinky_promise ... [--write]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from resolve_hand_collisions import (  # noqa: E402
    CLEARANCE, FRAME, HAND, LIBRARY_PATH, _contacts, _points, _preset, _vector, load_library,
)
from converter.hand_collision import RADII  # noqa: E402
from converter.hand_library import FINGERS  # noqa: E402


def target(points, over: str) -> np.ndarray:
    """over 손가락 둘째 마디 가운데에서 손바닥 반대쪽... 이 아니라 '주먹 바깥'(손바닥 방향 n)으로 붙은 자리."""
    mid = 0.5 * (np.array(points[over][1]) + np.array(points[over][2]))
    return mid + FRAME.palm_normal * (RADII[over][1] + RADII["thumb"][3])


def fit(preset, over):
    base = _vector(preset)
    pts0 = _points(preset)
    goal = target(pts0, over)
    best = None
    for c1, c2, c3, sp in itertools.product(range(0, 91, 6), range(0, 71, 7), range(0, 86, 8), range(-24, 21, 4)):
        x = base.copy(); x[0:4] = (c1, c2, c3, sp)
        pts = _points(_preset(preset, x))
        miss = float(np.linalg.norm(np.array(pts["thumb"][-1]) - goal))
        if best is not None and miss >= best[0]:
            continue
        # 엄지가 끼어든 접촉만 본다. 나머지 손가락은 호출 전에 따로 확인한다.
        if any("thumb" in c.a + c.b for c in _contacts(pts, -CLEARANCE)):
            continue
        best = (miss, x)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--over", default="middle")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    lib = load_library()
    raw = json.loads(Path(LIBRARY_PATH).read_text(encoding="utf-8"))
    for pid in a.ids:
        res = fit(lib.preset(pid), a.over)
        if res is None:
            print(pid, "해 없음"); continue
        miss, x = res
        print(f"{pid:14s} thumb curl {x[0:3].astype(int).tolist()} spread {int(x[3])}  목표와 {miss*1000:.1f}mm")
        raw["presets"][pid]["fingers"]["thumb"] = {"curl": [int(v) for v in x[0:3]], "spread": int(x[3])}
    if a.write:
        from resolve_hand_collisions import _dump
        _dump(raw)


if __name__ == "__main__":
    main()
