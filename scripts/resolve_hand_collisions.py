"""손 라이브러리에서 손가락 관통을 없앤다(04J-290).

관통하는 프리셋마다, 정의된 각도에서 **가장 적게** 바꾸면서 모든 마디·손바닥 사이에 여유
(기본 1mm)가 생기는 각도를 찾는다. 바꾼 각도는 정수(도)로 반올림한 뒤 다시 검사한다.

의도된 접촉은 지킨다.
- pinch·ok_sign·finger_heart·snap: 엄지 끝과 검지(snap은 중지) 끝이 닿아 있어야 한다(관통 X, 떨어짐 X).
- cross_fingers: 중지 끝이 검지 끝보다 엄지 쪽에 있어야 한다(꼬인 상태).

    python scripts/resolve_hand_collisions.py           # 검사만
    python scripts/resolve_hand_collisions.py --write   # 라이브러리 JSON 갱신
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converter.hand_collision import RADII, contacts  # noqa: E402
from converter.hand_geometry import hand_frame  # noqa: E402
from converter.hand_library import (  # noqa: E402
    CURL_LIMITS, FINGERS, LIBRARY_PATH, SPREAD_LIMITS, FingerPose, HandPreset, load_library,
)

_spec = importlib.util.spec_from_file_location("hand_preview", ROOT / "scripts" / "hand_library_preview.py")
preview = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preview)

CLEARANCE = 0.001        # 반올림 뒤에도 0.5mm 허용치를 넘지 않도록 1mm 여유
TOUCH = {"pinch": "index", "ok_sign": "index", "finger_heart": "index", "snap": "middle"}
HAND = preview.rest_hand("right")
FRAME = hand_frame("right", HAND["wrist"], HAND["bases"])


def _vector(preset: HandPreset) -> np.ndarray:
    return np.array([v for f in FINGERS for v in (*preset.fingers[f].curl, preset.fingers[f].spread)], float)


def _preset(base: HandPreset, x: np.ndarray) -> HandPreset:
    fingers = {f: FingerPose(tuple(float(v) for v in x[4 * i:4 * i + 3]), float(x[4 * i + 3]))
               for i, f in enumerate(FINGERS)}
    return HandPreset(base.preset_id, base.label_ko, base.vlm_hint, fingers, base.category)


def _points(preset: HandPreset):
    return preview.posed_points("right", preset, HAND)


def _contacts(points, tolerance):
    return contacts(points, palm_normal=FRAME.palm_normal, pinky_side=FRAME.pinky_side,
                    length_axis=FRAME.length_axis, tolerance=tolerance)


def _intent_error(preset_id: str, points) -> float:
    if preset_id in TOUCH:
        other = TOUCH[preset_id]
        gap = np.linalg.norm(np.array(points["thumb"][-1]) - np.array(points[other][-1]))
        touching = RADII["thumb"][3] + RADII[other][3] + 0.0012
        return (gap - touching) / 0.001
    if preset_id == "cross_fingers":
        across = float((np.array(points["middle"][-1]) - np.array(points["index"][-1])) @ -FRAME.pinky_side)
        return max(0.0, 0.004 - across) / 0.001       # 중지 끝이 검지보다 4mm 이상 엄지 쪽
    return 0.0


def _tips(points):
    return np.array([points[f][k] for f in FINGERS for k in (2, 3)], float)


TIP_WEIGHTS = np.array([0.15 if f == "thumb" else 4.0 for f in FINGERS for _ in (2, 3)])


def _cost(x, base_x, preset: HandPreset, base_tips):
    candidate = _preset(preset, x)
    points = _points(candidate)
    penetration = sum(c.depth ** 2 for c in _contacts(points, -CLEARANCE)) / 0.001 ** 2
    # 손이 '어떻게 보이는지'를 지킨다: 손가락 끝·셋째 마디 위치가 원래 의도에서 덜 움직일수록 좋다.
    # 엄지가 아닌 손가락의 모양을 강하게 지키고, 겹침은 주로 엄지가 바깥으로 감싸며 푼다(실제 주먹처럼).
    moved = float((((_tips(points) - base_tips) ** 2).sum(axis=1) * TIP_WEIGHTS).sum()) / 0.004 ** 2
    change = float(((x - base_x) / 30.0) @ ((x - base_x) / 30.0))
    return moved + change + 400.0 * penetration + 6.0 * _intent_error(preset.preset_id, points) ** 2


# 해부학 가동 범위(도). 엄지 CMC는 크게 접히지 않는다.
JOINT_LIMITS = {
    # 엄지 첫째 값은 CMC 굽힘+손바닥 쪽 벌림(대립)을 합친 것이라 단일 관절보다 크다.
    "thumb": [(-10, 90), (-10, 70), (-10, 85)],
    "other": [(-10, 95), (-10, 110), (-10, 90)],
}


def _bounds():
    out = []
    for finger in FINGERS:
        out += JOINT_LIMITS["thumb" if finger == "thumb" else "other"] + [SPREAD_LIMITS]
    return out


def resolve(preset: HandPreset) -> tuple[np.ndarray, list]:
    base_x = _vector(preset)
    base_tips = _tips(_points(preset))
    bounds = _bounds()
    clip = lambda v: np.array([min(max(val, lo), hi) for val, (lo, hi) in zip(v, bounds)])
    starts = [clip(base_x)]
    spread0 = base_x.copy(); spread0[3::4] = 0; starts.append(clip(spread0))
    thumb_out = base_x.copy(); thumb_out[3] += 15; thumb_out[0:3] *= 0.7; starts.append(clip(thumb_out))
    best = None
    for start in starts:
        result = minimize(_cost, start, args=(base_x, preset, base_tips), method="Powell", bounds=bounds,
                          options={"maxiter": 40000, "xtol": 0.05, "ftol": 1e-7})
        if best is None or result.fun < best.fun:
            best = result
    x = np.round(best.x)
    return x, _contacts(_points(_preset(preset, x)), 0.0005)


def _dump(raw: dict) -> None:
    text = json.dumps(raw, ensure_ascii=False, indent=2)
    text = re.sub(r'\{\n\s+"curl": \[\n\s+([-\d]+),\n\s+([-\d]+),\n\s+([-\d]+)\n\s+\],\n\s+"spread": ([-\d]+)\n\s+\}',
                  r'{"curl": [\1, \2, \3], "spread": \4}', text)
    text = re.sub(r'"fallback_order": \[\n\s+([^\]]+?)\n\s+\]',
                  lambda m: '"fallback_order": [' + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    Path(LIBRARY_PATH).write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    library = load_library()
    raw = json.loads(Path(LIBRARY_PATH).read_text(encoding="utf-8"))
    remaining = []
    for preset_id, preset in library.presets.items():
        points = _points(preset)
        found = _contacts(points, 0.0005)
        intent = abs(_intent_error(preset_id, points)) > 1.0
        if not found and not intent:
            continue
        x, after = resolve(preset)
        base_x = _vector(preset)
        changes = [f"{FINGERS[i // 4]}.{'curl' + str(i % 4 + 1) if i % 4 < 3 else 'spread'} "
                   f"{base_x[i]:g}→{x[i]:g}" for i in range(len(x)) if x[i] != base_x[i]]
        worst = max((c.depth for c in found), default=0.0)
        print(f"{preset_id:16s} 관통 {worst * 1000:5.1f}mm → {'없음' if not after else after}  |  {', '.join(changes)}")
        if after:
            remaining.append(preset_id)
        for i, finger in enumerate(FINGERS):
            raw["presets"][preset_id]["fingers"][finger] = {
                "curl": [int(v) for v in x[4 * i:4 * i + 3]], "spread": int(x[4 * i + 3])}
    if args.write:
        _dump(raw)
    print("남은 관통:", remaining or "없음")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
