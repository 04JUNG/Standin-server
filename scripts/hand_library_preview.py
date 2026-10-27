"""손 라이브러리 미리보기 데이터(04J-290).

변환기와 **같은 계산**(converter/hand_geometry.py)으로 프리셋마다 양손 관절 위치를 구해 JSON으로
쓴다. 시각화 페이지가 이 값을 그대로 그리고, 페이지 안의 각도 편집기는 같은 공식을 JS로 다시
계산해 이 값과 일치하는지 스스로 검사한다.

손 모양은 Mixamo T-pose 비율의 표준 손(Blender Z-up, 정면 -Y, 손바닥 아래)이다.
실제 캐릭터 메시 렌더가 아니다 — 그건 tests/converter/verify_hand_library.py(Blender)가 한다.

    python scripts/hand_library_preview.py out/hand_library_preview.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.hand_geometry import (  # noqa: E402
    THUMB_ACROSS_WEIGHT, FingerRest, finger_rotations, hand_frame,
)
from converter.hand_collision import (  # noqa: E402
    PALM_HALF_THICKNESS, RADII, ROOT_FRACTION, TOLERANCE, WRIST_HALF_WIDTH,
)
from converter.hand_library import FINGERS, LIBRARY_PATH, SIDES, load_library  # noqa: E402

HAND_LENGTH = 0.18  # 손목 → 중지 끝(m), 성인 평균 근처
# 손목 기준, 손 길이 비율. (x=팔 방향, y=위, z=정면) — pose_curation/hands/presets.py와 같은 비율.
_BASES = {
    "thumb": (0.14, -0.025, 0.13),
    "index": (0.43, 0.0, 0.12),
    "middle": (0.45, 0.0, 0.02),
    "ring": (0.43, 0.0, -0.08),
    "pinky": (0.39, 0.0, -0.18),
}
_SEGMENTS = {
    "thumb": ((0.145, -0.025, 0.13), (0.12, 0.0, 0.09), (0.10, 0.0, 0.05)),
    "index": ((0.19, 0, 0), (0.12, 0, 0), (0.10, 0, 0)),
    "middle": ((0.22, 0, 0), (0.16, 0, 0), (0.12, 0, 0)),
    "ring": ((0.205, 0, 0), (0.14, 0, 0), (0.105, 0, 0)),
    "pinky": ((0.145, 0, 0), (0.095, 0, 0), (0.075, 0, 0)),
}


def _blender(side: str, v) -> np.ndarray:
    """(팔 방향, 위, 정면) 비율 → Blender 좌표(m). 왼손은 +X, 오른손은 -X로 뻗는다."""
    x, y, z = v
    sign = 1.0 if side == "left" else -1.0
    return np.array([sign * x, -z, y]) * HAND_LENGTH


def _basis(direction: np.ndarray) -> np.ndarray:
    y = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, helper)
    x /= np.linalg.norm(x)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def rest_hand(side: str) -> dict:
    wrist = np.zeros(3)
    bases, rests, lengths = {}, {}, {}
    for finger in FINGERS:
        head = _blender(side, _BASES[finger])
        bases[finger] = head
        for joint, segment in enumerate(_SEGMENTS[finger]):
            vector = _blender(side, segment)
            length = float(np.linalg.norm(vector))
            rests[(finger, joint)] = FingerRest(head, vector / length, _basis(vector))
            lengths[(finger, joint)] = length
            head = head + vector
    return {"wrist": wrist, "bases": bases, "rests": rests, "lengths": lengths}


def posed_points(side: str, preset, hand: dict) -> dict[str, list[list[float]]]:
    """finger_rotations의 본 로컬 회전으로 FK. 손가락마다 [뿌리, 둘째, 셋째, 끝]."""
    frame = hand_frame(side, hand["wrist"], hand["bases"])
    local = finger_rotations(side, preset, frame, hand["rests"], hand["bases"])
    out = {}
    for finger in FINGERS:
        delta = np.eye(3)
        point = hand["rests"][(finger, 0)].head
        chain = [point]
        for joint in range(3):
            rest = hand["rests"][(finger, joint)]
            posed = delta @ rest.basis @ local[(finger, joint)]
            point = point + posed[:, 1] * hand["lengths"][(finger, joint)]
            chain.append(point)
            delta = posed @ rest.basis.T
        out[finger] = [[round(float(c), 6) for c in p] for p in chain]
    return out


def build() -> dict:
    library = load_library()
    hands = {side: rest_hand(side) for side in SIDES}
    frames = {side: hand_frame(side, h["wrist"], h["bases"]) for side, h in hands.items()}
    open_tips = {side: posed_points(side, library.preset("open"), hands[side]) for side in SIDES}
    presets = {}
    for preset in library.presets.values():
        sides = {}
        for side in SIDES:
            points = posed_points(side, preset, hands[side])
            normal = frames[side].palm_normal
            sides[side] = {
                "points": points,
                "tip_to_wrist": {f: round(float(np.linalg.norm(points[f][-1])), 5) for f in FINGERS},
                "tip_toward_palm_vs_open": {
                    f: round(float(np.dot(np.array(points[f][-1]) - open_tips[side][f][-1], normal)), 5)
                    for f in FINGERS},
            }
        presets[preset.preset_id] = {
            "fist_profile": None if preset.articulation is None else {
                "thumb_directions": preset.articulation.thumb_directions, "fan": dict(preset.articulation.fan)},
            "label_ko": preset.label_ko, "category": preset.category, "vlm_hint": preset.vlm_hint,
            "fingers": {f: {"curl": list(p.curl), "spread": p.spread} for f, p in preset.fingers.items()},
            "sides": sides,
        }
    rest = {side: {
        "palm_normal": [round(float(c), 9) for c in frames[side].palm_normal],
        "fingers": {f: [{"head": [round(float(c), 9) for c in h["rests"][(f, j)].head],
                         "dir": [round(float(c), 9) for c in h["rests"][(f, j)].direction],
                         "length": round(h["lengths"][(f, j)], 9)} for j in range(3)]
                    for f in FINGERS},
    } for side, h in hands.items()}
    return {
        "library_version": library.library_version,
        "library_sha256": library.sha256,
        "categories": dict(library.categories),
        "fallback_order": list(library.fallback_order),
        "hand_length_m": HAND_LENGTH,
        "thumb_across_weight": THUMB_ACROSS_WEIGHT,
        "collision": {
            "radii": {f: list(r) for f, r in RADII.items()},
            "palm_half_thickness": PALM_HALF_THICKNESS, "wrist_half_width": WRIST_HALF_WIDTH,
            "tolerance": TOLERANCE, "root_fraction": ROOT_FRACTION,
        },
        "rest": rest,
        "presets": presets,
        "library_json": json.loads(Path(LIBRARY_PATH).read_text(encoding="utf-8")),
    }


def write_viewer(data: dict, target: Path) -> None:
    """미리보기 데이터를 담은 단일 HTML(3D 갤러리·각도 실험·일치 검사)을 쓴다."""
    template = Path(__file__).with_name("hand_library_viewer.template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    target.write_text(template.replace("__DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "out/hand_library_preview.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    if target.suffix == ".html":
        write_viewer(data, target)
    else:
        target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{len(data['presets'])} presets -> {target}")
