"""손 라이브러리(04J-290) 계약: 정의 검증, 리그 독립 굽힘 방향, worker 요청 검증.

Blender 없이 도는 테스트다. 실제 남·여 캐릭터 적용은
``tests/converter/verify_hand_library.py``(Blender 5.2)로 확인한다.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from converter.hand_geometry import (
    FingerRest, axis_angle_matrix, finger_rotations, hand_frame, matrix_to_quaternion,
)
from converter.hand_library import (
    FINGERS, SIDES, FingerPose, HandLibraryError, LIBRARY_PATH, candidate_order, finger_bone_names,
    load_library, normalize_request, parse_library,
)

BASIC_SEVEN = {"open", "relaxed", "support", "cup", "grip", "fist", "pinch"}
# 상황별로 꼭 있어야 하는 손(무기·공·제스처). 이름을 바꾸면 VLM·클라이언트 계약이 깨진다.
SITUATIONAL = {"sword_grip", "gun_trigger", "bow_draw", "ball_palm", "ball_grip", "point",
               "v_sign", "thumbs_up", "ok_sign", "finger_heart", "hold_phone", "pen"}
CURLING = BASIC_SEVEN - {"open", "support"}


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _bone_basis(direction: np.ndarray, roll: float) -> np.ndarray:
    y = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, helper)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1) @ axis_angle_matrix(np.array([0.0, 1.0, 0.0]), roll)


def _mixamo_like_hand(side: str, rotation: np.ndarray, offset: np.ndarray, rng):
    """Mixamo T-pose 비율의 손(Blender Z-up, 정면 -Y, 손바닥 아래)을 강체 변환한다.

    본 roll은 무작위다. 결과가 roll·포즈·좌우와 무관해야 하기 때문이다.
    반환: wrist, bases, rests(FingerRest), lengths, palm(정답 손바닥 방향).
    """
    s = 1.0 if side == "left" else -1.0
    wrist = np.array([s * 0.7, 0.0, 1.4])
    knuckles = {"thumb": (0.02, -0.03, -0.015), "index": (0.09, -0.025, 0.0),
                "middle": (0.095, -0.005, 0.0), "ring": (0.09, 0.015, 0.0),
                "pinky": (0.08, 0.033, 0.0)}
    bases, rests, lengths = {}, {}, {}
    for finger, (x, y, z) in knuckles.items():
        base = wrist + np.array([s * x, y, z])
        step = (np.array([s * 0.025, -0.02, -0.01]) if finger == "thumb"
                else np.array([s * 0.03, 0.0, 0.0]))
        bases[finger] = rotation @ base + offset
        for joint in range(3):
            head = base + joint * step
            basis = _bone_basis(step, rng.uniform(-math.pi, math.pi))
            rests[(finger, joint)] = FingerRest(rotation @ head + offset,
                                                rotation @ (step / np.linalg.norm(step)),
                                                rotation @ basis)
            lengths[(finger, joint)] = float(np.linalg.norm(step))
    return rotation @ wrist + offset, bases, rests, lengths, rotation @ np.array([0.0, 0.0, -1.0])


def _tips(side, preset_id, wrist, bases, rests, lengths):
    """Blender와 같은 규약의 FK: child_pose = parent_delta @ child_rest @ local."""
    library = load_library()
    frame = hand_frame(side, wrist, bases)
    rotations = finger_rotations(side, library.preset(preset_id), frame, rests, bases)
    tips = {}
    for finger in FINGERS:
        delta = np.eye(3)
        position = rests[(finger, 0)].head
        for joint in range(3):
            rest = rests[(finger, joint)]
            posed = delta @ rest.basis @ rotations[(finger, joint)]
            position = position + posed[:, 1] * lengths[(finger, joint)]
            delta = posed @ rest.basis.T
        tips[finger] = position
    return frame, tips


def test_shipped_library_covers_basic_and_situational_hands():
    library = load_library()
    assert BASIC_SEVEN | SITUATIONAL <= set(library.preset_ids)
    assert len(library.presets) >= 70
    assert library.library_version == "hand-library-v1"
    assert set(library.categories) == {"basic", "hold", "weapon", "sports", "tool", "daily",
                                       "gesture", "emotion", "action"}
    for category in library.categories:
        assert library.in_category(category)
    for preset in library.presets.values():
        assert preset.label_ko and preset.vlm_hint
        assert set(preset.fingers) == set(FINGERS)
    assert len(finger_bone_names()) == 30


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema_version=2),
    lambda p: p["presets"]["fist"]["fingers"].pop("ring"),
    lambda p: p["presets"]["fist"]["fingers"]["ring"].update(curl=[1, 2]),
    lambda p: p["presets"]["fist"]["fingers"]["ring"].update(curl=[1, 2, 200]),
    lambda p: p["presets"]["fist"]["fingers"]["ring"].update(spread=float("nan")),
    lambda p: p["presets"]["fist"].pop("vlm_hint"),
    lambda p: p.update(fallback_order=["claw-x"]),
    lambda p: p.update(fallback_order=[]),
    lambda p: p["presets"]["fist"].update(category="dance"),
    lambda p: p["categories"].update(dance="춤"),
])
def test_invalid_definitions_fail_closed(mutate):
    payload = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    with pytest.raises(HandLibraryError):
        parse_library(json.dumps(payload).encode("utf-8"))


def test_request_normalization():
    assert normalize_request(None) is None
    assert normalize_request({"left": None, "right": None}) is None
    assert normalize_request({"right": "fist"}) == {"left": None, "right": "fist"}
    for bad in ({"left": "tentacle"}, {"both": "fist"}, {"left": 3}, ["fist"]):
        with pytest.raises(HandLibraryError):
            normalize_request(bad)


def test_candidate_order_puts_vlm_choices_then_same_category_then_defaults():
    library = load_library()
    assert candidate_order(["grip", "fist", "grip", None, "nope"])[:2] == ["grip", "fist"]
    weapons = candidate_order(["gun_trigger"])
    assert weapons[0] == "gun_trigger"
    assert all(library.presets[p].category == "weapon" for p in weapons)
    assert candidate_order([], limit=10) == list(library.fallback_order)
    assert len(candidate_order(["point"], limit=4)) == 4


@pytest.mark.parametrize("seed", range(6))
def test_every_preset_curls_toward_the_palm_for_any_roll_pose_and_side(seed):
    rng = np.random.default_rng(seed)
    rotation = np.eye(3) if seed == 0 else _random_rotation(rng)
    for side in SIDES:
        wrist, bases, rests, lengths, palm = _mixamo_like_hand(side, rotation, rng.normal(size=3), rng)
        frame, open_tips = _tips(side, "open", wrist, bases, rests, lengths)
        assert np.dot(frame.palm_normal, palm) > 0.99
        reach = {}
        for preset_id in BASIC_SEVEN:
            _, tips = _tips(side, preset_id, wrist, bases, rests, lengths)
            reach[preset_id] = np.linalg.norm(tips["middle"] - wrist)
            if preset_id == "open":
                continue
            for finger in ("index", "middle", "ring", "pinky"):
                assert np.dot(tips[finger] - open_tips[finger], palm) > 0, (side, preset_id, finger)
        assert reach["fist"] < reach["grip"] < reach["cup"] < reach["relaxed"] < reach["open"]


def test_left_and_right_results_are_mirror_images():
    rng = np.random.default_rng(42)
    mirror = np.diag([-1.0, 1.0, 1.0])
    left = _mixamo_like_hand("left", np.eye(3), np.zeros(3), rng)
    right = _mixamo_like_hand("right", np.eye(3), np.zeros(3), rng)
    for preset_id in BASIC_SEVEN:
        _, lt = _tips("left", preset_id, *left[:3], left[3])
        _, rt = _tips("right", preset_id, *right[:3], right[3])
        for finger in FINGERS:
            assert np.allclose(mirror @ lt[finger], rt[finger], atol=1e-9)


def test_swapped_finger_names_are_rejected_instead_of_curling_backwards():
    rng = np.random.default_rng(1)
    wrist, bases, *_ = _mixamo_like_hand("left", np.eye(3), np.zeros(3), rng)
    swapped = dict(bases, index=bases["pinky"], pinky=bases["index"])
    with pytest.raises(HandLibraryError, match="thumb"):
        hand_frame("left", wrist, swapped)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        m = _random_rotation(rng)
        w, x, y, z = matrix_to_quaternion(m)
        back = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        assert np.allclose(back, m, atol=1e-9)


# ---- worker 요청 계약 ---------------------------------------------------------

def _job(tmp_path, **extra):
    from tests.converter.test_worker_contract import _job as base_job
    path, payload = base_job(tmp_path)
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_worker_job_without_hands_is_unchanged(tmp_path):
    from converter.worker import load_job
    assert load_job(_job(tmp_path))["hands"] is None
    assert "bpy" not in sys.modules


def test_worker_job_accepts_known_presets(tmp_path):
    from converter.worker import load_job
    job = load_job(_job(tmp_path, hands={"left": "grip", "right": None}))
    assert job["hands"] == {"left": "grip", "right": None}


@pytest.mark.parametrize("hands", [{"left": "tentacle"}, {"left": 1}, "fist", {"middle": "fist"}])
def test_worker_job_rejects_unknown_hands(tmp_path, hands):
    from converter.worker import JobValidationError, load_job
    with pytest.raises(JobValidationError, match="hands"):
        load_job(_job(tmp_path, hands=hands))


def test_exact_v324_mode_cannot_bake_hands(tmp_path):
    from converter.worker import JobValidationError, load_job
    with pytest.raises(JobValidationError, match="hand"):
        load_job(_job(tmp_path, hands={"left": "fist"}, force_exact_v324=True))


def _gap(points, a, b):
    return float(np.linalg.norm(points[a][-1] - points[b][-1]))


def test_every_preset_stays_finite_and_never_bends_backwards():
    rng = np.random.default_rng(7)
    library = load_library()
    for side in SIDES:
        wrist, bases, rests, lengths, palm = _mixamo_like_hand(side, np.eye(3), np.zeros(3), rng)
        _, open_tips = _tips(side, "open", wrist, bases, rests, lengths)
        for preset_id, preset in library.presets.items():
            _, tips = _tips(side, preset_id, wrist, bases, rests, lengths)
            for finger in FINGERS:
                assert np.isfinite(tips[finger]).all()
                if finger != "thumb" and min(preset.fingers[finger].curl) >= 0:
                    # 굽힘이 0 이상이면 끝이 손등 쪽으로 가면 안 된다(벌림만으로 생기는 미세 오차 허용).
                    assert np.dot(tips[finger] - open_tips[finger], palm) > -1e-3, (preset_id, finger)


def test_thumb_wraps_across_the_palm_and_pinch_tips_meet():
    """엄지는 손바닥을 뚫고 내려가지 않고 새끼 쪽으로 가로지른다(대립)."""
    rng = np.random.default_rng(11)
    for side in SIDES:
        wrist, bases, rests, lengths, palm = _mixamo_like_hand(side, np.eye(3), np.zeros(3), rng)
        frame, fist = _tips(side, "fist", wrist, bases, rests, lengths)
        _, opened = _tips(side, "open", wrist, bases, rests, lengths)
        thumb_shift = fist["thumb"] - opened["thumb"]
        assert np.dot(thumb_shift, frame.pinky_side) > 0
        assert np.dot(thumb_shift, palm) > 0


# ---- 관통(04J-290) ------------------------------------------------------------

def _standard_hand(side):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "hand_library_preview.py"
    spec = importlib.util.spec_from_file_location("hand_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.rest_hand(side)


def test_no_preset_has_finger_or_palm_penetration_on_either_hand():
    from converter.hand_collision import contacts
    library = load_library()
    for side in SIDES:
        preview, hand = _standard_hand(side)
        frame = hand_frame(side, hand["wrist"], hand["bases"])
        for preset_id, preset in library.presets.items():
            points = preview.posed_points(side, preset, hand)
            found = contacts(points, palm_normal=frame.palm_normal, pinky_side=frame.pinky_side,
                             length_axis=frame.length_axis)
            assert not found, (side, preset_id, [(c.a, c.b, round(c.depth * 1000, 1)) for c in found])


def test_collision_check_catches_crossing_and_palm_penetration():
    from converter.hand_collision import contacts
    from converter.hand_library import HandPreset
    library = load_library()
    preview, hand = _standard_hand("right")
    frame = hand_frame("right", hand["wrist"], hand["bases"])
    check = lambda fingers: contacts(
        preview.posed_points("right", HandPreset("x", "", "", fingers, "basic"), hand),
        palm_normal=frame.palm_normal, pinky_side=frame.pinky_side, length_axis=frame.length_axis)
    crossed = dict(library.preset("open").fingers)
    crossed["middle"] = FingerPose((0.0, 0.0, 0.0), 30.0)          # 중지를 검지 쪽으로 눕혀 통과
    assert any({c.a, c.b} == {"index1", "middle1"} or "index" in c.a + c.b for c in check(crossed))
    dug = dict(library.preset("open").fingers)
    dug["middle"] = FingerPose((100.0, 110.0, 80.0), 0.0)         # 손끝을 손바닥 속으로
    assert any(c.b == "palm" or c.a == "palm" for c in check(dug))
    assert not check(dict(library.preset("open").fingers))


def test_intended_contacts_touch_without_penetrating():
    """집기·오케이·하트·튕기기는 끝이 닿아야 한다(떨어짐 1.5mm 이하, 관통 없음)."""
    from converter.hand_collision import RADII
    library = load_library()
    preview, hand = _standard_hand("right")
    for preset_id, other in {"pinch": "index", "ok_sign": "index", "finger_heart": "index",
                             "snap": "middle"}.items():
        points = preview.posed_points("right", library.preset(preset_id), hand)
        gap = np.linalg.norm(np.array(points["thumb"][-1]) - np.array(points[other][-1]))
        surface_gap = gap - RADII["thumb"][3] - RADII[other][3]
        assert -0.0005 <= surface_gap <= 0.0025, (preset_id, surface_gap)
