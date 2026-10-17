"""러프 손 모양 인식과 손 후보(04J-290, src/hand_shapes.py).

손은 FBX 손가락 굽기·후보에만 쓰인다. 검색 결과가 손 호출 때문에 바뀌면 안 된다.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.hand_shapes import (
    DEFAULT_PRESET, apply_hand_shapes, default_hands, detect_person_hands, parse_hand_shapes,
)
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.vlm import prompts
from src.vlm.client import MockVLMClient, _coerce
from converter.hand_library import load_library


def _side(shape, *, visible=True, alternative=None, confidence="high"):
    return {"visible": visible, "shape": shape, "alternative": alternative,
            "confidence": confidence}


def _analysis(num=2):
    return _coerce({
        "num_people": num, "shot": "full_half", "action": "standing", "view": "front",
        "relationship": "solo" if num == 1 else "talking",
        "approx_boxes": [{"x1": 0.4 * i, "y1": 0.1, "x2": 0.4 * i + 0.3, "y2": 0.9}
                         for i in range(num)],
    }, 100, 200)


def test_a_read_hand_is_baked_and_leads_the_candidates():
    [person] = parse_hand_shapes({"people": [
        {"left": _side("grip", alternative="fist"), "right": _side("open", confidence="low")},
    ]}, 1)
    assert person.status == "vlm"
    assert (person.left.preset, person.left.source, person.left.confidence) == ("grip", "vlm", "high")
    # VLM 1순위·대안 → 1순위와 같은 묶음(물건 쥐기)
    assert person.left.candidates == ("grip", "fist", "cup", "pinch")
    assert (person.right.preset, person.right.confidence) == ("open", "low")
    assert person.right.candidates[0] == "open"


def test_hidden_or_unknown_hands_fall_back_but_keep_the_hint_as_a_candidate():
    [person] = parse_hand_shapes({"people": [
        {"left": _side("fist", visible=False), "right": _side("tentacle")},
    ]}, 1)
    assert (person.left.preset, person.left.source, person.left.visible) == (DEFAULT_PRESET, "default", False)
    assert person.left.candidates[0] == "fist"
    assert (person.right.preset, person.right.source) == (DEFAULT_PRESET, "default")
    assert person.right.candidates == tuple(load_library().fallback_order[:4])


@pytest.mark.parametrize("payload", [None, {}, {"people": []}, {"people": [{}, {}, {}]}, "x"])
def test_misaligned_or_failed_answers_never_shift_hands_between_people(payload):
    people = parse_hand_shapes(payload, 2)
    assert [p.status for p in people] == ["unavailable", "unavailable"]
    assert all(p.left.preset == DEFAULT_PRESET and p.left.source == "default" for p in people)


def test_one_bad_entry_does_not_shift_the_next_person():
    people = parse_hand_shapes({"people": ["junk", {"left": _side("fist"), "right": None}]}, 2)
    assert people[0].status == "unavailable"
    assert people[1].left.preset == "fist" and people[1].right.source == "default"


def test_only_vlm_slots_get_read_hands():
    vlm = _analysis(1)
    assert detect_person_hands(vlm, 0).status == "not_requested"
    apply_hand_shapes(vlm, {"people": [{"left": _side("pinch"), "right": _side("pinch")}]})
    assert detect_person_hands(vlm, 0).left.preset == "pinch"
    assert detect_person_hands(vlm, None).status == "unavailable"


def test_prompt_lists_every_preset_and_the_anatomical_side_rule():
    library = load_library()
    text = prompts.hand_shapes_prompt(
        [(0.1, 0.1, 0.4, 0.9), None],
        [(p.preset_id, p.label_ko, p.vlm_hint) for p in library.presets.values()])
    for preset_id in library.preset_ids:
        assert f"- {preset_id} (" in text
    assert "1번: 위치 불명" in text and "길이는 반드시 2" in text
    assert "화면의 왼쪽/오른쪽이 아니다" in text
    # 분석 프롬프트는 손 기능 때문에 바뀌지 않는다.
    assert "손 모양" not in prompts.USER_TEMPLATE and "\"shape\"" not in prompts.USER_TEMPLATE


class _HandReadingVLM(MockVLMClient):
    calls = 0

    def read_hands(self, image, boxes):
        type(self).calls += 1
        return {"people": [{"left": _side("fist"), "right": _side("grip", alternative="pinch")}
                           for _ in boxes]}


class _Img(str):
    @property
    def hint(self):
        return str(self)


def _search_outcome(result):
    return result.route, [
        [(c.pose_id, c.view.value, round(c.distance, 9)) for c in candidates]
        for candidates in result.person_candidates
    ], [desc.refine_allowed for desc in result.descriptors]


@pytest.mark.parametrize("hint", ["full_half sitting side 2p", "full_half standing front 1p",
                                  "bust front 1p"])
def test_hand_shapes_change_nothing_in_search(monkeypatch, hint):
    pipeline = Pipeline(build_synthetic_index(), vlm_client=_HandReadingVLM(),
                        pose_model=MockPoseModel())
    monkeypatch.setattr(CFG, "vlm_hand_shapes", False)
    before = pipeline.process_cut(_Img(hint))
    assert all(d.hands is None or d.hands.status == "not_requested" for d in before.descriptors)
    calls = _HandReadingVLM.calls
    monkeypatch.setattr(CFG, "vlm_hand_shapes", True)
    after = pipeline.process_cut(_Img(hint))
    assert _HandReadingVLM.calls == calls + 1
    assert _search_outcome(after) == _search_outcome(before)
    vlm_people = [d for d in after.descriptors if d.slot_origin == "vlm"]
    assert vlm_people and all(d.hands.left.preset == "fist" and d.hands.right.preset == "grip"
                              for d in vlm_people)


_ALLOWED = {"src/hand_shapes.py", "src/schema.py", "src/descriptor.py", "src/pipeline.py",
            "src/vlm/client.py"}
_HAND_NAMES = {"person_hands", "detect_person_hands", "apply_hand_shapes", "read_hands"}


def test_only_metadata_code_touches_hand_shapes():
    users = set()
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = (node.attr if isinstance(node, ast.Attribute)
                    else node.id if isinstance(node, ast.Name) else None)
            if name in _HAND_NAMES:
                users.add(path.relative_to(ROOT).as_posix())
    assert users <= _ALLOWED, users - _ALLOWED


def test_default_hands_are_serializable_for_the_api():
    from dataclasses import asdict
    from api.models import PersonHandsOut

    out = PersonHandsOut(**asdict(default_hands()))
    assert out.left.preset == DEFAULT_PRESET and out.status == "not_requested"
    assert out.left.candidates[:2] == ["relaxed", "fist"]


def test_gemini_hand_call_and_pose_description_keep_separate_contracts():
    """Merging adjacent VLM methods must not replace either request/timeout."""
    import json
    from types import SimpleNamespace
    from PIL import Image
    from src.vlm.client import GeminiVLMClient

    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            payload = {"people": [{"left": _side("hold_phone"), "right": _side("open")}]}
        else:
            payload = {"people": [{"visible": True, "certainty": "high", "actions": ["kick"],
                "body_details": "한 다리를 앞으로 뻗음", "search_text": "한 다리를 앞으로 뻗어 발차기"}]}
        return SimpleNamespace(text=json.dumps(payload))

    client = GeminiVLMClient(client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    image = Image.new("RGB", (32, 32), "white")
    hands = client.read_hands(image, [(0, 0, 1, 1)])
    description = client.describe_people(image, [(0, 0, 1, 1)])
    assert hands["people"][0]["left"]["shape"] == "hold_phone"
    assert description.status == "ok" and description.descriptions[0].actions == ["kick"]
    assert len(calls) == 2
    assert calls[0]["config"].system_instruction == prompts.HAND_SHAPES_SYSTEM
    assert calls[0]["config"].http_options.timeout == CFG.vlm_hand_shapes_timeout_ms
    assert calls[1]["config"].system_instruction == prompts.POSE_DESCRIBE_SYSTEM
    assert calls[1]["config"].http_options.timeout == CFG.routing_describe_timeout_ms
