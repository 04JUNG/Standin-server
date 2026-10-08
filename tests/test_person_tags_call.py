"""인물별 태그를 분석 호출에서 떼어 낸 뒤의 동작.

핵심은 하나다 — 이 호출이 어떻게 되든(성공·실패·지연·헛소리) route·인원수·박스는
분석 호출이 정한 그대로여야 한다. 그래서 대부분의 테스트가 "분석 결과가 그대로인가"를 본다.
"""
from __future__ import annotations

import time

import pytest

from src.config import CFG
from src.person_tags import apply_person_tags
from src.schema import Action, View, VLMAnalysis, Shot, Relationship, BBox
from src.vlm import prompts


def _analysis(num: int = 2) -> VLMAnalysis:
    return VLMAnalysis(
        num_people=num, shot=Shot.FULL_HALF, action=Action.STANDING, view=View.FRONT,
        relationship=Relationship.SOLO,
        approx_boxes=[BBox(x1=10 * i, y1=0, x2=10 * i + 8, y2=40, source="vlm", score=0.5)
                      for i in range(num)],
        raw={},
    )


def test_prompt_lists_every_person_in_order():
    text = prompts.person_tags_prompt([(0.0, 0.1, 0.2, 0.9), (0.5, 0.1, 0.7, 0.9)])
    assert "0번" in text and "1번" in text
    assert "반드시 2" in text
    # 사람 수를 다시 세지 말라고 분명히 적어 둔다. 두 번째 호출이 인원수를 바꾸면
    # 첫 호출이 정한 슬롯과 어긋난다.
    assert "다시 세지 마라" in text


def test_missing_box_keeps_the_slot():
    text = prompts.person_tags_prompt([(0.0, 0.1, 0.2, 0.9), None])
    assert "1번: 위치 불명" in text


def test_tags_land_on_the_analysis():
    vlm = _analysis(2)
    changed = apply_person_tags(vlm, {"person_actions": ["sitting", "walking"],
                                      "person_views": ["side", None]})
    assert changed is True
    assert vlm.person_actions == [Action.SITTING, Action.WALKING]
    assert vlm.person_views == [View.SIDE, None]


def test_misaligned_arrays_are_dropped_whole():
    # 한 칸 밀리면 다른 사람의 태그가 붙는다. 반만 살리지 않는다.
    vlm = _analysis(2)
    changed = apply_person_tags(vlm, {"person_actions": ["sitting"], "person_views": []})
    assert changed is False
    assert vlm.person_actions == [None, None]
    assert vlm.person_views == [None, None]


def test_asking_is_recorded_even_when_the_answer_is_useless():
    # 물어본 Job과 묻지 않은 Job을 나중에 구분해야 한다(legacy_cut 경로가 갈린다).
    vlm = _analysis(1)
    apply_person_tags(vlm, {"person_actions": [None], "person_views": [None]})
    assert "person_actions" in vlm.raw and "person_views" in vlm.raw


def test_a_failed_call_changes_nothing():
    vlm = _analysis(2)
    assert apply_person_tags(vlm, None) is False
    assert vlm.person_actions == []
    assert vlm.raw == {}


class _Client:
    """분석은 늘 같은 값을 주고, 태그 호출만 시나리오대로 움직이는 가짜 클라이언트."""

    def __init__(self, behaviour="ok", delay=0.0, num=2):
        self.behaviour = behaviour
        self.delay = delay
        self.num = num
        self.tag_calls = 0

    def analyze(self, image, img_w, img_h):
        return _analysis(self.num)

    def tag_people(self, image, boxes):
        self.tag_calls += 1
        time.sleep(self.delay)
        if self.behaviour == "raise":
            raise RuntimeError("upstream down")
        if self.behaviour == "garbage":
            # 사람 수를 제멋대로 바꿔 보내는 경우.
            return {"person_actions": ["sitting"] * 9, "person_views": ["side"] * 9}
        return {"person_actions": ["sitting", "walking"][:self.num],
                "person_views": ["side", "front"][:self.num]}


def _pipeline(client):
    from src.pipeline import Pipeline

    return Pipeline(entries=[], vlm_client=client)


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(CFG, "vlm_person_tags", True)
    monkeypatch.setattr(CFG, "vlm_person_tags_timeout_ms", 2000)


def test_disabled_by_default_makes_no_second_call(monkeypatch):
    monkeypatch.setattr(CFG, "vlm_person_tags", False)
    client = _Client()
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    pipeline._start_person_tags(None, 100, 200, vlm)
    pipeline._join_person_tags(vlm)
    assert client.tag_calls == 0
    assert vlm.person_actions == []


def test_tags_are_joined_before_descriptors():
    client = _Client()
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    pipeline._start_person_tags(None, 100, 200, vlm)
    pipeline._join_person_tags(vlm)
    assert client.tag_calls == 1
    assert vlm.person_actions == [Action.SITTING, Action.WALKING]


def test_a_raising_tag_call_leaves_the_analysis_untouched():
    client = _Client(behaviour="raise")
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    before = (vlm.num_people, vlm.shot, len(vlm.approx_boxes))
    pipeline._start_person_tags(None, 100, 200, vlm)
    pipeline._join_person_tags(vlm)
    assert (vlm.num_people, vlm.shot, len(vlm.approx_boxes)) == before
    assert vlm.person_actions == [None, None] or vlm.person_actions == []


def test_a_slow_tag_call_does_not_hold_the_analysis(monkeypatch):
    monkeypatch.setattr(CFG, "vlm_person_tags_timeout_ms", 100)
    client = _Client(delay=1.5)
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    pipeline._start_person_tags(None, 100, 200, vlm)
    started = time.monotonic()
    pipeline._join_person_tags(vlm)
    assert time.monotonic() - started < 1.0
    assert vlm.person_actions == []


def test_a_tag_call_cannot_change_the_person_count():
    client = _Client(behaviour="garbage")
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    pipeline._start_person_tags(None, 100, 200, vlm)
    pipeline._join_person_tags(vlm)
    assert vlm.num_people == 2
    assert vlm.person_actions == [None, None]


def test_joining_twice_calls_once():
    client = _Client()
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    pipeline._start_person_tags(None, 100, 200, vlm)
    pipeline._join_person_tags(vlm)
    pipeline._join_person_tags(vlm)
    assert client.tag_calls == 1


def test_no_people_means_no_call():
    client = _Client(num=0)
    pipeline = _pipeline(client)
    vlm = client.analyze(None, 100, 200)
    pipeline._start_person_tags(None, 100, 200, vlm)
    pipeline._join_person_tags(vlm)
    assert client.tag_calls == 0
