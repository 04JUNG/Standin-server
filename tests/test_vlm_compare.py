"""VLM 프롬프트 A/B 비교 게이트(standin_eval/vlm_compare.py)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.vlm.client import _coerce
from standin_eval.dataset import EvalDataset
from standin_eval.vlm_compare import evaluate, matched_box_iou, run_vlm_compare, write_vlm_compare


class _Image:
    size = (100, 200)


def _payload(num=1, shot="full_half", relationship=None, actions=None, views=None, x=0.1):
    payload = {
        "num_people": num, "shot": shot, "action": "standing", "view": "front",
        "relationship": relationship or ("solo" if num == 1 else "talking"),
        "approx_boxes": [{"x1": x + 0.4 * i, "y1": 0.1, "x2": x + 0.4 * i + 0.3, "y2": 0.9}
                         for i in range(num)],
    }
    if actions is not None:
        payload["person_actions"] = actions
    if views is not None:
        payload["person_views"] = views
    return payload


class _ScriptedVLM:
    """프롬프트 버전과 그 버전의 호출 횟수로 응답을 고른다."""

    def __init__(self, script, clock=None, latency=None):
        self.script, self.calls, self.seen = script, {}, []
        self.clock, self.latency = clock, latency or {}

    def analyze(self, image, width, height):
        version = CFG.vlm_prompt_version
        count = self.calls.get(version, 0)
        self.calls[version] = count + 1
        self.seen.append(version)
        if self.clock is not None:
            self.clock.now += self.latency.get(version, 1.0)
        payload = self.script(version, count)
        if isinstance(payload, Exception):
            raise payload
        return _coerce(payload, width, height)


class _Clock:
    now = 0.0

    def __call__(self):
        return self.now


def _run(script, repeats=3, cuts=("c1", "c2"), **kwargs):
    client = _ScriptedVLM(script, **kwargs)
    samples = run_vlm_compare([{"cut_id": cut} for cut in cuts], prompt_a="p1-scope",
                              prompt_b="p2-person-tags", repeats=repeats, client=client,
                              load_image=lambda cut: _Image(),
                              clock=kwargs.get("clock") or _Clock())
    return client, evaluate(samples)


def _same(version, count):
    if version == "p2-person-tags":
        return _payload(actions=["standing"], views=["front"])
    return _payload()


def test_the_same_answers_pass_and_record_person_tag_coverage():
    _, report = _run(_same)
    assert report["passed"], report["checks"]
    assert report["metrics"]["route_change_excess"] == 0
    assert report["metrics"]["alignment_rate"] == 1.0
    assert report["metrics"]["person_tag_fill_rate"] == 1.0


def test_arms_alternate_and_the_prompt_setting_is_restored(monkeypatch):
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p1-scope")
    client, _ = _run(_same, repeats=2, cuts=("c1",))
    assert client.seen == ["p1-scope", "p2-person-tags", "p2-person-tags", "p1-scope"]
    assert CFG.vlm_prompt_version == "p1-scope"

    def boom(version, count):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(boom, repeats=2, cuts=("c1",))
    assert CFG.vlm_prompt_version == "p1-scope"


def test_jitter_shared_by_both_prompts_is_noise_not_regression():
    def jitter(version, count):
        num = 1 if count % 2 == 0 else 2
        tags = {"actions": ["standing"] * num, "views": ["front"] * num}
        return _payload(num=num, **tags) if version == "p2-person-tags" else _payload(num=num)

    _, report = _run(jitter, repeats=4)
    assert report["noise_floor"]["count_agreement"] < 1
    assert report["checks"]["count_agreement"], report["metrics"]


def test_a_route_flip_fails_the_gate():
    def flip(version, count):
        return _payload(shot="bust", actions=["standing"], views=["front"]) \
            if version == "p2-person-tags" else _payload()

    _, report = _run(flip)
    assert not report["passed"]
    assert not report["checks"]["route_change"]
    assert report["metrics"]["route_change_excess"] == 1.0


def test_misaligned_person_arrays_fail_alignment():
    def misaligned(version, count):
        return _payload(actions=["standing", "sitting"], views=["front"]) \
            if version == "p2-person-tags" else _payload()

    _, report = _run(misaligned)
    assert report["metrics"]["alignment_rate"] == 0.0
    assert not report["checks"]["alignment"]


def test_a_single_parse_failure_in_b_fails():
    def flaky(version, count):
        if version == "p2-person-tags" and count == 0:
            return ValueError("bad json")
        return _same(version, count)

    _, report = _run(flaky)
    assert report["metrics"]["parse_failures"] == {"A": 0, "B": 1}
    assert not report["checks"]["parse_failures_b"]


def test_slower_answers_fail_the_latency_check():
    clock = _Clock()
    _, report = _run(_same, clock=clock, latency={"p1-scope": 1.0, "p2-person-tags": 1.5})
    assert report["metrics"]["latency_p95_increase"] == pytest.approx(0.5)
    assert not report["checks"]["latency"]


def test_boxes_shifted_by_b_lower_the_iou():
    def shifted(version, count):
        return _payload(x=0.3, actions=["standing"], views=["front"]) \
            if version == "p2-person-tags" else _payload()

    _, report = _run(shifted)
    assert report["metrics"]["box_iou_median_drop"] > 0.03
    assert not report["checks"]["box_iou"]


def test_bad_arguments_stop_before_any_call():
    client = _ScriptedVLM(_same)
    with pytest.raises(ValueError):
        run_vlm_compare([{"cut_id": "c1"}], prompt_a="p1-scope", prompt_b="p9", repeats=3,
                        client=client, load_image=lambda cut: _Image())
    with pytest.raises(ValueError):
        run_vlm_compare([{"cut_id": "c1"}], prompt_a="p1-scope", prompt_b="p2-person-tags",
                        repeats=1, client=client, load_image=lambda cut: _Image())
    assert client.seen == []


def test_matched_box_iou_pairs_each_box_once():
    box = (0.1, 0.1, 0.4, 0.9)
    assert matched_box_iou([box], [box]) == 1.0
    assert matched_box_iou([box, None], [box, box]) == 1.0
    assert matched_box_iou([], [box]) is None


def test_report_from_a_dataset_keeps_only_ids_and_numbers(tmp_path):
    from PIL import Image

    image_path = tmp_path / "secret-rough-name.png"
    Image.new("RGB", (32, 48), "white").save(image_path)
    dataset = EvalDataset(root=tmp_path, manifest={"dataset_id": "tiny"},
                          cuts=[{"cut_id": "c1", "image_path": str(image_path)}], persons=[])

    report = write_vlm_compare(dataset, provider="mock", prompt_a="p1-scope",
                               prompt_b="p2-person-tags", repeats=2,
                               output_root=str(tmp_path / "out"))

    assert report["passed"], report["checks"]
    stored = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "secret-rough-name" not in stored
    assert json.loads(stored)["prompts"] == {"A": "p1-scope", "B": "p2-person-tags"}
