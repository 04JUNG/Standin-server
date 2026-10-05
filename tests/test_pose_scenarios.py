"""Scenario contracts: actual hand transforms, intent guards and review routing."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_pose_qa import candidate
from tests.test_pose_curation import library
from pose_curation.scenarios.recipes import recipe_for
from pose_curation.scenarios.build import set_hands
from pose_curation.scenarios.hands import orient_palms
from pose_curation.scenarios.validation import reach_error
from pose_curation.qa.checks import inspect_pose, preview_issues
from pose_curation.review.app import create_app
from pose_curation.storage import read_json, write_json
from src.bvh import parse_bvh, fk, channel_starts


def test_catalog_has_fifty_explicit_situations_per_category():
    catalog = read_json(Path("config/pose_scenarios_20261002.json"))
    scenes = catalog["scenes"]
    assert Counter(s["category"] for s in scenes) == dict.fromkeys(
        ["daily", "fantasy", "combat", "romance"], 50
    )
    assert len({s["label"] for s in scenes}) == len({s["id"] for s in scenes}) == 200
    assert all(recipe_for(s)["label"] == s["label"] for s in scenes)
    assert any(s["label"] == "앉아서 머리 말리기" for s in scenes)
    assert all(
        any(kind in s["props"] for s in scenes)
        for kind in ["spear", "sword", "cup", "crate"]
    )


def test_hand_styling_and_forearm_roll_preserve_body_endpoints(candidate):
    _, _, _, _, _, get = candidate
    path = get().bvh
    text = path.read_text(encoding="utf-8")
    for side in ["Left", "Right"]:
        for old, new in [("Shoulder", "Arm"), ("Elbow", "ForeArm"), ("Wrist", "Hand")]:
            text = text.replace(side + old, side + new)
    path.write_text(text, encoding="utf-8")
    joints, before = parse_bvh(str(path))
    set_hands(path, ["cup", "grip"])
    _, styled = parse_bvh(str(path))
    for j, start in zip(joints, channel_starts(joints)):
        if not any(
            "Hand" + f in j[0] for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        ):
            assert np.array_equal(
                before[0, start : start + len(j[3])],
                styled[0, start : start + len(j[3])],
            )
    positions = fk(joints, styled[0])
    changes = orient_palms(path, {"hands": ["cup", "grip"]})
    _, after = parse_bvh(str(path))
    result = fk(joints, after[0])
    for i, j in enumerate(joints):
        if not j[4] and not any(
            "Hand" + f in j[0] for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        ):
            assert np.allclose(positions[i], result[i], atol=1e-5)
    assert max(abs(v) for v in changes.values()) <= 95


def test_prop_change_invalidates_preview_and_reach_failure_blocks(candidate):
    _, _, batch, record, save, get = candidate
    record["scenario"] = {"id": "new"}
    record["checks"] = {
        "ik_target_adjustments_cm": dict.fromkeys(
            ["Leftarm", "Rightarm", "Leftleg", "Rightleg"], 0
        )
    }
    record["prop_guides"] = "cup:R"
    save()
    assert any("소품" in message for message in preview_issues(get()))
    report_path = batch / "render-results/fixture.json"
    report = read_json(report_path)
    report["scene_spec"] = {"prop_guides": "cup:R", "support": ""}
    write_json(report_path, report)
    assert preview_issues(get()) == []
    record["checks"]["ik_target_adjustments_cm"]["Rightarm"] = 12
    save()
    assert any(
        f["code"] == "scenario.reach" for f in inspect_pose(get(), {})["findings"]
    )


def test_category_filter_and_live_keyword_index(candidate):
    data, curation, _, record, save, _ = candidate
    record.update(
        category="daily",
        category_label="일상",
        style="앉아서 컵 들기",
        scenario={"id": "new"},
    )
    save()
    with TestClient(create_app(data, curation), base_url="http://127.0.0.1") as client:
        assert (
            client.get("/api/poses", params={"category": "daily"}).json()["total"] == 1
        )
        assert (
            client.get("/api/poses", params={"category": "combat"}).json()["total"] == 0
        )
        html = client.get("/scenarios").text
        assert "앉아서 컵 들기" in html and "검수 중" in html


@pytest.mark.parametrize("value", [5, float("nan"), -1])
def test_invalid_or_unreachable_targets_are_never_silently_accepted(value):
    values = dict.fromkeys(["Leftarm", "Rightarm", "Leftleg", "Rightleg"], 0)
    values["Leftarm"] = value
    assert reach_error({"ik_target_adjustments_cm": values})
