"""인물별 VLM 태그(src/person_tags.py)와 프롬프트 버전(src/vlm/prompts.py)."""
from __future__ import annotations

import ast
import io
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.library import build_synthetic_index
from src.person_tags import PersonTags, detect_person_tags
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import Action, View
from src.vlm import prompts
from src.vlm.client import MockVLMClient, _coerce, build_vlm_client
from standin_eval.fixtures import deserialize_vlm, serialize_vlm


def _analysis(num: int = 1, **extra):
    raw = {
        "num_people": num, "shot": "full_half", "action": "sitting", "view": "side",
        "relationship": "solo" if num == 1 else "talking",
        "approx_boxes": [{"x1": 0.4 * i, "y1": 0.1, "x2": 0.4 * i + 0.3, "y2": 0.9}
                         for i in range(num)],
    }
    raw.update(extra)
    return _coerce(raw, 100, 200)


class _Img(str):
    @property
    def hint(self):
        return str(self)


def test_aligned_person_tags_follow_box_order():
    vlm = _analysis(2, person_actions=["sitting", "running"], person_views=["side", "back"])

    assert vlm.person_actions == [Action.SITTING, Action.RUNNING]
    assert vlm.person_views == [View.SIDE, View.BACK]
    assert detect_person_tags(vlm, 1) == PersonTags(Action.RUNNING, View.BACK, "vlm_person")


def test_an_invalid_entry_does_not_shift_the_people_after_it():
    vlm = _analysis(3, person_actions=["jumping", 1, "running"], person_views=None)

    assert vlm.person_actions == [None, None, Action.RUNNING]
    assert vlm.person_views == [None, None, None]
    assert detect_person_tags(vlm, 2) == PersonTags(Action.RUNNING, None, "vlm_person")


def test_misaligned_arrays_stay_unknown():
    vlm = _analysis(2, person_actions=["sitting"], person_views=["side", "back", "front"])

    assert vlm.person_actions == [None, None]
    assert vlm.person_views == [None, None]
    assert detect_person_tags(vlm, 0) == PersonTags()


def test_stated_tags_do_not_inherit_parser_defaults():
    vlm = _analysis(1, action="flying", view="top")

    # 검색·라우팅 쪽 값은 예전처럼 기본값으로 채워진다.
    assert (vlm.action, vlm.view) == (Action.OTHER, View.FRONT)
    assert vlm.stated_tags == {"shot": "full_half", "action": None, "view": None,
                               "relationship": "solo"}


def test_a_solo_cut_from_the_old_prompt_reuses_the_stated_cut_label():
    assert detect_person_tags(_analysis(1), 0) == PersonTags(
        Action.SITTING, View.SIDE, "legacy_cut")


def test_the_cut_label_never_stands_in_for_several_people_or_for_defaults():
    assert detect_person_tags(_analysis(2), 0) == PersonTags()
    assert detect_person_tags(_analysis(1, action="flying", view="top"), 0) == PersonTags()
    # 인물별로 물었는데 비워 둔 경우도 모르는 것으로 둔다.
    assert detect_person_tags(
        _analysis(1, person_actions=[None], person_views=[None]), 0) == PersonTags()


def test_only_vlm_slots_get_person_tags():
    vlm = _analysis(1, person_actions=["sitting"], person_views=["side"])
    assert detect_person_tags(vlm, None) == PersonTags()


def test_p2_prompt_only_adds_the_person_tag_lines():
    p1 = prompts.user_template("p1-scope")
    p2 = prompts.user_template("p2-person-tags")

    assert p1 is prompts.USER_TEMPLATE
    assert p2.replace(prompts._PERSON_TAG_FIELDS, "").replace(prompts._PERSON_TAG_RULES, "") == p1
    actions, views = (re.findall(r'"([a-z_]+)"', line)
                      for line in prompts._PERSON_TAG_FIELDS.splitlines())
    assert set(actions) == {"person_actions"} | {member.value for member in Action}
    assert set(views) == {"person_views"} | {member.value for member in View}


def test_an_unknown_prompt_version_stops_client_creation(monkeypatch):
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p9-typo")
    with pytest.raises(ValueError, match="p9-typo"):
        build_vlm_client()


def test_the_mock_answers_per_person_only_when_the_prompt_asks(monkeypatch):
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p1-scope")
    old = MockVLMClient().analyze(_Img("full_half sitting side 2p"), 100, 200)
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p2-person-tags")
    new = MockVLMClient().analyze(_Img("full_half sitting side 2p"), 100, 200)

    assert old.person_actions == [] and "person_actions" not in old.raw
    assert new.person_actions == [Action.SITTING] * 2
    assert new.person_views == [View.SIDE] * 2


def _search_outcome(result):
    people = []
    for index, desc in enumerate(result.descriptors):
        people.append({
            "tags": desc.tag_dict(),
            "refine_allowed": desc.refine_allowed,
            "coverage": desc.coverage_class,
            "confidence": result.person_confidence[index],
            "candidates": [(c.pose_id, c.view.value, round(c.distance, 9))
                           for c in result.person_candidates[index]],
        })
    return result.route, result.count_confidence, people


@pytest.mark.parametrize("hint", ["full_half sitting side 2p", "full_half standing front 1p",
                                  "bust front 1p"])
def test_person_tags_change_nothing_in_search(monkeypatch, hint):
    pipeline = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                        pose_model=MockPoseModel())
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p1-scope")
    before = pipeline.process_cut(_Img(hint))
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p2-person-tags")
    after = pipeline.process_cut(_Img(hint))

    assert _search_outcome(after) == _search_outcome(before)
    assert all(desc.person_tags.source == "vlm_person" for desc in after.descriptors)


# 인물 태그를 다룰 수 있는 곳. 검색·라우팅·refine 모듈이 이 목록에 들면 안 된다.
_ALLOWED = {
    "src/person_tags.py", "src/schema.py", "src/descriptor.py", "src/pipeline.py",
    "src/vlm/client.py",
}
_TAG_NAMES = {"person_tags", "person_actions", "person_views", "stated_tags",
              "detect_person_tags"}


def test_only_metadata_code_touches_person_tags():
    users = set()
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = (node.attr if isinstance(node, ast.Attribute)
                    else node.id if isinstance(node, ast.Name)
                    else node.arg if isinstance(node, ast.keyword)
                    else None)
            if name in _TAG_NAMES:
                users.add(path.relative_to(ROOT).as_posix())
    assert users <= _ALLOWED, users - _ALLOWED
    # pipeline은 컷 태그를 결과에 옮기기만 한다.
    pipeline_source = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    assert pipeline_source.count("stated_tags") == pipeline_source.count("vlm_tags=vlm.stated_tags")


def test_analyze_reports_person_tags_cut_tags_and_prompt_version(monkeypatch):
    from PIL import Image
    from fastapi import UploadFile
    import api.app as api_app

    image_bytes = io.BytesIO()
    Image.new("RGB", (16, 12), color=(255, 255, 255)).save(image_bytes, format="PNG")
    image_bytes.seek(0)
    monkeypatch.setattr(CFG, "vlm_prompt_version", "p2-person-tags")
    monkeypatch.setattr(api_app, "STATE", {
        "pipeline": Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                             pose_model=MockPoseModel()),
        "provider": "mock", "pose_backend": "mock",
    })

    result = api_app.analyze(
        UploadFile(filename="cut.png", file=image_bytes),
        hint="full_half sitting side 1p",
    ).model_dump(mode="json")

    assert result["people"][0]["person_tags"] == {
        "action": "sitting", "view": "side", "source": "vlm_person"}
    assert result["vlm_tags"] == {"shot": "full_half", "action": "sitting", "view": "side",
                                  "relationship": "solo"}
    assert result["inference_metadata"]["vlm_prompt_version"] == "p2-person-tags"


def test_fixtures_keep_per_person_fields_and_read_old_payloads():
    vlm = _analysis(2, person_actions=["sitting", "running"], person_views=["side", None],
                    lower_body_visible=[True, False], body_scopes=["full", "half"])
    vlm.approx_boxes[1] = None

    replayed = deserialize_vlm(serialize_vlm(vlm))

    assert replayed.approx_boxes[1] is None
    for field in ("lower_body_visible", "lower_body_visibility_known", "body_scopes",
                  "person_actions", "person_views", "stated_tags"):
        assert getattr(replayed, field) == getattr(vlm, field), field

    old = serialize_vlm(vlm)
    for field in ("lower_body_visible", "lower_body_visibility_known", "body_scopes",
                  "person_actions", "person_views", "stated_tags"):
        old.pop(field)
    legacy = deserialize_vlm(old)
    # 예전 픽스처는 provider 원문에서 인물별 값을 다시 읽는다.
    assert legacy.lower_body_visible == [True, False]
    assert legacy.person_actions == [Action.SITTING, Action.RUNNING]
    assert legacy.stated_tags is None
