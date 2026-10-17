"""Independent unresolved-family review cannot modify control or resolved people."""

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace
import pytest
from PIL import Image
from src.vlm.client import _coerce, GeminiVLMClient
from src.config import CFG
from src.experimental.body_matching import integrated_review as review
from src.experimental.body_matching.family_review import validate_metadata
from src.experimental.body_matching.coarse import recommend
from tests.test_integrated_body import body, payload, ASSETS


def reading(family="female", certainty="tentative"):
    return dict(
        face="눈과 입의 그림 표현",
        clothing="상의 목선과 옷단",
        body_design="눈에 보이는 소체",
        family=family,
        certainty=certainty,
        reason="얼굴·의상·몸 표현을 함께 비교한 추정",
    )


def setup_scene(presentation="androgynous", certainty="unknown"):
    raw = payload()
    rows = [body(), body(1, "feminine")]
    rows[0]["axes"]["presentation"].update(value=presentation, certainty=certainty)
    raw["person_bodies"] = rows
    return raw, _coerce(raw, 200, 300)


def runner(analysis, response=None, error=None, deadline=30):
    calls = []

    def generate(**kw):
        calls.append(kw)
        if error:
            raise error
        return SimpleNamespace(
            text=json.dumps(response or {"people": [dict(person_id="V0", **reading())]})
        )

    vlm = SimpleNamespace(
        _clock=lambda: 10.0,
        _model="offline",
        _client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
    )
    cfg = SimpleNamespace(
        body_max_people=8, body_timeout_seconds=12, vlm_body_thinking_level="low"
    )
    review.review_uncertain(vlm, Image.new("RGB", (200, 300)), analysis, cfg, deadline)
    return calls


def test_unresolved_family_review_is_one_batched_call_and_only_changes_body():
    raw, analysis = setup_scene()
    before = asdict(analysis)
    original = copy.deepcopy(raw)
    calls = runner(analysis)
    assert len(calls) == 1 and calls[0]["config"].http_options.timeout == 12000
    assert calls[0]["config"].thinking_config.thinking_level.value.lower() == "low"
    assert calls[0]["config"].system_instruction is None
    assert calls[0]["contents"][0] == review.PROMPT
    assert calls[0]["config"].http_options.retry_options.attempts == 1
    assert analysis.raw == original
    after = asdict(analysis)
    before.pop("person_bodies")
    after.pop("person_bodies")
    assert before == after
    assert analysis.person_bodies[1] == _coerce(raw, 200, 300).person_bodies[1]
    o = analysis.person_bodies[0]
    validate_metadata(o["family_review"])
    assert o["family_review"]["status"] == "resolved"
    assert o["axes"]["presentation"]["certainty"] == "tentative"
    assert recommend(o, ASSETS)["recommended_body_id"] == "female-base-v0-3-4"
    # No author settings, old conclusion or expected preset is sent remotely.
    assert "androgynous" not in str(calls[0]["contents"])


@pytest.mark.parametrize(
    "value,certainty",
    [
        ("masculine", "tentative"),
        ("feminine", "tentative"),
        ("masculine", "clear"),
        ("feminine", "clear"),
    ],
)
def test_resolved_binary_families_are_never_reviewed(value, certainty):
    raw, analysis = setup_scene(value, certainty)
    before = copy.deepcopy(analysis.person_bodies)
    assert runner(analysis) == [] and analysis.person_bodies == before


@pytest.mark.parametrize(
    "kind", ["inverted", "missing", "ambiguous", "stick", "budget"]
)
def test_review_cannot_guess_invalid_target_or_reset_budget(kind):
    raw, analysis = setup_scene()
    if kind == "inverted":
        analysis.approx_boxes[0].x2 = 0
    elif kind == "missing":
        analysis.approx_boxes[0] = None
    elif kind == "ambiguous":
        analysis.person_bodies[0]["target"] = "ambiguous"
    elif kind == "stick":
        analysis.person_bodies[0]["representation"] = "stick"
    before = copy.deepcopy(analysis.person_bodies)
    calls = runner(analysis, deadline=10.5 if kind == "budget" else 30)
    assert not calls
    assert analysis.person_bodies[0]["axes"] == before[0]["axes"]
    if kind == "budget":
        assert analysis.person_bodies[0]["family_review"]["status"] == "error"


@pytest.mark.parametrize("kind", ["error", "malformed", "missing_id", "unresolved"])
def test_review_failure_keeps_primary_and_records_actual_failure(kind):
    raw, analysis = setup_scene()
    before = copy.deepcopy(analysis.person_bodies)
    response = {"people": [dict(person_id="V0", **reading("unresolved", "unknown"))]}
    if kind == "malformed":
        response = {"people": False}
    elif kind == "missing_id":
        response = {"people": []}
    calls = runner(
        analysis, response, error=TimeoutError() if kind == "error" else None
    )
    assert len(calls) == 1 and analysis.raw == raw
    assert analysis.person_bodies[0]["axes"] == before[0]["axes"]
    validate_metadata(analysis.person_bodies[0]["family_review"])
    assert analysis.person_bodies[0]["family_review"]["status"] == (
        "unresolved" if kind == "unresolved" else "error"
    )


def test_remaining_main_deadline_caps_review_timeout():
    _, analysis = setup_scene()
    assert runner(analysis, deadline=12.5)[0]["config"].http_options.timeout == 2500


def test_two_uncertain_people_share_one_review_rpc():
    _, analysis = setup_scene()
    analysis.person_bodies[1]["axes"]["presentation"].update(
        value="unknown", certainty="unknown"
    )
    calls = runner(
        analysis,
        {
            "people": [
                dict(person_id="V1", **reading("male", "clear")),
                dict(person_id="V0", **reading()),
            ]
        },
    )
    assert len(calls) == 1
    assert [o["axes"]["presentation"]["value"] for o in analysis.person_bodies] == [
        "feminine",
        "masculine",
    ]


def test_optional_review_exception_never_retries_successful_primary(monkeypatch):
    raw, _ = setup_scene()
    calls = []

    def generate(**kw):
        calls.append(kw)
        return SimpleNamespace(text=json.dumps(raw))

    def fail(*args):
        raise RuntimeError("independent review failure")

    monkeypatch.setattr(review, "review_uncertain", fail)
    monkeypatch.setattr(CFG, "body_matching_mode", "shadow")
    monkeypatch.setattr(CFG, "body_classification_strategy", "integrated")
    monkeypatch.setattr(CFG, "body_integrated_review_enabled", True)
    client = GeminiVLMClient(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        model="offline",
    )
    got = client.analyze(Image.new("RGB", (200, 300)), 200, 300)
    assert len(calls) == 1 and got.raw == raw


@pytest.mark.parametrize("presentation", ["androgynous", "unknown"])
def test_family_resolution_keeps_clear_independent_body_traits(presentation):
    raw, analysis = setup_scene(presentation)
    initial = analysis.person_bodies[0]
    for name, value in [
        ("muscle", "muscular"),
        ("fullness", "slim"),
        ("proportion", "elongated"),
    ]:
        initial["axes"][name].update(value=value, certainty="clear")
    expected = copy.deepcopy(initial["axes"])
    runner(analysis)
    for name in ("muscle", "fullness", "proportion"):
        assert analysis.person_bodies[0]["axes"][name] == expected[name]
    assert analysis.person_bodies[0]["axes"]["presentation"]["value"] == "feminine"


def test_actual_binary_family_reversal_still_downgrades_special_traits():
    from src.experimental.body_matching.family_review import apply_review

    _, analysis = setup_scene("masculine", "tentative")
    initial = analysis.person_bodies[0]
    initial["axes"]["muscle"].update(value="muscular", certainty="clear")
    got = apply_review(
        initial, reading("female", "clear"), preserve_unresolved_features=True
    )
    assert got["axes"]["muscle"]["certainty"] == "tentative"
    # Existing standalone coarse policy retains its previous BASE reset.
    initial["axes"]["presentation"].update(value="androgynous", certainty="unknown")
    old = apply_review(initial, reading("female", "clear"))
    assert old["axes"]["muscle"]["certainty"] == "tentative"
