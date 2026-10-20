import copy
import json
from types import SimpleNamespace
import pytest
from PIL import Image
from src.experimental.body_matching import family_review as fr
from src.experimental.body_matching.coarse import (
    GeminiCoarseClient,
    parse_response,
    recommend,
)
from tests.test_body_coarse import observation, ASSETS, CATALOG, Client
from src.experimental.body_matching.coarse_service import CoarseBodyService
from api.body_models import BodyMatchingOut
from tests.test_body_matching import cut


def reading(family="female", certainty="clear"):
    return dict(
        face="얼굴의 표현",
        clothing="의상의 목선과 옷단",
        body_design="몸의 디자인",
        family=family,
        certainty=certainty,
        reason="얼굴과 의상 및 체형의 종합 인상",
    )


def tentative(**kwargs):
    o = observation(**kwargs)
    o["axes"]["presentation"]["certainty"] = "tentative"
    return o


def test_review_corrects_family_without_id_or_manual_hint_and_starts_at_base():
    o = tentative(
        person_id="unseen-random", muscle="very_muscular", proportion="elongated"
    )
    original = copy.deepcopy(o)
    after = fr.apply_review(o, reading())
    rec = recommend(after, ASSETS)
    assert rec["recommended_family"] == "female" and rec["recommended_preset"] == "BASE"
    assert after["axes"]["muscle"]["value"] == "very_muscular"
    assert after["axes"]["muscle"]["certainty"] == "tentative"
    assert o == original
    assert after["family_review"]["initial_presentation"] == o["axes"]["presentation"]
    assert rec["auto_apply"] is False and rec["applied_body_id"] is None
    payload = {k: v for k, v in after.items() if k != "version"}
    assert (
        parse_response({"people": [payload]}, [o["person_id"]])[o["person_id"]] == after
    )


def test_same_family_confirmation_preserves_genuine_special_features():
    o = tentative(muscle="very_muscular")
    after = fr.apply_review(o, reading("male"))
    assert after["axes"]["muscle"] == o["axes"]["muscle"]
    assert recommend(after, ASSETS)["recommended_preset"] == "MUSCLE_MAX"


@pytest.mark.parametrize("kind", ["clear", "ambiguous", "stick", "unreadable"])
def test_review_cannot_override_clear_family_or_wrong_target(kind):
    o = observation() if kind == "clear" else tentative()
    if kind == "ambiguous":
        o["target"] = "ambiguous"
    elif kind not in ("clear", "ambiguous"):
        o["representation"] = kind
    assert not fr.eligible(o)
    assert fr.apply_review(o, reading()) == o


@pytest.mark.parametrize(
    "review,error",
    [
        (reading("unresolved", "unknown"), None),
        (reading("female", "unknown"), None),
        (None, "TimeoutError"),
    ],
)
def test_unresolved_or_failed_review_keeps_primary_recommendation(review, error):
    o = tentative(muscle="muscular")
    after = fr.apply_review(o, review, error=error)
    assert after["axes"] == o["axes"]
    assert recommend(after, ASSETS) == recommend(o, ASSETS)
    fr.validate_metadata(after["family_review"])


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing", "wrong", "blank", "extra", "bad_family"]
)
def test_review_rejects_misaligned_or_invalid_evidence(kind):
    p = dict(person_id="opaque", **reading())
    raw = dict(people=[p])
    if kind == "duplicate":
        raw["people"].append(copy.deepcopy(p))
    elif kind == "missing":
        raw["people"] = []
    elif kind == "wrong":
        p["person_id"] = "neighbor"
    elif kind == "blank":
        p["clothing"] = " "
    elif kind == "extra":
        p["hint"] = "label"
    else:
        p["family"] = "BASE"
    with pytest.raises(ValueError):
        fr.parse_response(raw, ["opaque"])


def test_service_keeps_review_metadata_and_other_family_alternative_on_wire():
    class Reviewed(Client):
        def analyze(self, crops):
            return {
                ident: fr.apply_review(tentative(person_id=ident), reading())
                for ident, _ in crops
            }

    out = CoarseBodyService(CATALOG, client=Reviewed(), provider="gemini").analyze(
        Image.new("RGB", (100, 100)), cut(1)
    )
    wire = BodyMatchingOut.model_validate(out).model_dump()
    p = wire["people"][0]
    assert p["coarse_recommendation"]["recommended_family"] == "female"
    assert p["coarse_classification"]["family_review"]["status"] == "resolved"
    assert (
        p["candidates"] == []
        and p["selected_asset"] is None
        and p["applied_body_id"] is None
    )


@pytest.mark.parametrize("failure", [False, True])
def test_client_only_reviews_uncertain_person_using_same_crop_pair_and_remaining_budget(
    failure,
):
    pytest.importorskip("google.genai")
    from src.experimental.body_matching.coarse_input import prepare_input

    first = [observation(person_id="sure"), tentative(person_id="unsure")]
    for o in first:
        o.pop("version")
    calls = []

    def generate(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return SimpleNamespace(text=json.dumps(dict(people=first)))
        if failure:
            raise TimeoutError("should not appear in diagnostics")
        return SimpleNamespace(
            text=json.dumps(dict(people=[dict(person_id="unsure", **reading())]))
        )

    c = GeminiCoarseClient.__new__(GeminiCoarseClient)
    c.model, c.family_review_model, c.timeout_seconds = "primary", "review", 30
    c.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    item = prepare_input(Image.new("RGB", (300, 500)), (80, 150, 130, 450))
    out = c.analyze([("sure", item), ("unsure", item)])
    assert len(calls) == 2 and calls[1]["model"] == "review"
    parts = calls[1]["contents"]
    assert "TARGET person_id=sure" not in parts and "TARGET person_id=unsure" in parts
    assert len([p for p in parts if not isinstance(p, str)]) == 2
    assert 0 < calls[1]["config"].http_options.timeout <= 30000
    assert "family_review" not in out["sure"]
    assert out["unsure"]["family_review"]["status"] == (
        "error" if failure else "resolved"
    )
    if failure:
        assert out["unsure"]["family_review"]["error"] == "TimeoutError"
        assert out["unsure"]["axes"] == first[1]["axes"]


def test_exhausted_total_budget_does_not_start_second_request():
    pytest.importorskip("google.genai")
    o = tentative()
    o.pop("version")
    calls = []

    def generate(**kw):
        calls.append(kw)
        return SimpleNamespace(text=json.dumps(dict(people=[o])))

    c = GeminiCoarseClient.__new__(GeminiCoarseClient)
    c.model, c.family_review_model, c.timeout_seconds = "primary", "review", 0
    c.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    out = c.analyze([("p0", Image.new("RGB", (20, 20)))])
    assert len(calls) == 1
    assert out["p0"]["family_review"]["error"] == "TimeoutError"
    assert out["p0"]["axes"] == o["axes"]


@pytest.mark.parametrize(
    "initial_family,review_family", [("masculine", "female"), ("feminine", "male")]
)
def test_equal_confidence_opposite_review_cannot_flip_existing_recommendation(
    initial_family, review_family
):
    o = tentative(presentation=initial_family, muscle="muscular")
    after = fr.apply_review(o, reading(review_family, "tentative"))
    assert after["axes"] == o["axes"]
    assert after["family_review"]["status"] == "conflict"
    assert recommend(after, ASSETS) == recommend(o, ASSETS)
    fr.validate_metadata(after["family_review"])


def test_tentative_review_can_resolve_previously_unassigned_family():
    o = tentative(presentation="androgynous")
    after = fr.apply_review(o, reading("female", "tentative"))
    assert after["axes"]["presentation"]["value"] == "feminine"
    assert after["family_review"]["status"] == "resolved"
    fr.validate_metadata(after["family_review"])


def test_nullable_review_field_from_typed_wire_cache_keeps_primary_observation():
    from api.body_models import CoarseClassificationOut

    original = observation()
    cached = CoarseClassificationOut.model_validate(original).model_dump()
    assert cached["family_review"] is None
    cached.pop("version")
    parsed = parse_response({"people": [cached]}, ["p0"])["p0"]
    assert parsed["axes"] == original["axes"]
    assert recommend(parsed, ASSETS) == recommend(original, ASSETS)
