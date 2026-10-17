import copy, json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from src.experimental.body_matching.coarse import (
    AXES,
    VERSION,
    parse_response,
    recommend,
    unknown,
)
from src.experimental.body_matching.coarse_service import CoarseBodyService
from src.experimental.body_matching.service import attach_body_matching
from src.config import Config
from api.body_models import BodyMatchingOut
from tests.test_body_matching import cut

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "config/body_morphology.v1.json"
ASSETS = json.loads(CATALOG.read_text())["assets"]


def observation(
    presentation="masculine",
    muscle="ordinary",
    fullness="average",
    proportion="standard",
    certainty="clear",
    person_id="p0",
    **flags
):
    p = dict(
        person_id=person_id,
        target="clear",
        representation="character",
        proportion_readable=True,
        axes={
            k: dict(value=v, certainty=certainty, reason="그림에서 보이는 특징")
            for k, v in dict(
                presentation=presentation,
                muscle=muscle,
                fullness=fullness,
                proportion=proportion,
            ).items()
        },
    )
    p.update(flags)
    return parse_response({"people": [p]}, [person_id])[person_id]


@pytest.mark.parametrize("family", ["masculine", "feminine"])
def test_family_first_ordinary_is_base(family):
    r = recommend(observation(family), ASSETS)
    assert r["recommended_preset"] == "BASE"
    assert r["recommendation_basis"] == "presentation_default"
    assert all(
        c["family"] == ("male" if family == "masculine" else "female")
        for c in r["candidates"]
    )


@pytest.mark.parametrize(
    "family,feature,value,preset",
    [
        ("masculine", "muscle", "muscular", "MUSCULAR"),
        ("masculine", "muscle", "very_muscular", "MUSCLE_MAX"),
        ("feminine", "muscle", "muscular", "MUSCLE_MAX"),
        ("feminine", "muscle", "very_muscular", "NEW_MUSCULAR"),
        ("masculine", "proportion", "elongated", "IDEAL83"),
        ("feminine", "proportion", "elongated", "IDEAL83"),
        ("masculine", "proportion", "compact", "TOON64"),
        ("feminine", "proportion", "compact", "TOON64"),
        ("masculine", "fullness", "heavy", "FAT_MAX"),
        ("feminine", "fullness", "heavy", "FAT_MAX"),
        ("masculine", "fullness", "soft", "SOFT"),
        ("feminine", "fullness", "soft", "SOFT"),
        ("feminine", "fullness", "slim", "SLIM"),
        ("masculine", "fullness", "slim", "BASE"),
    ],
)
def test_clear_feature_routes_to_existing_same_family_preset(
    family, feature, value, preset
):
    r = recommend(observation(presentation=family, **{feature: value}), ASSETS)
    assert r["recommended_preset"] == preset
    assert all(c["family"] == r["recommended_family"] for c in r["candidates"])
    assert any(c["preset"] == "BASE" for c in r["candidates"])
    assert r["recommended_body_id"] in {a["body_id"] for a in ASSETS}


@pytest.mark.parametrize(
    "feature,value",
    [("muscle", "very_muscular"), ("fullness", "heavy"), ("proportion", "elongated")],
)
def test_tentative_feature_cannot_override_base(feature, value):
    o = observation()
    o["axes"][feature] = dict(value=value, certainty="tentative", reason="약한 인상")
    assert recommend(o, ASSETS)["recommended_preset"] == "BASE"


def test_missing_lower_body_does_not_erase_clear_muscle():
    o = observation(
        muscle="very_muscular", proportion="compact", proportion_readable=False
    )
    assert o["axes"]["proportion"]["value"] == "unknown"
    assert recommend(o, ASSETS)["recommended_preset"] == "MUSCLE_MAX"


def test_unknown_body_still_gets_family_base():
    o = observation(
        "feminine", muscle="unknown", fullness="unknown", proportion="unknown"
    )
    r = recommend(o, ASSETS)
    assert r["recommended_family"] == "female" and r["recommended_preset"] == "BASE"


def test_muscular_female_never_switches_to_male():
    r = recommend(
        observation("feminine", muscle="very_muscular", fullness="heavy"), ASSETS
    )
    assert r["recommended_preset"] == "NEW_MUSCULAR"
    assert {c["family"] for c in r["candidates"]} == {"female"}


def test_every_ambiguous_case_records_explicit_fallback():
    r = recommend(unknown("p0", "가려짐"), ASSETS)
    assert r["recommended_body_id"] == "male-base-v0-7-2"
    assert (
        r["observed_family"] is None and r["recommendation_basis"] == "catalog_default"
    )
    assert {c["family"] for c in r["candidates"]} == {"male", "female"}


@pytest.mark.parametrize("representation", ["stick", "face_only", "unreadable"])
def test_no_shape_feature_from_unreadable_representation(representation):
    o = observation(
        muscle="very_muscular",
        fullness="heavy",
        proportion="compact",
        representation=representation,
    )
    assert all(
        o["axes"][k]["value"] == "unknown" for k in ("muscle", "fullness", "proportion")
    )
    assert recommend(o, ASSETS)["recommended_preset"] == "BASE"


def test_no_age_or_missing_fine_asset_invention():
    o = observation(proportion="compact")
    r = recommend(o, ASSETS)
    assert r["recommended_preset"] == "TOON64"
    assert not any(
        "child" in c["body_id"] or "teen" in c["body_id"] for c in r["candidates"]
    )


def test_duplicate_identity_rejected():
    o = observation()
    o.pop("version")
    with pytest.raises(ValueError):
        parse_response({"people": [o, o]}, ["p0"])


class Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def analyze(self, crops):
        self.calls += 1
        if self.fail:
            raise TimeoutError("fixture")
        return {
            ident: observation("feminine", muscle="muscular", person_id=ident)
            for ident, _ in crops
        }


def test_service_keeps_all_recommendations_and_bff_selection_contract():
    client = Client()
    service = CoarseBodyService(CATALOG, client=client, provider="gemini")
    result = cut(2)
    result.descriptors[0].skeleton_state = "failed"
    before = copy.deepcopy(result.person_candidates)
    out = service.analyze(
        Image.new("RGB", (200, 200)), result, mode="auto", pose_is_mock=True
    )
    validated = BodyMatchingOut.model_validate(out)
    assert client.calls == 1 and len(validated.people) == 2
    for p in validated.people:
        assert p.coarse_recommendation.recommended_preset == "MUSCLE_MAX"
        assert p.candidates == []
        assert p.coarse_recommendation.observed_family == "female"
        assert (
            p.auto_body_id is None
            and p.applied_body_id is None
            and not p.render_required
        )
    assert [[p.pose_id for p in x] for x in before] == [
        [p.pose_id for p in x] for x in result.person_candidates
    ]


def test_provider_error_records_fallback_for_every_person():
    out = CoarseBodyService(
        CATALOG, client=Client(fail=True), provider="gemini"
    ).analyze(Image.new("RGB", (200, 200)), cut(2))
    BodyMatchingOut.model_validate(out)
    assert out["status"] == "partial"
    assert all(
        p["coarse_recommendation"]["recommendation_basis"] == "catalog_default"
        for p in out["people"]
    )
    assert all(
        p["coarse_recommendation"]["recommended_preset"] == "BASE"
        for p in out["people"]
    )


def test_strategy_opt_in_keeps_legacy_default():
    assert (
        Config.__dataclass_fields__["body_classification_strategy"].default == "legacy"
    )
    with pytest.raises(ValueError):
        Config(body_classification_strategy="invalid")
    cfg = Config(
        body_matching_mode="shadow",
        body_classification_strategy="coarse",
        body_coarse_catalog_path=str(CATALOG),
        body_vlm_provider="mock",
    )
    pipe = SimpleNamespace(pose=None)
    result = cut()
    attach_body_matching(pipe, Image.new("RGB", (200, 200)), result, cfg)
    assert isinstance(pipe.body_matcher, CoarseBodyService)
    assert (
        result.body_matching["people"][0]["coarse_recommendation"]["recommended_preset"]
        == "BASE"
    )
