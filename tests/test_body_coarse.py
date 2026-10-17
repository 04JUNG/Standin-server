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


def test_head_context_and_edge_target_do_not_recenter_on_neighbor():
    from src.experimental.body_matching.coarse_input import prepare_input

    im = Image.new("RGB", (440, 550), "white")
    im.putpixel((410, 10), (0, 0, 255))  # target head above detector box
    im.putpixel((370, 130), (255, 0, 0))  # adjacent person's face
    before = im.tobytes()
    item = prepare_input(im, (401.6, 43.5, 445.5, 394))
    assert item.target_box == (401, 43, 440, 394)
    assert item.crop_box[1] == 0
    assert item.crop_box[0] > 370
    assert item.crop.getpixel((410 - item.crop_box[0], 10)) == (0, 0, 255)
    assert item.crop.size == item.context.size != im.size
    assert im.tobytes() == before
    assert item.crop.tobytes() != item.context.tobytes()


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 0, 10),
        (20, 0, 40, 10),
        (0, float("nan"), 10, 10),
        (0, 0, float("inf"), 10),
    ],
)
def test_invalid_target_cannot_silently_send_whole_image(box):
    from src.experimental.body_matching.coarse_input import prepare_input

    with pytest.raises(ValueError, match="invalid_coarse_target_box"):
        prepare_input(Image.new("RGB", (20, 20)), box)


def test_service_sends_shared_prepared_inputs_with_stable_identity():
    from src.experimental.body_matching.coarse_input import CoarseInput, prepare_input

    im = Image.new("RGB", (200, 200), "white")
    result = cut(2)

    class InspectClient(Client):
        def analyze(self, crops):
            assert len(crops) == 2 and crops[0][0] != crops[1][0]
            for i, (ident, item) in enumerate(crops):
                expected = prepare_input(im, result.descriptors[i].box.as_list())
                assert isinstance(item, CoarseInput)
                assert item.target_box == expected.target_box
                assert item.crop.tobytes() == expected.crop.tobytes()
                assert item.context.tobytes() == expected.context.tobytes()
            return super().analyze(crops)

    out = CoarseBodyService(CATALOG, client=InspectClient(), provider="gemini").analyze(
        im, result
    )
    assert out["status"] == "ok"
    for i, person in enumerate(out["people"]):
        expected = prepare_input(im, result.descriptors[i].box.as_list())
        assert person["observations"]["crop_box"] == list(expected.crop_box)
    assert all(
        p["coarse_recommendation"]["recommended_family"] == "female"
        for p in out["people"]
    )


def test_gemini_payload_contains_only_same_person_crop_pair():
    import io
    from src.experimental.body_matching.coarse_input import prepare_input
    from src.experimental.body_matching.coarse import GeminiCoarseClient

    pytest.importorskip("google.genai")

    item = prepare_input(Image.new("RGB", (600, 900)), (400, 100, 450, 500))
    o = observation(person_id="opaque")
    o.pop("version")
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text=json.dumps({"people": [o]}))

    client = GeminiCoarseClient.__new__(GeminiCoarseClient)
    client.model = "fixture"
    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    got = client.analyze([("opaque", item)])
    parts = [p for p in captured["contents"] if not isinstance(p, str)]
    assert len(parts) == 2 and set(got) == {"opaque"}
    decoded = [Image.open(io.BytesIO(p.inline_data.data)) for p in parts]
    assert [x.size for x in decoded] == [item.context.size, item.crop.size]
    assert decoded[0].tobytes() == item.context.tobytes()
    assert decoded[1].tobytes() == item.crop.tobytes()
    assert all(x.size != (600, 900) for x in decoded)


def test_eval_input_freezes_source_identity_and_uses_live_preparer(tmp_path):
    import hashlib
    from scripts.eval_body_coarse import load_cases
    from src.experimental.body_matching.coarse_input import prepare_input

    source = tmp_path / "source.png"
    im = Image.new("RGB", (100, 200), "white")
    im.save(source)
    box = (20.5, 50, 70, 180)
    protocol = tmp_path / "protocol.json"
    case = dict(
        id="p0",
        image=str(source),
        box=box,
        image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    protocol.write_text(json.dumps({"cases": [case]}))
    cases, inputs = load_cases(protocol, tmp_path / "out")
    expected = prepare_input(im, box)
    assert cases[0]["target_box"] == list(expected.target_box)
    assert inputs["p0"].context.tobytes() == expected.context.tobytes()
    assert (
        hashlib.sha256(Path(cases[0]["context"]).read_bytes()).hexdigest()
        == cases[0]["context_sha256"]
    )
    case["image_sha256"] = "tampered"
    protocol.write_text(json.dumps({"cases": [case]}))
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        load_cases(protocol, tmp_path / "out2")


@pytest.mark.parametrize("presentation", ["masculine", "feminine"])
@pytest.mark.parametrize("muscle", ["ordinary", "very_muscular"])
def test_tentative_family_records_other_base_without_overriding_recommendation(
    presentation, muscle
):
    o = observation(presentation, muscle=muscle)
    o["axes"]["presentation"]["certainty"] = "tentative"
    r = recommend(o, ASSETS)
    expected = "male" if presentation == "masculine" else "female"
    assert r["recommended_family"] == expected
    assert r["candidates"][0]["body_id"] == r["recommended_body_id"]
    assert r["candidates"][-1]["family"] != expected
    assert r["candidates"][-1]["preset"] == "BASE"
    assert any(
        c["preset"] == "BASE" and c["family"] == expected for c in r["candidates"]
    )
    assert len(r["candidates"]) <= 3
    assert [c["rank"] for c in r["candidates"]] == list(
        range(1, len(r["candidates"]) + 1)
    )
    assert r["auto_apply"] is False and r["applied_body_id"] is None


@pytest.mark.parametrize("context_input", [False, True])
def test_recorded_inputs_keep_legacy_replay_and_reject_context_mismatch(
    tmp_path, monkeypatch, context_input
):
    import hashlib
    from scripts.eval_body_coarse import load_cases, main

    source = tmp_path / "source.png"
    Image.new("RGB", (50, 100), "white").save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = tmp_path / "protocol.json"
    c = (
        dict(id="p0", image=str(source), image_sha256=digest, box=[10, 30, 40, 90])
        if context_input
        else dict(
            case_id="p0",
            crop=str(source),
            crop_sha256=digest,
            image_sha256="old-source-hash",
        )
    )
    protocol.write_text(json.dumps({"cases": [c]}))
    cases, _ = load_cases(protocol, tmp_path / "initial")
    obs = observation(person_id="p0")
    obs["version"] = "body-coarse.v2.family-first"
    previous = dict(
        version=obs["version"],
        implementation_sha256="source-code-hash",
        rows=[
            dict(
                case_id="p0",
                crop=cases[0]["crop"],
                crop_sha256=cases[0]["crop_sha256"],
                observation=obs,
                recommendation=recommend(obs, ASSETS),
                provider_error=None,
            )
        ],
    )
    if context_input:
        previous["rows"][0]["input_metadata"] = {
            k: cases[0][k]
            for k in ("input_version", "context_sha256", "image_sha256", "target_box")
        }
        previous["rows"][0]["input_metadata"]["context_sha256"] = "tampered"
    recorded = tmp_path / "recorded.json"
    recorded.write_text(json.dumps(previous))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval",
            "--protocol",
            str(protocol),
            "--catalog",
            str(CATALOG),
            "--out",
            str(out),
            "--recorded",
            str(recorded),
        ],
    )
    if context_input:
        with pytest.raises(ValueError, match="recorded_context_mismatch"):
            main()
    else:
        main()
        replay = json.load(open(out / "results.json"))
        assert replay["classification_versions"] == ["body-coarse.v2.family-first"]
        assert (
            replay["source_classification_implementation_sha256"] == "source-code-hash"
        )
        assert (
            replay["rows"][0]["recommendation"]["recommended_body_id"]
            == "male-base-v0-7-2"
        )


def test_confirmed_scoring_does_not_count_alternative_or_default_as_observed_success():
    from scripts.eval_body_coarse import validate_labels, score_labels

    labels = validate_labels(
        dict(
            source="user_confirmed_character_setting",
            rows=[
                dict(case_id="p0", expected_family="male"),
                dict(case_id="p1", expected_family="female"),
            ],
        ),
        {"p0", "p1"},
    )
    rows = [
        dict(case_id="p0", recommendation=recommend(observation(), ASSETS)),
        dict(
            case_id="p1", recommendation=recommend(unknown("p1", "ambiguous"), ASSETS)
        ),
    ]
    before = copy.deepcopy(rows)
    scored = score_labels(rows, labels)
    assert scored["recommended_correct"] == 1 and scored["observed_correct"] == 1
    assert scored["total"] == 2 and scored["recommendation_accuracy"] == 0.5
    assert rows == before
    # A fallback that happens to match a label still is not visual recognition.
    scored = score_labels(rows, {"p1": "male"})
    assert scored["recommended_correct"] == 1 and scored["observed_correct"] == 0


@pytest.mark.parametrize(
    "rows",
    [
        [dict(case_id="p0", expected_family="male")] * 2,
        [dict(case_id="wrong", expected_family="female")],
        [dict(case_id="p0", expected_family="unknown")],
    ],
)
def test_confirmed_labels_reject_wrong_identity_duplicates_and_nonbinary_ground_truth(
    rows,
):
    from scripts.eval_body_coarse import validate_labels

    with pytest.raises(ValueError, match="invalid_confirmed_label"):
        validate_labels(
            dict(source="user_confirmed_character_setting", rows=rows), {"p0"}
        )


@pytest.mark.parametrize("family", ["masculine", "feminine"])
def test_tentative_family_alternative_survives_http_response_validation(family):
    from api.body_models import CoarseRecommendationOut

    o = observation(family)
    o["axes"]["presentation"]["certainty"] = "tentative"
    rec = recommend(o, ASSETS)
    wire = CoarseRecommendationOut.model_validate(rec)
    assert wire.model_dump()["candidates"] == rec["candidates"]
    assert wire.auto_apply is False and wire.applied_body_id is None
    bad = copy.deepcopy(rec)
    bad["unresolved_axes"] = []
    with pytest.raises(ValueError, match="coarse_family_mismatch"):
        CoarseRecommendationOut.model_validate(bad)
    bad = copy.deepcopy(rec)
    bad["candidates"][-1]["preset"] = "MUSCULAR"
    with pytest.raises(ValueError, match="coarse_family_mismatch"):
        CoarseRecommendationOut.model_validate(bad)
    bad = copy.deepcopy(rec)
    bad["observed_family"] = "female" if rec["observed_family"] == "male" else "male"
    with pytest.raises(ValueError, match="coarse_family_mismatch"):
        CoarseRecommendationOut.model_validate(bad)


def test_clear_androgynous_reading_still_allows_both_base_alternatives():
    from api.body_models import CoarseRecommendationOut

    rec = recommend(observation("androgynous"), ASSETS)
    assert rec["observed_family"] is None
    wire = CoarseRecommendationOut.model_validate(rec)
    assert {c.family for c in wire.candidates} == {"male", "female"}
