import copy
import json
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from src.config import CFG
from src.vlm.client import _coerce, GeminiVLMClient
from src.vlm import prompts
from src.experimental.body_matching import integrated
from src.experimental.body_matching.coarse import recommend
from src.experimental.body_matching.coarse_service import CoarseBodyService
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from api.body_models import BodyMatchingOut

CATALOG = Path(__file__).resolve().parents[1] / "config/body_morphology.v1.json"
ASSETS = json.loads(CATALOG.read_text())["assets"]


def body(index=0, presentation="masculine"):
    return dict(
        person_index=index,
        evidence="얼굴과 의상 목선·옷단 및 몸의 표현을 함께 읽음",
        target="clear",
        representation="character",
        proportion_readable=True,
        axes={
            k: dict(value=v, certainty="clear")
            for k, v in dict(
                presentation=presentation,
                muscle="ordinary",
                fullness="average",
                proportion="standard",
            ).items()
        },
    )


def payload():
    return dict(
        num_people=2,
        shot="full_half",
        action="standing",
        view="front",
        relationship="talking",
        approx_boxes=[
            dict(x1=0.6, y1=0.05, x2=0.95, y2=0.95),
            dict(x1=0.05, y1=0.05, x2=0.4, y2=0.95),
        ],
        lower_body_visible=[True, True],
        body_scopes=["full", "full"],
    )


def test_optional_body_output_does_not_change_control_fields():
    raw = payload()
    old = _coerce(raw, 200, 300)
    newraw = copy.deepcopy(raw)
    newraw["person_bodies"] = [body(), body(1, "feminine")]
    new = _coerce(newraw, 200, 300)
    for k in [
        "num_people",
        "shot",
        "action",
        "view",
        "relationship",
        "approx_boxes",
        "lower_body_visible",
        "lower_body_visibility_known",
        "body_scopes",
        "stated_tags",
    ]:
        assert getattr(new, k) == getattr(old, k)
    assert old.person_bodies == []
    assert [recommend(o, ASSETS)["recommended_family"] for o in new.person_bodies] == [
        "male",
        "female",
    ]


@pytest.mark.parametrize(
    "kind", ["length", "duplicate", "foreign", "bool_index", "not_list"]
)
def test_misaligned_body_block_never_shifts_people(kind):
    rows = [body(), body(1, "feminine")]
    if kind == "length":
        rows.pop()
    elif kind == "duplicate":
        rows[1]["person_index"] = 0
    elif kind == "foreign":
        rows[1]["person_index"] = 3
    elif kind == "bool_index":
        rows[1]["person_index"] = True
    else:
        rows = {}
    raw = payload()
    raw["person_bodies"] = rows
    got = _coerce(raw, 200, 300)
    assert got.num_people == 2 and len(got.approx_boxes) == 2
    assert [o["axes"]["presentation"]["value"] for o in got.person_bodies] == [
        "unknown",
        "unknown",
    ]


def test_one_invalid_body_keeps_the_other_person_and_ordinary_default():
    rows = [body(), body(1, "feminine")]
    rows[0]["axes"]["muscle"]["value"] = "imaginary"
    parsed = integrated.parse_bodies(rows, 2)
    assert parsed[0]["target"] == "ambiguous"
    assert recommend(parsed[1], ASSETS)["recommended_body_id"] == "female-base-v0-3-4"


def test_explicit_indices_are_mapped_before_array_order_and_copied():
    parsed = integrated.parse_bodies([body(1, "feminine"), body(0)], 2)
    assert parsed[0]["axes"]["presentation"]["value"] == "masculine"
    vlm = SimpleNamespace(person_bodies=parsed)
    copy_body = integrated.detect_body(vlm, 1)
    copy_body["axes"]["presentation"]["value"] = "unknown"
    assert vlm.person_bodies[1]["axes"]["presentation"]["value"] == "feminine"
    assert integrated.detect_body(vlm, None) is None


def signature(result):
    return dict(
        route=result.route,
        vlm_count=result.vlm_count,
        detector_count=result.detector_count,
        count_confidence=result.count_confidence,
        people=[
            dict(
                box=d.box.as_list() if d.box else None,
                points=d.skeleton.keypoints.tolist() if d.skeleton else None,
                scores=d.skeleton.scores.tolist() if d.skeleton else None,
                mask=(
                    d.valid_joint_mask.tolist()
                    if d.valid_joint_mask is not None
                    else None
                ),
                refine=d.refine_allowed,
                scope=d.output_scope.detected,
                slot=d.quality_trace.get("vlm_person_index"),
                hands=asdict(d.hands) if d.hands else None,
            )
            for d in result.descriptors
        ],
        poses=[
            [(c.pose_id, c.view.value, c.distance) for c in cs]
            for cs in result.person_candidates
        ],
        dynamic=[
            [
                (x["pose_id"], x["rank_score"])
                for x in p.get("fullbody_search", {}).get("selected", [])
            ]
            for p in result.person_routing
        ],
    )


@pytest.mark.parametrize("fullbody", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_body_metadata_keeps_count_boxes_pose_rank_hand_and_fullbody_flow(
    monkeypatch, fullbody, invalid
):
    from tests.test_fullbody_routing import Camera, Semantic
    from src.fullbody_routing import FullBodySearch
    from src.search_routing import DescribeOutcome, PoseDescription

    entries = build_synthetic_index()
    ids = list(dict.fromkeys(e.pose_id for e in entries))[:3]

    class Reader:
        def analyze(self, *args):
            raw = payload()
            raw["person_bodies"] = [body(), body(1, "feminine")]
            if invalid:
                raw["person_bodies"][0]["evidence"] = ""
            return _coerce(raw, 200, 300)

        def describe_people(self, image, boxes):
            return DescribeOutcome(
                "ok",
                descriptions={
                    i: PoseDescription(
                        visible=True,
                        certainty="high",
                        postures=["standing"],
                        actions=["stand"],
                        body_details="상체를 세우고 양팔을 아래로 내림",
                        search_text="서서 양팔을 아래로 내린 자세",
                    )
                    for i in range(len(boxes))
                },
            )

    monkeypatch.setattr(CFG, "vlm_person_tags", False)
    monkeypatch.setattr(CFG, "vlm_hand_shapes", False)
    monkeypatch.setattr(CFG, "search_routing_enabled", False)
    monkeypatch.setattr(CFG, "body_classification_strategy", "integrated")
    monkeypatch.setattr(CFG, "vlm_provider", "gemini")
    monkeypatch.setattr(CFG, "body_coarse_catalog_path", str(CATALOG))
    engine = FullBodySearch(entries, Semantic(ids), Camera(ids)) if fullbody else None
    pipe = Pipeline(
        entries, vlm_client=Reader(), pose_model=MockPoseModel(), fullbody_search=engine
    )
    image = Image.new("RGB", (200, 300))
    monkeypatch.setattr(CFG, "body_matching_mode", "off")
    before = pipe.process_cut(image, 200, 300)
    monkeypatch.setattr(CFG, "body_matching_mode", "shadow")
    after = pipe.process_cut(image, 200, 300)
    assert signature(after) == signature(before)
    assert len(after.body_matching["people"]) == len(after.descriptors)
    wire = BodyMatchingOut.model_validate(after.body_matching).model_dump()
    for d, p in zip(after.descriptors, wire["people"]):
        if d.quality_trace["vlm_person_index"] == 1:
            assert p["coarse_recommendation"]["recommended_family"] == "female"
        assert (
            p["auto_body_id"] is None
            and p["applied_body_id"] is None
            and p["candidates"] == []
        )
    # Original VLM order is right-to-left; API descriptor order is left-to-right.
    assert after.descriptors[0].quality_trace["vlm_person_index"] == 1


def test_integrated_service_never_initializes_or_calls_a_second_model(monkeypatch):
    import src.experimental.body_matching.coarse_service as mod

    def forbidden(*args, **kw):
        raise AssertionError("extra body request")

    monkeypatch.setattr(mod, "GeminiCoarseClient", forbidden)
    svc = CoarseBodyService(CATALOG, provider="gemini", integrated=True)
    assert svc.client is None


@pytest.mark.parametrize(
    "mode,strategy,extension",
    [
        ("off", "integrated", False),
        ("shadow", "legacy", False),
        ("shadow", "integrated", True),
    ],
)
def test_first_gemini_call_contains_optional_body_extension_without_extra_request(
    monkeypatch, mode, strategy, extension
):
    monkeypatch.setattr(CFG, "body_matching_mode", mode)
    monkeypatch.setattr(CFG, "body_classification_strategy", strategy)
    monkeypatch.setattr(CFG, "vlm_body_thinking_level", "minimal")
    calls = []

    def generate(**kw):
        calls.append(kw)
        raw = payload()
        if extension:
            raw["person_bodies"] = [body(), body(1, "feminine")]
        return SimpleNamespace(text=json.dumps(raw))

    client = GeminiVLMClient(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    result = client.analyze(Image.new("RGB", (200, 300)), 200, 300)
    assert len(calls) == 1 and result.num_people == 2
    base = prompts.user_template(CFG.vlm_prompt_version)
    assert calls[0]["contents"][0] == (
        integrated.analysis_prompt(base) if extension else base
    )
    assert bool(calls[0]["config"].thinking_config) == extension


def test_compact_response_keeps_the_same_public_recommendation():
    original = body(0, "feminine")
    compact = copy.deepcopy(original)
    compact["evidence"] = ["둥근 눈과 입", "의상 목선과 옷단", "보통 체형"]
    compact["axes"] = {
        k: [v["value"], v["certainty"]] for k, v in original["axes"].items()
    }
    parsed = integrated.parse_bodies([compact], 1)[0]
    rec = recommend(parsed, ASSETS)
    assert rec["recommended_body_id"] == "female-base-v0-3-4"
    assert (
        parsed["axes"]["presentation"]["reason"]
        == "둥근 눈과 입 / 의상 목선과 옷단 / 보통 체형"
    )
    compact["axes"]["presentation"].reverse()
    assert (
        integrated.parse_bodies([compact], 1)[0]["axes"]["presentation"]["value"]
        == "unknown"
    )


def test_missing_body_fields_with_real_control_data_keep_pose_and_explicit_fallback(
    monkeypatch,
):
    raw = payload()
    raw["person_bodies"] = None
    vlm = _coerce(raw, 200, 300)
    assert vlm.num_people == 2 and len(vlm.approx_boxes) == 2
    assert all(o["target"] == "ambiguous" for o in vlm.person_bodies)
    assert all(recommend(o, ASSETS)["recommended_body_id"] for o in vlm.person_bodies)


def test_integrated_strategy_does_not_reuse_old_network_matcher_or_claim_mock_is_real(
    monkeypatch,
):
    from src.experimental.body_matching.service import attach_body_matching
    from src.vlm.client import MockVLMClient
    from tests.test_body_matching import cut

    class Old:
        def analyze(self, *args, **kw):
            raise AssertionError("old network sidecar must not run")

    pipe = SimpleNamespace(
        body_matcher=Old(), vlm=MockVLMClient(), pose=MockPoseModel()
    )
    monkeypatch.setattr(CFG, "body_classification_strategy", "integrated")
    monkeypatch.setattr(CFG, "body_coarse_catalog_path", str(CATALOG))
    monkeypatch.setattr(CFG, "body_matching_mode", "shadow")
    monkeypatch.setattr(CFG, "vlm_provider", "gemini")
    result = attach_body_matching(pipe, Image.new("RGB", (100, 100)), cut(1), CFG)
    assert result.body_matching["provider_actual"] == "mock"
    assert result.body_matching["is_mock"] is True
    assert pipe.body_matcher.integrated and pipe.body_matcher.client is None
    assert result.body_matching["people"][0]["coarse_recommendation"][
        "recommended_body_id"
    ]


def test_approved_body_metadata_is_not_a_search_hand_or_refine_input():
    import ast

    root = Path(__file__).resolve().parents[1]
    allowed = {
        "src/schema.py",
        "src/vlm/client.py",
        "src/descriptor.py",
        "src/experimental/body_matching/integrated.py",
        "src/experimental/body_matching/coarse_service.py",
    }
    found = set()
    for p in (root / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            name = (
                node.attr
                if isinstance(node, ast.Attribute)
                else node.id if isinstance(node, ast.Name) else None
            )
            if name in {"person_bodies", "body_classification", "detect_body"}:
                found.add(p.relative_to(root).as_posix())
    assert found <= allowed, found - allowed


def test_eval_matches_original_boxes_globally_instead_of_using_output_order():
    from scripts.eval_integrated_body import match_people

    cases = [{"id": "a", "box": [0, 0, 10, 10]}, {"id": "b", "box": [20, 0, 30, 10]}]
    people = [{"box": [20, 0, 30, 10]}, {"box": [0, 0, 10, 10]}]
    got = match_people(cases, people)
    assert got["a"]["index"] == 1 and got["b"]["index"] == 0
    assert got["a"]["iou"] == 1
    missing = match_people(cases, [{"box": None}, {"box": [20, 0, 30, 10]}])
    assert missing["a"] is None and missing["b"]["index"] == 1
    assert match_people(cases, []) == {"a": None, "b": None}


def test_eval_uses_dynamic_pose_contract_and_never_counts_errors_as_equal():
    from scripts.eval_integrated_body import pose_signature, same_successful_count

    a = {
        "candidates": [],
        "dynamic_search": {"selected": [{"pose_id": "a", "kind": "library_original"}]},
    }
    b = {
        "candidates": [],
        "dynamic_search": {"selected": [{"pose_id": "b", "kind": "library_original"}]},
    }
    assert pose_signature(a) != pose_signature(b)
    assert pose_signature({"candidates": []}) == []
    failure = {"error": {"type": "JSONDecodeError"}, "result": None}
    assert not same_successful_count(failure, failure)
    success = {"error": None, "result": {"vlm_count": 2}}
    assert same_successful_count(success, success)


def test_eval_keeps_malformed_primary_as_failure_and_accepts_provider_fences():
    from scripts.eval_integrated_body import primary_payload

    assert primary_payload([{"kind": "analysis", "text": "{broken"}]) is None
    assert primary_payload(
        [{"kind": "analysis", "text": '```json\n{"num_people":1}\n```'}]
    ) == {"num_people": 1}


@pytest.mark.parametrize("active", [False, True])
def test_openai_first_call_keeps_original_prompt_when_disabled(monkeypatch, active):
    from src.vlm.client import OpenAIVLMClient

    monkeypatch.setattr(CFG, "body_classification_strategy", "integrated")
    monkeypatch.setattr(CFG, "body_matching_mode", "shadow" if active else "off")
    calls = []

    def create(**kw):
        calls.append(kw)
        raw = payload()
        if active:
            raw["person_bodies"] = [body(), body(1, "feminine")]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw)))]
        )

    client = OpenAIVLMClient.__new__(OpenAIVLMClient)
    client._model = "offline-test"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    got = client.analyze("data:image/png;base64,offline", 200, 300)
    assert len(calls) == 1 and got.num_people == 2
    assert calls[0]["messages"][0]["content"] == prompts.SYSTEM + (
        integrated.SYSTEM_EXTENSION if active else ""
    )
    assert calls[0]["messages"][1]["content"][0]["text"] == prompts.user_template(
        CFG.vlm_prompt_version
    ) + (integrated.PROMPT_EXTENSION if active else "")
