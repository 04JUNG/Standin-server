"""Offline contracts and fault injection; no provider requests or target quality claims."""
from dataclasses import FrozenInstanceError
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.refine_evidence import (bind_evidence_to_slots, parse_refine_evidence,
                                 evidence_schema, extend_analysis_schema, hand_response_schema)
from src.joint_provenance import initialize_joint_provenance, transfer_joint_provenance
from src.schema import Skeleton
from src.skeleton_repair import _skeleton_from
from src.vlm.client import GeminiVLMClient, _coerce
from src.vlm import prompts


def visibility(**overrides):
    return {"region": "left_wrist", "state": "visible", "evidence": "direct_visible",
            "confidence": "high", **overrides}


def relation(**overrides):
    return {"id": "r0", "subject_region": "left_wrist", "object_region": "head",
            "kind": "above", "frame": "image", "side_certainty": "certain",
            "evidence": "direct_visible", "confidence": "high", **overrides}


def observations(ids=(0,), **overrides):
    return {"people": [{"person_id": pid, "body_scope": "full", "visibility": [visibility()],
                        "relations": [relation()], "preserve": ["r0"], **overrides} for pid in ids]}


def contact(**overrides):
    return {"effector_region": "left_palm", "target": "self_region", "target_ref": "left_thigh",
            "kind": "touching", "side_certainty": "certain", "chain_visible": True,
            "evidence": "direct_visible", "confidence": "high", **overrides}


@pytest.mark.parametrize("ids", [(0, 0), (1,), (False,), (), (0, 1)])
def test_id_set_mismatch_discards_whole_optional_bundle(ids):
    bundle = parse_refine_evidence(observations(ids), expected_person_ids=(0,))
    assert bundle.status == "unavailable" and not bundle.people
    assert bundle.reasons == ("person_id_set_mismatch",)


def test_bounded_groups_are_discarded_not_truncated_and_other_groups_survive():
    bundle = parse_refine_evidence(observations(visibility=[visibility()] * 17), expected_person_ids=(0,))
    assert bundle.status == "partial"
    person = bundle.to_serializable()["people"][0]
    assert person["visibility"] == [] and len(person["relations"]) == 1
    assert person["disposition"] == "diagnostic_only"


@pytest.mark.parametrize("bad", [None, [], {}, 1, True, "screen_left"])
def test_malformed_enum_is_optional_only(bad):
    bundle = parse_refine_evidence(observations(visibility=[visibility(state=bad)]), expected_person_ids=(0,))
    assert bundle.status == "partial" and bundle.people[0].visibility == ()


def test_coordinates_and_executable_preserve_are_rejected():
    raw = observations(relations=[relation(x=4)], preserve=["import os"])
    bundle = parse_refine_evidence(raw, expected_person_ids=(0,))
    assert bundle.status == "partial"
    assert bundle.people[0].relations == bundle.people[0].preserve == ()


def test_explicit_reordered_ids_are_mapped_not_array_position():
    raw = observations((1, 0))
    raw["people"][0]["visibility"] = [visibility(region="right_wrist")]
    bundle = parse_refine_evidence(raw, expected_person_ids=(0, 1))
    assert bundle.people[0].person_id == 0
    assert bundle.to_serializable()["people"][0]["visibility"][0]["region"] == "left_wrist"


def test_hand_subset_maps_zero_to_original_slot_not_global_zero():
    raw = {"people": [{"person_id": 0, "body_scope": "full", "contacts": [contact()]}]}
    bundle = parse_refine_evidence(raw, expected_person_ids=(0,), kind="contacts")
    bound = bind_evidence_to_slots(bundle, {0: 3}, provenance={"image_digest": "sha",
                                   "crop_transform": "full-image-v1", "slot_map_version": "v1"})
    assert bound.people[0].person_id == 3
    assert bundle.people[0].person_id == 0
    assert bind_evidence_to_slots(bundle, {0: 3}, provenance={}).status == "unavailable"


def test_other_person_contacts_validate_both_ends_then_remap():
    raw = {"people": [{"person_id": pid, "body_scope": "full", "contacts": [contact(target="other_person", target_ref=1-pid)]}
                       for pid in (0, 1)]}
    bundle = parse_refine_evidence(raw, expected_person_ids=(0, 1), kind="contacts")
    bound = bind_evidence_to_slots(bundle, {0: 4, 1: 7}, provenance={"image_digest": "s", "crop_transform": "f", "slot_map_version": "1"})
    assert bound.to_serializable()["people"][0]["contacts"][0]["target_ref"] == 7
    raw["people"][0]["contacts"][0]["target_ref"] = 99
    invalid = parse_refine_evidence(raw, expected_person_ids=(0, 1), kind="contacts")
    assert invalid.status == "partial" and not invalid.people[0].contacts


def test_scope_and_side_ambiguity_fail_closed():
    bundle = parse_refine_evidence(observations(relations=[relation(side_certainty="unknown")]), expected_person_ids=(0,))
    assert bundle.to_serializable()["people"][0]["relations"][0]["evidence"] == "unknown"
    assert parse_refine_evidence(observations(), expected_person_ids=(0,), body_scopes={0: "head"}).status == "partial"
    assert parse_refine_evidence(observations(body_scope="head"), expected_person_ids=(0,)).people[0].visibility == ()


def test_snapshots_are_immutable_and_mutating_input_cannot_change_hash():
    raw = observations()
    bundle = parse_refine_evidence(raw, expected_person_ids=(0,))
    before = bundle.to_serializable()
    raw["people"][0]["visibility"][0]["region"] = "right_wrist"
    assert bundle.to_serializable() == before
    with pytest.raises(FrozenInstanceError):
        bundle.status = "valid"


def skeleton():
    return Skeleton(np.arange(34, dtype=np.float32).reshape(17, 2), np.ones(17, np.float32))


def test_per_joint_repair_preserves_originals_and_distinguishes_estimates():
    old = skeleton()
    old.joint_provenance = initialize_joint_provenance(old, source="full_image", person_id=2)
    before = deepcopy(old.joint_provenance)
    from src.schema import COCO17
    person = {"keypoints": [{"name": name, "state": "visible", "x": float(i), "y": float(i), "score": 1}
                            for i, name in enumerate(COCO17)]}
    person["keypoints"][13] = {"name": COCO17[13], "state": "occluded_estimated", "x": 500, "y": 600}
    person["keypoints"][15] = {"name": COCO17[15], "state": "visible", "x": 500, "y": 800}
    revised, changed = _skeleton_from(person, old, 1000, 1000, .4)
    assert set(changed) == {COCO17[13], COCO17[15]}
    assert revised.joint_provenance[9] == before[9]  # Original visible arm retained.
    assert revised.joint_provenance[13].source == "vlm_estimated"
    assert revised.joint_provenance[13].visibility == "occluded_estimated"
    assert revised.joint_provenance[15].source == "vlm_visible"
    assert revised.joint_provenance[15].visibility == "model_inferred"
    assert revised.joint_provenance[13].original == before[13].original
    assert old.joint_provenance == before
    old.keypoints[:] = -999
    assert before[13].original != tuple(old.keypoints[13])


def test_crop_and_rescue_keep_originals_and_only_changed_joint_sources():
    old = skeleton()
    old.joint_provenance = initialize_joint_provenance(old, "full_image", 0)
    crop = skeleton()
    crop.keypoints[9] += 2
    crop.joint_provenance = transfer_joint_provenance(old, crop, source="crop_retry", person_id=0)
    assert crop.joint_provenance[8] == old.joint_provenance[8]
    rescue = Skeleton(crop.keypoints.copy(), crop.scores.copy())
    rescue.keypoints[13] += 3
    rescue.joint_provenance = transfer_joint_provenance(crop, rescue, source="fallback_full_image", person_id=0)
    assert rescue.joint_provenance[9].source == "crop_retry"
    assert rescue.joint_provenance[13].source == "fallback_full_image"
    assert rescue.joint_provenance[13].original == old.joint_provenance[13].original
    assert all(p.visibility != "direct_observed" for p in rescue.joint_provenance)


def core_response(**extra):
    return {"num_people": 1, "approx_boxes": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}],
            "shot": "full_half", "action": "standing", "view": "front", "relationship": "solo", **extra}


def test_malformed_optional_keeps_all_core_control_fields(monkeypatch):
    monkeypatch.setenv("REFINE_V3_EVIDENCE_ENABLED", "1")
    core = _coerce(core_response(), 100, 100)
    enriched = _coerce(core_response(refine_observations={"people": [{"person_id": 99}]}), 100, 100)
    assert enriched.num_people == core.num_people and enriched.shot == core.shot
    assert enriched.approx_boxes == core.approx_boxes
    assert enriched.refine_evidence["status"] == "unavailable"


class FakeModels:
    def __init__(self, text):
        self.text, self.calls, self.requests = text, 0, []
    def generate_content(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        return SimpleNamespace(text=self.text, usage_metadata=SimpleNamespace(prompt_token_count=40,
                               candidates_token_count=20, total_token_count=60))


def client(monkeypatch, response):
    models = FakeModels(json.dumps(response))
    instance = GeminiVLMClient(client=SimpleNamespace(models=models), model="fake-model")
    monkeypatch.setattr(instance, "_to_part", lambda image: "image")
    return instance, models


@pytest.mark.parametrize("enabled", [False, True])
def test_analyze_reuses_exactly_one_logical_call_and_tracks_usage(monkeypatch, enabled):
    monkeypatch.setenv("REFINE_V3_EVIDENCE_ENABLED", "1" if enabled else "0")
    from src.config import CFG
    monkeypatch.setattr(CFG, "body_matching_mode", "off")
    monkeypatch.setattr(CFG, "body_integrated_review_enabled", False)
    instance, models = client(monkeypatch, core_response(refine_observations=observations()))
    result = instance.analyze("ignored", 100, 100)
    assert models.calls == 1
    if enabled:
        trace = result.refine_evidence["call_trace"]
        assert trace["additional_logical_calls"] == trace["solver_logical_calls"] == 0
        assert trace["provider_attempts"] == 1 and trace["total_tokens"] == 60
        schema = models.requests[0]["config"].response_json_schema
        assert "refine_observations" in schema["properties"]
        assert "refine_observations" not in schema["required"]
    else:
        assert result.refine_evidence is None
        assert models.requests[0]["contents"][0] == prompts.user_template(CFG.vlm_prompt_version)


@pytest.mark.parametrize("optional", [None, {"people": [{"person_id": 9}]}, {"people": [{"person_id": 0, "body_scope": "full", "contacts": [contact()]}]}])
def test_hands_reuses_one_existing_call_and_keeps_core_on_optional_failure(monkeypatch, optional):
    monkeypatch.setenv("REFINE_V3_EVIDENCE_ENABLED", "1")
    hands = {"left": {"visible": True, "shape": "relaxed", "alternative": None, "confidence": "high"},
             "right": {"visible": False, "shape": None, "alternative": None, "confidence": "low"}}
    instance, models = client(monkeypatch, {"people": [hands], "refine_contacts": optional})
    response = instance.read_hands("ignored", [[0, 0, 1, 1]])
    assert models.calls == 1 and response["people"] == [hands]
    assert response["refine_evidence"]["provenance"]["additional_logical_calls"] == 0


def test_provider_schema_is_optional_bounded_and_does_not_mutate_core():
    original = {"properties": {"num_people": {"type": "integer"}}, "required": ["num_people"]}
    expanded = extend_analysis_schema(original)
    assert expanded["required"] == original["required"] and "refine_observations" not in original["properties"]
    assert evidence_schema()["properties"]["people"]["maxItems"] == 20
    assert hand_response_schema()["required"] == ["people"]


def test_gemini_transport_avoids_nested_bounds_without_changing_core_contract():
    from src.experimental.body_matching.integrated import response_schema
    core = response_schema()
    before = deepcopy(core)
    wire = extend_analysis_schema(core)
    assert core == before
    assert {k:v for k,v in wire["properties"].items() if k != "refine_observations"} == core["properties"]
    assert wire["required"] == core["required"]
    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    unsupported = {"maxItems", "minItems", "minimum", "maximum", "maxLength", "minLength"}
    for field in (wire["properties"]["refine_observations"], hand_response_schema()["properties"]["refine_contacts"]):
        assert not keys(field) & unsupported
        assert {"enum", "required", "additionalProperties"} <= keys(field)


def test_compact_provider_grammar_does_not_expand_runtime_admission():
    from src.refine_evidence import MAX_BYTES, MAX_PEOPLE, LIMITS
    too_many = observations(tuple(range(MAX_PEOPLE+1)))
    assert parse_refine_evidence(too_many, expected_person_ids=tuple(range(MAX_PEOPLE+1))).status == "unavailable"
    oversized = observations(visibility=[visibility()]*(LIMITS["visibility"]+1))
    parsed = parse_refine_evidence(oversized, expected_person_ids=(0,))
    assert parsed.status == "partial" and not parsed.people[0].visibility
    assert parsed.people[0].disposition == "diagnostic_only"
    oversized = observations(untrusted="x"*MAX_BYTES)
    assert parse_refine_evidence(oversized, expected_person_ids=(0,)).reasons == ("evidence_size_limit",)


def test_contact_schema_only_offers_effectors_the_strict_parser_admits():
    schema = evidence_schema("contacts")
    offered = schema["properties"]["people"]["items"]["properties"]["contacts"]["items"]["properties"]["effector_region"]["enum"]
    for effector in offered:
        raw = {"people": [{"person_id": 0, "body_scope": "full", "contacts": [contact(effector_region=effector)]}]}
        assert parse_refine_evidence(raw, expected_person_ids=(0,), kind="contacts").status == "valid"
    assert "right_hand" not in offered and "left_foot" not in offered
    raw = {"people": [{"person_id": 0, "body_scope": "full", "contacts": [contact(effector_region="right_hand")]}]}
    assert parse_refine_evidence(raw, expected_person_ids=(0,), kind="contacts").status == "partial"


def test_stored_evidence_checksum_and_binding_revalidation():
    stored = parse_refine_evidence(observations(), expected_person_ids=(0,)).to_serializable()
    bound = bind_evidence_to_slots(stored, {0: 7}, provenance={"image_digest": "sha", "crop_transform": "full", "slot_map_version": "v1"})
    assert bound.people[0].person_id == 7
    stored["people"][0]["visibility"][0]["region"] = "right_wrist"
    assert bind_evidence_to_slots(stored, {0: 7}, provenance={}).status == "unavailable"


def test_pipeline_hook_skips_detector_only_and_preserves_hand_subset(monkeypatch):
    from src.refine_evidence import bind_analysis_evidence, bind_hand_contact_evidence
    monkeypatch.setenv("REFINE_V3_EVIDENCE_ENABLED", "1")
    vlm = _coerce(core_response(num_people=2, refine_observations=observations((0, 1))), 100, 200)
    slots = [SimpleNamespace(slot_id=1, slot_origin="vlm"), SimpleNamespace(slot_id=5, slot_origin="rtm_provisional"),
             SimpleNamespace(slot_id=0, slot_origin="vlm")]
    contact_bundle = parse_refine_evidence({"people": [{"person_id": 0, "body_scope": "full", "contacts": [contact()]}]},
                                           expected_person_ids=(0,), kind="contacts").to_serializable()
    vlm._refine_contact_evidence = bind_hand_contact_evidence({"refine_evidence": contact_bundle}, (1,), image_digest="same", image_size=(100, 200))
    mapped = bind_analysis_evidence(vlm, slots, "same", (100, 200))
    assert set(mapped) == {0, 1} and mapped[0]["contacts"] is None
    assert mapped[1]["contacts"]["people"][0]["person_id"] == 1
    vlm._refine_contact_evidence["people"][0]["person_id"] = 0
    assert bind_analysis_evidence(vlm, slots, "same", (100, 200))[1]["contacts"] is None
    monkeypatch.setenv("REFINE_V3_EVIDENCE_ENABLED", "0")
    assert bind_analysis_evidence(vlm, slots, "same", (100, 200)) == {}
