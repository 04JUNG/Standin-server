"""Opt-in recovery keeps valid owners/qualified poses and exposes uncertainty."""

import copy
from types import SimpleNamespace
import numpy as np
import pytest
from src.schema import BBox
from src.skeleton_extraction import assign_candidates
from tests.test_skeleton_extraction import _skeleton
from tests.test_fullbody_routing import engine, decision, evidence, Generator
from src.fullbody_routing import FullBodySearch
from src.camera_search import CameraIndex
from src.library import _synthetic_pose
from src.semantic_search import SemanticSearchResult
from api.models import DynamicSearchOut, WholePoseCandidateOut


def scene():
    boxes = [
        BBox(20, 10, 180, 260, "vlm"),
        BBox(240, 10, 220, 260, "vlm"),
        BBox(420, 10, 580, 260, "vlm"),
    ]
    return boxes, [_skeleton(cx=500), _skeleton(cx=100), _skeleton(cx=300)]


def test_unique_invalid_box_recovers_only_the_unassigned_detector():
    boxes, people = scene()
    before = assign_candidates(boxes, people, 600, 300, expected_count=3)
    after = assign_candidates(
        boxes, people, 600, 300, expected_count=3, recover_invalid=True
    )
    assert before.slots[1].skeleton is None
    assert after.slots[1].skeleton is people[2]
    assert after.slots[1].slot_id == 1 and after.slots[1].slot_origin == "vlm"
    assert after.slots[1].box_recovery["detector_index"] == 2
    assert (
        after.slots[1].assignment_cost is None
    )  # never fabricate VLM overlap evidence
    for i in [0, 2]:
        assert before.slots[i].assigned_rtm_index == after.slots[i].assigned_rtm_index
        assert before.slots[i].vlm_box == after.slots[i].vlm_box
        assert before.slots[i].reasons == after.slots[i].reasons
    assert not after.unmatched_candidate_indices and len(after.slots) == 3
    assert boxes[1].x1 > boxes[1].x2  # input is not silently sorted/rewritten


@pytest.mark.parametrize(
    "bad",
    [
        "multiple_missing",
        "count_mismatch",
        "duplicate",
        "unowned_peer",
        "weak_structure",
    ],
)
def test_box_recovery_does_not_guess_an_ambiguous_owner(bad):
    boxes, people = scene()
    if bad == "multiple_missing":
        boxes[0] = None
    elif bad == "count_mismatch":
        people.append(_skeleton(cx=700))
    elif bad == "duplicate":
        people[2] = copy.deepcopy(people[1])
    elif bad == "unowned_peer":
        boxes[0] = BBox(10, 270, 40, 299, "vlm")
    else:
        people[2].scores[:] = 0
    after = assign_candidates(
        boxes, people, 800, 300, expected_count=3, recover_invalid=True
    )
    assert after.slots[1].skeleton is None and not after.slots[1].box_recovery


def test_valid_assignment_is_identical_when_recovery_enabled():
    boxes, people = scene()
    boxes[1] = BBox(220, 10, 380, 260, "vlm")
    a = assign_candidates(boxes, people, 600, 300, expected_count=3)
    b = assign_candidates(
        boxes, people, 600, 300, expected_count=3, recover_invalid=True
    )
    assert [s.assigned_rtm_index for s in a.slots] == [
        s.assigned_rtm_index for s in b.slots
    ]
    assert [s.reasons for s in a.slots] == [s.reasons for s in b.slots]
    assert not any(s.box_recovery for s in b.slots)


def constrained_scene(posture="standing", distance_limit=1.0):
    p = _synthetic_pose("standing")
    camera = CameraIndex(["a"], [p], [np.ones(17)], hashes={"a": "sha"})

    class Text:
        core_records = {}
        body = {}
        service = SimpleNamespace(
            index=SimpleNamespace(
                lexicon=SimpleNamespace(labels={("object", "phone"): "휴대폰"})
            )
        )

        def search(self, q, top_k):
            return SemanticSearchResult("no_result")

    gen = Generator()
    e = FullBodySearch(
        [SimpleNamespace(pose_id="a", meta={})],
        Text(),
        camera,
        generator=gen,
        geometry_max_distance=distance_limit,
    )
    d = decision()
    d.semantic.description.update(
        postures=[posture],
        actions=["hold"],  # phone fixture must not inherit decision()'s fold_arms
        body_details="서서 양손으로 휴대폰을 쥔다",
        search_text="휴대폰을 양손으로 쥐고 서기",
        objects=["phone"],
        held_object_hand_counts={"phone": "two"},
        arm_states={},
        body_observations={},
    )
    ev = evidence()
    ev["keypoints"] = p[:, :2].copy()
    ev["keypoints"][:, 1] *= -1
    return e, d, ev, gen


def test_text_overconstraint_gets_review_geometry_without_claiming_qualified_match():
    e, d, ev, gen = constrained_scene()
    old = e.search(d, ev)
    assert old["selected"] == [] and gen.calls == 1
    new = e.search(d, ev, review_geometry=True)
    assert gen.calls == 2  # existing empty-qualified-search generation is preserved
    [pick] = new["selected"]
    assert pick["pose_id"] == "a" and pick["match_level"] == "geometry_fallback"
    assert (
        pick["review_required"]
        and pick["not_exact_match"]
        and pick["constraints_satisfied"] is False
    )
    assert not pick["refine_allowed"] and not pick["release_allowed"]
    assert pick["core_conditions"]["whole_pose_validation"]["status"] != "passed"
    wire = DynamicSearchOut.model_validate(new).model_dump()
    assert wire["selected"][0]["review_required"] is True
    assert wire["geometry_review_fallback"]["qualified_count"] == 0


def test_review_geometry_never_overrides_existing_qualified_selection():
    e = engine()
    d = decision()
    ev = evidence()
    a = e.search(d, ev)
    b = e.search(d, ev, review_geometry=True)
    assert a["selected"] == b["selected"] and e.generator.calls == 0
    assert "geometry_review_fallback" not in b


@pytest.mark.parametrize(
    "mode", ["posture_conflict", "ownership_block", "bad_joints", "search_error"]
)
def test_review_geometry_keeps_hard_safety_gates(mode):
    if mode == "posture_conflict":
        e, d, ev, g = constrained_scene("lying")
    else:
        e, d, ev, g = constrained_scene()
        if mode == "ownership_block":
            d.joint.policy_block = "cross_person_owner"
        elif mode == "bad_joints":
            ev["joint_mask"][:] = False
        else:

            def fail(*args, **kwargs):
                raise RuntimeError("provider error")

            e.semantic.search = fail
    out = e.search(d, ev, review_geometry=True)
    assert not out["selected"]


def test_wire_never_erases_or_accepts_false_review_certification():
    e, d, ev, g = constrained_scene()
    pick = e.search(d, ev, review_geometry=True)["selected"][0]
    for field, value in [
        ("review_required", False),
        ("not_exact_match", False),
        ("constraints_satisfied", True),
    ]:
        bad = copy.deepcopy(pick)
        bad[field] = value
        with pytest.raises(ValueError):
            WholePoseCandidateOut.model_validate(bad)
