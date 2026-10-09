"""Per-person framing lineage and geometry-based search/refine safety gates."""
import io
from dataclasses import asdict

import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

from src.body_scope import (BodyScope, ScopeDetection, detect_scope, observed_legs,
                            promote_for_observed_legs)
from src.descriptor import build_slot_descriptors
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import Skeleton
from src.skeleton_extraction import PersonSlot
from src.vlm.client import MockVLMClient, _coerce


def payload(**overrides):
    return {"num_people": 2, "shot": "full_half", "approx_boxes": [
        {"x1": .6, "y1": .1, "x2": .9, "y2": .9},
        {"x1": .1, "y1": .1, "x2": .4, "y2": .9},
    ], **overrides}


@pytest.mark.parametrize("raw", [None, "half", ["half"], ["full", "bust", "head"]])
def test_missing_or_misaligned_scope_is_unknown(raw):
    vlm = _coerce(payload(body_scopes=raw, lower_body_visible=[False, False]), 512, 768)
    assert vlm.body_scopes == [None, None]
    assert detect_scope(vlm, 0).detected is None


def test_bad_entry_and_box_do_not_shift_person_scope():
    vlm = _coerce(payload(approx_boxes=[None, {"x1": .1, "y1": .1, "x2": .4, "y2": .9}],
                          body_scopes=[{"malformed": True}, "head"]), 512, 768)
    assert vlm.body_scopes == [None, BodyScope.HEAD]
    assert detect_scope(vlm, 0).detected is None
    assert detect_scope(vlm, 1).detected == BodyScope.HEAD
    assert detect_scope(vlm, None).detected is None


def test_legacy_cut_label_is_only_used_for_a_single_person():
    assert detect_scope(_coerce(payload(shot="bust"), 512, 768), 0).detected is None
    single = _coerce(payload(num_people=1, shot="bust"), 512, 768)
    assert asdict(detect_scope(single, 0)) == {"detected": BodyScope.BUST, "source": "legacy_shot"}
    uncertain = _coerce(payload(num_people=1, shot="bust", body_scopes=[None]), 512, 768)
    assert detect_scope(uncertain, 0).detected is None


def test_slot_identity_survives_sort_and_detector_provisional_is_unknown():
    vlm = _coerce(payload(body_scopes=["head", "half"]), 512, 768)
    slots = [PersonSlot(1, "vlm"), PersonSlot(0, "vlm"), PersonSlot(0, "rtm_provisional")]
    descs = build_slot_descriptors(vlm, slots)
    assert [d.output_scope.detected for d in descs] == [BodyScope.HALF, BodyScope.HEAD, None]
    assert all(not d.refine_allowed for d in descs)


@pytest.mark.parametrize("hint,route,scope", [("face", "skip", "head")])
def test_skipped_search_keeps_people_without_inventing_skeleton(hint, route, scope):
    class NoPose:
        def estimate(self, *args):
            raise AssertionError("Skipped route must not infer joints")

    pipe = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(), pose_model=NoPose())
    result = pipe.process_cut(f"{hint} 2p")
    assert result.route == route
    assert len(result.descriptors) == 2
    assert result.person_candidates == [[], []]
    assert all(d.skeleton is None and not d.refine_allowed for d in result.descriptors)
    assert [d.output_scope.detected.value for d in result.descriptors] == [scope, scope]


def test_full_geometry_is_preserved_when_crop_metadata_conflicts():
    class ScopedMock(MockVLMClient):
        def __init__(self, scope):
            self.scope = scope

        def analyze(self, *args):
            vlm = super().analyze(*args)
            vlm.body_scopes = [self.scope] * vlm.num_people
            return vlm

    index = build_synthetic_index()
    results = [Pipeline(index, vlm_client=ScopedMock(scope), pose_model=MockPoseModel())
               .process_cut("standing") for scope in [BodyScope.FULL, BodyScope.HALF]]
    left, right = results
    assert left.route == right.route == "core"
    assert [[(c.pose_id, c.distance) for c in cs] for cs in left.person_candidates] == [
        [(c.pose_id, c.distance) for c in cs] for cs in right.person_candidates]
    assert [(d.refine_allowed, d.lower_body_observed) for d in left.descriptors] == [
        (d.refine_allowed, d.lower_body_observed) for d in right.descriptors]


def test_api_serializes_per_person_detection_on_mixed_route(monkeypatch):
    import api.app as api_app

    vlm = _coerce(payload(shot="bust", body_scopes=["bust", "head"]), 512, 768)
    class FixedVLM:
        def analyze(self, *args):
            return vlm

    pipe = Pipeline(build_synthetic_index(), vlm_client=FixedVLM(), pose_model=MockPoseModel())
    monkeypatch.setitem(api_app.STATE, "pipeline", pipe)
    data = io.BytesIO()
    Image.new("RGB", (512, 768)).save(data, format="PNG")
    data.seek(0)
    result = api_app.analyze(UploadFile(file=data, filename="scope.png"), hint="", rescue="")
    people = result.model_dump(mode="json")["people"]
    # The mock pose draws knees for the bust person, so that label cannot crop legs.
    assert [p["output_scope"] for p in people] == [
        {"detected": "head", "source": "vlm_person"},
        {"detected": "full", "source": "observed_legs"},
    ]
    assert people[1]["quality_trace"]["output_scope_promotion"]["vlm_detected"] == "bust"
    assert [p["index"] for p in people] == [0, 1]
    assert people[0]["candidates"] == []
    assert "head_search_unsupported" in people[0]["quality_reasons"]
    assert people[1]["candidates"]


def standing_skeleton(leg_score=0.9):
    """Front-facing person fully inside a 600x500 frame: hips y=250, knees 350, ankles 450."""
    points = np.zeros((17, 2), dtype=np.float32)
    points[:5] = [150, 40]
    points[5:11] = [[190, 100], [110, 100], [220, 150], [80, 150], [230, 205], [70, 205]]
    points[11:] = [[175, 250], [125, 250], [175, 350], [125, 350], [175, 450], [125, 450]]
    scores = np.full(17, 0.9, dtype=np.float32)
    scores[13:] = leg_score
    return Skeleton(points, scores)


def test_knee_must_be_observed_inside_the_frame_and_owned():
    points = standing_skeleton().keypoints
    mask = np.ones(17, dtype=bool)
    assert observed_legs(points, mask, (), (600, 500)) == ("left_leg", "right_leg")
    # A knee predicted below the frame edge is a guess about an off-screen joint.
    assert observed_legs(points, mask, (), (600, 350)) == ()
    hidden_left_knee = mask.copy()
    hidden_left_knee[13] = False
    assert observed_legs(points, hidden_left_knee, (), (600, 500)) == ("right_leg",)
    assert observed_legs(points, mask, ("left_leg", "right_leg"), (600, 500)) == ()


@pytest.mark.parametrize("scope", [BodyScope.HALF, BodyScope.BUST])
def test_observed_legs_promote_only_leg_cutting_scopes(scope):
    for source in ("vlm_person", "legacy_shot"):
        promoted = promote_for_observed_legs(ScopeDetection(scope, source), ("left_leg",))
        assert promoted == ScopeDetection(BodyScope.FULL, "observed_legs")
        unchanged = ScopeDetection(scope, source)
        assert promote_for_observed_legs(unchanged, ()) is unchanged
    for kept in (ScopeDetection(), ScopeDetection(BodyScope.FULL, "vlm_person"),
                 ScopeDetection(BodyScope.HEAD, "vlm_person")):
        assert promote_for_observed_legs(kept, ("left_leg", "right_leg")) is kept


class _FixedPose:
    self_detecting = True

    def __init__(self, skeleton):
        self.skeleton = skeleton

    def estimate(self, *args):
        return [self.skeleton]

    def estimate_crop_candidates(self, *args):
        return []


def _half_labelled_cut(skeleton):
    class VLM:
        def analyze(self, *args):
            return _coerce({
                "num_people": 1, "shot": "full_half", "body_scopes": ["half"],
                "approx_boxes": [{"x1": .05, "y1": .02, "x2": .45, "y2": .98}],
            }, 600, 500)

    pipe = Pipeline(build_synthetic_index(), vlm_client=VLM(), pose_model=_FixedPose(skeleton))
    return pipe.process_cut(None, 600, 500)


def test_half_label_with_observed_knees_is_output_as_full():
    result = _half_labelled_cut(standing_skeleton())
    desc = result.descriptors[0]
    assert desc.output_scope == ScopeDetection(BodyScope.FULL, "observed_legs")
    assert desc.quality_trace["output_scope_promotion"] == {
        "vlm_detected": "half", "vlm_source": "vlm_person",
        "observed_legs": ["left_leg", "right_leg"],
    }
    # Output framing only: the fail-closed refine visibility gate is untouched.
    assert desc.lower_body_observed is False
    assert result.person_candidates[0]


def test_half_label_without_observed_knees_stays_half():
    desc = _half_labelled_cut(standing_skeleton(leg_score=0.0)).descriptors[0]
    assert desc.output_scope == ScopeDetection(BodyScope.HALF, "vlm_person")
    assert "output_scope_promotion" not in desc.quality_trace
