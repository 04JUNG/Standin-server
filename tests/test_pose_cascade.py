"""Executable contract for current-X -> Human-Art M rescue cascade."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG
import src.pipeline as pipeline_module
from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import BasePoseModel, MockPoseModel, RTMPoseModel
from src.pose_cascade import CascadePoseModel
from src.pose_rescue import parse_rescue_request
from src.schema import Action, BBox, Relationship, Shot, VLMAnalysis, View
from src.vlm.client import BaseVLMClient
from scripts.compare_pose_canary import cascade_promotion_decision


class _Image:
    pass


class FixedVLM(BaseVLMClient):
    def __init__(self, boxes):
        self.boxes = list(boxes)

    def analyze(self, image, img_w, img_h):
        count = len(self.boxes)
        return VLMAnalysis(
            count, Shot.FULL_HALF, Action.STANDING, View.FRONT,
            Relationship.SOLO if count == 1 else Relationship.TALKING,
            list(self.boxes),
            lower_body_visible=[True] * count,
            lower_body_visibility_known=[True] * count,
        )


def _skeleton(box):
    return MockPoseModel().estimate(_Image(), [box], 400, 300)[0]


class FakeCascadePose(BasePoseModel):
    self_detecting = True

    def __init__(self, primary, fallback, *, stage="canary-100",
                 rescue_error: Exception | None = None):
        self.primary = list(primary)
        self.fallback = list(fallback)
        self.stage = stage
        self.rescue_error = rescue_error
        self.full_calls = 0
        self.crop_calls = 0
        self.rescue_calls = 0

    def estimate(self, image, boxes, img_w, img_h):
        self.full_calls += 1
        return list(self.primary)

    def estimate_crop_candidates(self, image, box, img_w, img_h):
        self.crop_calls += 1
        return []

    def rescue_candidates(self, image, img_w, img_h):
        self.rescue_calls += 1
        if self.rescue_error is not None:
            raise self.rescue_error
        return list(self.fallback)

    def fallback_kpt_threshold(self):
        return 0.35

    def rescue_stage(self):
        return self.stage


def _pipeline(boxes, pose):
    return Pipeline(
        build_synthetic_index(), vlm_client=FixedVLM(boxes), pose_model=pose,
    )


def test_no_unresolved_slot_never_calls_fallback():
    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose([_skeleton(box)], [_skeleton(box)])
    result = _pipeline([box], pose).process_cut(_Image(), 400, 300)
    assert pose.full_calls == 1
    assert pose.rescue_calls == 0
    assert result.descriptors[0].skeleton_source == "full_image"


def test_automatic_rescue_accepts_only_missing_slot_and_stays_low():
    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose([], [_skeleton(box)])
    result = _pipeline([box], pose).process_cut(_Image(), 400, 300)
    descriptor = result.descriptors[0]
    assert pose.rescue_calls == 1
    assert descriptor.skeleton_source == "fallback_full_image"
    assert descriptor.quality_trace["pose_rescue"]["accepted"] is True
    summary = descriptor.quality_trace["pose_rescue"]["cut_summary"]
    assert summary["triggered"] is True
    assert summary["unresolved_before"] == 1
    assert summary["accepted"] == 1
    assert summary["error"] is None
    assert result.person_candidates[0]
    assert result.person_confidence[0] == "low"
    assert descriptor.refine_allowed is False
    assert np.allclose(descriptor.skeleton.scores, 0.0)


def test_humanart_rescue_uses_one_conservative_pos_search_even_if_global_metric_differs():
    box = BBox(0, 0, 200, 300, "vlm")
    fallback = _skeleton(box)
    fallback.scores[9] = 0.0  # 불완전 왼팔: conservative mask가 elbow까지 제거
    pose = FakeCascadePose([], [fallback])
    original_metric = CFG.distance_metric
    original_top_k = CFG.top_k_final
    original_knn = pipeline_module.knn_geometric
    calls = []

    def recording_knn(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_knn(*args, **kwargs)

    try:
        CFG.distance_metric = "angle"
        CFG.top_k_final = 2
        pipeline_module.knn_geometric = recording_knn
        result = _pipeline([box], pose).process_cut(_Image(), 400, 300)
    finally:
        pipeline_module.knn_geometric = original_knn
        CFG.distance_metric = original_metric
        CFG.top_k_final = original_top_k

    descriptor = result.descriptors[0]
    trace = descriptor.quality_trace
    search_mask = np.asarray(trace["search_valid_joint_mask"], dtype=bool)
    assert descriptor.skeleton_source == "fallback_full_image"
    assert descriptor.distance_metric == "pos"
    assert trace["search_scope"] == "humanart_conservative"
    assert len(calls) == 1
    assert calls[0]["metric"] == "pos"
    assert calls[0]["top_k"] == 5
    assert not search_mask[7] and not search_mask[9]
    assert len(result.person_candidates[0]) == 5
    assert trace["search_top_k"] == 5
    assert trace["candidate_count"] == 5
    assert result.person_confidence[0] == "low"
    assert descriptor.refine_allowed is False


def test_humanart_rescue_returns_x_when_pos_library_has_no_candidate():
    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose([], [_skeleton(box)])
    result = Pipeline(
        [], vlm_client=FixedVLM([box]), pose_model=pose,
    ).process_cut(_Image(), 400, 300)

    descriptor = result.descriptors[0]
    assert descriptor.skeleton_source == "fallback_full_image"
    assert descriptor.distance_metric == "pos"
    assert descriptor.quality_trace["search_scope"] == "humanart_conservative"
    assert descriptor.quality_trace["search_top_k"] == 5
    assert descriptor.quality_trace["candidate_count"] == 0
    assert result.person_candidates[0] == []  # API의 X 계약
    assert result.person_confidence[0] == "low"
    assert descriptor.refine_allowed is False


def test_cross_model_duplicate_is_removed_without_creating_ghost_slot():
    left = BBox(0, 0, 190, 300, "vlm")
    right = BBox(210, 0, 400, 300, "vlm")
    left_pose = _skeleton(left)
    right_pose = _skeleton(right)
    pose = FakeCascadePose([left_pose], [left_pose, right_pose])
    result = _pipeline([left, right], pose).process_cut(_Image(), 400, 300)
    assert len(result.descriptors) == 2
    assert pose.rescue_calls == 1
    sources = [descriptor.skeleton_source for descriptor in result.descriptors]
    assert sources == ["full_image", "fallback_full_image"]
    assert any("duplicate_of_resolved" in note for note in result.notes)


def test_cross_slot_fallback_candidate_is_rejected():
    left = BBox(0, 0, 190, 300, "vlm")
    right = BBox(210, 0, 400, 300, "vlm")
    left_pose = _skeleton(left)
    crossing_pose = _skeleton(right)
    crossing_pose.keypoints[7] = np.asarray([120.0, 120.0], dtype=np.float32)
    crossing_pose.keypoints[9] = np.asarray([90.0, 160.0], dtype=np.float32)
    pose = FakeCascadePose([left_pose], [crossing_pose])
    result = _pipeline([left, right], pose).process_cut(_Image(), 400, 300)
    right_descriptor = result.descriptors[1]
    trace = right_descriptor.quality_trace["pose_rescue"]
    assert trace["accepted"] is False
    assert trace["rejected_reason"] == "cross_slot_ownership"
    assert right_descriptor.skeleton is None


def test_shadow_runs_assignment_but_never_replaces_slot():
    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose([], [_skeleton(box)], stage="shadow")
    result = _pipeline([box], pose).process_cut(_Image(), 400, 300)
    trace = result.descriptors[0].quality_trace["pose_rescue"]
    assert pose.rescue_calls == 1
    assert trace["would_accept"] is True
    assert trace["accepted"] is False
    assert result.descriptors[0].skeleton is None
    assert result.person_candidates[0] == []


def test_fallback_error_preserves_primary_failure_without_raising():
    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose([], [], rescue_error=RuntimeError("boom"))
    result = _pipeline([box], pose).process_cut(_Image(), 400, 300)
    assert result.descriptors[0].skeleton is None
    assert result.person_candidates[0] == []
    assert any("fallback_error:RuntimeError" in note for note in result.notes)
    summary = result.descriptors[0].quality_trace["pose_rescue"]["cut_summary"]
    assert summary["error"] == "fallback_error:RuntimeError"


def test_manual_all_can_replace_resolved_slot_but_keeps_safety_policy():
    box = BBox(0, 0, 200, 300, "vlm")
    primary = _skeleton(box)
    fallback = _skeleton(BBox(10, 0, 190, 300, "pose"))
    pose = FakeCascadePose([primary], [fallback])
    result = _pipeline([box], pose).process_cut(
        _Image(), 400, 300, rescue_request="all"
    )
    descriptor = result.descriptors[0]
    assert pose.rescue_calls == 1
    assert descriptor.skeleton_source == "fallback_full_image"
    assert descriptor.quality_trace["pose_rescue"]["trigger"] == "manual"
    assert result.person_confidence[0] == "low"
    assert descriptor.refine_allowed is False


def test_manual_unknown_person_index_is_reported_without_fallback_call():
    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose([_skeleton(box)], [], stage="canary-100")
    result = _pipeline([box], pose).process_cut(
        _Image(), 400, 300, rescue_request="3"
    )
    assert pose.rescue_calls == 0
    assert any("unknown_person_index:3" in note for note in result.notes)


def test_analyze_forwards_manual_rescue_form_field():
    from fastapi import UploadFile
    from PIL import Image
    import api.app as api_app

    box = BBox(0, 0, 200, 300, "vlm")
    pose = FakeCascadePose(
        [_skeleton(box)],
        [_skeleton(BBox(10, 0, 190, 300, "pose"))],
    )
    image_bytes = io.BytesIO()
    Image.new("RGB", (400, 300), color=(255, 255, 255)).save(
        image_bytes, format="PNG",
    )
    image_bytes.seek(0)

    previous_state = dict(api_app.STATE)
    api_app.STATE.clear()
    api_app.STATE.update({
        "pipeline": _pipeline([box], pose),
        "provider": "mock",
        "pose_backend": "mock",
    })
    try:
        result = api_app.analyze(
            UploadFile(filename="cut.png", file=image_bytes),
            rescue="all",
        )
    finally:
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)

    assert pose.rescue_calls == 1
    assert result.people[0].skeleton_source == "fallback_full_image"


def test_analyze_rejects_invalid_rescue_selector_as_422():
    from fastapi import HTTPException, UploadFile
    import api.app as api_app

    try:
        api_app.analyze(
            UploadFile(filename="cut.png", file=io.BytesIO(b"unused")),
            rescue="0,0",
        )
    except HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["code"] == "invalid_rescue_selector"
    else:
        raise AssertionError("invalid rescue selector was accepted")


def test_crop_retry_budget_has_absolute_cap_two():
    boxes = [BBox(i * 100, 0, (i + 1) * 100, 300, "vlm") for i in range(6)]
    pose = FakeCascadePose([], [], stage="off")
    previous_floor = CFG.slot_crop_max_per_cut
    previous_cap = CFG.slot_crop_hard_cap
    try:
        CFG.slot_crop_max_per_cut = 2
        CFG.slot_crop_hard_cap = 2
        _pipeline(boxes, pose).process_cut(_Image(), 600, 300)
    finally:
        CFG.slot_crop_max_per_cut = previous_floor
        CFG.slot_crop_hard_cap = previous_cap
    assert pose.crop_calls == 2


def test_crop_retry_budget_never_expands_past_two():
    boxes = [BBox(i * 100, 0, (i + 1) * 100, 300, "vlm") for i in range(3)]
    pose = FakeCascadePose([], [], stage="off")
    previous_floor = CFG.slot_crop_max_per_cut
    previous_cap = CFG.slot_crop_hard_cap
    try:
        CFG.slot_crop_max_per_cut = 2
        CFG.slot_crop_hard_cap = 2
        _pipeline(boxes, pose).process_cut(_Image(), 300, 300)
    finally:
        CFG.slot_crop_max_per_cut = previous_floor
        CFG.slot_crop_hard_cap = previous_cap
    assert pose.crop_calls == 2


def test_crop_retry_rejects_neighbor_already_resolved_by_current_x():
    left = BBox(0, 0, 190, 300, "vlm")
    right = BBox(210, 0, 400, 300, "vlm")
    left_pose = _skeleton(left)

    class WrongNeighborCrop(FakeCascadePose):
        def estimate_crop_candidates(self, image, box, img_w, img_h):
            self.crop_calls += 1
            return [left_pose]

    pose = WrongNeighborCrop([left_pose], [], stage="off")
    result = _pipeline([left, right], pose).process_cut(_Image(), 400, 300)

    assert pose.crop_calls == 1
    right_descriptor = result.descriptors[1]
    assert right_descriptor.skeleton is None
    assert right_descriptor.quality_trace["crop_mapping"]["accepted"] is False
    assert right_descriptor.quality_trace["crop_mapping"]["rejected_reason"] \
        == "duplicate_of_resolved"


def test_crop_retry_same_person_cannot_fill_two_failed_slots():
    left = BBox(0, 0, 190, 300, "vlm")
    right = BBox(210, 0, 400, 300, "vlm")
    left_pose = _skeleton(left)

    class RepeatedPersonCrop(FakeCascadePose):
        def estimate_crop_candidates(self, image, box, img_w, img_h):
            self.crop_calls += 1
            return [left_pose]

    pose = RepeatedPersonCrop([], [], stage="off")
    result = _pipeline([left, right], pose).process_cut(_Image(), 400, 300)

    assert pose.crop_calls == 2
    assert [item.skeleton_source for item in result.descriptors] == [
        "crop_retry", "none",
    ]
    assert result.descriptors[1].quality_trace["crop_mapping"][
        "rejected_reason"
    ] == "duplicate_of_resolved"


def test_crop_retry_maps_shuffled_candidates_to_their_unique_owner():
    left = BBox(0, 0, 190, 300, "vlm")
    right = BBox(210, 0, 400, 300, "vlm")
    left_pose = _skeleton(left)
    right_pose = _skeleton(right)

    class ShuffledCrop(FakeCascadePose):
        def estimate_crop_candidates(self, image, box, img_w, img_h):
            self.crop_calls += 1
            return (
                [right_pose, left_pose]
                if box.x1 < 200 else [left_pose, right_pose]
            )

    pose = ShuffledCrop([], [], stage="off")
    result = _pipeline([left, right], pose).process_cut(_Image(), 400, 300)

    assert pose.crop_calls == 2
    assert [item.skeleton_source for item in result.descriptors] == [
        "crop_retry", "crop_retry",
    ]
    assert [
        item.quality_trace["crop_mapping"]["selected_candidate_index"]
        for item in result.descriptors
    ] == [1, 1]


def test_rescue_selector_validation():
    assert parse_rescue_request("").mode == "auto"
    assert parse_rescue_request("all").mode == "all"
    assert parse_rescue_request("0,2").person_indices == (0, 2)
    for invalid in ("-1", "0,0", "one", "1,", "100"):
        try:
            parse_rescue_request(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid rescue selector accepted: {invalid}")


def test_promotion_gate_rolls_back_on_first_wrong_owner():
    assert cascade_promotion_decision(0.90, 1) \
        == "rollback_current_x_and_stop_promotion"
    assert cascade_promotion_decision(0.49, 0) \
        == "remove_cascade_and_return_current_x"
    assert cascade_promotion_decision(0.50, None) \
        == "blocked_pending_wrong_owner_review"
    assert cascade_promotion_decision(0.50, 0) \
        == "eligible_for_next_canary_stage"
    assert cascade_promotion_decision(0.90, 0, local_checks_pass=False) \
        == "blocked_failed_local_shadow_checks"
    # D10의 즉시 롤백/제거 결정은 일반 local gate 실패보다 우선한다.
    assert cascade_promotion_decision(0.90, 1, local_checks_pass=False) \
        == "rollback_current_x_and_stop_promotion"
    assert cascade_promotion_decision(0.49, 0, local_checks_pass=False) \
        == "remove_cascade_and_return_current_x"


class _FakeBundle:
    build_id = "humanart-test"
    detector_path = Path("/tmp/yolox-shared.onnx")
    detector_input_size_wh = (640, 640)
    calibration = {
        "profile_id": "humanart-rescue-v1",
        "profile_sha256": "profile-hash",
        "skeleton_kpt_threshold": 0.35,
    }

    def identity(self):
        return {
            "model_id": "humanart-m",
            "build_id": self.build_id,
            "status": "canary",
            "license_review": "approved",
            "calibration_profile_id": self.calibration["profile_id"],
            "calibration_profile_sha256": self.calibration["profile_sha256"],
        }


def test_cascade_lazy_loader_is_singleton_under_concurrency():
    box = BBox(0, 0, 200, 300, "vlm")
    primary = FakeCascadePose([_skeleton(box)], [], stage="off")
    fallback = FakeCascadePose([_skeleton(box)], [], stage="off")
    factory_calls = []

    def factory(bundle):
        factory_calls.append(bundle)
        return fallback

    cascade = CascadePoseModel(
        primary=primary,
        fallback_bundle=_FakeBundle(),
        fallback_factory=factory,
        canary_stage="canary-100",
    )
    assert cascade.runtime_identity()["fallback_initialized"] is False
    cascade.estimate(_Image(), None, 400, 300)
    assert factory_calls == []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _index: cascade.rescue_candidates(_Image(), 400, 300),
            range(8),
        ))
    assert len(factory_calls) == 1
    assert all(len(result) == 1 for result in results)
    assert cascade.runtime_identity()["fallback_initialized"] is True


def test_cascade_reuses_exact_primary_detector_context():
    detector = object()
    box = BBox(0, 0, 200, 300, "vlm")

    class Primary(BasePoseModel):
        self_detecting = True

        def detector_contract(self):
            return {
                "model_path": str(_FakeBundle.detector_path),
                "input_size_wh": _FakeBundle.detector_input_size_wh,
                "component": detector,
            }

        def estimate_with_rescue_context(self, image, boxes, img_w, img_h):
            return [], {
                "detector_model_path": str(_FakeBundle.detector_path.resolve()),
                "detector_input_size_wh": _FakeBundle.detector_input_size_wh,
                "detected_bboxes": np.asarray([[0, 0, 200, 300]], dtype=np.float32),
            }

        def estimate(self, image, boxes, img_w, img_h):
            raise AssertionError("context-aware path must be used")

    class Fallback:
        def __init__(self):
            self.detected = None

        def estimate_with_detections(self, image, detected, img_w, img_h):
            self.detected = np.asarray(detected).copy()
            return [_skeleton(box)]

        def estimate(self, image, boxes, img_w, img_h):
            raise AssertionError("fallback detector must not run again")

    fallback = Fallback()
    cascade = CascadePoseModel(
        primary=Primary(), fallback_bundle=_FakeBundle(),
        fallback_factory=lambda _bundle: fallback,
        canary_stage="canary-100",
    )
    _primary, context = cascade.estimate_with_rescue_context(
        _Image(), None, 400, 300
    )
    candidates = cascade.rescue_candidates_with_context(
        _Image(), 400, 300, context
    )
    assert cascade.runtime_identity()["shared_detector_session"] is True
    assert len(candidates) == 1
    assert np.array_equal(fallback.detected, context["detected_bboxes"])


def test_pipeline_passes_request_local_detector_context_to_rescue():
    box = BBox(0, 0, 200, 300, "vlm")

    class ContextPose(FakeCascadePose):
        def estimate_with_rescue_context(self, image, boxes, img_w, img_h):
            self.full_calls += 1
            return [], {"request_token": "same-cut"}

        def rescue_candidates_with_context(
            self, image, img_w, img_h, context,
        ):
            assert context == {"request_token": "same-cut"}
            self.rescue_calls += 1
            return list(self.fallback)

    pose = ContextPose([], [_skeleton(box)])
    result = _pipeline([box], pose).process_cut(_Image(), 400, 300)
    assert pose.full_calls == 1
    assert pose.rescue_calls == 1
    assert result.descriptors[0].skeleton_source == "fallback_full_image"


def test_current_x_detector_runs_once_and_returns_reusable_boxes():
    class Detector:
        onnx_model = "/tmp/yolox.onnx"
        model_input_size = (640, 640)

        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            return np.asarray([[10, 20, 110, 220]], dtype=np.float32)

    class Pose:
        def __init__(self):
            self.boxes = None

        def __call__(self, image, *, bboxes):
            self.boxes = np.asarray(bboxes).copy()
            return (
                np.zeros((1, 17, 2), dtype=np.float32),
                np.ones((1, 17), dtype=np.float32),
            )

    detector = Detector()
    pose = Pose()
    model = RTMPoseModel.__new__(RTMPoseModel)
    model.model = type("Body", (), {
        "one_stage": False,
        "det_model": detector,
        "pose_model": pose,
    })()
    skeletons, context = model.estimate_with_rescue_context(
        np.zeros((240, 120, 3), dtype=np.uint8), None, 120, 240
    )
    assert detector.calls == 1
    assert len(skeletons) == 1
    assert np.array_equal(context["detected_bboxes"], pose.boxes)
    assert context["detector_input_size_wh"] == (640, 640)


def main() -> int:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
