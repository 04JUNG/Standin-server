"""Torso search eligibility must not erase safe observed coordinates."""
import numpy as np
import pytest

from api.models import SkeletonOut
from src.descriptor import build_slot_descriptors
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import BBox, Skeleton
from src.skeleton_extraction import (
    PersonSlot, analyze_skeleton, finalize_slot, observed_joints_without_torso,
)
from src.vlm.client import MockVLMClient, _coerce
from standin_eval.pose_benchmark import _prediction_person


def fixture():
    points = MockPoseModel._CANON * 100 + np.array([100, 120])
    scores = np.zeros(17, dtype=np.float32)
    scores[[0, 1, 2, 3, 4, 5, 7, 9]] = .9
    skeleton = Skeleton(points.astype(np.float32), scores)
    box = BBox(0, 0, 200, 250, source='vlm')
    evidence = analyze_skeleton(skeleton, box, owner_box=box)
    slot = PersonSlot(0, 'vlm', vlm_box=box, skeleton=skeleton,
                      skeleton_box=box, evidence=evidence,
                      state=evidence.state, skeleton_source='full_image',
                      reasons=list(evidence.reasons), assigned_rtm_index=0)
    finalize_slot(slot)
    vlm = _coerce(dict(num_people=1, shot='bust', approx_boxes=[
        dict(x1=0, y1=0, x2=1, y2=1)], body_scopes=['bust']), 200, 250)
    return slot, vlm


def test_torso_failure_preserves_only_existing_observed_coordinates():
    slot, vlm = fixture()
    original = slot.skeleton.keypoints.copy()
    desc = build_slot_descriptors(vlm, [slot], image_size=(200, 250))[0]
    assert desc.skeleton is not None
    np.testing.assert_array_equal(desc.skeleton.keypoints, original)
    assert desc.quality_trace['observed_joint_mask'] == slot.evidence.valid_joint_mask.tolist()
    assert desc.skeleton_state == 'invalid' and desc.coverage_class == 'insufficient'
    assert desc.feature is None and not desc.refine_allowed
    assert not desc.skeleton.scores.any()  # old clients cannot refine these points
    SkeletonOut(**desc.skeleton.to_serializable())  # finite HTTP contract
    np.testing.assert_array_equal(slot.skeleton.keypoints, original)


@pytest.mark.parametrize('reason', ['assignment_ambiguous', 'duplicate_candidate',
    'merge_suspected', 'torso_cross_slot', 'left_arm_cross_slot',
    'torso_outside_slot', 'torso_degenerate', 'unstable_search_after_retry'])
def test_ownership_or_structural_failures_still_withhold_output(reason):
    slot, vlm = fixture()
    slot.reasons.append(reason)
    desc = build_slot_descriptors(vlm, [slot], image_size=(200, 250))[0]
    assert desc.skeleton is None
    assert 'observed_joint_mask' not in desc.quality_trace


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_input_cannot_reenter_the_http_output(value):
    slot, vlm = fixture()
    slot.skeleton.keypoints[16, 0] = value
    assert build_slot_descriptors(vlm, [slot], image_size=(200, 250))[0].skeleton is None


def test_observation_mask_enforces_owner_and_image_bounds_without_zero_sentinel():
    slot, _ = fixture()
    slot.skeleton.keypoints[0] = [0, 0]  # a real image-origin point
    slot.skeleton.keypoints[1] = [-1, 20]
    slot.skeleton.keypoints[2] = [200, 20]
    slot.skeleton.keypoints[3] = [20, 250]
    mask = observed_joints_without_torso(slot, (200, 250))
    assert mask[0] and not mask[1:4].any()
    slot.vlm_box = BBox(60, 10, 140, 100)
    mask = observed_joints_without_torso(slot, (200, 250))
    assert mask is not None and not mask[9]  # wrist outside padded owner


def test_missing_or_provisional_person_is_not_fabricated():
    slot, _ = fixture()
    slot.skeleton = None
    assert observed_joints_without_torso(slot, (200, 250)) is None
    slot, _ = fixture()
    slot.slot_origin = 'rtm_provisional'
    assert observed_joints_without_torso(slot, (200, 250)) is None
    slot.slot_origin = 'vlm'
    assert observed_joints_without_torso(slot, None) is None


def test_isolated_body_endpoint_is_not_restored_without_a_valid_segment():
    slot, _ = fixture()
    slot.skeleton.scores[:] = 0
    slot.skeleton.scores[[0, 1, 10, 14]] = .9
    slot.evidence = analyze_skeleton(slot.skeleton, slot.vlm_box)
    mask = observed_joints_without_torso(slot, (200, 250))
    assert mask[:2].all()
    assert not mask[5:].any()  # floating wrist/knee are not body observations
    slot.skeleton.scores[:2] = 0
    slot.evidence = analyze_skeleton(slot.skeleton, slot.vlm_box)
    assert observed_joints_without_torso(slot, (200, 250)) is None


def test_observation_only_output_never_becomes_search_or_refine_eligible():
    slot, vlm = fixture()
    pipe = Pipeline([], vlm_client=MockVLMClient(), pose_model=MockPoseModel())
    outcome = pipe._evaluate_slot(vlm, slot, 1.0, image_size=(200, 250))
    assert outcome.descriptor.skeleton is not None
    assert outcome.candidates == [] and outcome.confidence == 'low'
    assert not outcome.descriptor.refine_allowed
    assert not outcome.descriptor.skeleton.scores.any()
    assert outcome.descriptor.feature is None
    assert outcome.descriptor.quality_trace['observed_joint_mask'][0]


def test_existing_valid_output_is_unchanged_and_has_no_observation_override():
    slot, vlm = fixture()
    slot.skeleton.scores[:] = .9
    slot.evidence = analyze_skeleton(slot.skeleton, slot.vlm_box)
    slot.state = slot.evidence.state
    desc = build_slot_descriptors(vlm, [slot], image_size=(200, 250))[0]
    assert 'observed_joint_mask' not in desc.quality_trace
    np.testing.assert_array_equal(desc.skeleton.keypoints, slot.skeleton.keypoints)
    np.testing.assert_array_equal(desc.skeleton.scores, slot.evidence.effective_scores)


def test_evaluator_respects_observation_mask_and_separate_observation_scores():
    slot, vlm = fixture()
    desc = build_slot_descriptors(vlm, [slot], image_size=(200, 250))[0]
    row = dict(keypoints=desc.skeleton.keypoints.tolist(),
               scores=desc.skeleton.scores.tolist(), quality_trace=desc.quality_trace)
    points = _prediction_person(row, .3)
    assert points[0] == row['keypoints'][0] and points[6] is None
    row['quality_trace']['observed_joint_mask'] = [1] * 17
    with pytest.raises(ValueError, match='17 explicit booleans'):
        _prediction_person(row, None)
    row.pop('quality_trace')
    assert _prediction_person(row, None) == row['keypoints']  # prior runs unchanged
