"""docs/SKELETON_EXTRACTION_IMPROVEMENT.md의 P0~슬롯 코어 계약."""
import copy
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.features import angle_distance, bone_dirs, normalize_skeleton, pose_distance
from src.library import build_entries_from_pose, build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.repo import build_db
from src.schema import (
    Action, BBox, LibraryEntry, PoseCandidate, Relationship, Shot,
    Skeleton, VLMAnalysis, View,
)
from src.search import candidate_stability, pose_family_id
from src.vlm.client import BaseVLMClient, MockVLMClient
from src.skeleton_extraction import (
    analyze_skeleton,
    are_duplicate_skeletons,
    assign_candidates,
    select_crop_candidate,
    sort_slots_left_to_right,
)


CANON = np.array(MockPoseModel._CANON, dtype=np.float32)


class _Img(str):
    @property
    def hint(self):
        return str(self)


def _skeleton(cx=100.0, cy=120.0, scale=100.0, scores=None):
    keypoints = CANON * scale + np.array([cx, cy], dtype=np.float32)
    if scores is None:
        scores = np.full(17, 0.9, dtype=np.float32)
    return Skeleton(keypoints, np.asarray(scores, dtype=np.float32))


def test_pose_distance_excludes_explicitly_masked_query_joint():
    a = np.zeros((17, 2), dtype=np.float32)
    b = a.copy()
    b[9] = [10.0, 0.0]
    mask = np.ones(17, dtype=bool)
    mask[9] = False
    assert pose_distance(a, b) > 0.0
    assert np.isclose(pose_distance(a, b, mask), 0.0)


def test_zero_coordinate_is_valid_when_explicit_mask_says_so():
    feature = np.zeros((17, 2), dtype=np.float32)
    feature[5] = [0.0, 0.0]
    feature[7] = [1.0, 0.0]
    _, legacy = bone_dirs(feature)
    _, explicit = bone_dirs(feature, np.ones(17, dtype=bool))
    assert not legacy[0]
    assert explicit[0]


def test_angle_distance_uses_both_joint_masks():
    a = np.zeros((17, 2), dtype=np.float32)
    b = np.zeros((17, 2), dtype=np.float32)
    a[5], a[7] = [0, 0], [1, 0]
    b[5], b[7] = [0, 0], [-1, 0]
    mask = np.zeros(17, dtype=bool)
    mask[[5, 7]] = True
    assert np.isclose(angle_distance(a, b, mask, mask), 2.0)
    mask[7] = False
    assert np.isclose(angle_distance(a, b, mask, mask), 2.0)  # 유효 뼈 없음 sentinel


def test_coverage_classes_are_based_on_complete_limbs():
    full = analyze_skeleton(_skeleton(), BBox(0, 0, 200, 250))
    assert full.coverage_class == "full" and full.state == "valid"

    reduced_scores = np.full(17, 0.9, dtype=np.float32)
    reduced_scores[[8, 10, 14, 16]] = 0.0
    reduced = analyze_skeleton(
        _skeleton(scores=reduced_scores), BBox(0, 0, 200, 250)
    )
    assert reduced.coverage_class == "reduced"
    assert set(reduced.refinable_limbs) == {"left_arm", "left_leg"}

    sparse_scores = np.full(17, 0.9, dtype=np.float32)
    sparse_scores[[7, 8, 9, 10, 13, 14, 15, 16]] = 0.0
    sparse = analyze_skeleton(
        _skeleton(scores=sparse_scores), BBox(0, 0, 200, 250)
    )
    assert sparse.coverage_class == "sparse"
    assert not sparse.high_confidence_eligible
    assert sparse.refinable_limbs == ()


def test_bad_normalization_anchor_is_insufficient_before_search():
    scores = np.full(17, 0.9, dtype=np.float32)
    scores[5] = 0.0
    evidence = analyze_skeleton(_skeleton(scores=scores), BBox(0, 0, 200, 250))
    assert evidence.coverage_class == "insufficient"
    assert "invalid_torso_anchors" in evidence.reasons


def test_metric_and_coverage_thresholds_are_independent():
    cfg = copy.copy(CFG)
    cfg.fallback_pos_full = 0.4
    cfg.fallback_pos_reduced = 0.2
    cfg.fallback_angle_full = 0.7
    assert cfg.fallback_threshold("pos", "full") == 0.4
    assert cfg.fallback_threshold("pos", "reduced") == 0.2
    assert cfg.fallback_threshold("angle", "full") == 0.7
    assert cfg.fallback_threshold("pos", "sparse") is None


def test_stability_folds_mirror_ids_into_same_pose_family():
    assert pose_family_id("Stand_01_mirror") == "Stand_01"
    a = [PoseCandidate(f"pose_{i}", View.FRONT, 0.1, {}) for i in range(5)]
    b = [PoseCandidate(f"pose_{i}_mirror", View.BACK, 0.1, {}) for i in range(3)]
    b += [PoseCandidate(f"other_{i}", View.FRONT, 0.2, {}) for i in range(2)]
    stability = candidate_stability(a, b, [])
    assert stability["family_overlap"] == 3
    assert stability["status"] == "stable"


def test_incomplete_bvh_body_mapping_fails_before_entry_creation():
    joints = np.zeros((17, 3), dtype=np.float32)
    scores = np.ones(17, dtype=np.float32)
    scores[9] = 0.0
    try:
        build_entries_from_pose("broken", joints, {}, scores=scores)
    except ValueError as exc:
        assert "incomplete BVH body mapping" in str(exc)
    else:
        raise AssertionError("불완전 body mapping을 허용했습니다")


def test_db_rejects_wrong_shape_and_non_finite_feature():
    for feature in (np.zeros(33, np.float32),
                    np.full(34, np.nan, np.float32)):
        entry = LibraryEntry("bad", View.FRONT, feature, {})
        with tempfile.TemporaryDirectory() as directory:
            try:
                build_db([entry], os.path.join(directory, "poses.db"))
            except ValueError:
                pass
            else:
                raise AssertionError("잘못된 feature를 DB에 저장했습니다")


def test_global_assignment_is_one_to_one_and_final_order_is_left_to_right():
    left_box = BBox(20, 10, 180, 260, "vlm")
    right_box = BBox(220, 10, 380, 260, "vlm")
    right = _skeleton(cx=300)
    left = _skeleton(cx=100)
    result = assign_candidates([left_box, right_box], [right, left], 400, 300)
    assert len(result.slots) == 2
    assert {id(slot.skeleton) for slot in result.slots} == {id(left), id(right)}
    ordered = sort_slots_left_to_right(result.slots)
    assert ordered[0].skeleton is left
    assert ordered[1].skeleton is right


def test_vlm_boxless_path_promotes_only_nonduplicate_structural_rtm():
    one = assign_candidates([], [_skeleton()], 400, 300)
    assert len(one.slots) == 1
    assert one.slots[0].slot_origin == "rtm_provisional"

    duplicate = assign_candidates([], [_skeleton(), _skeleton()], 400, 300)
    assert duplicate.slots == []
    assert duplicate.unmatched_candidate_indices == [0, 1]


def test_equal_counts_do_not_make_duplicate_skeletons_valid():
    boxes = [BBox(20, 0, 200, 280, "vlm"), BBox(40, 0, 220, 280, "vlm")]
    result = assign_candidates(boxes, [_skeleton(cx=100), _skeleton(cx=100)], 400, 300)
    assert len(result.slots) == 2
    assert all(slot.state == "suspect" for slot in result.slots)
    assert all("duplicate_candidate" in slot.reasons for slot in result.slots)


def test_high_iou_people_are_not_duplicates_when_keypoints_differ():
    first = _skeleton(cx=100)
    second = _skeleton(cx=100)
    second.keypoints[[7, 9, 8, 10]] += np.array(
        [[-80, -40], [-120, -60], [80, 40], [120, 60]], dtype=np.float32
    )
    first_box = BBox(20, 10, 180, 260)
    second_box = BBox(20, 10, 180, 260)
    assert not are_duplicate_skeletons(first, second, first_box, second_box)


def test_cross_slot_limb_is_suspect_but_not_immediately_masked():
    skeleton = _skeleton(cx=100, scale=140)
    skeleton.keypoints[7] = [190, 110]
    skeleton.keypoints[9] = [240, 110]
    owner = BBox(0, 0, 200, 280, "vlm")
    peer = BBox(180, 0, 380, 280, "vlm")
    evidence = analyze_skeleton(
        skeleton, BBox(20, 10, 240, 260), owner_box=owner,
        peer_boxes=[peer],
    )
    assert "left_arm" in evidence.suspect_limbs
    assert "left_arm_cross_slot" in evidence.reasons
    assert evidence.valid_joint_mask[7] and evidence.valid_joint_mask[9]
    assert evidence.effective_scores[7] > 0 and evidence.effective_scores[9] > 0
    assert evidence.refine_scores[7] == 0 and evidence.refine_scores[9] == 0
    assert evidence.state == "partial"


def test_extreme_distal_joint_masks_only_the_bad_endpoint():
    skeleton = _skeleton()
    skeleton.keypoints[9] = skeleton.keypoints[7] + np.array([300, 0], np.float32)
    evidence = analyze_skeleton(skeleton, BBox(0, 0, 400, 300))
    assert "left_arm_length_outlier" in evidence.reasons
    assert evidence.valid_joint_mask[7]
    assert not evidence.valid_joint_mask[9]
    assert "left_arm" not in evidence.refinable_limbs


def test_crop_candidate_does_not_replace_better_original():
    slot = assign_candidates(
        [BBox(20, 10, 180, 260, "vlm")], [_skeleton(cx=100)], 400, 300
    ).slots[0]
    weak_scores = np.full(17, 0.9, dtype=np.float32)
    weak_scores[[7, 8, 9, 10, 13, 14, 15, 16]] = 0.0
    assert select_crop_candidate(slot, [_skeleton(cx=100, scores=weak_scores)]) is None


def test_normal_pipeline_calls_full_pose_once_and_skips_crop():
    class CountingPose(MockPoseModel):
        self_detecting = True

        def __init__(self):
            self.full_calls = 0
            self.crop_calls = 0

        def estimate(self, image, boxes, img_w, img_h):
            self.full_calls += 1
            return [_skeleton(cx=img_w * 0.5, cy=img_h * 0.5, scale=img_h * 0.4)]

        def estimate_crop_candidates(self, image, box, img_w, img_h):
            self.crop_calls += 1
            return []

    pose = CountingPose()
    result = Pipeline(
        build_synthetic_index(), vlm_client=MockVLMClient(), pose_model=pose
    ).process_cut(
        _Img("full_half standing front 1p"), 512, 768
    )
    assert result.route == "core"
    assert pose.full_calls == 1
    assert pose.crop_calls == 0
    assert not any("ab_knn=" in note for note in result.notes)


def test_pipeline_person_index_is_stable_when_rtm_order_is_shuffled():
    class ShuffledPose(MockPoseModel):
        self_detecting = True

        def estimate(self, image, boxes, img_w, img_h):
            return [_skeleton(cx=img_w * 0.75), _skeleton(cx=img_w * 0.25)]

    result = Pipeline(
        build_synthetic_index(), vlm_client=MockVLMClient(),
        pose_model=ShuffledPose(),
    ).process_cut(
        _Img("full_half standing front 2p"), 400, 300
    )
    xs = [descriptor.box.x1 for descriptor in result.descriptors]
    assert xs == sorted(xs)


def test_sparse_query_keeps_top5_but_never_becomes_high_confidence():
    class SparsePose(MockPoseModel):
        self_detecting = True

        def estimate(self, image, boxes, img_w, img_h):
            scores = np.full(17, 0.9, dtype=np.float32)
            scores[[7, 8, 9, 10, 13, 14, 15, 16]] = 0.0
            return [_skeleton(cx=img_w * 0.5, scores=scores)]

    result = Pipeline(
        build_synthetic_index(), vlm_client=MockVLMClient(), pose_model=SparsePose()
    ).process_cut(
        _Img("full_half standing front 1p"), 400, 300
    )
    assert result.person_candidates[0]
    assert result.person_confidence[0] == "low"
    assert not result.descriptors[0].refine_allowed
    assert np.allclose(result.descriptors[0].skeleton.scores, 0.0)
    assert any("coverage=sparse" in note for note in result.notes)


def test_partial_query_runs_family_aware_ab_stability_only_when_mask_changes():
    class PartialPose(MockPoseModel):
        self_detecting = True

        def estimate(self, image, boxes, img_w, img_h):
            scores = np.full(17, 0.9, dtype=np.float32)
            scores[10] = 0.0  # 오른 전완만 불완전 → 나머지 3사지는 완성
            return [_skeleton(cx=img_w * 0.5, scores=scores)]

    result = Pipeline(
        build_synthetic_index(), vlm_client=MockVLMClient(), pose_model=PartialPose()
    ).process_cut(
        _Img("full_half standing front 1p"), 400, 300
    )
    assert result.person_confidence[0] == "high"
    assert result.descriptors[0].refine_allowed
    assert "right_arm" not in result.descriptors[0].refinable_limbs
    assert any("family_overlap=" in note and "ab_knn=" in note
               for note in result.notes)


def test_unstable_partial_retries_crop_once_then_researches():
    partial_scores = np.full(17, 0.9, dtype=np.float32)
    partial_scores[10] = 0.0
    partial = _skeleton(cx=200, cy=150, scale=120, scores=partial_scores)
    full = _skeleton(cx=200, cy=150, scale=120)
    valid_mask = partial_scores >= CFG.skeleton_kpt_threshold
    query = normalize_skeleton(partial.keypoints, partial.scores,
                               valid_mask=valid_mask).reshape(17, 2)
    entries = []
    stable_body = [index for index in range(5, 17) if index not in (8, 10)]
    for index in range(5):
        feature = query.copy()
        feature[stable_body, 0] += 0.04 + index * 0.001
        entries.append(LibraryEntry(
            f"arm_match_{index}", View.FRONT, feature.reshape(-1), {}
        ))
    for index in range(5):
        feature = query.copy()
        feature[8, 0] += 1.0 + index * 0.01
        entries.append(LibraryEntry(
            f"body_match_{index}", View.FRONT, feature.reshape(-1), {}
        ))

    class RetryPose(MockPoseModel):
        self_detecting = True

        def __init__(self):
            self.crop_calls = 0

        def estimate(self, image, boxes, img_w, img_h):
            return [partial]

        def estimate_crop_candidates(self, image, box, img_w, img_h):
            self.crop_calls += 1
            return [full]

    pose = RetryPose()
    result = Pipeline(
        entries, vlm_client=MockVLMClient(), pose_model=pose
    ).process_cut(_Img("full_half standing front 1p"), 400, 300)
    assert pose.crop_calls == 1
    assert result.descriptors[0].skeleton_source == "crop_retry"
    assert result.descriptors[0].search_stability == "not_required"
    assert result.person_candidates[0]
    assert result.person_confidence[0] == "low"
    assert any("crop_retry:unstable_search" in note for note in result.notes)


def test_unstable_partial_above_distance_threshold_becomes_hard_fallback():
    scores = np.full(17, 0.9, dtype=np.float32)
    scores[10] = 0.0
    partial = _skeleton(cx=200, cy=150, scale=120, scores=scores)
    query = normalize_skeleton(
        partial.keypoints, partial.scores,
        valid_mask=scores >= CFG.skeleton_kpt_threshold,
    ).reshape(17, 2)
    entries = []
    stable_body = [index for index in range(5, 17) if index not in (8, 10)]
    for index in range(5):
        feature = query.copy()
        feature[stable_body, 0] += 0.60 + index * 0.001
        entries.append(LibraryEntry(
            f"far_arm_{index}", View.FRONT, feature.reshape(-1), {}
        ))
    for index in range(5):
        feature = query.copy()
        feature[8, 0] += 7.0 + index * 0.01
        entries.append(LibraryEntry(
            f"far_body_{index}", View.FRONT, feature.reshape(-1), {}
        ))

    class FailedRetryPose(MockPoseModel):
        self_detecting = True

        def __init__(self):
            self.crop_calls = 0

        def estimate(self, image, boxes, img_w, img_h):
            return [partial]

        def estimate_crop_candidates(self, image, box, img_w, img_h):
            self.crop_calls += 1
            return []

    pose = FailedRetryPose()
    result = Pipeline(
        entries, vlm_client=MockVLMClient(), pose_model=pose
    ).process_cut(_Img("full_half standing front 1p"), 400, 300)
    assert pose.crop_calls == 1
    assert result.person_candidates[0] == []
    assert result.descriptors[0].skeleton_state == "invalid"
    assert not result.descriptors[0].refine_allowed
    assert any("hard fallback" in note for note in result.notes)


def test_crop_recovered_slot_remains_low_confidence():
    class RecoveringPose(MockPoseModel):
        self_detecting = True

        def estimate(self, image, boxes, img_w, img_h):
            return []

        def estimate_crop_candidates(self, image, box, img_w, img_h):
            return [_skeleton(cx=(box.x1 + box.x2) * 0.5)]

    result = Pipeline(
        build_synthetic_index(), vlm_client=MockVLMClient(),
        pose_model=RecoveringPose(),
    ).process_cut(
        _Img("full_half standing front 1p"), 400, 300
    )
    assert result.person_candidates[0]
    assert result.person_confidence[0] == "low"
    assert not result.descriptors[0].refine_allowed
    assert np.allclose(result.descriptors[0].skeleton.scores, 0.0)
    assert any("복원" in note for note in result.notes)


def test_missing_middle_slot_keeps_person_index_alignment():
    class ThreePeopleVLM(BaseVLMClient):
        def analyze(self, image, img_w, img_h):
            boxes = [
                BBox(0, 0, 120, img_h, "vlm"),
                BBox(140, 0, 260, img_h, "vlm"),
                BBox(280, 0, 400, img_h, "vlm"),
            ]
            return VLMAnalysis(
                3, Shot.FULL_HALF, Action.STANDING, View.FRONT,
                Relationship.TALKING, boxes,
            )

    class MissingMiddlePose(MockPoseModel):
        self_detecting = True

        def estimate(self, image, boxes, img_w, img_h):
            return [_skeleton(cx=60), _skeleton(cx=340)]

        def estimate_crop_candidates(self, image, box, img_w, img_h):
            return []

    result = Pipeline(
        build_synthetic_index(), vlm_client=ThreePeopleVLM(),
        pose_model=MissingMiddlePose(),
    ).process_cut(_Img("three"), 400, 300)
    assert len(result.descriptors) == 3
    assert result.person_candidates[0]
    assert result.person_candidates[1] == []
    assert result.person_candidates[2]
    xs = [descriptor.box.x1 for descriptor in result.descriptors]
    assert xs == sorted(xs)


if __name__ == "__main__":
    import traceback

    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    failed = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {function.__name__}")
            traceback.print_exc()
    print(f"\n{len(functions) - failed}/{len(functions)} passed")
    raise SystemExit(1 if failed else 0)
