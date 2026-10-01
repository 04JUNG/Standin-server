"""Observed-region retrieval: geometry invariance, ownership and refine isolation."""
import numpy as np
import pytest

from src.body_scope import BodyScope
from src.config import CFG
from src.features import normalize_skeleton
from src.library import build_synthetic_index
from src.partial_pose import shoulder_frame
from src.pipeline import Pipeline
from src.schema import BBox, LibraryEntry, Skeleton, View
from src.search import PositionSearchIndex, knn_upper_body
from src.skeleton_extraction import analyze_skeleton
from src.vlm.client import _coerce


def upper_skeleton():
    points = np.zeros((17, 2), dtype=np.float32)
    points[:5] = [150, 40]
    points[5:11] = [[190, 100], [110, 100], [220, 150], [80, 150], [230, 205], [70, 205]]
    points[11:13] = [[175, 250], [125, 250]]
    points[13:] = [[175, 350], [125, 350], [175, 450], [125, 450]]
    scores = np.full(17, 0.9, dtype=np.float32)
    scores[11:] = 0
    return Skeleton(points, scores)


def entry(name, points, **kwargs):
    return LibraryEntry(name, View.FRONT, normalize_skeleton(points), {}, **kwargs)


def test_affine_symmetry_and_hidden_hips_do_not_change_ranking():
    skeleton = upper_skeleton()
    target = entry("matching", skeleton.keypoints)
    wrong = skeleton.keypoints.copy()
    wrong[7:11, 1] = 50
    library = [entry("arms_up", wrong), target]
    mask = analyze_skeleton(skeleton).valid_joint_mask
    expected = None
    for vectorized in (False, True):
        index = PositionSearchIndex.build(library) if vectorized else None
        for scale, offset in [(1, [0, 0]), (3.7, [-930, 472])]:
            query = skeleton.keypoints * scale + offset
            query[11:] = 100000  # hidden coordinates cannot change center or scale
            candidates = knn_upper_body(library, query, mask, search_index=index)
            assert candidates[0].pose_id == "matching"
            assert candidates[0].distance < 1e-5
            current = [(c.pose_id, round(c.distance, 5)) for c in candidates]
            if expected is None:
                expected = current
            assert current == expected


def test_library_lower_body_and_torso_scale_do_not_change_upper_distance():
    sk = upper_skeleton()
    second = sk.keypoints.copy()
    second[11:] = second[11:] * [2, 4] + [10, 90]
    candidates = knn_upper_body(
        [entry("a", sk.keypoints), entry("b", second)], sk.keypoints,
        analyze_skeleton(sk).valid_joint_mask,
    )
    assert all(c.distance < 1e-5 for c in candidates)


def test_missing_wrist_is_ignored_even_if_its_coordinate_is_extreme():
    sk = upper_skeleton()
    sk.scores[10] = 0
    mask = analyze_skeleton(sk).valid_joint_mask
    library = [entry("a", sk.keypoints)]
    sk.keypoints[10] = [1e6, 1e6]
    assert knn_upper_body(library, sk.keypoints, mask)[0].distance < 1e-5


def test_upper_search_preserves_quarantine_family_dedup_and_stable_ties(monkeypatch):
    import src.search as search
    sk = upper_skeleton()
    entries = [entry(name, sk.keypoints) for name in ["bad", "a", "a_mirror", "b"]]
    monkeypatch.setattr(search, "load_pose_quarantine", lambda cfg: {"bad"})
    candidates = knn_upper_body(entries, sk.keypoints, analyze_skeleton(sk).valid_joint_mask)
    assert [c.pose_id for c in candidates] == ["a", "b"]


@pytest.mark.parametrize("case", ["shoulder_missing", "shoulder_collapsed", "no_arms", "nonfinite", "leg_conflict"])
def test_insufficient_or_conflicting_geometry_is_not_upper_coverage(case):
    sk = upper_skeleton()
    if case == "shoulder_missing":
        sk.scores[5] = 0
    elif case == "shoulder_collapsed":
        sk.keypoints[6] = sk.keypoints[5]
    elif case == "no_arms":
        sk.scores[7:11] = 0
    elif case == "nonfinite":
        sk.keypoints[11] = np.nan
    else:
        sk.scores[15] = .9
    assert analyze_skeleton(sk).coverage_class == "insufficient"


def test_upper_coverage_never_uses_face_or_lower_body_and_cannot_refine():
    sk = upper_skeleton()
    evidence = analyze_skeleton(sk)
    assert evidence.coverage_class == "upper_only"
    assert evidence.searchable and not evidence.high_confidence_eligible
    assert evidence.valid_joint_mask[5:11].all()
    assert not evidence.valid_joint_mask[:5].any()
    assert not evidence.valid_joint_mask[11:].any()
    assert not evidence.refine_scores.any() and not evidence.refinable_limbs


def test_observed_elbows_are_enough_but_single_arm_segment_is_not():
    sk = upper_skeleton()
    sk.scores[9:11] = 0
    assert analyze_skeleton(sk).coverage_class == "upper_only"
    sk.scores[8] = 0
    assert analyze_skeleton(sk).coverage_class == "insufficient"


def test_owner_overlap_and_exploded_limbs_are_rejected():
    sk = upper_skeleton()
    owner = BBox(30, 10, 270, 220)
    peer = BBox(90, 60, 200, 220)
    assert analyze_skeleton(sk, owner_box=owner, peer_boxes=[peer]).coverage_class == "insufficient"
    sk.keypoints[7:11] *= 10
    assert analyze_skeleton(sk, owner_box=owner).coverage_class == "insufficient"


def test_degenerate_library_projection_is_skipped_and_input_is_unchanged():
    sk = upper_skeleton()
    degenerate = sk.keypoints.copy()
    degenerate[6] = degenerate[5]
    entries = [entry("side", degenerate), entry("good", sk.keypoints)]
    before = np.stack([e.feature.copy() for e in entries])
    result = knn_upper_body(entries, sk.keypoints, analyze_skeleton(sk).valid_joint_mask)
    assert [c.pose_id for c in result] == ["good"]
    np.testing.assert_array_equal(before, np.stack([e.feature for e in entries]))
    frame, valid = shoulder_frame(degenerate)
    assert not valid and np.isfinite(frame).all()


class FixedPose:
    self_detecting = True

    def __init__(self, skeletons):
        self.skeletons = skeletons
        self.calls = 0

    def estimate(self, *args):
        self.calls += 1
        return self.skeletons

    def estimate_crop_candidates(self, *args):
        return []


def pipeline(scopes=("bust",), shot="bust", skeletons=None):
    class VLM:
        def analyze(self, *args):
            return _coerce({
                "num_people": len(scopes), "shot": shot, "body_scopes": list(scopes),
                "relationship": "solo" if len(scopes) == 1 else "side_by_side",
                "approx_boxes": [{"x1": (30 + 300 * i) / 600, "y1": .02,
                                  "x2": (270 + 300 * i) / 600, "y2": .46}
                                 for i in range(len(scopes))],
            }, 600, 500)
    model = FixedPose(skeletons if skeletons is not None else [upper_skeleton()])
    return Pipeline(build_synthetic_index(), vlm_client=VLM(), pose_model=model), model


@pytest.mark.parametrize("scope", ["half", "bust", None])
@pytest.mark.parametrize("v2", [False, True])
def test_pipeline_searches_observed_upper_body_without_relaxing_refine(monkeypatch, scope, v2):
    monkeypatch.setattr(CFG, "refine_v2_enabled", v2)
    pipe, model = pipeline(scopes=(scope,))
    result = pipe.process_cut(None, 600, 500)
    assert result.route == "core" and model.calls == 1
    assert result.count_confidence == "high"
    desc = result.descriptors[0]
    assert desc.coverage_class == "upper_only"
    assert result.person_candidates[0] and result.person_confidence == ["low"]
    assert desc.confidence_threshold is None and desc.distance_metric == "upper_pos"
    assert not desc.refine_allowed and not desc.refinable_limbs
    assert not desc.skeleton.scores.any()


def test_mixed_face_cut_preserves_upper_person_and_blocks_head_person():
    second = upper_skeleton()
    second.keypoints[:, 0] += 300
    pipe, _ = pipeline(scopes=("bust", "head"), shot="face", skeletons=[second, upper_skeleton()])
    result = pipe.process_cut(None, 600, 500)
    assert len(result.descriptors) == 2
    assert result.person_candidates[0] and result.person_candidates[1] == []
    assert result.descriptors[0].output_scope.detected == BodyScope.BUST
    assert result.descriptors[1].skeleton is None
    assert "head_search_unsupported" in result.descriptors[1].quality_reasons


def test_head_only_does_not_call_body_pose_model():
    pipe, model = pipeline(scopes=("head",), shot="face")
    result = pipe.process_cut(None, 600, 500)
    assert result.route == "skip" and model.calls == 0
    assert result.person_candidates == [[]]


def test_http_returns_upper_candidates_with_frozen_refine_lineage(monkeypatch):
    import io
    from fastapi import UploadFile
    from PIL import Image
    import api.app as api_app
    from src.refine_policy import structural_refine_allowed

    pipe, _ = pipeline()
    monkeypatch.setitem(api_app.STATE, "pipeline", pipe)
    data = io.BytesIO()
    Image.new("RGB", (600, 500)).save(data, format="PNG")
    data.seek(0)
    result = api_app.analyze(UploadFile(file=data, filename="fixture.png"), hint="", rescue="")
    person = result.model_dump(mode="json")["people"][0]
    assert person["coverage_class"] == "upper_only" and person["candidates"]
    assert person["output_scope"]["detected"] == "bust"
    assert person["confidence"] == "low" and person["confidence_threshold"] is None
    assert person["scores"] == [0.0] * 17 and not person["refine_allowed"]
    # Even a forged allowed-limbs hint cannot satisfy the endpoint's shared gate.
    assert not structural_refine_allowed(
        skeleton_state="partial", coverage_class="upper_only", refinable_limbs=["left_arm"],
        slot_origin="vlm", skeleton_source="full_image",
    )
