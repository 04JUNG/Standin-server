"""스모크 테스트: shot/사람수 분기 + 기하 매칭 + 신뢰도 폴백 계약 검증."""
import os, sys
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockSelfDetectingPoseModel
from src.config import CFG
from src.runtime_guard import (
    MockBackendError,
    actual_backend_names,
    ensure_production_backends,
)
from src.thumbnails import find_thumbnail, thumbnail_url


class _Img(str):
    @property
    def hint(self): return str(self)


def _pipe():
    return Pipeline(build_synthetic_index())


def test_core_route_and_geometric_topk():
    res = _pipe().process_cut(_Img("full_half standing front 1p"))
    assert res.route == "core"
    assert 1 <= len(res.candidates) <= CFG.top_k_final
    # mock 스켈레톤은 서기 → 기하 매칭 1위는 서기 포즈
    assert res.candidates[0].tags["action"] == "standing"


def test_face_skips():
    res = _pipe().process_cut(_Img("face front 1p"))
    assert res.route == "skip"
    assert res.candidates == []


def test_bust_route_skips_search():
    res = _pipe().process_cut(_Img("bust front 1p"))
    assert res.route == "bust"
    assert res.candidates == []


def test_count_mismatch_is_low_conf():
    res = _pipe().process_cut(_Img("full_half standing front 2p miss"))
    assert res.count_confidence == "low"     # 검출 1 vs VLM 2
    assert res.vlm_count == 2


def test_count_match_is_high_conf():
    res = _pipe().process_cut(_Img("full_half standing front 2p"))
    assert res.count_confidence == "high"


def test_person_confidence_high_on_good_match():
    res = _pipe().process_cut(_Img("full_half standing front 1p"))
    # 서기 쿼리 ≈ 서기 라이브러리 → 거리 작음 → high
    assert res.person_confidence and res.person_confidence[0] == "high"


def test_two_person_two_results():
    res = _pipe().process_cut(_Img("full_half standing front 2p"))
    assert len(res.person_candidates) == 2


def test_selfdetect_recovers_missing_person():
    p = Pipeline(build_synthetic_index(), pose_model=MockSelfDetectingPoseModel())
    res = p.process_cut(_Img("full_half standing front 2p miss"))
    assert res.detector_count == 1
    assert res.count_confidence == "low"
    assert len(res.person_candidates) == 2
    assert any("복원" in note for note in res.notes)


def test_load_bgr_ndarray_channels():
    import numpy as np
    from src.pose import _load_bgr
    assert _load_bgr(np.zeros((5, 4), np.uint8)).shape == (5, 4, 3)
    assert _load_bgr(np.zeros((5, 4, 4), np.uint8)).shape == (5, 4, 3)
    assert _load_bgr(np.zeros((5, 4, 1), np.uint8)).shape == (5, 4, 3)
    out = _load_bgr(np.zeros((5, 4, 3), np.uint8))
    assert out.shape == (5, 4, 3) and out.flags["C_CONTIGUOUS"]


def test_load_bgr_pil_modes():
    try:
        from PIL import Image
    except ImportError:
        return
    import numpy as np
    from src.pose import _load_bgr
    for mode in ("RGBA", "L", "P", "RGB"):
        out = _load_bgr(Image.new(mode, (6, 4)))
        assert out.shape == (4, 6, 3) and out.dtype == np.uint8


def test_selfdetect_count_match_high():
    p = Pipeline(build_synthetic_index(), pose_model=MockSelfDetectingPoseModel())
    res = p.process_cut(_Img("full_half standing front 2p"))
    assert res.detector_count == 2
    assert res.count_confidence == "high"
    assert not any("복원" in note for note in res.notes)


def test_production_rejects_silent_mock_fallback():
    """설정이 실백엔드여도 실제 객체가 mock이면 프로덕션 기동을 막는다."""
    pipeline = Pipeline(build_synthetic_index())
    try:
        ensure_production_backends(
            pipeline,
            is_production=True,
            requested_vlm="gemini",
            requested_pose="rtmlib",
        )
    except MockBackendError as exc:
        message = str(exc)
        assert "VLM_PROVIDER=gemini" in message
        assert "POSE_BACKEND=rtmlib" in message
        assert "MockVLMClient" in message
        assert "MockPoseModel" in message
    else:
        raise AssertionError("프로덕션 mock 백엔드를 허용했습니다.")


def test_development_allows_mock_backends_and_reports_actual_names():
    pipeline = Pipeline(build_synthetic_index())
    ensure_production_backends(
        pipeline,
        is_production=False,
        requested_vlm="gemini",
        requested_pose="rtmlib",
    )
    assert actual_backend_names(pipeline, "gemini", "rtmlib") == ("mock", "mock")


def test_order_left_to_right_sorts_by_box_x1():
    """좌→우 정렬 헬퍼: 뒤섞인 박스를 왼쪽부터 정렬한다."""
    from src.descriptor import order_left_to_right
    from src.schema import BBox
    boxes = [BBox(300, 0, 400, 10), BBox(0, 0, 50, 10), BBox(120, 0, 220, 10)]
    _, ordered = order_left_to_right([None, None, None], boxes)
    xs = [b.x1 for b in ordered]
    assert xs == [0.0, 120.0, 300.0]


def test_order_left_to_right_falls_back_to_skeleton_x():
    """박스가 없으면 스켈레톤 x중앙값으로 좌→우 정렬한다."""
    import numpy as np
    from src.descriptor import order_left_to_right
    from src.schema import Skeleton
    right = Skeleton(np.array([[0.9, 0.5]] * 17, np.float32), np.ones(17, np.float32))
    left = Skeleton(np.array([[0.1, 0.5]] * 17, np.float32), np.ones(17, np.float32))
    ordered, _ = order_left_to_right([right, left], [None, None])
    assert float(ordered[0].keypoints[0][0]) < float(ordered[1].keypoints[0][0])


def test_person_index_is_left_to_right():
    """파이프라인 결과의 인물 순서(person_index)가 좌→우여야 한다."""
    res = _pipe().process_cut(_Img("full_half standing front 2p"))
    xs = [d.box.x1 for d in res.descriptors if d.box is not None]
    assert xs == sorted(xs)


def test_thumbnail_lookup_uses_pose_and_view_without_path_escape():
    with tempfile.TemporaryDirectory() as tmp:
        thumbs = Path(tmp) / "thumbs"
        thumbs.mkdir()
        expected = thumbs / "Wave Pose__front.png"
        expected.write_bytes(b"png")

        assert find_thumbnail(tmp, "Wave Pose", "front") == expected.resolve()
        assert thumbnail_url(tmp, "Wave Pose", "front") == (
            "/pose/Wave%20Pose/thumbnail?view=front"
        )
        assert find_thumbnail(tmp, "../escape", "front") is None
        assert find_thumbnail(tmp, "Wave Pose", "diagonal") is None


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
