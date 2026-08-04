"""스모크 테스트: shot/사람수 분기 + 기하 매칭 + 신뢰도 폴백 계약 검증."""
import io
import os, sys
import tempfile
from pathlib import Path
from pydantic import ValidationError
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.pose import MockPoseModel, MockSelfDetectingPoseModel
from src.config import CFG
from src.runtime_guard import (
    MockBackendError,
    actual_backend_names,
    ensure_production_backends,
)
from src.thumbnails import find_thumbnail, thumbnail_url
from src.vlm.client import MockVLMClient
from api.models import SkeletonOut, ImageInfoOut, InferenceMetadataOut


def _coco17_keypoints():
    return [[float(i), float(i + 1)] for i in range(17)]


def test_api_models_include_skeleton_and_version_lineage():
    skeleton = SkeletonOut(keypoints=_coco17_keypoints(), scores=[0.9] * 17)
    image = ImageInfoOut(width=100, height=200)
    metadata = InferenceMetadataOut(
        deployment_version="sha",
        vlm_provider="mock",
        vlm_model="mock",
        pose_backend="mock",
        pose_model_version="1",
        pose_library_version="v1",
        feature_version=1,
    )
    assert skeleton.schema_version == "coco17-v1"
    assert image.height == 200
    assert metadata.pose_library_version == "v1"


def _assert_invalid_skeleton(**overrides):
    payload = {"keypoints": _coco17_keypoints(), "scores": [0.9] * 17}
    payload.update(overrides)
    try:
        SkeletonOut(**payload)
    except ValidationError:
        return
    raise AssertionError(f"invalid COCO-17 skeleton was accepted: {overrides}")


def test_api_model_rejects_non_coco17_shapes():
    for count in (16, 18):
        _assert_invalid_skeleton(keypoints=_coco17_keypoints()[:count]
                                 if count < 17 else _coco17_keypoints() + [[17.0, 18.0]])
        _assert_invalid_skeleton(scores=[0.9] * count)
    _assert_invalid_skeleton(keypoints=[[0.0]] + _coco17_keypoints()[1:])
    _assert_invalid_skeleton(keypoints=[[0.0, 1.0, 2.0]] + _coco17_keypoints()[1:])
    _assert_invalid_skeleton(schema_version="other")


class _Img(str):
    @property
    def hint(self): return str(self)


def _pipe():
    return Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                    pose_model=MockPoseModel())


def test_analyze_route_maps_full_response_contract():
    from PIL import Image
    from fastapi import UploadFile
    import api.app as api_app

    image_bytes = io.BytesIO()
    Image.new("RGB", (16, 12), color=(255, 255, 255)).save(image_bytes, format="PNG")
    image_bytes.seek(0)

    previous_state = dict(api_app.STATE)
    api_app.STATE.clear()
    api_app.STATE.update({
        "pipeline": _pipe(),
        "provider": "mock",
        "pose_backend": "mock",
    })
    try:
        result = api_app.analyze(
            UploadFile(filename="cut.png", file=image_bytes),
            hint="full_half standing front 1p",
        ).model_dump()
    finally:
        api_app.STATE.clear()
        api_app.STATE.update(previous_state)

    assert result["image"] == {"width": 16, "height": 12}
    assert len(result["people"]) == 1
    skeleton = result["people"][0]["skeleton"]
    assert skeleton["schema_version"] == "coco17-v1"
    assert len(skeleton["keypoints"]) == 17
    assert all(len(point) == 2 for point in skeleton["keypoints"])
    assert len(skeleton["scores"]) == 17
    assert result["people"][0]["candidates"]
    metadata = result["inference_metadata"]
    assert metadata["deployment_version"] == CFG.deployment_version
    assert metadata["vlm_provider"] == "mock"
    assert metadata["pose_backend"] == "mock"
    assert metadata["pose_library_version"] == CFG.pose_library_version
    assert isinstance(metadata["feature_version"], int)


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
    p = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                 pose_model=MockSelfDetectingPoseModel())
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
    p = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                 pose_model=MockSelfDetectingPoseModel())
    res = p.process_cut(_Img("full_half standing front 2p"))
    assert res.detector_count == 2
    assert res.count_confidence == "high"
    assert not any("복원" in note for note in res.notes)


def test_production_rejects_silent_mock_fallback():
    """설정이 실백엔드여도 실제 객체가 mock이면 프로덕션 기동을 막는다."""
    pipeline = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                        pose_model=MockPoseModel())
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
    pipeline = Pipeline(build_synthetic_index(), vlm_client=MockVLMClient(),
                        pose_model=MockPoseModel())
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


# ---- refine (docs/REFINE_DESIGN.md) ---------------------------------------
# 핵심 계약은 하나: **좋아지거나, 그대로.** 아래 테스트는 그 보장을 검증한다.

def _synthetic_bvh(tmpdir, name="t.bvh"):
    """의존성 없는 최소 BVH(어깨·팔꿈치·고관절·무릎 = refine 파라미터 전부 포함)."""
    import os
    text = """HIERARCHY
ROOT Hips
{ OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  { OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftArm
    { OFFSET 5 5 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftForeArm
      { OFFSET 0 -8 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftHand
        { OFFSET 0 -8 0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site { OFFSET 0 -2 0 }
        }
      }
    }
    JOINT RightArm
    { OFFSET -5 5 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightForeArm
      { OFFSET 0 -8 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightHand
        { OFFSET 0 -8 0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site { OFFSET 0 -2 0 }
        }
      }
    }
    JOINT Head
    { OFFSET 0 8 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site { OFFSET 0 3 0 }
    }
  }
  JOINT LeftUpLeg
  { OFFSET 3 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLeg
    { OFFSET 0 -10 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      { OFFSET 0 -10 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site { OFFSET 0 -2 0 }
      }
    }
  }
  JOINT RightUpLeg
  { OFFSET -3 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLeg
    { OFFSET 0 -10 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      { OFFSET 0 -10 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site { OFFSET 0 -2 0 }
      }
    }
  }
}
MOTION
Frames: 1
Frame Time: 0.033333
"""
    n_ch = 6 + 3 * 15
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        f.write(text + " ".join(["0"] * n_ch) + "\n")
    return path


def _bvh_with_rotation(tmpdir, name, joint_suffix, deg):
    """같은 스켈레톤에서 한 관절만 돌린 변형 → refine의 타깃으로 쓴다."""
    import numpy as np
    from src.bvh import parse_bvh, find_joint, rotation_channel_indices, \
        write_single_frame_bvh
    src = _synthetic_bvh(tmpdir, "src.bvh")
    joints, data = parse_bvh(src)
    fr = data[0].copy()
    ch = rotation_channel_indices(joints, find_joint(joints, joint_suffix))
    fr[ch[0]] = deg
    import os
    return write_single_frame_bvh(src, fr, os.path.join(tmpdir, name))


def _bvh_with_rotations(tmpdir, name, rotations):
    """같은 스켈레톤에서 여러 관절을 함께 돌린 refine 타깃."""
    from src.bvh import (parse_bvh, find_joint, rotation_channel_indices,
                         write_single_frame_bvh)
    src = _synthetic_bvh(tmpdir, "src_multi.bvh")
    joints, data = parse_bvh(src)
    fr = data[0].copy()
    for joint_suffix, deg in rotations.items():
        channels = rotation_channel_indices(joints, find_joint(joints, joint_suffix))
        fr[channels[0]] = deg
    return write_single_frame_bvh(src, fr, os.path.join(tmpdir, name))


def _target_kp(bvh, view="front"):
    """BVH → 이미지 좌표(y-down) 2D 스켈레톤. 러프 대용."""
    from src.bvh import load_coco17
    from src.library import project_3d_to_2d, view_angle
    kp3, sc = load_coco17(bvh)
    kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
    kp2[:, 1] *= -1
    return kp2, sc


def _collision_pose(left_inside=False, scale=1.0):
    """P3 순수 기하 테스트용 COCO17 3D 자세."""
    import numpy as np
    kp = np.zeros((17, 3), dtype=np.float64)
    kp[5], kp[6] = (-0.30, 1.0, 0.0), (0.30, 1.0, 0.0)
    kp[11], kp[12] = (-0.20, 0.0, 0.0), (0.20, 0.0, 0.0)
    if left_inside:
        kp[7], kp[9] = (-0.45, 0.75, 0.0), (0.0, 0.45, 0.0)
    else:
        kp[7], kp[9] = (-0.55, 0.75, 0.0), (-0.65, 0.45, 0.0)
    kp[8], kp[10] = (0.55, 0.75, 0.0), (0.65, 0.45, 0.0)
    return kp * float(scale), np.ones(17, dtype=np.float32)


def test_refine_matches_rotated_target():
    """베이스와 타깃이 한 관절만 다르면 refine이 그 차이를 줄여야 한다."""
    import tempfile
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 15.0)
        kp, sc = _target_kp(tgt)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh")
        assert res.refined, f"조정이 폐기됨: {res.reason}"
        assert res.loss_final < res.loss_base * 0.5, (res.loss_base, res.loss_final)
        assert os.path.exists(res.bvh_path)


def test_refine_output_bvh_is_reloadable():
    """조정본이 유효한 BVH여야 한다(동원이 그대로 소비하므로 계약)."""
    import tempfile
    from src.bvh import load_coco17, parse_bvh
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftForeArm", 30.0)
        kp, sc = _target_kp(tgt)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh")
        assert res.refined
        j_base, _ = parse_bvh(base)
        j_new, data = parse_bvh(res.bvh_path)
        # 계층·뼈 길이는 그대로, 회전각만 달라져야 한다
        assert [x[0] for x in j_base] == [x[0] for x in j_new]
        assert len(data) == 1
        kp3, _ = load_coco17(res.bvh_path)
        assert kp3.shape == (17, 3)


def test_refine_gate_already_matched_returns_base():
    """이미 맞으면 개선 여지가 없다 → 손대지 않고 베이스 그대로."""
    import tempfile
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        kp, sc = _target_kp(base)                      # 자기 자신이 타깃
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh")
        assert not res.refined and res.reason in (
            "already_matched", "no_gain", "low_observability"
        )
        assert res.bvh_path == base
        assert not os.path.exists(f"{d}/out.bvh"), "폐기했는데 파일을 썼다"


def test_refine_gate_base_mismatch():
    """검색이 실패한 컷(Top-1 거리 초과)에는 refine을 돌리지 않는다."""
    import tempfile
    from src.config import CFG
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 35.0)
        kp, sc = _target_kp(tgt)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh",
                         search_distance=CFG.fallback_distance + 0.5)
        assert not res.refined and res.reason == "base_mismatch"
        assert res.bvh_path == base


def test_refine_gate_low_skeleton_score():
    """추출 실패 스켈레톤을 타깃으로 삼지 않는다."""
    import tempfile
    import numpy as np
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 35.0)
        kp, _ = _target_kp(tgt)
        res = refine_bvh(base, kp, np.zeros(17, np.float32), "front",
                         out_path=f"{d}/out.bvh")
        assert not res.refined and res.reason == "low_skeleton_score"


def test_refine_respects_skeleton_policy_allowed_limbs():
    """추출 품질 단계가 허용하지 않은 사지는 score가 높아도 풀지 않는다."""
    import tempfile
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 25.0)
        kp, sc = _target_kp(tgt)
        res = refine_bvh(
            base, kp, sc, "front", out_path=f"{d}/out.bvh",
            allowed_limbs=[],
        )
        assert not res.refined and res.reason == "insufficient_target_bones"


def test_refine_disabled_switch():
    """REFINE_ENABLED=0은 시연 중 비상 스위치 — 항상 베이스."""
    import copy, tempfile
    from src.config import CFG
    from src.refine import refine_bvh
    off = copy.copy(CFG); off.refine_enabled = False
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        kp, sc = _target_kp(base)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/o.bvh", cfg=off)
        assert not res.refined and res.reason == "disabled"


def test_refine_respects_max_delta_bound():
    """조정은 '미세'조정이다 — 채널당 베이스에서 max_delta_deg를 넘지 않는다."""
    import tempfile
    import numpy as np
    from src.bvh import parse_bvh
    from src.config import CFG
    from src.refine import refine_bvh, _Forward
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 120.0)   # 바운드 밖 목표
        kp, sc = _target_kp(tgt)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh")
        if not res.refined:
            return
        j, data = parse_bvh(base)
        fwd = _Forward(j, data[0], "front")
        _, new = parse_bvh(res.bvh_path)
        delta = np.abs(new[0][fwd.param_idx] - data[0][fwd.param_idx])
        assert delta.max() <= CFG.refine_max_delta_deg + 1e-6, delta.max()


def test_refine_observability_gate_drops_weak_limbs():
    """
    관측 감도 게이트: 이 컷에서 '잘 안 보이는' 사지는 조정 대상에서 빠지고,
    그 사지의 회전 채널은 한 값도 바뀌지 않아야 한다.

    '다리를 켜냐 끄냐'는 잘못된 질문이라 전역 스위치 대신 컷마다 측정해 고른다
    (다리 관측 감도는 컷에 따라 8배 범위로 흩어진다 — REFINE_DESIGN.md §6-5).
    임계값을 0.99로 올려 '가장 잘 보이는 사지 하나만' 남게 만들어 배선을 검증한다.
    """
    import copy, tempfile
    import numpy as np
    from src.bvh import parse_bvh, find_joint, rotation_channel_indices
    from src.config import CFG
    from src.refine import refine_bvh, LIMBS
    strict = copy.copy(CFG)
    strict.refine_observability_gate = True
    strict.refine_min_observability = 0.99      # 최고 사지 외 전부 탈락
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 35.0)
        kp, sc = _target_kp(tgt)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh", cfg=strict)
        if not res.refined:
            return
        assert len(res.limbs) < len(LIMBS), res.limbs        # 일부는 탈락해야 함
        assert res.observability, "감도가 기록되지 않음"
        j, before = parse_bvh(base)
        _, after = parse_bvh(res.bvh_path)
        for limb in LIMBS:
            if limb in res.limbs:
                continue
            for suf in LIMBS[limb][0]:
                for gi in rotation_channel_indices(j, find_joint(j, suf)):
                    assert np.isclose(before[0][gi], after[0][gi]), \
                        f"동결했어야 할 {suf} 채널 {gi}이 바뀜"


def test_refine_observability_gate_rejects_when_all_limbs_are_weak():
    """모든 사지가 절대 하한 미만이면 전부 다시 켜지 말고 베이스를 반환한다."""
    import copy, tempfile
    import src.refine as R
    from src.config import CFG
    strict = copy.copy(CFG)
    strict.refine_observability_gate = True
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 35.0)
        kp, sc = _target_kp(tgt)
        original = R.limb_observability
        try:
            R.limb_observability = lambda *a, **k: 0.0
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=strict)
        finally:
            R.limb_observability = original
        assert not res.refined and res.reason == "low_observability"


def test_refine_p1a_axis_lambda_only_strengthens_weak_axes():
    """P1a는 기존 lambda를 약화하지 않고 같은 관절의 저감도 축만 강화한다."""
    import numpy as np
    from src.refine import axis_lambda_multipliers
    obs = np.array([1.0, 0.1, 0.0, 0.5, 0.5, 0.5])
    groups = ["LeftArm"] * 3 + ["LeftForeArm"] * 3
    mult = axis_lambda_multipliers(obs, groups, max_mult=100.0)
    assert np.all(mult >= 1.0)
    assert np.all(mult <= 100.0)
    assert np.allclose(mult[:3], [1.0, 10.0, 100.0])
    assert np.allclose(mult[3:], [1.0, 1.0, 1.0])


def test_refine_p1b_finds_combination_null_space_missed_by_p1a():
    """각 축 노름은 같아도 X-Y 조합이 null이면 P1b가 그 방향만 강하게 묶는다."""
    import numpy as np
    from src.refine import axis_lambda_multipliers, svd_lambda_basis
    jac = np.array([[1.0, 1.0]])
    # P1a가 보는 축별 감도는 같아서 둘 다 1x다.
    p1a = axis_lambda_multipliers(np.linalg.norm(jac, axis=0), ["J", "J"])
    assert np.allclose(p1a, [1.0, 1.0])

    vt, singular, mult = svd_lambda_basis(jac, max_mult=100.0)
    assert singular[0] > 0.0 and np.isclose(singular[1], 0.0)
    assert np.isclose(mult[0], 1.0) and np.isclose(mult[1], 100.0)
    null = vt[1]
    assert np.linalg.norm(jac @ null) < 1e-10
    assert abs(abs(float(np.dot(null, np.array([1.0, -1.0]) / np.sqrt(2)))) - 1.0) < 1e-10


def test_refine_p1b_block_basis_does_not_mix_limbs():
    """P1b 기저는 사지별 블록 대각이어야 P2 부분 롤백과 결합되지 않는다."""
    import tempfile
    import numpy as np
    from src.bvh import parse_bvh
    from src.features import _BONES
    from src.refine import (_Forward, ARM_LIMBS, ARM_MASK, LIMBS,
                            _limb_param_columns, block_svd_lambda_basis)
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        joints, frames = parse_bvh(base)
        suffixes = tuple(s for limb in ARM_LIMBS for s in LIMBS[limb][0])
        fwd = _Forward(joints, frames[0], "front", suffixes)
        p0 = fwd.base_frame[fwd.param_idx].copy()
        vt, _, _ = block_svd_lambda_basis(
            fwd, p0, ARM_MASK, np.ones(len(_BONES)), ARM_LIMBS, max_mult=2.0
        )
        left = _limb_param_columns(fwd, "left_arm")
        right = _limb_param_columns(fwd, "right_arm")
        assert np.allclose(vt[np.ix_(left, right)], 0.0, atol=1e-12)
        assert np.allclose(vt[np.ix_(right, left)], 0.0, atol=1e-12)


def test_refine_p2_threshold_boundaries_are_inclusive():
    """0.20/0.35 경계값 자체는 통과하고, 초과한 경우만 탈락한다."""
    import copy
    from src.config import CFG
    from src.refine import movement_gate_reason
    cfg = copy.copy(CFG)
    cfg.refine_max_move_mean = 0.20
    cfg.refine_max_move_max = 0.35
    assert movement_gate_reason(0.20, 0.35, cfg) == "ok"
    assert movement_gate_reason(0.200001, 0.35, cfg) == "mean_move"
    assert movement_gate_reason(0.10, 0.350001, cfg) == "max_endpoint_move"
    assert movement_gate_reason(0.30, 0.40, cfg) == "max_endpoint_move"


def test_refine_p2_rolls_back_only_failed_limb():
    """한 팔이 과이동해도 반대팔 조정은 유지하고, 실패한 팔 채널만 복구한다."""
    import copy, tempfile
    import numpy as np
    import src.refine as R
    from src.bvh import parse_bvh, find_joint, rotation_channel_indices
    from src.config import CFG
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = True
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 15.0)
        kp, sc = _target_kp(tgt)
        original = R.limb_movement
        try:
            R.limb_movement = lambda _a, _b, limb: (
                (0.10, 0.15) if limb == "left_arm" else (0.40, 0.50)
            )
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=cfg)
        finally:
            R.limb_movement = original

        assert res.refined and res.reason == "ok_partial", res.to_dict()
        assert res.limbs == ("left_arm",), res.limbs
        assert res.limb_decisions["left_arm"]["accepted"]
        assert res.limb_decisions["right_arm"]["reason"] == "max_endpoint_move"

        joints, before = parse_bvh(base)
        _, after = parse_bvh(res.bvh_path)
        for suffix in R.LIMBS["right_arm"][0]:
            for channel in rotation_channel_indices(joints, find_joint(joints, suffix)):
                assert np.isclose(before[0][channel], after[0][channel])
        left_channels = [
            channel
            for suffix in R.LIMBS["left_arm"][0]
            for channel in rotation_channel_indices(joints, find_joint(joints, suffix))
        ]
        assert not np.allclose(before[0][left_channels], after[0][left_channels])


def test_refine_p2_returns_base_when_all_limbs_fail():
    """이동량 게이트 뒤 남은 사지가 없으면 파일을 쓰지 않고 베이스를 반환한다."""
    import copy, tempfile
    import src.refine as R
    from src.config import CFG
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = True
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 15.0)
        kp, sc = _target_kp(tgt)
        original = R.limb_movement
        try:
            R.limb_movement = lambda *_a, **_k: (0.40, 0.50)
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=cfg)
        finally:
            R.limb_movement = original
        assert not res.refined and res.reason == "movement_gate", res.to_dict()
        assert res.bvh_path == base
        assert not os.path.exists(f"{d}/out.bvh")
        assert all(not d["accepted"] for d in res.limb_decisions.values())


def test_refine_p2_rechecks_global_gain_after_partial_rollback():
    """부분 복구 뒤 전체 개선량이 부족해지면 안전한 사지도 포함해 베이스로 돌아간다."""
    import copy, tempfile
    import src.refine as R
    from src.bvh import (parse_bvh, find_joint, rotation_channel_indices,
                         write_single_frame_bvh)
    from src.config import CFG
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = True
    cfg.refine_min_gain = 0.40
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        joints, frames = parse_bvh(base)
        target_frame = frames[0].copy()
        for suffix in ("LeftArm", "RightArm"):
            channels = rotation_channel_indices(joints, find_joint(joints, suffix))
            target_frame[channels[0]] = 15.0
        target = write_single_frame_bvh(base, target_frame, f"{d}/target.bvh")
        kp, sc = _target_kp(target)

        original = R.limb_movement
        try:
            R.limb_movement = lambda _a, _b, limb: (
                (0.10, 0.15) if limb == "left_arm" else (0.40, 0.50)
            )
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=cfg)
        finally:
            R.limb_movement = original

        assert not res.refined and res.reason == "global_no_gain", res.to_dict()
        assert res.limb_decisions["left_arm"]["reason"] == "global_no_gain"
        assert res.limb_decisions["right_arm"]["reason"] == "max_endpoint_move"
        assert not os.path.exists(f"{d}/out.bvh")


def test_refine_p3_capsule_detects_deep_penetration_and_is_scale_invariant():
    """몸통 밖 전완은 0, 내부 관통은 양수이며 전체 스케일에는 불변이어야 한다."""
    import numpy as np
    from src.collision import arm_torso_penetration
    outside, scores = _collision_pose(left_inside=False)
    inside, _ = _collision_pose(left_inside=True)
    outside_m = arm_torso_penetration(outside, "left_arm", scores)
    inside_m = arm_torso_penetration(inside, "left_arm", scores)
    scaled_m = arm_torso_penetration(inside * 10.0, "left_arm", scores)
    assert outside_m.available and np.isclose(outside_m.depth, 0.0)
    assert inside_m.available and inside_m.depth > 0.02
    assert np.isclose(inside_m.depth, scaled_m.depth, atol=1e-12)


def test_refine_p3_hand_segment_catches_collision_outside_forearm():
    """손목·전완은 밖이어도 손끝 방향이 몸을 가로지르면 손 충돌로 잡아야 한다."""
    import numpy as np
    from src.collision import arm_torso_penetration
    outside, scores = _collision_pose(left_inside=False)
    wrist_only = arm_torso_penetration(outside, "left_arm", scores)
    with_hand = arm_torso_penetration(
        outside, "left_arm", scores, hand_tip=np.array([0.0, 0.45, 0.0])
    )
    assert np.isclose(wrist_only.depth, 0.0)
    assert with_hand.depth > 0.05
    assert with_hand.part == "hand"


def test_refine_p3_classifies_only_new_penetration():
    """베이스에 이미 있던 충돌은 신규 실패가 아니고, 밖→안 변화만 양성이다."""
    from src.collision import arm_torso_penetration, collision_status
    outside, scores = _collision_pose(left_inside=False)
    inside, _ = _collision_pose(left_inside=True)
    clear = arm_torso_penetration(outside, "left_arm", scores)
    deep = arm_torso_penetration(inside, "left_arm", scores)
    assert collision_status(clear, deep, 0.02, 0.01) == "new_penetration"
    assert collision_status(deep, deep, 0.02, 0.01) == "in_base"
    assert collision_status(clear, clear, 0.02, 0.01) == "clear"


def test_refine_p3_diagnostic_is_recorded_when_gate_is_off():
    """P3 하드 게이트를 꺼도 충돌 보정용 진단값은 계속 남긴다."""
    import copy, tempfile
    from src.config import CFG
    from src.refine import refine_bvh
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = False
    cfg.refine_collision_gate = False
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        target = _bvh_with_rotation(d, "target.bvh", "LeftArm", 15.0)
        kp, sc = _target_kp(target)
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh", cfg=cfg)
        assert res.refined, res.to_dict()
        for limb in ("left_arm", "right_arm"):
            collision = res.limb_decisions[limb]["collision"]
            assert collision["checked"]
            assert collision["status"] in ("clear", "new_penetration", "in_base")
            assert collision["final_depth"] is not None


def test_refine_p3_rolls_back_only_colliding_limb():
    """왼팔 충돌 시 왼팔만 원복하고 유용한 오른팔 P1 조정은 보존한다."""
    import copy, tempfile
    import numpy as np
    import src.refine as R
    from src.bvh import parse_bvh, find_joint, rotation_channel_indices
    from src.config import CFG
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = False
    cfg.refine_collision_gate = True
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        target = _bvh_with_rotations(
            d, "target.bvh", {"LeftArm": 15.0, "RightArm": 15.0}
        )
        kp, sc = _target_kp(target)
        original = R.collision_status
        statuses = iter(("new_penetration", "clear"))
        try:
            R.collision_status = lambda *_a, **_k: next(statuses)
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=cfg)
        finally:
            R.collision_status = original

        assert res.refined and res.reason == "ok_partial", res.to_dict()
        assert res.limbs == ("right_arm",), res.limbs
        assert res.limb_decisions["left_arm"]["reason"] == "self_collision"
        assert res.limb_decisions["right_arm"]["accepted"]
        left_collision = res.limb_decisions["left_arm"]["collision"]
        assert np.isclose(left_collision["final_depth"], left_collision["base_depth"])

        joints, before = parse_bvh(base)
        _, after = parse_bvh(res.bvh_path)
        for suffix in R.LIMBS["left_arm"][0]:
            for channel in rotation_channel_indices(joints, find_joint(joints, suffix)):
                assert np.isclose(before[0][channel], after[0][channel])
        right_channels = [
            channel
            for suffix in R.LIMBS["right_arm"][0]
            for channel in rotation_channel_indices(joints, find_joint(joints, suffix))
        ]
        assert not np.allclose(before[0][right_channels], after[0][right_channels])


def test_refine_p3_returns_base_when_all_limbs_collide():
    """양팔이 모두 신규 관통이면 조정 파일을 쓰지 않고 전체 베이스를 반환한다."""
    import copy, tempfile
    import src.refine as R
    from src.config import CFG
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = False
    cfg.refine_collision_gate = True
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        target = _bvh_with_rotations(
            d, "target.bvh", {"LeftArm": 15.0, "RightArm": 15.0}
        )
        kp, sc = _target_kp(target)
        original = R.collision_status
        try:
            R.collision_status = lambda *_a, **_k: "new_penetration"
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=cfg)
        finally:
            R.collision_status = original
        assert not res.refined and res.reason == "collision_gate", res.to_dict()
        assert res.bvh_path == base
        assert not os.path.exists(f"{d}/out.bvh")


def test_refine_p3_rechecks_global_gain_after_collision_rollback():
    """충돌 팔을 복구한 뒤 개선량이 사라지면 전체 베이스로 돌아간다."""
    import copy, tempfile
    import src.refine as R
    from src.config import CFG
    cfg = copy.copy(CFG)
    cfg.refine_observability_gate = False
    cfg.refine_move_gate = False
    cfg.refine_collision_gate = True
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        target = _bvh_with_rotation(d, "target.bvh", "LeftArm", 15.0)
        kp, sc = _target_kp(target)
        original = R.collision_status
        statuses = iter(("new_penetration", "clear"))
        try:
            R.collision_status = lambda *_a, **_k: next(statuses)
            res = R.refine_bvh(base, kp, sc, "front",
                               out_path=f"{d}/out.bvh", cfg=cfg)
        finally:
            R.collision_status = original
        assert not res.refined and res.reason == "global_no_gain", res.to_dict()
        assert res.limb_decisions["left_arm"]["reason"] == "self_collision"
        assert res.limb_decisions["right_arm"]["reason"] == "global_no_gain"
        assert not os.path.exists(f"{d}/out.bvh")


def test_refine_p1a_records_axis_diagnostics_and_never_solves_hands():
    """P1a 감도·lambda가 기록되고 손/발 rotation은 최적화 변수에 들어가지 않는다."""
    import copy, tempfile
    from src.config import CFG
    from src.refine import refine_bvh
    cfg = copy.copy(CFG)
    cfg.refine_axis_observability = True
    cfg.refine_svd_observability = True
    cfg.refine_limbs = "arms"
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 35.0)
        kp, sc = _target_kp(tgt)
        res = refine_bvh(base, kp, sc, "front",
                         out_path=f"{d}/out.bvh", cfg=cfg)
        assert res.axis_observability, res.reason
        assert res.axis_lambda_mult, res.reason
        assert all(v >= 1.0 for v in res.axis_lambda_mult.values())
        assert res.svd_singular_values, res.reason
        assert res.svd_lambda_mult, res.reason
        assert all(v >= 1.0 for v in res.svd_lambda_mult)
        assert not any("Hand" in k or "Foot" in k
                       for k in res.axis_observability)


def test_refine_freezes_limb_with_invisible_target():
    """러프에서 안 보이는 사지는 통째로 동결한다(한쪽 뼈만 맞추면 관절이 크게 돈다)."""
    import tempfile
    import numpy as np
    from src.refine import refine_bvh
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 35.0)
        kp, sc = _target_kp(tgt)
        sc = np.asarray(sc, dtype=np.float32).copy()
        sc[[6, 8, 10]] = 0.0             # 오른팔 전체를 안 보이게
        res = refine_bvh(base, kp, sc, "front", out_path=f"{d}/out.bvh")
        assert res.limb_decisions["right_arm"]["reason"] == "invisible_target"
        if res.refined:
            assert "right_arm" not in res.limbs, res.limbs


def test_refine_backends_agree():
    """
    scipy 백엔드와 numpy 폴백이 **같은 판정**을 내야 한다.

    이 테스트가 없어서 놓친 실제 버그(2026-08-01):
    scipy의 기본 유한차분 보폭(상대 1.5e-8)은 float32로 양자화된 우리 목적함수에서
    변화를 못 만든다 → 야코비안이 0 → scipy가 nfev=1로 시작점을 반환 → 조정이
    전혀 안 일어났는데 `no_gain`("개선 여지 없음")으로 보고됐다.
    numpy 폴백은 보폭이 커서 정상 동작했기 때문에, 환경에 따라 결과가 갈렸다.
    """
    import tempfile
    import src.refine as R
    with tempfile.TemporaryDirectory() as d:
        base = _synthetic_bvh(d, "base.bvh")
        tgt = _bvh_with_rotation(d, "tgt.bvh", "LeftArm", 15.0)
        kp, sc = _target_kp(tgt)

        results = {}
        original = R._solve_scipy
        try:
            for name in ("scipy", "numpy"):
                if name == "numpy":
                    def _no_scipy(*a, **k):
                        raise ImportError("forced")
                    R._solve_scipy = _no_scipy
                results[name] = R.refine_bvh(base, kp, sc, "front",
                                             out_path=f"{d}/{name}.bvh")
        finally:
            R._solve_scipy = original

        for name, r in results.items():
            # 개선 가능한 케이스인데 '개선 없음'이 나오면 최적화기가 안 돈 것이다.
            assert r.refined, f"{name} 백엔드가 조정을 못 했다: {r.reason}"
            assert r.loss_final < r.loss_base * 0.5, (name, r.loss_base, r.loss_final)
        a, b = results["scipy"], results["numpy"]
        assert a.refined == b.refined, (a.reason, b.reason)
        assert a.limb_decisions == b.limb_decisions


def test_feature_space_symmetry_shared_function():
    """색인과 refine이 같은 pose_to_feature를 통과한다(불변식 4)."""
    import numpy as np
    from src.library import build_entries_from_pose, pose_to_feature, _synthetic_pose
    joints = _synthetic_pose("sitting")
    entries = build_entries_from_pose("x", joints, {"shot": "full_half"})
    for e in entries:
        direct = pose_to_feature(joints, e.view)
        assert np.allclose(e.feature, direct), e.view


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
