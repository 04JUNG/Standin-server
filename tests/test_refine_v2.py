"""승인된 Refine v2의 feature-flag 경로와 안전 계약 테스트."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest import SkipTest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bvh import (coco17_from_fk, fk, parse_bvh,
                     rotation_channel_indices, find_joint)
from src.config import CFG, Config
from src.refine import REFINE_V2_CODE_VERSION, refine_bvh
from src.library import pose_to_feature
from src.refine_v2 import _projected_joint_angle
from tests.test_smoke import (_bvh_with_rotation, _bvh_with_rotations,
                              _synthetic_bvh, _target_kp, _pipe, _Img)


def _v2_cfg(**overrides):
    cfg = copy.copy(CFG)
    cfg.refine_enabled = True
    cfg.refine_v2_enabled = True
    cfg.refine_v2_lower_body = True
    cfg.refine_v2_torso_enabled = False
    cfg.refine_observability_gate = False
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_v25_product_defaults_are_safe_aggressive():
    fields = Config.__dataclass_fields__
    assert fields["refine_v2_enabled"].default is True
    assert fields["refine_default_mode"].default == "aggressive"
    assert fields["refine_v25_selector_enabled"].default is True
    assert fields["refine_v25_joint_nme_weight"].default > 0.0
    assert fields["refine_v25_skip_inactive_aggressive"].default is True
    assert 0.0 < fields[
        "refine_v25_inactive_skip_budget_fraction"
    ].default < 1.0
    assert fields["refine_v253_single_leg_extension_enabled"].default is True
    assert fields["refine_v253_up_leg_max_delta_deg"].default == 18.0


def test_v253_foreshortened_single_leg_extension_closes_131211_fixture():
    fixture_path = Path(__file__).parent / "fixtures/refine_v253_single_leg_extension.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parent.parent
    base_path = repo / fixture["base_bvh"]
    if not base_path.exists():
        # data/는 Mixamo·CMU 원본 재배포 금지 정책으로 레포에 커밋하지 않는다
        # (.gitignore). 포즈 라이브러리를 내려받은 환경에서만 실제 BVH 회귀를 돈다.
        raise SkipTest(f"pose library BVH unavailable: {fixture['base_bvh']}")
    cfg = copy.copy(CFG)
    cfg.refine_enabled = True
    cfg.refine_v2_enabled = True
    cfg.refine_v2_lower_body = True
    cfg.refine_v2_torso_enabled = False
    cfg.refine_observability_gate = True
    cfg.refine_default_mode = "aggressive"
    cfg.refine_v25_selector_enabled = True
    cfg.refine_timeout_seconds = 20.0
    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, "final.bvh")
        result = refine_bvh(
            str(base_path), np.asarray(fixture["keypoints"]),
            np.asarray(fixture["scores"]), fixture["view"],
            out_path=output, allowed_limbs=fixture["allowed_limbs"],
            lower_body_observed=True, refine_mode="aggressive", cfg=cfg,
        )
        assert result.refined, result.to_dict()
        assert result.diagnostics["mode_applied"] == "aggressive"
        right = result.limb_decisions["right_leg"]
        assert right["accepted"]
        assert right["reason"] == "ok_foreshortened_extension"
        evidence = result.diagnostics["single_leg_extension"]["right_leg"]
        assert evidence["accepted"]
        assert evidence["final"]["angle_gate"]
        assert evidence["final"]["ankle_gate"]
        final_gate = result.diagnostics[
            "final_single_leg_extension_postcheck"
        ]
        assert final_gate["passed"] and not final_gate["skipped"]
        assert final_gate["checks"]["right_leg"]["accepted"]

        joints, base_frames = parse_bvh(str(base_path))
        _, final_frames = parse_bvh(output)
        kp, scores = coco17_from_fk(joints, fk(joints, final_frames[0]))
        feature = pose_to_feature(kp, fixture["view"], scores)
        final_angle = _projected_joint_angle(feature, 12, 14, 16)
        assert final_angle >= cfg.refine_v253_straight_angle_min_deg

        up_leg = find_joint(joints, "RightUpLeg")
        channels = rotation_channel_indices(joints, up_leg)
        delta = (final_frames[0][channels] - base_frames[0][channels] + 180.0) \
            % 360.0 - 180.0
        assert float(np.max(np.abs(delta))) \
            <= cfg.refine_v253_up_leg_max_delta_deg + 1e-6


def test_v2_uses_positive_gain_and_records_base_solved_adopted_losses():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "out.bvh"),
            allowed_limbs=["left_arm"], cfg=cfg,
        )
        assert result.refined, result.to_dict()
        assert result.refine_version == REFINE_V2_CODE_VERSION
        assert result.refine_outcome == "improved"
        diagnostics = result.diagnostics
        assert diagnostics["hybrid_loss_adopted"] < (
            diagnostics["hybrid_loss_base"] - cfg.refine_gain_epsilon
        )
        left = diagnostics["losses"]["left_arm"]
        assert set(left["hybrid"]) == {"base", "solved", "adopted"}
        assert left["position"]["adopted"] <= left["position"]["base"]
        assert diagnostics["solver_profile"]["prepared_target_reused"]
        assert diagnostics["solver_profile"]["prepared_base_reused"]
        assert diagnostics["time_budget"]["prepare_ms"] >= 0.0
        phase_profile = diagnostics["phases"]["conservative"]["solver_profile"]
        assert phase_profile["nfev"] == result.iterations
        assert phase_profile["parameter_count"] > 0


def test_v2_does_not_use_search_distance_as_execution_gate():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "out.bvh"),
            search_distance=CFG.fallback_distance + 10.0,
            allowed_limbs=["left_arm"], cfg=cfg,
        )
        assert result.refined, result.to_dict()
        assert result.reason != "base_mismatch"
        assert result.diagnostics["context"]["search_distance"] > CFG.fallback_distance


def test_v24_d0_can_capture_raw_aggressive_candidate_without_changing_final_contract():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 20.0)
        keypoints, scores = _target_kp(target)
        candidate_path = os.path.join(directory, "raw-candidate.bvh")
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "final.bvh"),
            allowed_limbs=["left_arm"], refine_mode="aggressive",
            diagnostic_candidate_out_path=candidate_path, cfg=cfg,
        )
        assert os.path.isfile(candidate_path), result.to_dict()
        _, candidate_frames = parse_bvh(candidate_path)
        assert len(candidate_frames) == 1
        assert result.bvh_path != candidate_path


def test_v2_pipeline_preserves_scores_when_only_search_confidence_is_low():
    previous_flag = CFG.refine_v2_enabled
    previous_threshold = CFG.fallback_pos_full
    try:
        CFG.refine_v2_enabled = True
        CFG.fallback_pos_full = 0.0
        result = _pipe().process_cut(_Img("full_half standing front 1p"))
        descriptor = result.descriptors[0]
        assert result.person_confidence[0] == "low"
        assert descriptor.refine_allowed
        assert descriptor.quality_trace["refine_policy"] == "v2_structural"
        assert float(np.asarray(descriptor.skeleton.scores).sum()) > 0.0
    finally:
        CFG.refine_v2_enabled = previous_flag
        CFG.fallback_pos_full = previous_threshold


def test_v252_pipeline_removes_all_legs_when_vlm_marks_half_body():
    previous_flag = CFG.refine_v2_enabled
    previous_lower = CFG.refine_v2_lower_body
    try:
        CFG.refine_v2_enabled = True
        CFG.refine_v2_lower_body = True
        result = _pipe().process_cut(
            _Img("full_half standing front 1p half_body")
        )
        descriptor = result.descriptors[0]
        assert descriptor.lower_body_observed is False
        assert set(descriptor.refinable_limbs).isdisjoint(
            {"left_leg", "right_leg"}
        )
        assert descriptor.quality_trace["lower_body_policy"] \
            == "all_lower_frozen"
    finally:
        CFG.refine_v2_enabled = previous_flag
        CFG.refine_v2_lower_body = previous_lower


def test_v2_lower_body_is_solved_with_safe_partial_alpha():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftUpLeg", 15.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "out.bvh"),
            allowed_limbs=["left_leg"], lower_body_observed=True, cfg=cfg,
        )
        assert result.refined, result.to_dict()
        decision = result.limb_decisions["left_leg"]
        assert decision["accepted"]
        assert decision["alpha"] in (1.0, 0.75, 0.5, 0.25)
        assert decision["foot"]["status"] == "ok"
        assert "leg_leg" in decision["collision"]


def test_v252_unobserved_lower_body_freezes_every_leg_channel():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotations(
            directory, "target.bvh",
            {"LeftArm": 15.0, "LeftUpLeg": 20.0},
        )
        keypoints, scores = _target_kp(target)
        output = os.path.join(directory, "out.bvh")
        result = refine_bvh(
            base, keypoints, scores, "front", out_path=output,
            allowed_limbs=["left_arm", "left_leg"],
            lower_body_observed=False, cfg=cfg,
        )
        assert result.refined, result.to_dict()
        assert "left_arm" in result.limbs
        assert "left_leg" not in result.limbs
        assert result.diagnostics["lower_body_policy"] == "all_lower_frozen"
        joints, before = parse_bvh(base)
        _, after = parse_bvh(result.bvh_path)
        for suffix in ("LeftUpLeg", "LeftLeg", "LeftFoot",
                       "RightUpLeg", "RightLeg", "RightFoot"):
            index = find_joint(joints, suffix)
            channels = rotation_channel_indices(joints, index)
            assert np.array_equal(before[0][channels], after[0][channels]), suffix


def test_v252_missing_lower_body_lineage_is_fail_closed_for_leg_only_request():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftUpLeg", 15.0)
        keypoints, scores = _target_kp(target)
        output = os.path.join(directory, "out.bvh")
        result = refine_bvh(
            base, keypoints, scores, "front", out_path=output,
            allowed_limbs=["left_leg"], cfg=cfg,
        )
        assert not result.refined
        assert result.bvh_path == base
        assert not os.path.exists(output)
        assert result.diagnostics["lower_body_policy"] == "all_lower_frozen"


def test_v2_leg_collision_proxy_is_scale_invariant():
    from src.collision import leg_leg_penetration

    pose = np.zeros((17, 3), dtype=np.float64)
    pose[5], pose[6] = (-0.3, 1.0, 0.0), (0.3, 1.0, 0.0)
    pose[11], pose[12] = (-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)
    pose[13], pose[15] = (-0.1, -0.5, 0.0), (0.15, -1.0, 0.0)
    pose[14], pose[16] = (0.1, -0.5, 0.0), (-0.15, -1.0, 0.0)
    scores = np.ones(17, dtype=np.float64)
    crossing = leg_leg_penetration(pose, scores)
    scaled = leg_leg_penetration(pose * 10.0, scores)
    assert crossing.available and crossing.depth > 0.0
    assert np.isclose(crossing.depth, scaled.depth, atol=1e-12)


def test_v24_lap_contact_clearance_is_signed_and_scale_invariant():
    from src.collision import hand_leg_surface_clearance

    pose = np.zeros((17, 3), dtype=np.float64)
    pose[5], pose[6] = (-0.3, 1.0, 0.0), (0.3, 1.0, 0.0)
    pose[11], pose[12] = (-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)
    pose[13], pose[15] = (-0.2, -0.5, 0.0), (-0.2, -1.0, 0.0)
    pose[9] = (-0.13, -0.25, 0.0)  # hand+leg radii 합과 같은 표면 거리
    scores = np.ones(17, dtype=np.float64)

    surface = hand_leg_surface_clearance(
        pose, "left_arm", "left_leg", scores
    )
    scaled = hand_leg_surface_clearance(
        pose * 10.0, "left_arm", "left_leg", scores
    )
    assert surface.available and abs(surface.clearance) < 1e-12
    assert np.isclose(surface.clearance, scaled.clearance, atol=1e-12)

    pose[9] = (-0.2, -0.25, 0.0)
    overlap = hand_leg_surface_clearance(
        pose, "left_arm", "left_leg", scores
    )
    assert overlap.available and overlap.clearance < 0.0
    assert overlap.part in ("hand_thigh", "hand_knee")


def test_v24_ankle_counter_rotation_changes_only_foot_local_channels():
    from src.refine_v2 import _counter_rotate_feet

    cfg = _v2_cfg(
        refine_v2_foot_direction_deg=0.0,
        refine_v2_ankle_counter_max_delta_deg=18.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        turned = _bvh_with_rotation(
            directory, "turned.bvh", "LeftUpLeg", 25.0
        )
        joints, base_frames = parse_bvh(base)
        _, turned_frames = parse_bvh(turned)
        corrected, diagnostics, _ = _counter_rotate_feet(
            joints, base_frames[0], turned_frames[0], "front",
            ("left_leg",), cfg, None,
        )

        foot = find_joint(joints, "LeftFoot")
        foot_channels = set(rotation_channel_indices(joints, foot))
        changed = set(np.flatnonzero(
            np.abs(corrected - turned_frames[0]) > 1e-8
        ).tolist())
        row = diagnostics["left_leg"]
        assert row["accepted"] and row["after_deg"] < row["before_deg"]
        assert changed and changed <= foot_channels
        assert max(abs(value) for value in row["rotation_delta_deg"].values()) \
            <= cfg.refine_v2_ankle_counter_max_delta_deg + 1e-6
        assert np.allclose(
            fk(joints, corrected)[foot], fk(joints, turned_frames[0])[foot]
        )


def test_v2_arm_leg_collision_distinguishes_contact_from_new_penetration():
    from src.collision import (arm_leg_penetration, collision_relation,
                               collision_status)

    pose = np.zeros((17, 3), dtype=np.float64)
    pose[5], pose[6] = (-0.3, 1.0, 0.0), (0.3, 1.0, 0.0)
    pose[11], pose[12] = (-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)
    pose[13], pose[15] = (-0.2, -0.5, 0.0), (-0.2, -1.0, 0.0)
    pose[7], pose[9] = (-0.8, 0.2, 0.0), (-0.8, -0.4, 0.0)
    scores = np.ones(17, dtype=np.float64)
    base = arm_leg_penetration(pose, "left_arm", "left_leg", scores)

    shallow_pose = pose.copy()
    shallow_pose[7], shallow_pose[9] = (-0.275, 0.0, 0.0), (-0.275, -0.5, 0.0)
    shallow = arm_leg_penetration(
        shallow_pose, "left_arm", "left_leg", scores
    )
    shallow_status = collision_status(base, shallow, 0.01, 0.005)
    assert 0.0 < shallow.depth < 0.01
    assert shallow_status != "new_penetration"
    assert collision_relation(base, shallow, shallow_status, 0.01) == "shallow_contact"

    penetrating_pose = pose.copy()
    penetrating_pose[7], penetrating_pose[9] = (
        (-0.2, 0.0, 0.0), (-0.2, -0.5, 0.0)
    )
    penetrating = arm_leg_penetration(
        penetrating_pose, "left_arm", "left_leg", scores
    )
    penetrating_status = collision_status(base, penetrating, 0.01, 0.005)
    assert penetrating_status == "new_penetration"
    assert np.isclose(
        penetrating.depth,
        arm_leg_penetration(
            penetrating_pose * 10.0, "left_arm", "left_leg", scores
        ).depth,
        atol=1e-12,
    )


def test_v2_lower_pair_uses_common_alpha_and_improves_pair_loss():
    cfg = _v2_cfg(
        refine_max_move_mean=1.0,
        refine_max_move_max=1.0,
        refine_v2_foot_direction_deg=180.0,
        refine_v2_ground_tolerance=1.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotations(
            directory, "target.bvh",
            {"LeftUpLeg": -15.0, "RightUpLeg": 15.0},
        )
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "out.bvh"),
            allowed_limbs=["left_leg", "right_leg"],
            lower_body_observed=True, cfg=cfg,
        )
        assert result.refined, result.to_dict()
        pair = result.diagnostics["lower_pair"]
        assert pair["active"] and pair["adoption"]["attempted"]
        assert pair["adoption"]["accepted"], pair
        assert pair["adopted"]["loss"] < pair["base"]["loss"]
        assert result.limb_decisions["left_leg"]["alpha"] == (
            result.limb_decisions["right_leg"]["alpha"]
        )
        assert result.limb_decisions["left_leg"]["reason"] == "ok_lower_pair"


def test_v2_returns_exact_base_when_every_alpha_fails_safety():
    cfg = _v2_cfg(refine_max_move_mean=0.0, refine_max_move_max=0.0)
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftUpLeg", 15.0)
        keypoints, scores = _target_kp(target)
        output = os.path.join(directory, "out.bvh")
        result = refine_bvh(
            base, keypoints, scores, "front", out_path=output,
            allowed_limbs=["left_leg"], lower_body_observed=True, cfg=cfg,
        )
        assert not result.refined and result.reason == "safety_gate", result.to_dict()
        assert result.bvh_path == base
        assert not os.path.exists(output)
        assert result.limb_decisions["left_leg"]["alpha"] == 0.0


def test_v2_torso_changes_only_allowlisted_local_channels_and_keeps_root():
    cfg = _v2_cfg(
        refine_v2_torso_enabled=True,
        refine_max_move_mean=1.0,
        refine_max_move_max=1.0,
        refine_v2_foot_direction_deg=180.0,
        refine_v2_ground_tolerance=1.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "Spine", 6.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "out.bvh"),
            allowed_limbs=list(("left_arm", "right_arm", "left_leg", "right_leg")),
            lower_body_observed=True,
            cfg=cfg,
        )
        assert result.refined, result.to_dict()
        torso = result.diagnostics["torso"]
        assert torso["attempted"] and torso["accepted"], torso
        joints, before = parse_bvh(base)
        _, after = parse_bvh(result.bvh_path)
        root_rotations = rotation_channel_indices(joints, 0)
        assert np.array_equal(before[0][root_rotations], after[0][root_rotations])
        spine = rotation_channel_indices(joints, find_joint(joints, "Spine"))
        assert not np.array_equal(before[0][spine], after[0][spine])
        assert max(abs(v) for v in torso["rotation_delta_deg"].values()) \
            <= cfg.refine_v2_torso_max_delta_deg + 1e-6


def test_v2_timeout_falls_back_without_writing():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        output = os.path.join(directory, "out.bvh")
        result = refine_bvh(
            base, keypoints, scores, "front", out_path=output,
            allowed_limbs=["left_arm"], deadline=time.monotonic() - 1.0,
            cfg=cfg,
        )
        assert not result.refined and result.reason == "timeout", result.to_dict()
        assert result.bvh_path == base and not os.path.exists(output)


def test_v25_api_returns_inline_bvh_and_leaves_no_local_artifact():
    """조정본은 응답 본문으로만 나가고 디스크에는 아무것도 남지 않는다.

    handle 기반 로컬 캐시는 제거됐다(docs/REFINE_HANDOFF.md §3 4단계). 재계산을
    막는 멱등성은 BFF의 refined_artifacts PK가 담당하므로 서버는 무상태다.
    """
    import api.app as api_app
    from api.models import RefineRequest

    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        request = RefineRequest(
            pose_id="pose", view="front", keypoints=keypoints.tolist(),
            scores=scores.tolist(), refine_allowed=True,
            refinable_limbs=["left_arm"], lower_body_observed=False,
            skeleton_state="valid", coverage_class="full", slot_origin="vlm",
            skeleton_source="full_image", gap_type="unknown",
        )
        assert request.refine_mode is None
        original_meta = api_app.get_pose_meta
        original_flag = CFG.refine_v2_enabled
        original_obs = CFG.refine_observability_gate
        original_state = dict(api_app.STATE)
        before = sorted(os.listdir(directory))
        try:
            api_app.STATE["db_path"] = os.path.join(directory, "unused.db")
            api_app.get_pose_meta = lambda *_args: {
                "bvh_path": base, "set_id": None,
            }
            CFG.refine_v2_enabled = True
            CFG.refine_observability_gate = False
            response = api_app.refine(request)
        finally:
            api_app.get_pose_meta = original_meta
            CFG.refine_v2_enabled = original_flag
            CFG.refine_observability_gate = original_obs
            api_app.STATE.clear()
            api_app.STATE.update(original_state)

        assert response.refined, response.reason
        # 조정본을 얻는 유일한 경로는 응답 본문이다.
        assert response.bvh and response.bvh.startswith("HIERARCHY")
        assert "\r\n" not in response.bvh
        # bvh_url은 refined 여부와 무관하게 항상 베이스를 가리킨다.
        assert response.bvh_url == "/pose/pose/bvh"
        assert response.refine_version == REFINE_V2_CODE_VERSION
        assert response.diagnostics["mode_effective"] == "aggressive"
        assert response.diagnostics["mode_applied"] in {
            "aggressive", "conservative", "base",
        }
        # 요청이 끝난 뒤 로컬에 남은 산출물이 없어야 한다.
        assert sorted(os.listdir(directory)) == before
        assert not os.path.exists(os.path.splitext(base)[0] + ".refined.v2.bvh")


def test_v25_api_base_fallback_has_no_inline_bvh():
    """refined=false면 bvh는 None이고 bvh_url은 베이스다."""
    import api.app as api_app
    from api.models import RefineRequest

    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        request = RefineRequest(
            pose_id="pose", view="front",
            keypoints=np.zeros((17, 2)).tolist(),
            refine_allowed=False,
        )
        original_meta = api_app.get_pose_meta
        original_state = dict(api_app.STATE)
        try:
            api_app.STATE["db_path"] = os.path.join(directory, "unused.db")
            api_app.get_pose_meta = lambda *_args: {
                "bvh_path": base, "set_id": None,
            }
            response = api_app.refine(request)
        finally:
            api_app.get_pose_meta = original_meta
            api_app.STATE.clear()
            api_app.STATE.update(original_state)

        assert response.refined is False
        assert response.reason == "skeleton_policy"
        assert response.bvh is None
        assert response.bvh_url == "/pose/pose/bvh"
        assert response.refine_outcome == "not_attempted"


def test_v25_selector_accepts_only_common_metric_positive_gain():
    from src.refine_selector import select_aggressive
    from src.refine_v2 import _JOINT_DELTA_LIMITS

    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        conservative = _bvh_with_rotation(
            directory, "conservative.bvh", "LeftArm", 5.0
        )
        aggressive = _bvh_with_rotation(
            directory, "aggressive.bvh", "LeftArm", 10.0
        )
        keypoints, scores = _target_kp(aggressive)
        # Freeze a focused target mask so unrelated pair/contact cohorts are
        # inactive while torso alignment and the changed arm remain measurable.
        scores = scores.copy()
        scores[[8, 10, 13, 14, 15, 16]] = 0.0
        accepted = select_aggressive(
            policy_base_path=base, conservative_path=conservative,
            aggressive_path=aggressive, conservative_mode="conservative",
            target_keypoints=keypoints, target_scores=scores, view="front",
            allowed_limbs=["left_arm"], deadline=None, cfg=cfg,
            trust_limits=_JOINT_DELTA_LIMITS,
        )
        assert accepted.candidate_accepted, accepted.to_dict()
        assert accepted.selected_mode == "aggressive"

        rejected = select_aggressive(
            policy_base_path=base, conservative_path=aggressive,
            aggressive_path=conservative, conservative_mode="conservative",
            target_keypoints=keypoints, target_scores=scores, view="front",
            allowed_limbs=["left_arm"], deadline=None, cfg=cfg,
            trust_limits=_JOINT_DELTA_LIMITS,
        )
        assert not rejected.candidate_accepted
        assert rejected.fallback_reason == "candidate_non_regression"


def test_v25_selector_enforces_cumulative_trust_from_original_base():
    from src.refine_selector import select_aggressive
    from src.refine_v2 import _JOINT_DELTA_LIMITS

    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        conservative = _bvh_with_rotation(
            directory, "conservative.bvh", "LeftForeArm", 30.0
        )
        candidate = _bvh_with_rotation(
            directory, "candidate.bvh", "LeftForeArm", 50.0
        )
        keypoints, scores = _target_kp(candidate)
        decision = select_aggressive(
            policy_base_path=base, conservative_path=conservative,
            aggressive_path=candidate, conservative_mode="conservative",
            target_keypoints=keypoints, target_scores=scores, view="front",
            allowed_limbs=["left_arm"], deadline=None, cfg=cfg,
            trust_limits=_JOINT_DELTA_LIMITS,
        )
        assert not decision.candidate_accepted
        assert decision.fallback_reason == "candidate_structural_gate"
        violations = decision.structural_checks["violations"]
        assert any(row["type"] == "rotation_trust_region_exceeded"
                   for row in violations), violations


def test_v25_selector_recovers_safe_partial_global_blend():
    from src.refine_selector import select_aggressive
    from src.refine_v2 import _JOINT_DELTA_LIMITS

    cfg = _v2_cfg(refine_v25_partial_alphas="0.75,0.5,0.25")
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        conservative = _bvh_with_rotation(
            directory, "conservative.bvh", "LeftForeArm", 10.0
        )
        candidate = _bvh_with_rotation(
            directory, "candidate.bvh", "LeftForeArm", 25.0
        )
        keypoints, scores = _target_kp(candidate)
        decision = select_aggressive(
            policy_base_path=base, conservative_path=conservative,
            aggressive_path=candidate, conservative_mode="conservative",
            target_keypoints=keypoints, target_scores=scores, view="front",
            allowed_limbs=["left_arm"], deadline=None, cfg=cfg,
            trust_limits=_JOINT_DELTA_LIMITS,
        )
        try:
            assert decision.candidate_accepted, decision.to_dict()
            assert decision.selected_variant == "global_blend"
            assert decision.selected_alpha in (0.5, 0.25)
            assert os.path.isfile(decision.selected_path)
            assert not decision.common_metrics["regressions"]
            assert decision.common_metrics["positive_gains"]
        finally:
            if (decision.candidate_accepted
                    and decision.selected_path not in {conservative, candidate}
                    and os.path.exists(decision.selected_path)):
                os.unlink(decision.selected_path)


def test_v252_final_collision_postcheck_reverts_conservative_to_exact_base():
    import src.refine_selector as selector_module

    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        output = os.path.join(directory, "out.bvh")
        original = selector_module.final_collision_safety
        selector_module.final_collision_safety = lambda *_a, **_k: {
            "passed": False,
            "violations": [{"type": "final_new_collision", "pair": "left_arm:torso"}],
            "checks": {},
        }
        try:
            result = refine_bvh(
                base, keypoints, scores, "front", out_path=output,
                allowed_limbs=["left_arm"], refine_mode="conservative", cfg=cfg,
            )
        finally:
            selector_module.final_collision_safety = original
        assert not result.refined
        assert result.reason == "final_collision_gate"
        assert result.bvh_path == base
        assert not os.path.exists(output)
        assert result.diagnostics["mode_applied"] == "base"
        assert result.diagnostics["selector"]["fallback_stage"] == "final_collision"
        assert result.diagnostics["final_collision_postcheck"]["passed"] is False


def test_v25_aggressive_skips_when_final_check_budget_is_not_available():
    cfg = _v2_cfg(
        refine_v25_aggressive_min_remaining_seconds=10.0,
        refine_v25_final_check_reserve_seconds=1.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "result.bvh"),
            allowed_limbs=["left_arm"], refine_mode="aggressive",
            deadline=time.monotonic() + 2.0, cfg=cfg,
        )
        assert result.refined
        assert result.diagnostics["mode_applied"] == "conservative"
        assert not result.diagnostics["aggressive_attempted"]
        assert (result.diagnostics["selector"]["fallback_reason"]
                == "aggressive_insufficient_time")


def test_v25_inactive_aggressive_skips_only_when_budget_risk_is_high():
    cfg = _v2_cfg(
        refine_v25_skip_inactive_aggressive=True,
        refine_v25_inactive_skip_budget_fraction=0.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "result.bvh"),
            allowed_limbs=["left_arm"], refine_mode="aggressive",
            deadline=time.monotonic() + 5.0, cfg=cfg,
        )
        assert result.refined, result.to_dict()
        assert result.diagnostics["mode_applied"] == "conservative"
        assert not result.diagnostics["aggressive_attempted"]
        assert result.diagnostics["aggressive_activity"]["active"] is False
        assert result.diagnostics["selector"]["fallback_reason"] \
            == "aggressive_objectives_inactive_budget_risk"


def test_v25_api_resolves_default_and_explicit_mode():
    import api.app as api_app
    from api.models import RefineRequest

    original_flag = CFG.refine_v2_enabled
    original_default = CFG.refine_default_mode
    try:
        CFG.refine_v2_enabled = True
        CFG.refine_default_mode = "aggressive"
        assert api_app._effective_refine_mode(None) == "aggressive"
        assert api_app._effective_refine_mode("conservative") == "conservative"
        CFG.refine_default_mode = "conservative"
        assert api_app._effective_refine_mode(None) == "conservative"
        assert api_app._effective_refine_mode("aggressive") == "aggressive"
        CFG.refine_v2_enabled = False
        CFG.refine_default_mode = "aggressive"
        assert api_app._effective_refine_mode(None) == "conservative"
        capability = api_app._refine_capability()
        assert capability["default_mode"] == "conservative"
        assert not capability["selector_enabled"]
    finally:
        CFG.refine_v2_enabled = original_flag
        CFG.refine_default_mode = original_default


def test_v1_api_revalidates_structural_lineage_before_solver_call():
    import api.app as api_app
    from api.models import RefineRequest

    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        request = RefineRequest(
            pose_id="pose", view="front", keypoints=keypoints.tolist(),
            scores=scores.tolist(), refine_allowed=True,
            refinable_limbs=["left_arm"], skeleton_state="valid",
            coverage_class="full", slot_origin="vlm",
            skeleton_source="crop_retry",
        )
        original_meta = api_app.get_pose_meta
        original_solver = api_app.refine_bvh
        original_v2 = CFG.refine_v2_enabled
        original_state = dict(api_app.STATE)
        try:
            api_app.STATE["db_path"] = os.path.join(directory, "unused.db")
            api_app.get_pose_meta = lambda *_args: {
                "bvh_path": base, "set_id": None,
            }
            CFG.refine_v2_enabled = False

            def must_not_run(*_args, **_kwargs):
                raise AssertionError("invalid production lineage reached the v1 solver")

            api_app.refine_bvh = must_not_run
            response = api_app.refine(request)
            assert response.refined is False
            assert response.reason == "skeleton_policy"
            assert response.refine_outcome == "not_attempted"
        finally:
            api_app.get_pose_meta = original_meta
            api_app.refine_bvh = original_solver
            CFG.refine_v2_enabled = original_v2
            api_app.STATE.clear()
            api_app.STATE.update(original_state)


def test_refine_request_rejects_nan_and_wrong_shape():
    from pydantic import ValidationError
    from api.models import RefineRequest

    good = [[float(i), float(i + 1)] for i in range(17)]
    for keypoints in (good[:16], [[float("nan"), 0.0]] + good[1:]):
        try:
            RefineRequest(pose_id="p", view="front", keypoints=keypoints)
        except ValidationError:
            continue
        raise AssertionError("invalid keypoints were accepted")

    try:
        RefineRequest(
            pose_id="p", view="front", keypoints=good,
            refine_mode="unbounded",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("unknown refine mode was accepted")


def test_v24_aggressive_contact_regression_falls_back_to_conservative_exactly():
    cfg = _v2_cfg(
        refine_max_move_mean=1.0,
        refine_max_move_max=1.0,
        refine_v2_foot_direction_deg=180.0,
        refine_v2_ground_tolerance=1.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotations(
            directory, "target.bvh", {
                "LeftArm": -10.0, "RightArm": 10.0,
                "LeftUpLeg": -10.0, "RightUpLeg": 10.0,
            },
        )
        keypoints, scores = _target_kp(target)
        conservative_path = os.path.join(directory, "conservative.bvh")
        aggressive_path = os.path.join(directory, "aggressive.bvh")
        conservative = refine_bvh(
            base, keypoints, scores, "front", out_path=conservative_path,
            allowed_limbs=[
                "left_arm", "right_arm", "left_leg", "right_leg",
            ],
            lower_body_observed=True,
            refine_mode="conservative", cfg=cfg,
        )
        aggressive = refine_bvh(
            base, keypoints, scores, "front", out_path=aggressive_path,
            allowed_limbs=[
                "left_arm", "right_arm", "left_leg", "right_leg",
            ],
            lower_body_observed=True,
            refine_mode="aggressive", cfg=cfg,
        )
        assert conservative.refined and aggressive.refined
        assert aggressive.diagnostics["mode_requested"] == "aggressive"
        assert aggressive.diagnostics["mode_applied"] == "conservative"
        assert aggressive.diagnostics["aggressive_attempted"]
        objectives = aggressive.diagnostics["phase_objectives"]["aggressive"]
        assert objectives["hand_pair"]["active"]
        assert (objectives["hand_pair"]["solved"]["loss"]
                < objectives["hand_pair"]["base"]["loss"])
        assert objectives["lap_contact"]["active"]
        assert (objectives["lap_contact"]["adopted"]["loss"]
                <= objectives["lap_contact"]["base"]["loss"])
        assert objectives["lower_pair"]["adoption"]["reason"] \
            == "lap_contact_regression"
        with open(conservative_path, "rb") as source:
            conservative_bytes = source.read()
        with open(aggressive_path, "rb") as source:
            aggressive_bytes = source.read()
        assert aggressive_bytes == conservative_bytes


def test_v2_synthetic_loop_writes_fixed_pair_manifest():
    from scripts.eval_refine_v2_synthetic import run_synthetic_loop

    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "pose_a.bvh")
        variant = _bvh_with_rotations(
            directory, "pose_b.bvh", {"LeftArm": 12.0, "LeftUpLeg": 8.0}
        )
        output = os.path.join(directory, "eval")
        manifest = run_synthetic_loop(
            [base, variant], output, cfg=_v2_cfg()
        )
        assert manifest["source_count"] == 2
        assert len(manifest["records"]) == 2
        assert all(row["target_pose_id"] != row["base_pose_id"]
                   for row in manifest["records"])
        assert os.path.exists(os.path.join(output, "manifest.json"))


def test_v25_inline_and_file_delivery_produce_identical_geometry():
    """전달 방식이 solver 결과를 바꾸지 않는다.

    out_path를 준 호출(평가 harness)과 out_path=None 호출(제품 API)은 같은
    solver·selector를 타므로 최종 BVH 바이트가 정확히 같아야 한다. 이 등가성이
    깨지면 D0 수치와 제품 결과가 갈라진다.
    """
    cfg = _v2_cfg(refine_default_mode="aggressive")
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotations(
            directory, "target.bvh",
            {"LeftArm": 14.0, "RightArm": -12.0},
        )
        keypoints, scores = _target_kp(target)
        common = dict(view="front", allowed_limbs=["left_arm", "right_arm"],
                      lower_body_observed=False, refine_mode="aggressive",
                      cfg=cfg)
        written = refine_bvh(
            base, keypoints, scores,
            out_path=os.path.join(directory, "file.bvh"), **common
        )
        inline = refine_bvh(base, keypoints, scores, out_path=None, **common)

        assert written.refined == inline.refined
        assert written.reason == inline.reason
        assert written.bvh_text == inline.bvh_text
        if written.refined:
            assert inline.bvh_text
            on_disk = open(written.bvh_path, encoding="utf-8").read()
            assert on_disk == inline.bvh_text, "파일과 인라인 본문이 다르다"
            assert inline.bvh_path is None, "인라인 호출이 경로를 남겼다"
        # 인라인 경로는 어떤 파일도 남기지 않는다.
        assert not os.path.exists(os.path.splitext(base)[0] + ".refined.v2.bvh")


def test_v25_inline_delivery_leaves_no_scratch_directory():
    """요청 수명 임시 디렉터리가 반드시 정리된다."""
    import glob
    cfg = _v2_cfg(refine_default_mode="aggressive")
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "refine-v25-*")))
        refine_bvh(base, keypoints, scores, "front", out_path=None,
                   allowed_limbs=["left_arm"], refine_mode="aggressive", cfg=cfg)
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "refine-v25-*")))
        assert after == before, f"임시 디렉터리가 남았다: {after - before}"


def test_v1_path_also_freezes_lower_body_when_unobserved():
    """REFINE_V2_ENABLED=0으로 되돌려도 하체 fail-closed가 유지된다.

    API가 정책 판정에만 쓰고 solver에는 원본 refinable_limbs를 넘기면, v2는
    내부에서 다시 막지만 v1 경로에는 하체 게이트가 없어 조용히 다리가 움직인다.
    비상 복구 스위치를 내린 순간 안전 계약이 달라지면 안 된다.
    """
    import api.app as api_app
    from api.models import RefineRequest

    seen = {}

    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftArm", 15.0)
        keypoints, scores = _target_kp(target)
        request = RefineRequest(
            pose_id="pose", view="front", keypoints=keypoints.tolist(),
            scores=scores.tolist(), refine_allowed=True,
            refinable_limbs=["left_arm", "left_leg", "right_leg"],
            lower_body_observed=False,          # 반신 컷 — 하체 비관측
            skeleton_state="valid", coverage_class="full", slot_origin="vlm",
            skeleton_source="full_image",
        )
        original_meta = api_app.get_pose_meta
        original_solver = api_app.refine_bvh
        original_flag = CFG.refine_v2_enabled
        original_state = dict(api_app.STATE)

        def capture(*args, **kwargs):
            seen["allowed_limbs"] = list(kwargs.get("allowed_limbs") or [])
            return original_solver(*args, **kwargs)

        try:
            api_app.STATE["db_path"] = os.path.join(directory, "unused.db")
            api_app.get_pose_meta = lambda *_a: {"bvh_path": base, "set_id": None}
            api_app.refine_bvh = capture
            CFG.refine_v2_enabled = False        # 비상 복구 경로
            api_app.refine(request)
        finally:
            api_app.get_pose_meta = original_meta
            api_app.refine_bvh = original_solver
            CFG.refine_v2_enabled = original_flag
            api_app.STATE.clear()
            api_app.STATE.update(original_state)

    assert "left_leg" not in seen["allowed_limbs"], seen
    assert "right_leg" not in seen["allowed_limbs"], seen
    assert seen["allowed_limbs"] == ["left_arm"]


if __name__ == "__main__":
    import traceback
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_")]
    passed = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
            passed += 1
        except SkipTest as exc:
            print(f"SKIP {function.__name__}: {exc}")
            passed += 1
        except Exception:
            print(f"FAIL {function.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(functions)} passed")
    raise SystemExit(0 if passed == len(functions) else 1)
