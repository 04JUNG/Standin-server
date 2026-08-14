"""승인된 Refine v2의 feature-flag 경로와 안전 계약 테스트."""
from __future__ import annotations

import copy
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bvh import fk, parse_bvh, rotation_channel_indices, find_joint
from src.config import CFG
from src.refine import REFINE_V2_CODE_VERSION, refine_bvh
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


def test_v2_lower_body_is_solved_with_safe_partial_alpha():
    cfg = _v2_cfg()
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        target = _bvh_with_rotation(directory, "target.bvh", "LeftUpLeg", 15.0)
        keypoints, scores = _target_kp(target)
        result = refine_bvh(
            base, keypoints, scores, "front",
            out_path=os.path.join(directory, "out.bvh"),
            allowed_limbs=["left_leg"], cfg=cfg,
        )
        assert result.refined, result.to_dict()
        decision = result.limb_decisions["left_leg"]
        assert decision["accepted"]
        assert decision["alpha"] in (1.0, 0.75, 0.5, 0.25)
        assert decision["foot"]["status"] == "ok"
        assert "leg_leg" in decision["collision"]


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
            allowed_limbs=["left_leg", "right_leg"], cfg=cfg,
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
            allowed_limbs=["left_leg"], cfg=cfg,
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


def test_v2_api_cache_reuses_content_addressed_result():
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
            skeleton_source="full_image", gap_type="unknown",
        )
        assert request.refine_mode == "conservative"
        aggressive_request = request.model_copy(
            update={"refine_mode": "aggressive"}
        )
        assert api_app._refine_handle(request, base) != api_app._refine_handle(
            aggressive_request, base
        )
        original_meta = api_app.get_pose_meta
        original_dir = api_app.REFINE_DIR
        original_solver = api_app.refine_bvh
        original_flag = CFG.refine_v2_enabled
        original_obs = CFG.refine_observability_gate
        original_state = dict(api_app.STATE)
        try:
            api_app.STATE["db_path"] = os.path.join(directory, "unused.db")
            api_app.get_pose_meta = lambda *_args: {
                "bvh_path": base, "set_id": None,
            }
            api_app.REFINE_DIR = os.path.join(directory, "cache")
            CFG.refine_v2_enabled = True
            CFG.refine_observability_gate = False
            first = api_app.refine(request)
            assert first.refined and first.diagnostics["cache_hit"] is False

            def must_not_run(*_args, **_kwargs):
                raise AssertionError("cache miss")
            api_app.refine_bvh = must_not_run
            second = api_app.refine(request)
            assert second.refined and second.diagnostics["cache_hit"] is True
            assert second.bvh_url == first.bvh_url
        finally:
            api_app.get_pose_meta = original_meta
            api_app.REFINE_DIR = original_dir
            api_app.refine_bvh = original_solver
            CFG.refine_v2_enabled = original_flag
            CFG.refine_observability_gate = original_obs
            api_app.STATE.clear()
            api_app.STATE.update(original_state)


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
            refine_mode="conservative", cfg=cfg,
        )
        aggressive = refine_bvh(
            base, keypoints, scores, "front", out_path=aggressive_path,
            allowed_limbs=[
                "left_arm", "right_arm", "left_leg", "right_leg",
            ],
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
        except Exception:
            print(f"FAIL {function.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(functions)} passed")
    raise SystemExit(0 if passed == len(functions) else 1)
