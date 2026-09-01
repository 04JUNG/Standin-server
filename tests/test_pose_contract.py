"""Self-contained tests for the opt-in pose bundle/decoder contract."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG
from src.descriptor import build_descriptors
from src.pose import MockPoseModel, build_pose_model
from src.pose_contract import (
    PoseContractError,
    decode_simcc_scores,
    installed_version,
    load_pose_bundle,
    validate_calibration_against_config,
    validate_rescue_calibration,
)
from src.schema import Action, BBox, Relationship, Shot, Skeleton, VLMAnalysis, View
from src.runtime_guard import MockBackendError, ensure_production_backends


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(pose_bytes=b"pose", detector_bytes=b"detector") -> dict:
    return {
        "schema_version": 1,
        "model_id": "humanart-m",
        "build_id": "test-build",
        "status": "candidate",
        "source": {"license_review": "pending"},
        "artifact": {"path": "pose.onnx", "sha256": _sha(pose_bytes)},
        "detector": {
            "model_id": "yolox-x-humanart",
            "path": "detector.onnx",
            "sha256": _sha(detector_bytes),
            "input_size_wh": [640, 640],
        },
        "contract": {
            "task": "top_down_pose",
            "keypoint_schema": "COCO17",
            "input_size_wh": [192, 256],
            "tensor_layout": "NCHW",
            "tensor_dtype": "float32",
            "tensor_shape": [1, 3, 256, 192],
            "color_order_at_api": "BGR",
            "color_order_to_pose": "RGB",
            "bbox_format": "xyxy",
            "preprocess_owner": "rtmlib-explicit-rgb-v1",
            "postprocess_owner": "rtmlib",
            "score_decoder": "rtmlib-average-v1",
            "simcc_split_ratio": 2.0,
            "flip_test": False,
            "observed_outputs": [
                {"name": "simcc_x", "shape": ["batch", 17, 384], "dtype": "float32"},
                {"name": "simcc_y", "shape": ["batch", 17, 512], "dtype": "float32"},
            ],
        },
        "runtime": {
            "onnxruntime": installed_version("onnxruntime"),
            "rtmlib": installed_version("rtmlib"),
            "execution_provider": "CPUExecutionProvider",
        },
        "calibration": {
            "profile_id": "test-profile",
            "profile_sha256": "test-profile-sha",
            "skeleton_kpt_threshold": CFG.skeleton_kpt_threshold,
            "min_skeleton_score": CFG.min_skeleton_score,
            "distance_metric": CFG.distance_metric,
            "fallback_pos_full": CFG.fallback_pos_full,
            "fallback_pos_reduced": CFG.fallback_pos_reduced,
            "fallback_angle_full": CFG.fallback_angle_full,
            "fallback_angle_reduced": CFG.fallback_angle_reduced,
            "fallback_hybrid_full": CFG.fallback_hybrid_full,
            "fallback_hybrid_reduced": CFG.fallback_hybrid_reduced,
        },
    }


def test_bundle_hash_and_calibration_are_atomic():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "pose.onnx").write_bytes(b"pose")
        (root / "detector.onnx").write_bytes(b"detector")
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
        bundle = load_pose_bundle(manifest_path, expected_model_id="humanart-m")
        validate_calibration_against_config(bundle, CFG)
        assert bundle.pose_sha256 == _sha(b"pose")
        assert bundle.identity()["score_decoder"] == "rtmlib-average-v1"

        payload = _manifest()
        payload["artifact"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_pose_bundle(manifest_path, expected_model_id="humanart-m")
        except PoseContractError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("bad artifact hash must fail closed")


def test_cascade_rescue_calibration_is_manifest_owned_but_direct_stays_exact():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "pose.onnx").write_bytes(b"pose")
        (root / "detector.onnx").write_bytes(b"detector")
        payload = _manifest()
        payload["calibration"]["skeleton_kpt_threshold"] = 0.35
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        bundle = load_pose_bundle(manifest_path, expected_model_id="humanart-m")
        validate_rescue_calibration(bundle)
        try:
            validate_calibration_against_config(bundle, CFG)
        except PoseContractError as exc:
            assert "skeleton_kpt_threshold" in str(exc)
        else:
            raise AssertionError(
                "direct Human-Art calibration mismatch must remain fail-closed"
            )


def test_decoder_contract_distinguishes_rtmlib_and_mmpose_scores():
    x = np.array([[[0.1, 0.8], [0.4, 0.2]]], dtype=np.float32)
    y = np.array([[[0.3, 0.6], [0.9, 0.1]]], dtype=np.float32)
    average = decode_simcc_scores(x, y, decoder="rtmlib-average-v1")
    minimum = decode_simcc_scores(x, y, decoder="mmpose-min-v1")
    assert np.allclose(average, [[0.7, 0.65]])
    assert np.allclose(minimum, [[0.6, 0.4]])


def test_descriptor_uses_runtime_threshold_single_source():
    old_threshold = CFG.skeleton_kpt_threshold
    try:
        CFG.skeleton_kpt_threshold = 0.2
        keypoints = np.arange(34, dtype=np.float32).reshape(17, 2)
        scores = np.full(17, 0.25, dtype=np.float32)
        skeleton = Skeleton(keypoints, scores)
        vlm = VLMAnalysis(
            1, Shot.FULL_HALF, Action.OTHER, View.FRONT, Relationship.SOLO,
            [BBox(0, 0, 100, 200, source="vlm")],
        )
        descriptor = build_descriptors(vlm, [skeleton], vlm.approx_boxes)[0]
        assert descriptor.feature is not None
        assert np.count_nonzero(descriptor.feature) > 0
    finally:
        CFG.skeleton_kpt_threshold = old_threshold


def test_humanart_factory_is_default_off_and_strict():
    old = (
        CFG.pose_model_variant, CFG.pose_backend, CFG.pose_strict,
        CFG.pose_model_manifest, CFG.pose_canary_stage,
    )
    try:
        CFG.pose_model_variant = "current-x"
        CFG.pose_backend = "mock"
        CFG.pose_strict = False
        assert isinstance(build_pose_model(), MockPoseModel)

        CFG.pose_model_variant = "humanart-m"
        CFG.pose_backend = "rtmlib"
        CFG.pose_strict = True
        CFG.pose_model_manifest = ""
        CFG.pose_canary_stage = "off"
        try:
            build_pose_model()
        except PoseContractError as exc:
            assert "POSE_MODEL_MANIFEST" in str(exc)
        else:
            raise AssertionError("strict Human-Art without a manifest must fail")
    finally:
        (
            CFG.pose_model_variant, CFG.pose_backend, CFG.pose_strict,
            CFG.pose_model_manifest, CFG.pose_canary_stage,
        ) = old


def test_cascade_factory_bad_manifest_is_strict_and_never_mocks():
    old = (
        CFG.pose_model_variant, CFG.pose_backend, CFG.pose_strict,
        CFG.pose_model_manifest, CFG.pose_canary_stage,
    )
    try:
        CFG.pose_model_variant = "cascade"
        CFG.pose_backend = "rtmlib"
        CFG.pose_strict = True
        CFG.pose_model_manifest = "/definitely/missing/humanart-manifest.json"
        CFG.pose_canary_stage = "shadow"
        try:
            build_pose_model()
        except PoseContractError as exc:
            assert "manifest not found" in str(exc)
        else:
            raise AssertionError("strict cascade with a bad manifest must fail closed")
    finally:
        (
            CFG.pose_model_variant, CFG.pose_backend, CFG.pose_strict,
            CFG.pose_model_manifest, CFG.pose_canary_stage,
        ) = old


def test_production_guard_requires_candidate_license_and_status():
    class RealVLM:
        pass

    class CandidatePose:
        def __init__(self, license_review="pending", status="candidate"):
            self.license_review = license_review
            self.status = status

        def runtime_identity(self):
            return {
                "model_id": "humanart-m",
                "build_id": "test",
                "backend": "rtmlib",
                "license_review": self.license_review,
                "status": self.status,
            }

    class Pipe:
        vlm = RealVLM()
        pose = CandidatePose()

    try:
        ensure_production_backends(
            Pipe(), is_production=True, requested_vlm="real",
            requested_pose="rtmlib", requested_pose_variant="humanart-m",
        )
    except MockBackendError as exc:
        assert "license" in str(exc)
    else:
        raise AssertionError("pending license must fail in production")

    Pipe.pose = CandidatePose("approved", "canary")
    ensure_production_backends(
        Pipe(), is_production=True, requested_vlm="real",
        requested_pose="rtmlib", requested_pose_variant="humanart-m",
    )


def test_production_guard_accepts_verified_cascade_contract():
    class RealVLM:
        pass

    class CascadePose:
        def runtime_identity(self):
            return {
                "model_id": "cascade",
                "build_id": "current-x->humanart-test",
                "backend": "rtmlib",
                "canary_stage": "shadow",
                "fallback_contract_ready": True,
                "primary": {"model_id": "current-x"},
                "fallback": {
                    "model_id": "humanart-m",
                    "license_review": "approved",
                    "status": "shadow",
                },
            }

    class Pipe:
        vlm = RealVLM()
        pose = CascadePose()

    ensure_production_backends(
        Pipe(), is_production=True, requested_vlm="real",
        requested_pose="rtmlib", requested_pose_variant="cascade",
    )


def test_production_guard_rejects_unapproved_cascade_contract():
    class RealVLM:
        pass

    class CascadePose:
        def __init__(self, *, license_review="pending", status="candidate"):
            self.license_review = license_review
            self.status = status

        def runtime_identity(self):
            return {
                "model_id": "cascade",
                "build_id": "current-x->humanart-test",
                "backend": "rtmlib",
                "canary_stage": "shadow",
                "fallback_contract_ready": True,
                "primary": {"model_id": "current-x"},
                "fallback": {
                    "model_id": "humanart-m",
                    "license_review": self.license_review,
                    "status": self.status,
                },
            }

    class Pipe:
        vlm = RealVLM()
        pose = CascadePose()

    for pose, expected_message in (
        (CascadePose(license_review="pending", status="shadow"), "license"),
        (CascadePose(license_review="approved", status="candidate"), "shadow/canary/promoted"),
    ):
        Pipe.pose = pose
        try:
            ensure_production_backends(
                Pipe(), is_production=True, requested_vlm="real",
                requested_pose="rtmlib", requested_pose_variant="cascade",
            )
        except MockBackendError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError("unapproved cascade contract must fail in production")


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
