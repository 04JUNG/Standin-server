"""Fail-closed provisioning contract for remote Human-Art model bundles."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.app as api_app
from src.config import CFG
from src.pose_contract import installed_version, load_pose_bundle
from src.pose_model_source import (
    PoseModelFetchError,
    PoseModelProvisioningResult,
    _copy_limited,
    _download_s3,
    ensure_pose_model,
)


_MANIFEST_URI = (
    "s3://assets/pose-models/humanart-m/"
    "20260828.codex1-fp32-cpu-raw-simcc/manifest.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    pose: bytes,
    detector: bytes,
    *,
    pose_path: str = "model.onnx",
    pose_sha: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "model_id": "humanart-m",
        "build_id": "20260828.codex1-fp32-cpu-raw-simcc",
        "status": "candidate",
        "source": {"license_review": "pending"},
        "artifact": {
            "path": pose_path,
            "sha256": pose_sha or _sha(pose),
            "size_bytes": len(pose),
        },
        "detector": {
            "model_id": "yolox-x-humanart",
            "path": "detector.onnx",
            "sha256": _sha(detector),
            "size_bytes": len(detector),
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
                {
                    "name": "simcc_x",
                    "shape": ["batch", 17, 384],
                    "dtype": "float32",
                },
                {
                    "name": "simcc_y",
                    "shape": ["batch", 17, 512],
                    "dtype": "float32",
                },
            ],
        },
        "runtime": {
            "onnxruntime": installed_version("onnxruntime"),
            "rtmlib": installed_version("rtmlib"),
            "execution_provider": "CPUExecutionProvider",
        },
        "calibration": {
            "profile_id": "humanart-test",
            "profile_sha256": "test-profile-sha",
            "skeleton_kpt_threshold": 0.35,
            "min_skeleton_score": 0.2,
            "distance_metric": "pos",
            "fallback_pos_full": 0.45,
            "fallback_pos_reduced": 0.45,
            "fallback_angle_full": 0.45,
            "fallback_angle_reduced": 0.45,
            "fallback_hybrid_full": 0.45,
            "fallback_hybrid_reduced": 0.45,
        },
    }


def _objects(manifest: dict, pose: bytes, detector: bytes) -> dict[str, bytes]:
    prefix = _MANIFEST_URI.rsplit("/", 1)[0]
    return {
        _MANIFEST_URI: json.dumps(manifest).encode("utf-8"),
        f"{prefix}/model.onnx": pose,
        f"{prefix}/detector.onnx": detector,
    }


def _downloader(objects: dict[str, bytes], calls: list[str]):
    def download(uri, destination, *, max_bytes, expected_size, deadline):
        calls.append(uri)
        payload = objects[uri]
        with destination.open("wb") as output:
            _copy_limited(
                io.BytesIO(payload),
                output,
                max_bytes=max_bytes,
                expected_size=expected_size,
                deadline=deadline,
            )

    return download


def test_model_download_stream_enforces_declared_expected_and_actual_sizes() -> None:
    invalid = (
        {"declared_size": 5, "expected_size": 4, "max_bytes": 10},
        {"declared_size": 4, "expected_size": 4, "max_bytes": 3},
        {"declared_size": None, "expected_size": 4, "max_bytes": 10},
    )
    for options in invalid:
        try:
            _copy_limited(io.BytesIO(b"12345"), io.BytesIO(), **options)
        except PoseModelFetchError:
            continue
        raise AssertionError(f"invalid model download size passed: {options}")


def test_model_download_stream_enforces_total_deadline() -> None:
    try:
        _copy_limited(
            io.BytesIO(b"model"),
            io.BytesIO(),
            max_bytes=10,
            deadline=time.monotonic() - 1.0,
        )
    except PoseModelFetchError as exc:
        assert "시간 예산" in str(exc)
    else:
        raise AssertionError("expired model download budget was accepted")


def test_copy_limited_never_mutates_botocore_streaming_body_socket() -> None:
    class BufferedStreamingBody(io.BytesIO):
        def set_socket_timeout(self, _timeout):
            raise AttributeError("'NoneType' object has no attribute 'raw'")

    output = io.BytesIO()
    copied = _copy_limited(
        BufferedStreamingBody(b"manifest"),
        output,
        max_bytes=32,
        expected_size=8,
        deadline=time.monotonic() + 10.0,
    )
    assert copied == 8
    assert output.getvalue() == b"manifest"


def test_s3_download_configures_client_timeouts_and_no_hidden_retries() -> None:
    captured: dict = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeClient:
        def get_object(self, *, Bucket, Key):
            captured["bucket"] = Bucket
            captured["key"] = Key
            return {"Body": io.BytesIO(b"model"), "ContentLength": 5}

    fake_boto3 = types.ModuleType("boto3")

    def client(service, *, config):
        captured["service"] = service
        captured["client_config"] = config
        return FakeClient()

    fake_boto3.client = client
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = FakeConfig
    previous_boto3 = sys.modules.get("boto3")
    previous_botocore_config = sys.modules.get("botocore.config")
    try:
        sys.modules["boto3"] = fake_boto3
        sys.modules["botocore.config"] = fake_botocore_config
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.onnx"
            _download_s3(
                "s3://assets/prefix/model.onnx",
                destination,
                max_bytes=10,
                expected_size=5,
                deadline=time.monotonic() + 10.0,
            )
            assert destination.read_bytes() == b"model"
    finally:
        if previous_boto3 is None:
            sys.modules.pop("boto3", None)
        else:
            sys.modules["boto3"] = previous_boto3
        if previous_botocore_config is None:
            sys.modules.pop("botocore.config", None)
        else:
            sys.modules["botocore.config"] = previous_botocore_config

    assert captured["service"] == "s3"
    assert captured["bucket"] == "assets"
    assert captured["key"] == "prefix/model.onnx"
    assert 0 < captured["config"]["connect_timeout"] <= 10.0
    assert 0 < captured["config"]["read_timeout"] <= 10.0
    assert captured["config"]["retries"]["total_max_attempts"] == 1


def test_remote_bundle_is_verified_and_atomically_published() -> None:
    pose = b"humanart-pose"
    detector = b"humanart-detector"
    payload = _manifest(pose, detector)
    objects = _objects(payload, pose, detector)
    calls: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        result = ensure_pose_model(
            _MANIFEST_URI,
            directory,
            downloader=_downloader(objects, calls),
        )
        assert result.fetched is True
        assert result.manifest_path == (
            Path(directory).resolve()
            / "humanart-m"
            / payload["build_id"]
            / "manifest.json"
        )
        assert result.manifest_path.is_file()
        assert result.elapsed_seconds >= 0
        assert (result.manifest_path.parent / "model.onnx").read_bytes() == pose
        assert (result.manifest_path.parent / "detector.onnx").read_bytes() == detector
        bundle = load_pose_bundle(result.manifest_path, expected_model_id="humanart-m")
        assert bundle.pose_sha256 == _sha(pose)
        assert calls == [
            _MANIFEST_URI,
            _MANIFEST_URI.rsplit("/", 1)[0] + "/model.onnx",
            _MANIFEST_URI.rsplit("/", 1)[0] + "/detector.onnx",
        ]


def test_existing_immutable_build_is_revalidated_without_redownloading_onnx() -> None:
    pose = b"pose"
    detector = b"detector"
    objects = _objects(_manifest(pose, detector), pose, detector)
    with tempfile.TemporaryDirectory() as directory:
        first_calls: list[str] = []
        ensure_pose_model(
            _MANIFEST_URI,
            directory,
            downloader=_downloader(objects, first_calls),
        )
        second_calls: list[str] = []
        result = ensure_pose_model(
            _MANIFEST_URI,
            directory,
            downloader=_downloader(objects, second_calls),
        )
        assert result.fetched is False
        assert second_calls == [_MANIFEST_URI]


def test_bad_hash_never_publishes_a_partial_build() -> None:
    pose = b"pose"
    detector = b"detector"
    payload = _manifest(pose, detector, pose_sha="0" * 64)
    objects = _objects(payload, pose, detector)
    with tempfile.TemporaryDirectory() as directory:
        try:
            ensure_pose_model(
                _MANIFEST_URI,
                directory,
                downloader=_downloader(objects, []),
            )
        except PoseModelFetchError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("bad model hash was accepted")
        assert not (Path(directory) / "humanart-m" / payload["build_id"]).exists()


def test_remote_manifest_rejects_absolute_or_traversing_artifact_paths() -> None:
    for unsafe in ("/tmp/model.onnx", "../model.onnx", "nested/model.onnx"):
        pose = b"pose"
        detector = b"detector"
        payload = _manifest(pose, detector, pose_path=unsafe)
        objects = {_MANIFEST_URI: json.dumps(payload).encode("utf-8")}
        with tempfile.TemporaryDirectory() as directory:
            try:
                ensure_pose_model(
                    _MANIFEST_URI,
                    directory,
                    downloader=_downloader(objects, []),
                )
            except PoseModelFetchError as exc:
                assert "단일 상대 파일명" in str(exc)
            else:
                raise AssertionError(f"unsafe artifact path was accepted: {unsafe}")


def test_startup_uses_provisioned_local_manifest_without_mutating_env() -> None:
    previous = (
        CFG.pose_model_variant,
        CFG.pose_model_uri,
        CFG.pose_models_root,
        CFG.pose_model_download_budget_seconds,
        CFG.pose_model_manifest,
    )
    original = api_app.ensure_pose_model
    original_env = os.environ.get("POSE_MODEL_MANIFEST")
    try:
        CFG.pose_model_variant = "cascade"
        CFG.pose_model_uri = _MANIFEST_URI
        CFG.pose_models_root = "/models"
        CFG.pose_model_download_budget_seconds = 123.0
        CFG.pose_model_manifest = ""
        expected = Path("/models/humanart-m/build/manifest.json")
        captured: dict = {}

        def provision(*args, **kwargs):
            captured.update(kwargs)
            return PoseModelProvisioningResult(
                expected, "humanart-m", "build", True, 1.25
            )

        api_app.ensure_pose_model = provision
        result = api_app._ensure_pose_model_bundle()
        assert result is not None
        assert CFG.pose_model_manifest == str(expected)
        assert captured["total_budget_seconds"] == 123.0
        assert os.environ.get("POSE_MODEL_MANIFEST") == original_env
    finally:
        api_app.ensure_pose_model = original
        (
            CFG.pose_model_variant,
            CFG.pose_model_uri,
            CFG.pose_models_root,
            CFG.pose_model_download_budget_seconds,
            CFG.pose_model_manifest,
        ) = previous


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
