"""Immutable pose-model bundle contract used by opt-in canary adapters.

The current-X default path does not depend on this module. Candidate models use
it to bind the ONNX files, preprocessing/decoder semantics, runtime versions,
and score calibration into one fail-closed rollback unit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any


class PoseContractError(RuntimeError):
    """A candidate bundle does not match its declared runtime contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _required(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise PoseContractError(f"missing {where}.{key}")
    return mapping[key]


def _resolve_artifact(manifest_path: Path, value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise PoseContractError(f"missing {label} artifact: {path}")
    return path


@dataclass(frozen=True)
class PoseBundle:
    manifest_path: Path
    manifest_sha256: str
    model_id: str
    build_id: str
    status: str
    license_review: str
    pose_path: Path
    pose_sha256: str
    detector_model_id: str
    detector_path: Path
    detector_sha256: str
    input_size_wh: tuple[int, int]
    detector_input_size_wh: tuple[int, int]
    calibration: dict[str, Any]
    contract: dict[str, Any]
    runtime: dict[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "build_id": self.build_id,
            "status": self.status,
            "license_review": self.license_review,
            "manifest_sha256": self.manifest_sha256,
            "pose_sha256": self.pose_sha256,
            "detector_model_id": self.detector_model_id,
            "detector_sha256": self.detector_sha256,
            "calibration_profile_id": self.calibration["profile_id"],
            "calibration_profile_sha256": self.calibration["profile_sha256"],
            "input_size_wh": list(self.input_size_wh),
            "detector_input_size_wh": list(self.detector_input_size_wh),
            "score_decoder": self.contract["score_decoder"],
            "color_order_to_pose": self.contract["color_order_to_pose"],
            "flip_test": self.contract["flip_test"],
        }


_EXPECTED_CONTRACT = {
    "task": "top_down_pose",
    "keypoint_schema": "COCO17",
    "tensor_layout": "NCHW",
    "tensor_dtype": "float32",
    "color_order_at_api": "BGR",
    "color_order_to_pose": "RGB",
    "bbox_format": "xyxy",
    "preprocess_owner": "rtmlib-explicit-rgb-v1",
    "postprocess_owner": "rtmlib",
    "score_decoder": "rtmlib-average-v1",
    "simcc_split_ratio": 2.0,
    "flip_test": False,
}


def load_pose_bundle(path: str | Path, *, expected_model_id: str) -> PoseBundle:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise PoseContractError(f"pose manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseContractError(f"invalid pose manifest: {manifest_path}") from exc
    if payload.get("schema_version") != 1:
        raise PoseContractError("pose manifest schema_version must be 1")
    model_id = str(_required(payload, "model_id", "manifest"))
    if model_id != expected_model_id:
        raise PoseContractError(
            f"pose model identity mismatch: requested={expected_model_id}, manifest={model_id}"
        )
    status = str(_required(payload, "status", "manifest"))
    if status not in {"candidate", "verified", "shadow", "canary", "promoted"}:
        raise PoseContractError(f"unsupported pose bundle status: {status}")

    source = _required(payload, "source", "manifest")
    artifact = _required(payload, "artifact", "manifest")
    detector = _required(payload, "detector", "manifest")
    contract = _required(payload, "contract", "manifest")
    runtime = _required(payload, "runtime", "manifest")
    calibration = _required(payload, "calibration", "manifest")
    if not all(isinstance(item, dict) for item in (
        source, artifact, detector, contract, runtime, calibration,
    )):
        raise PoseContractError("pose manifest sections must be objects")

    for key, expected in _EXPECTED_CONTRACT.items():
        actual = _required(contract, key, "contract")
        if actual != expected:
            raise PoseContractError(
                f"pose contract mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )
    input_size = tuple(_required(contract, "input_size_wh", "contract"))
    tensor_shape = list(_required(contract, "tensor_shape", "contract"))
    if input_size != (192, 256) or tensor_shape != [1, 3, 256, 192]:
        raise PoseContractError(
            f"Human-Art input signature mismatch: wh={input_size}, tensor={tensor_shape}"
        )
    expected_outputs = [
        {"name": "simcc_x", "shape": ["batch", 17, 384], "dtype": "float32"},
        {"name": "simcc_y", "shape": ["batch", 17, 512], "dtype": "float32"},
    ]
    if contract.get("observed_outputs") != expected_outputs:
        raise PoseContractError("Human-Art output signature is not approved raw SimCC")

    pose_path = _resolve_artifact(
        manifest_path, str(_required(artifact, "path", "artifact")), "pose"
    )
    detector_path = _resolve_artifact(
        manifest_path, str(_required(detector, "path", "detector")), "detector"
    )
    pose_sha = str(_required(artifact, "sha256", "artifact"))
    detector_sha = str(_required(detector, "sha256", "detector"))
    observed_pose_sha = sha256_file(pose_path)
    observed_detector_sha = sha256_file(detector_path)
    if observed_pose_sha != pose_sha:
        raise PoseContractError(
            f"pose artifact hash mismatch: expected={pose_sha}, actual={observed_pose_sha}"
        )
    if observed_detector_sha != detector_sha:
        raise PoseContractError(
            "detector artifact hash mismatch: "
            f"expected={detector_sha}, actual={observed_detector_sha}"
        )

    for package in ("onnxruntime", "rtmlib"):
        expected = str(_required(runtime, package, "runtime"))
        actual = installed_version(package)
        if expected != actual:
            raise PoseContractError(
                f"runtime mismatch for {package}: expected={expected}, actual={actual}"
            )
    if runtime.get("execution_provider") != "CPUExecutionProvider":
        raise PoseContractError("only CPUExecutionProvider is approved for this build")

    required_calibration = {
        "profile_id", "profile_sha256", "skeleton_kpt_threshold",
        "min_skeleton_score", "distance_metric", "fallback_pos_full",
        "fallback_pos_reduced", "fallback_angle_full", "fallback_angle_reduced",
        "fallback_hybrid_full", "fallback_hybrid_reduced",
    }
    missing = sorted(required_calibration - set(calibration))
    if missing:
        raise PoseContractError(f"missing calibration fields: {', '.join(missing)}")

    return PoseBundle(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        model_id=model_id,
        build_id=str(_required(payload, "build_id", "manifest")),
        status=status,
        license_review=str(_required(source, "license_review", "source")),
        pose_path=pose_path,
        pose_sha256=pose_sha,
        detector_model_id=str(_required(detector, "model_id", "detector")),
        detector_path=detector_path,
        detector_sha256=detector_sha,
        input_size_wh=(int(input_size[0]), int(input_size[1])),
        detector_input_size_wh=tuple(map(int, _required(
            detector, "input_size_wh", "detector"
        ))),
        calibration=dict(calibration),
        contract=dict(contract),
        runtime=dict(runtime),
    )


def validate_calibration_against_config(bundle: PoseBundle, cfg) -> None:
    """Prevent model-only or threshold-only rollout/rollback."""
    mapping = {
        "skeleton_kpt_threshold": "skeleton_kpt_threshold",
        "min_skeleton_score": "min_skeleton_score",
        "distance_metric": "distance_metric",
        "fallback_pos_full": "fallback_pos_full",
        "fallback_pos_reduced": "fallback_pos_reduced",
        "fallback_angle_full": "fallback_angle_full",
        "fallback_angle_reduced": "fallback_angle_reduced",
        "fallback_hybrid_full": "fallback_hybrid_full",
        "fallback_hybrid_reduced": "fallback_hybrid_reduced",
    }
    mismatches = []
    for manifest_key, config_attr in mapping.items():
        expected = bundle.calibration[manifest_key]
        actual = getattr(cfg, config_attr)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            matches = abs(float(expected) - float(actual)) <= 1e-12
        else:
            matches = expected == actual
        if not matches:
            mismatches.append(f"{config_attr}: manifest={expected}, runtime={actual}")
    if mismatches:
        raise PoseContractError(
            "pose calibration profile does not match runtime config: "
            + "; ".join(mismatches)
        )


def validate_rescue_calibration(bundle: PoseBundle) -> None:
    """Validate the manifest-owned subset used by the cascade fallback.

    A direct Human-Art deployment owns the whole runtime calibration and must
    exactly match ``CFG``.  A cascade deliberately keeps current-X search
    calibration while using Human-Art's own keypoint threshold for candidate
    acceptance, so equality with the primary profile would be incorrect.
    """
    try:
        threshold = float(bundle.calibration["skeleton_kpt_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PoseContractError(
            "Human-Art rescue calibration has an invalid skeleton threshold"
        ) from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise PoseContractError(
            "Human-Art rescue skeleton threshold must be finite and non-negative"
        )
    profile_id = bundle.calibration.get("profile_id")
    profile_sha256 = bundle.calibration.get("profile_sha256")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise PoseContractError("Human-Art rescue calibration profile_id is invalid")
    if not isinstance(profile_sha256, str) or not profile_sha256.strip():
        raise PoseContractError(
            "Human-Art rescue calibration profile_sha256 is invalid"
        )


def decode_simcc_scores(
    simcc_x, simcc_y, *, decoder: str = "rtmlib-average-v1",
):
    """Small executable specification for the approved score decoder."""
    import numpy as np

    x = np.asarray(simcc_x, dtype=np.float32)
    y = np.asarray(simcc_y, dtype=np.float32)
    if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
        raise PoseContractError(
            f"invalid SimCC shapes: x={x.shape}, y={y.shape}"
        )
    max_x = x.max(axis=2)
    max_y = y.max(axis=2)
    if decoder == "rtmlib-average-v1":
        return (max_x + max_y) * 0.5
    if decoder == "mmpose-min-v1":
        return np.minimum(max_x, max_y)
    raise PoseContractError(f"unsupported SimCC score decoder: {decoder}")
