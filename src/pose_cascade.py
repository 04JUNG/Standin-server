"""current-X primary with a lazy Human-Art M full-image rescue fallback."""
from __future__ import annotations

from threading import RLock
from time import perf_counter
from typing import Callable
from pathlib import Path

from .config import CFG
from .pose import BasePoseModel, RTMPoseModel
from .pose_contract import (
    PoseBundle,
    PoseContractError,
    load_pose_bundle,
    sha256_file,
    validate_rescue_calibration,
)


_ACTIVE_STAGES = {
    "shadow", "canary-5", "canary-25", "canary-50", "canary-100",
}


def _component_identity(component, default_model_id: str) -> dict:
    runtime_identity = getattr(component, "runtime_identity", None)
    if callable(runtime_identity):
        identity = dict(runtime_identity())
        identity.setdefault("model_id", default_model_id)
        identity.setdefault("adapter", type(component).__name__)
        return identity
    model_id = "mock" if "mock" in type(component).__name__.lower() else default_model_id
    return {
        "model_id": model_id,
        "adapter": type(component).__name__,
        "initialized": True,
    }


class CascadePoseModel(BasePoseModel):
    """Delegate normal inference to current-X and rescue calls to Human-Art M.

    The Human-Art bundle contract is checked eagerly, while its ONNX sessions
    are initialized once on the first rescue call.  This keeps the normal path
    unchanged without allowing a bad manifest to pass production startup.
    """

    self_detecting = True
    model_id = "cascade"

    def __init__(
        self,
        *,
        primary: BasePoseModel | None = None,
        fallback_bundle: PoseBundle | None = None,
        fallback_factory: Callable[[PoseBundle], BasePoseModel] | None = None,
        canary_stage: str | None = None,
    ):
        self.canary_stage = (canary_stage or CFG.pose_canary_stage).strip().lower()
        if self.canary_stage not in _ACTIVE_STAGES:
            raise PoseContractError(
                "cascade is default-off; POSE_CANARY_STAGE must be shadow or canary-*"
            )
        if fallback_bundle is None:
            if not CFG.pose_model_manifest:
                raise PoseContractError(
                    "POSE_MODEL_MANIFEST is required for POSE_MODEL_VARIANT=cascade"
                )
            fallback_bundle = load_pose_bundle(
                CFG.pose_model_manifest, expected_model_id="humanart-m"
            )
        validate_rescue_calibration(fallback_bundle)
        self.bundle = fallback_bundle
        # Validate the rollback unit before allocating the primary RTM session.
        # Strict/production startup must report a bad fallback contract directly.
        self.primary = primary or RTMPoseModel()
        detector_contract_method = getattr(self.primary, "detector_contract", None)
        detector_contract = (
            detector_contract_method()
            if callable(detector_contract_method) else None
        )
        self._shared_detector = None
        if detector_contract is not None:
            primary_detector_path = Path(
                detector_contract["model_path"]
            ).resolve()
            same_artifact = primary_detector_path == self.bundle.detector_path.resolve()
            if not same_artifact and primary_detector_path.is_file():
                same_artifact = (
                    sha256_file(primary_detector_path)
                    == self.bundle.detector_sha256
                )
            same_input = (
                tuple(detector_contract["input_size_wh"])
                == tuple(self.bundle.detector_input_size_wh)
            )
            if same_artifact and same_input:
                self._shared_detector = detector_contract["component"]
        if fallback_factory is None:
            from .pose_humanart import HumanArtPoseModel

            fallback_factory = lambda bundle: HumanArtPoseModel(
                bundle=bundle, calibration_owner="manifest-rescue",
                shared_detector=self._shared_detector,
            )
        self._fallback_factory = fallback_factory
        self._fallback: BasePoseModel | None = None
        self._fallback_lock = RLock()
        self._fallback_init_error: Exception | None = None
        self._fallback_last_error: str | None = None
        self._fallback_init_ms = 0.0

    def estimate(self, image, boxes, img_w: int, img_h: int):
        return self.primary.estimate(image, boxes, img_w, img_h)

    def estimate_with_rescue_context(self, image, boxes, img_w: int, img_h: int):
        method = getattr(self.primary, "estimate_with_rescue_context", None)
        if self._shared_detector is not None and callable(method):
            return method(image, boxes, img_w, img_h)
        return self.estimate(image, boxes, img_w, img_h), None

    def estimate_crop(self, image, box, img_w: int, img_h: int):
        return self.primary.estimate_crop(image, box, img_w, img_h)

    def estimate_crop_candidates(self, image, box, img_w: int, img_h: int):
        return self.primary.estimate_crop_candidates(image, box, img_w, img_h)

    def _ensure_fallback(self) -> BasePoseModel:
        if self._fallback is not None:
            return self._fallback
        with self._fallback_lock:
            if self._fallback is not None:
                return self._fallback
            if self._fallback_init_error is not None:
                raise RuntimeError("Human-Art fallback initialization previously failed") \
                    from self._fallback_init_error
            started = perf_counter()
            try:
                fallback = self._fallback_factory(self.bundle)
            except Exception as exc:
                self._fallback_init_error = exc
                self._fallback_last_error = f"fallback_init:{type(exc).__name__}"
                raise
            finally:
                self._fallback_init_ms = (perf_counter() - started) * 1000.0
            self._fallback = fallback
            self._fallback_last_error = None
            return fallback

    def rescue_candidates(self, image, img_w: int, img_h: int):
        try:
            fallback = self._ensure_fallback()
            candidates = fallback.estimate(image, None, img_w, img_h)
        except Exception as exc:
            if self._fallback is not None:
                self._fallback_last_error = f"fallback_inference:{type(exc).__name__}"
            raise
        self._fallback_last_error = None
        return candidates

    def rescue_candidates_with_context(
        self, image, img_w: int, img_h: int, context,
    ):
        if self._shared_detector is None or not isinstance(context, dict):
            return self.rescue_candidates(image, img_w, img_h)
        expected_path = str(self.bundle.detector_path.resolve())
        if (
            context.get("detector_model_path") != expected_path
            or tuple(context.get("detector_input_size_wh", ()))
            != tuple(self.bundle.detector_input_size_wh)
        ):
            return self.rescue_candidates(image, img_w, img_h)
        try:
            fallback = self._ensure_fallback()
            candidates = fallback.estimate_with_detections(
                image, context.get("detected_bboxes"), img_w, img_h
            )
        except Exception as exc:
            if self._fallback is not None:
                self._fallback_last_error = f"fallback_inference:{type(exc).__name__}"
            raise
        self._fallback_last_error = None
        return candidates

    def fallback_kpt_threshold(self) -> float:
        return float(self.bundle.calibration["skeleton_kpt_threshold"])

    def rescue_stage(self) -> str:
        return self.canary_stage

    def fallback_init_ms(self) -> float:
        return float(self._fallback_init_ms)

    def runtime_identity(self) -> dict:
        primary = _component_identity(self.primary, "current-x")
        primary.setdefault(
            "skeleton_kpt_threshold", float(CFG.skeleton_kpt_threshold)
        )
        fallback = self.bundle.identity()
        fallback.update({
            "adapter": (
                type(self._fallback).__name__
                if self._fallback is not None else "HumanArtPoseModel"
            ),
            "initialized": self._fallback is not None,
            "calibration_owner": "manifest-rescue",
            "skeleton_kpt_threshold": self.fallback_kpt_threshold(),
        })
        return {
            "model_id": self.model_id,
            "build_id": f"current-x->{self.bundle.build_id}",
            "backend": "rtmlib",
            "adapter": type(self).__name__,
            "initialized": True,
            "canary_stage": self.canary_stage,
            "primary": primary,
            "fallback": fallback,
            "fallback_contract_ready": True,
            "fallback_initialized": self._fallback is not None,
            "fallback_last_error": self._fallback_last_error,
            "fallback_init_ms": round(float(self._fallback_init_ms), 3),
            "shared_detector_session": self._shared_detector is not None,
        }


__all__ = ["CascadePoseModel"]
