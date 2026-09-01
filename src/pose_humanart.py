"""Default-off Human-Art M ONNX canary adapter.

The detector consumes OpenCV BGR. The pose model receives an explicit RGB
conversion and raw-SimCC ONNX outputs are decoded by the pinned rtmlib runtime.
No request-level fallback to current-X or mock is performed here.
"""
from __future__ import annotations

from threading import RLock
from typing import Optional

import numpy as np

from .config import CFG
from .pose import BasePoseModel, _load_bgr
from .pose_contract import (
    PoseBundle,
    PoseContractError,
    load_pose_bundle,
    validate_calibration_against_config,
    validate_rescue_calibration,
)
from .schema import BBox, Skeleton


class HumanArtPoseModel(BasePoseModel):
    """Human-Art YOLOX + RTMPose-M experimental deployment bundle."""

    self_detecting = True
    model_id = "humanart-m"

    def __init__(self, manifest_path: str | None = None, *,
                 bundle: PoseBundle | None = None,
                 calibration_owner: str = "runtime",
                 shared_detector=None):
        manifest_path = manifest_path or CFG.pose_model_manifest
        if bundle is None and not manifest_path:
            raise PoseContractError(
                "POSE_MODEL_MANIFEST is required for POSE_MODEL_VARIANT=humanart-m"
            )
        if CFG.pose_canary_stage not in {
            "shadow", "canary-5", "canary-25", "canary-50", "canary-100",
        }:
            raise PoseContractError(
                "Human-Art is default-off; set POSE_CANARY_STAGE to an approved "
                "shadow/canary stage"
            )
        self.bundle = bundle or load_pose_bundle(
            manifest_path, expected_model_id=self.model_id
        )
        if calibration_owner == "runtime":
            validate_calibration_against_config(self.bundle, CFG)
        elif calibration_owner == "manifest-rescue":
            validate_rescue_calibration(self.bundle)
        else:
            raise PoseContractError(
                f"unsupported Human-Art calibration owner: {calibration_owner}"
            )
        self.calibration_owner = calibration_owner

        from rtmlib import RTMPose

        if shared_detector is None:
            from rtmlib import YOLOX
            self.detector = YOLOX(
                str(self.bundle.detector_path),
                model_input_size=self.bundle.detector_input_size_wh,
                backend="onnxruntime",
                device="cpu",
            )
            self.detector_session_owner = "humanart-m"
        else:
            self.detector = shared_detector
            self.detector_session_owner = "current-x-shared"
        self.pose = RTMPose(
            str(self.bundle.pose_path),
            model_input_size=self.bundle.input_size_wh,
            backend="onnxruntime",
            device="cpu",
        )
        self._lock = RLock()
        self._validate_live_signature()

    def _validate_live_signature(self) -> None:
        session = self.pose.session
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise PoseContractError(f"expected one pose input, got {len(inputs)}")
        if [item.name for item in outputs] != ["simcc_x", "simcc_y"]:
            raise PoseContractError(
                f"unexpected pose output names: {[item.name for item in outputs]}"
            )
        width, height = self.bundle.input_size_wh
        probe = np.zeros((1, 3, height, width), dtype=np.float32)
        observed = session.run(
            [item.name for item in outputs], {inputs[0].name: probe}
        )
        shapes = [tuple(value.shape) for value in observed]
        if shapes != [(1, 17, 384), (1, 17, 512)]:
            raise PoseContractError(f"unexpected live raw-SimCC shapes: {shapes}")
        if not all(np.isfinite(value).all() for value in observed):
            raise PoseContractError("non-finite output from pose startup probe")

    @staticmethod
    def _pose_rgb(image_bgr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image_bgr[:, :, ::-1])

    @staticmethod
    def _skeletons(keypoints, scores, offset=None) -> list[Skeleton]:
        kp = np.asarray(keypoints, dtype=np.float32)
        sc = np.asarray(scores, dtype=np.float32)
        if kp.ndim != 3 or kp.shape[1:] != (17, 2):
            raise PoseContractError(f"invalid keypoint output shape: {kp.shape}")
        if sc.shape != kp.shape[:2]:
            raise PoseContractError(f"invalid score output shape: {sc.shape}")
        if offset is not None:
            kp = kp + np.asarray(offset, dtype=np.float32)
        return [Skeleton(points.copy(), score.copy()) for points, score in zip(kp, sc)]

    def estimate(self, image, boxes, img_w: int, img_h: int) -> list[Skeleton]:
        bgr = _load_bgr(image)
        with self._lock:
            detected = self.detector(bgr)
        return self._estimate_bgr_with_detections(bgr, detected)

    def _estimate_bgr_with_detections(
        self, bgr: np.ndarray, detected,
    ) -> list[Skeleton]:
        boxes = np.asarray(detected, dtype=np.float32)
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise PoseContractError(
                f"invalid shared detector boxes: shape={boxes.shape}"
            )
        with self._lock:
            keypoints, scores = self.pose(
                self._pose_rgb(bgr), bboxes=boxes
            )
        return self._skeletons(keypoints, scores)

    def estimate_with_detections(
        self, image, detected, img_w: int, img_h: int,
    ) -> list[Skeleton]:
        """Run only Human-Art M pose using current-X's identical YOLOX boxes."""
        return self._estimate_bgr_with_detections(_load_bgr(image), detected)

    def estimate_crop_candidates(
        self, image, box: BBox, img_w: int, img_h: int,
    ) -> list[Skeleton]:
        bgr = _load_bgr(image)
        h, w = bgr.shape[:2]
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        padding = CFG.slot_crop_padding
        x1 = max(0, int(box.x1 - padding * bw))
        y1 = max(0, int(box.y1 - padding * bh))
        x2 = min(w, int(box.x2 + padding * bw))
        y2 = min(h, int(box.y2 + padding * bh))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        with self._lock:
            detected = self.detector(crop)
            if len(detected) == 0:
                return []
            keypoints, scores = self.pose(
                self._pose_rgb(crop), bboxes=detected
            )
        return self._skeletons(keypoints, scores, offset=[x1, y1])

    def estimate_crop(
        self, image, box: BBox, img_w: int, img_h: int,
    ) -> Optional[Skeleton]:
        candidates = self.estimate_crop_candidates(image, box, img_w, img_h)
        if not candidates:
            return None
        index = int(np.argmax([
            np.asarray(item.scores, dtype=np.float32).mean()
            for item in candidates
        ]))
        return candidates[index]

    def runtime_identity(self) -> dict:
        identity = self.bundle.identity()
        identity.update({
            "backend": "rtmlib",
            "adapter": type(self).__name__,
            "canary_stage": CFG.pose_canary_stage,
            "calibration_owner": self.calibration_owner,
            "skeleton_kpt_threshold": float(
                self.bundle.calibration["skeleton_kpt_threshold"]
            ),
            "initialized": True,
            "detector_session_owner": self.detector_session_owner,
        })
        return identity
