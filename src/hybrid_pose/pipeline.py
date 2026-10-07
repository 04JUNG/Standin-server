"""Reusable v3.2 extraction, with no GT, dataset IDs, or visual re-evaluation."""

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import merge_repair, validate_evaluation
from .render import context, overlay, png_bytes
from .schema import COCO17, keypoint_bounds, validate_prediction


def normalize_rtm(skeletons, width, height, threshold):
    """Original RTM confidence is a localization filter, NOT a VLM routing gate."""
    rows = []
    for skeleton in skeletons:
        points = np.asarray(skeleton.keypoints, dtype=np.float32).reshape(17, 2)
        scores = np.asarray(skeleton.scores, dtype=np.float32).reshape(17)
        mask = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(scores)
            & (scores >= threshold)
        )
        center = float(np.median(points[mask, 0])) if mask.any() else float("inf")
        rows.append((center, points, scores, mask))
    people = []
    for pid, (_, points, scores, mask) in enumerate(sorted(rows, key=lambda r: r[0])):
        joints = []
        for j, name in enumerate(COCO17):
            joints.append(
                {
                    "name": name,
                    "x": (
                        float(np.clip(points[j, 0] / width * 1000, 0, 1000))
                        if mask[j]
                        else None
                    ),
                    "y": (
                        float(np.clip(points[j, 1] / height * 1000, 0, 1000))
                        if mask[j]
                        else None
                    ),
                    "state": "visible" if mask[j] else "not_present",
                    "score": (
                        float(scores[j]) if math.isfinite(float(scores[j])) else None
                    ),
                }
            )
        located = [p for p in joints if p["state"] != "not_present"]
        bbox = {
            "x1": min((p["x"] for p in located), default=0),
            "y1": min((p["y"] for p in located), default=0),
            "x2": max((p["x"] for p in located), default=0),
            "y2": max((p["y"] for p in located), default=0),
        }
        people.append({"person_index": pid, "bbox": bbox, "keypoints": joints})
    return validate_prediction({"people": people})


class HybridPosePipeline:
    """Inject a fresh-inference pose model and a single-attempt reviewer.

    The model can be warm/shared across requests; per-cut data remains local.
    artifact(name, value) optionally records bytes/JSON for human inspection.
    Failure returns final=None plus the preserved RTM proposal, never a pass.
    """

    def __init__(self, pose_model, reviewer, threshold=0.3):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be finite within 0..1")
        self.pose_model = pose_model
        self.reviewer = reviewer
        self.threshold = threshold

    def extract(self, image_path, artifact=None):
        started = time.perf_counter()
        save = artifact if artifact is not None else lambda name, value: None
        result = {
            "protocol": "hybrid-evaluate-repair-v3.2",
            "status": "error",
            "delivery_status": "incomplete",
            "quality_review": "unreviewed",
            "visual_reassessment": False,
            "rtm": None,
            "evaluation": None,
            "repair": None,
            "final": None,
            "audit": None,
            "api_calls_started": 0,
            "timings_ms": {},
            "reviewer": copy.deepcopy(getattr(self.reviewer, "metadata", {})),
            "rtm_localization_threshold": self.threshold,
        }
        stage = "input"
        stage_started = started
        try:
            with Image.open(Path(image_path)) as source:
                image = source.convert("RGB")
            width, height = image.size
            original = png_bytes(image)
            result["input"] = {
                "width": width,
                "height": height,
                "normalized_png_sha256": hashlib.sha256(original).hexdigest(),
            }
            save("original.png", original)
            result["timings_ms"][stage] = (time.perf_counter() - stage_started) * 1000
            stage, stage_started = "rtmpose", time.perf_counter()
            # RTMPose's ndarray contract is BGR.
            bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
            base = normalize_rtm(
                self.pose_model.estimate(bgr, [], width, height),
                width,
                height,
                self.threshold,
            )
            result["rtm"] = copy.deepcopy(base)
            save("rtm.json", base)
            result["timings_ms"][stage] = (time.perf_counter() - stage_started) * 1000
            stage, stage_started = "input_preparation", time.perf_counter()
            marked = png_bytes(overlay(image, base))
            ctx = context(base, width, height)
            save("rtm-overlay.png", marked)
            save("context.json", ctx)
            result["timings_ms"][stage] = (time.perf_counter() - stage_started) * 1000

            stage, stage_started = "evaluation", time.perf_counter()
            result["api_calls_started"] += 1
            evaluation, raw = self.reviewer.request(stage, original, marked, ctx)
            save("evaluation.response.json", raw)
            validate_evaluation(evaluation, base)
            result["evaluation"] = copy.deepcopy(evaluation)
            save("evaluation.json", evaluation)
            result["timings_ms"][stage] = (time.perf_counter() - stage_started) * 1000

            if evaluation["decision"] == "repair":
                stage, stage_started = "repair", time.perf_counter()
                result["api_calls_started"] += 1
                patch, raw = self.reviewer.request(
                    stage, original, marked, ctx, evaluation
                )
                save("repair.response.json", raw)
                result["repair"] = copy.deepcopy(patch)
                save("repair.json", patch)
                result["timings_ms"][stage] = (
                    time.perf_counter() - stage_started
                ) * 1000
                stage, stage_started = "code_validation", time.perf_counter()
                final, audit = merge_repair(base, evaluation, patch)
                delivery = (
                    "repair_unverified_review_needed"
                    if audit["unresolved"] or audit["uncertainties"]
                    else "repair_unverified"
                )
            else:
                stage, stage_started = "code_validation", time.perf_counter()
                final = copy.deepcopy(base)
                audit = {
                    "changes": [],
                    "unresolved": [],
                    "uncertainties": evaluation["uncertainties"],
                    "visual_reassessment": False,
                }
                delivery = (
                    "review_needed"
                    if evaluation["decision"] == "review_needed"
                    else "rtm_kept"
                )
            validate_prediction(final)
            result.update(
                final=final, audit=audit, keypoint_bounds=keypoint_bounds(final)
            )
            save("final.json", final)
            result.update(status="ok", delivery_status=delivery)
            result["timings_ms"][stage] = (time.perf_counter() - stage_started) * 1000
        except Exception as error:
            result["timings_ms"][stage] = (time.perf_counter() - stage_started) * 1000
            # Arbitrary exception strings may contain a credential, prompt or image.
            # Only our provider adapter's deliberately safe message is exposed.
            from .gemini import GeminiStageError

            result.update(
                status="error",
                delivery_status="incomplete",
                final=None,
                error={
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": (
                        str(error)
                        if isinstance(error, GeminiStageError)
                        else "Stage failed; no automatic retry or success fallback."
                    ),
                },
            )
        finally:
            result["timings_ms"]["total"] = (time.perf_counter() - started) * 1000
            result["within_20_seconds"] = (
                result["status"] == "ok" and result["timings_ms"]["total"] <= 20000
            )
        return result
