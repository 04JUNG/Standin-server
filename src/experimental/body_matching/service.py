"""Request-local observations + live catalog snapshot -> per-person body decisions."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import logging
import math
from pathlib import Path
from threading import RLock
from time import perf_counter
import uuid

from .catalog import file_sha256, load_catalog
from .observation import (GeminiBodyAttributeClient, MockBodyAttributeClient, PROMPT_VERSION,
                          geometry_ratios)
from .schema import GEOMETRY_VERSION, OBSERVATION_VERSION, SCHEMA_VERSION, parse_person, unknown_attributes
from .selection import choose_body

_INIT_LOCK = RLock()


def _image(image):
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError("body_image_unavailable")


def _box(desc, width, height):
    if desc.box is None:
        return None
    raw = desc.box.as_list()
    if not all(math.isfinite(x) for x in raw):
        return None
    x1, y1, x2, y2 = raw
    box = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2)))
    return box if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else None


def _overlapping(a, b):
    if a is None or b is None:
        return False
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return overlap / area > 0.35  # conservative ownership gate, not an accuracy threshold


class BodyMatchingService:
    def __init__(self, catalog_path, *, client=None, provider="mock", model="gemini-2.5-flash",
                 timeout_seconds=12.0, max_people=8, cache_size=128):
        self.catalog_path = str(catalog_path)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or not 1 <= max_people <= 20:
            raise ValueError("invalid_body_runtime_limits")
        if provider not in {"mock", "gemini"}:
            raise ValueError("unsupported_body_provider")
        self.provider_requested = provider
        self.init_error = None
        if client is None:
            try:
                client = GeminiBodyAttributeClient(model, timeout_seconds) if provider == "gemini" else MockBodyAttributeClient()
            except Exception as exc:
                self.init_error = "body_provider_init_" + type(exc).__name__
                client = MockBodyAttributeClient()
        self.client = client
        self.max_people = max_people
        self.cache_size = max(0, cache_size)
        self.cache = OrderedDict()
        self.lock = RLock()

    def analyze(self, image, result, *, mode="auto", pose_is_mock=False):
        if mode not in {"auto", "shadow"}:
            raise ValueError("invalid_body_matching_mode")
        start = perf_counter()
        output = {"schema_version": SCHEMA_VERSION, "mode": mode, "status": "ok", "people": [],
                  "provider_requested": self.provider_requested, "provider_actual": self.client.provider,
                  "model": self.client.model, "is_mock": self.client.is_mock,
                  "prompt_version": PROMPT_VERSION, "pose_order_preserved": True,
                  "rendering_executed": False}
        if result.route != "core":
            return {**output, "status": "not_applicable", "reason": "route_" + result.route}
        # Reload atomically published JSON for every cut: no hard-coded model list or restart.
        catalog = None
        catalog_error = None
        try:
            catalog = load_catalog(self.catalog_path)
            output.update(catalog_version=catalog.version, catalog_sha256=catalog.sha256,
                          catalog_issues=list(catalog.issues))
        except Exception as exc:
            catalog_error = "catalog_" + type(exc).__name__
        pil = None
        image_error = None
        try:
            pil = _image(image)
            input_hash = hashlib.sha256(str(pil.size).encode() + pil.tobytes()).hexdigest()
        except Exception:
            input_hash = None
            image_error = "body_image_unavailable"
        cut_id = input_hash or ("unavailable-" + uuid.uuid4().hex)
        output["input_sha256"] = input_hash
        boxes = [_box(d, *pil.size) if pil else None for d in result.descriptors]
        observations, crops, cache_keys = [], [], {}
        for index, desc in enumerate(result.descriptors):
            person_id = cut_id + ":p" + str(index)
            owned = desc.skeleton_state in {"valid", "partial"} and not any(
                _overlapping(boxes[index], b) for j, b in enumerate(boxes) if j != index)
            obs = {"schema_version": OBSERVATION_VERSION, "person_id": person_id,
                   "attributes": unknown_attributes(), "geometry": {}, "geometry_version": GEOMETRY_VERSION,
                   "clothing_occlusion": False, "foreshortening": False,
                   "ownership_ambiguous": not owned, "crop_box": list(boxes[index]) if boxes[index] else None,
                   "provider_error": self.init_error, "reason_codes": []}
            if owned and not pose_is_mock and desc.skeleton is not None and desc.valid_joint_mask is not None:
                obs["geometry"] = geometry_ratios(desc.skeleton.keypoints, desc.valid_joint_mask)
            if not owned:
                obs["reason_codes"].append("ownership_ambiguous")
            elif image_error:
                obs["reason_codes"].append(image_error)
            elif boxes[index] is None:
                obs["reason_codes"].append("invalid_crop")
            elif index >= self.max_people:
                obs["reason_codes"].append("body_batch_budget_exceeded")
            elif not self.client.is_mock:
                key = (person_id, boxes[index], self.client.provider, self.client.model, PROMPT_VERSION)
                cache_keys[person_id] = key
                with self.lock:
                    cached = self.cache.get(key)
                    if cached is not None:
                        self.cache.move_to_end(key)
                if cached is not None:
                    obs.update(json.loads(cached))
                    obs["cache_hit"] = True
                else:
                    crop = pil.crop(boxes[index])
                    crop.thumbnail((768, 768))
                    crops.append((person_id, crop))
            if self.client.is_mock:
                obs["reason_codes"].append("mock_not_visual_evidence")
            observations.append(obs)
        if crops:
            try:
                batch = self.client.analyze(crops)
                expected = {key for key, _ in crops}
                if not isinstance(batch, dict) or set(batch) != expected:
                    raise ValueError("body_batch_identity_error")
                # Validate injected clients too; do not partially accept a malformed batch.
                parsed = {key: parse_person(batch[key], key) for key in expected}
                for obs in observations:
                    key = obs["person_id"]
                    if key in parsed:
                        obs.update(parsed[key])
                        with self.lock:
                            self.cache[cache_keys[key]] = json.dumps(parsed[key])
                            self.cache.move_to_end(cache_keys[key])
                            while len(self.cache) > self.cache_size:
                                self.cache.popitem(last=False)
            except Exception as exc:
                for obs in observations:
                    if obs["person_id"] in {key for key, _ in crops}:
                        obs["provider_error"] = "body_provider_" + type(exc).__name__
        for index, obs in enumerate(observations):
            if obs["ownership_ambiguous"]:
                obs["attributes"] = unknown_attributes()
                obs["geometry"] = {}
            poses = []
            for candidate in (result.person_candidates[index] if index < len(result.person_candidates) else []):
                pose_hash = None
                try:
                    if candidate.bvh_path and Path(candidate.bvh_path).is_file():
                        pose_hash = file_sha256(candidate.bvh_path)
                except OSError:
                    obs["reason_codes"].append("pose_asset_unreadable_geometry_not_scored")
                poses.append({"pose_id": candidate.pose_id, "view": candidate.view.value, "pose_sha256": pose_hash})
            if catalog is not None:
                decision = choose_body(catalog, obs, poses)
            else:
                decision = {"auto_body_id": None, "selected_asset": None, "candidates": [],
                            "selection_source": None, "diagnostic": "catalog_error", "reason_codes": [catalog_error],
                            "acceptance_probability": None}
            selected = decision["auto_body_id"]
            person = {"person_index": index, "person_id": obs["person_id"], **decision,
                      "observations": obs, "pose_bindings": poses,
                      "applied_body_id": selected if mode == "auto" else None,
                      "selection_state": "selected" if selected else "unavailable",
                      "render_required": bool(selected and mode == "auto"),
                      "manual_change_scope": "current_cut_person"}
            output["people"].append(person)
        if catalog_error:
            output["status"] = "unavailable"
        elif any(p["selection_state"] == "unavailable" for p in output["people"]):
            output["status"] = "partial"
        output["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
        return output


def attach_body_matching(pipe, image, result, cfg):
    from ...pose import MockPoseModel
    try:
        with _INIT_LOCK:
            if getattr(pipe, "body_matcher", None) is None:
                pipe.body_matcher = BodyMatchingService(
                    cfg.body_catalog_path, provider=cfg.body_vlm_provider, model=cfg.body_vlm_model,
                    timeout_seconds=cfg.body_timeout_seconds, max_people=cfg.body_max_people)
            service = pipe.body_matcher
        result.body_matching = service.analyze(image, result, mode=cfg.body_matching_mode,
                                               pose_is_mock=isinstance(pipe.pose, MockPoseModel))
    except Exception as exc:
        logging.getLogger(__name__).warning("body_matching_failed type=%s", type(exc).__name__)
        result.body_matching = {"schema_version": SCHEMA_VERSION, "mode": cfg.body_matching_mode,
                                "status": "unavailable", "reason": "body_execution_" + type(exc).__name__,
                                "people": [], "pose_order_preserved": True, "rendering_executed": False}
    return result
