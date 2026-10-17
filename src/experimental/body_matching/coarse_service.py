"""Opt-in /analyze sidecar: family first, BASE by default, clear traits override.

Recommendations are independent of pose/contour quality and never grant asset QA.
Legacy runtime selection/default handling remains in the default strategy.
"""

from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from time import perf_counter
from .coarse import GeminiCoarseClient, VERSION, parse_response, recommend, unknown
from .coarse_input import prepare_input
from .schema import (
    SCHEMA_VERSION,
    OBSERVATION_VERSION,
    GEOMETRY_VERSION,
    unknown_attributes,
)


class CoarseBodyService:
    def __init__(
        self,
        catalog_path,
        *,
        client=None,
        provider="mock",
        model="gemini-flash-lite-latest",
        timeout_seconds=12,
        max_people=8,
        family_review_model="",
    ):
        if (
            provider not in ("mock", "gemini")
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or not 1 <= max_people <= 20
        ):
            raise ValueError("invalid_coarse_config")
        self.catalog_path = str(catalog_path)
        self.provider = provider
        self.model = model
        self.family_review_model = family_review_model
        self.max_people = max_people
        self.init_error = None
        self.client = client
        if client is None and provider == "gemini":
            try:
                self.client = GeminiCoarseClient(
                    model, timeout_seconds, family_review_model=family_review_model
                )
            except Exception as e:
                self.init_error = "coarse_provider_init_" + type(e).__name__

    def analyze(self, image, result, *, mode="shadow", pose_is_mock=False):
        from .service import _image, _box

        if mode not in ("auto", "shadow"):
            raise ValueError("invalid_body_matching_mode")
        start = perf_counter()
        out = dict(
            schema_version=SCHEMA_VERSION,
            mode=mode,
            status="ok",
            people=[],
            provider_requested=self.provider,
            provider_actual="gemini" if self.client else "mock",
            model=self.model,
            family_review_model=self.family_review_model or None,
            is_mock=self.client is None,
            prompt_version=VERSION,
            classification_strategy="coarse",
            pose_order_preserved=True,
            rendering_executed=False,
        )
        if result.route not in ("core", "bust") or not result.descriptors:
            return {
                **out,
                "status": "not_applicable",
                "reason": "route_" + result.route,
            }
        raw = Path(self.catalog_path).read_bytes()
        catalog = json.loads(raw)
        assets = catalog["assets"]
        out["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
        out["catalog_version"] = catalog["schema_version"]
        im = _image(image)
        key = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        out["input_sha256"] = key
        cases = []
        crops = []
        for i, desc in enumerate(result.descriptors):
            ident = key + ":p" + str(i)
            box = _box(desc, *im.size)
            prepared = None
            err = self.init_error or (
                "mock_not_visual_classification" if self.client is None else None
            )
            if box is None:
                err = "coarse_crop_unavailable"
            elif i >= self.max_people:
                err = "coarse_batch_budget_exceeded"
            elif self.client:
                prepared = prepare_input(im, box)
                crops.append((ident, prepared))
            cases.append(
                dict(
                    person_id=ident,
                    box=box,
                    crop_box=prepared.crop_box if prepared else None,
                    error=err,
                    observation=unknown(ident, err or "classification_unavailable"),
                )
            )
        if crops:
            try:
                batch = self.client.analyze(crops)
                # Validate injected/cached clients at this boundary too.
                payload = {
                    "people": [
                        {k: v for k, v in batch[ident].items() if k != "version"}
                        for ident, _ in crops
                    ]
                }
                if set(batch) != {x[0] for x in crops}:
                    raise ValueError("coarse_batch_identity")
                parsed = parse_response(payload, [x[0] for x in crops])
                for c in cases:
                    if c["person_id"] in parsed:
                        c["observation"] = parsed[c["person_id"]]
                        c["error"] = None
            except Exception as e:
                for c in cases:
                    if c["person_id"] in {x[0] for x in crops}:
                        c["error"] = "coarse_provider_" + type(e).__name__
        for i, c in enumerate(cases):
            classified = c["observation"]
            rec = recommend(classified, assets)
            poses = []
            for p in (
                result.person_candidates[i] if i < len(result.person_candidates) else []
            ):
                poses.append(
                    dict(pose_id=p.pose_id, view=p.view.value, pose_sha256=None)
                )
            obs = dict(
                schema_version=OBSERVATION_VERSION,
                person_id=c["person_id"],
                attributes=unknown_attributes(),
                geometry={},
                geometry_version=GEOMETRY_VERSION,
                clothing_occlusion=False,
                foreshortening=False,
                ownership_ambiguous=classified["target"] == "ambiguous",
                crop_box=list(c["crop_box"]) if c["crop_box"] else None,
                provider_error=c["error"],
                reason_codes=[],
            )
            out["people"].append(
                dict(
                    person_index=i,
                    person_id=c["person_id"],
                    auto_body_id=None,
                    applied_body_id=None,
                    selected_asset=None,
                    selection_source=None,
                    selection_state="unavailable",
                    diagnostic="coarse_recommendation",
                    reason_codes=[c["error"]] if c["error"] else [],
                    acceptance_probability=None,
                    # Legacy candidates imply runtime catalog eligibility.
                    # Coarse recommendations live only in their explicit field.
                    candidates=[],
                    observations=obs,
                    pose_bindings=poses,
                    render_required=False,
                    manual_change_scope="current_cut_person",
                    decision="suggest_only",
                    suggested_body_id=None,
                    suggested_asset=None,
                    coarse_classification=classified,
                    coarse_recommendation=rec,
                )
            )
        if any(c["error"] for c in cases):
            out["status"] = "partial"
        out["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
        return out
