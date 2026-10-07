"""Deterministic catalog ranking. No changes to pose IDs, order or cameras."""
from __future__ import annotations
import statistics

from .schema import AXES, NOMINAL_AXES, VISIBILITY

SCORER_VERSION = "body-score.v2.presentation-gate.uncalibrated"


def choose_body(catalog, observation, poses):
    candidates = catalog.eligible([p["pose_id"] for p in poses])
    return _select(candidates, observation, poses, catalog.default_body_id,
                   unavailable_reason=("no_pose_candidates" if not poses else
                                       "asset_incompatible" if catalog.assets else "catalog_empty"),
                   require_poses=True, presentation_defaults=catalog.presentation_defaults)


def compare_body_shapes(assets, observation, *, default_body_id=None, presentation_defaults=None):
    """Offline shape experiment; does not grant QA eligibility or rendering permission."""
    result = _select(list(assets), observation, [], default_body_id,
                     unavailable_reason="catalog_empty", require_poses=False, presentation_defaults=presentation_defaults)
    result.update(evaluation_scope="shape_only", applied_body_id=None, render_required=False)
    return result


def _select(candidates, observation, poses, default_body_id, *, unavailable_reason, require_poses, presentation_defaults=None):
    base = {"auto_body_id": None, "selected_asset": None, "candidates": [],
            "selection_source": None, "diagnostic": "catalog_empty", "reason_codes": [],
            "acceptance_probability": None, "scorer_version": SCORER_VERSION,
            "calibrator_version": None, "geometry_hypotheses": [], "evidence_axes": []}
    if not candidates or (require_poses and not poses):
        return {**base, "diagnostic": unavailable_reason}
    from .presentation import presentation_candidates, presentation_tie_rank
    candidates, default_body_id, presentation_trace = presentation_candidates(
        candidates, observation, default_body_id, presentation_defaults or {})
    base["presentation_selection"] = presentation_trace
    attrs = observation["attributes"]
    # Same axes for every candidate. Missing metadata cannot make a candidate win.
    axes = [axis for axis in AXES if attrs[axis]["value"] is not None
            and all(a["attributes"][axis] is not None for a in candidates)]
    reasons = list(presentation_trace["reason_codes"])
    if any(attrs[k]["value"] is not None and k not in axes for k in AXES):
        reasons.append("incomplete_catalog_attributes_excluded")
    geometry = observation["geometry"]
    hypotheses = []
    for pose in poses:
        if pose["pose_sha256"] is None:
            continue
        records = {}
        for asset in candidates:
            record = next((p for p in asset["projections"] if all(
                p[k] == pose[k] for k in ("pose_id", "view", "pose_sha256"))), None)
            if record:
                records[asset["body_id"]] = record["ratios"]
        if len(records) != len(candidates):
            continue
        common = sorted(set(geometry).intersection(*(set(r) for r in records.values())))
        if common:
            hypotheses.append((pose, records, common))
    if geometry and not hypotheses:
        reasons.append("target_projection_missing_geometry_not_scored")
    # Foreshortening does not prohibit projected comparison, but weakens semantic ratios.
    def terms(asset):
        items = []
        for axis in axes:
            values = AXES[axis]
            q, c = attrs[axis]["value"], asset["attributes"][axis]
            loss = float(q != c) if axis in NOMINAL_AXES else abs(values.index(q) - values.index(c)) / (len(values) - 1)
            weight = VISIBILITY[attrs[axis]["visibility"]]
            if observation.get("foreshortening") and axis in {"head_ratio_class", "limb_proportion"}:
                weight *= 0.25
            items.append((loss, weight))
        return items
    ranked = []
    for asset in candidates:
        scores = []
        for _, records, common in hypotheses or [(None, None, [])]:
            items = terms(asset)
            for axis in common:
                q, c = geometry[axis], records[asset["body_id"]][axis]
                items.append((abs(q - c) / max(abs(q), abs(c), 1e-8), 1.0))
            denom = sum(w for _, w in items)
            if denom:
                scores.append(sum(d * w for d, w in items) / denom)
        if scores:
            ranked.append({"body_id": asset["body_id"], "body_version": asset["body_version"],
                           "rank_score": float(statistics.median(scores)),
                           "score_spread": float(max(scores) - min(scores)),
                           "acceptance_probability": None})
    by_id = {a["body_id"]: a for a in candidates}
    def tie_key(asset):
        return (presentation_tie_rank(asset, observation), -asset.get("quality_priority", 0), asset["body_id"] != default_body_id, asset["body_id"])
    if ranked:
        ranked.sort(key=lambda r: (round(r["rank_score"], 12), *tie_key(by_id[r["body_id"]])))
        selected = by_id[ranked[0]["body_id"]]
        source, diagnostic = "auto_best_effort", "uncertain"
        reasons.extend(["uncalibrated_rank_score", "library_coverage_unvalidated"])
    else:
        selected = next((a for a in candidates if a["body_id"] == default_body_id),
                        min(candidates, key=tie_key))
        source = "auto_default"
        diagnostic = "provider_error" if observation.get("provider_error") else "insufficient_evidence"
        reasons.append("no_comparable_evidence")
    base.update(auto_body_id=selected["body_id"], selection_source=source, diagnostic=diagnostic,
                tied_body_ids=[r["body_id"] for r in ranked if abs(r["rank_score"] - ranked[0]["rank_score"]) < 1e-12],
                selected_asset={key: selected[key] for key in (
                    "body_id", "body_version", "asset_sha256", "rig_version", "measurement_version")},
                candidates=[dict(r, rank=i + 1) for i, r in enumerate(ranked[:3])],
                evidence_axes=axes + sorted({a for _, _, common in hypotheses for a in common}),
                geometry_hypotheses=[dict(p) for p, _, _ in hypotheses], reason_codes=reasons)
    return base
