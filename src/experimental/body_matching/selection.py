"""Deterministic catalog ranking. No changes to pose IDs, order or cameras."""
from __future__ import annotations
import statistics

from .schema import AXES, NOMINAL_AXES, VISIBILITY
from .evidence import mask_geometry
from .policy import POLICY_VERSION, application_policy

SCORER_VERSION = "body-score.v4.regional-gate.uncalibrated"


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
            "calibrator_version": None, "geometry_hypotheses": [], "evidence_axes": [],
            "decision": "asset_unavailable", "policy_version": POLICY_VERSION,
            "suggested_body_id": None, "suggested_asset": None}
    if not candidates or (require_poses and not poses):
        return {**base, "diagnostic": unavailable_reason}
    from .presentation import presentation_candidates, presentation_tie_rank
    candidates, default_body_id, presentation_trace = presentation_candidates(
        candidates, observation, default_body_id, presentation_defaults or {})
    base["presentation_selection"] = presentation_trace
    attrs = observation["attributes"]
    # Same axes for every candidate. Missing metadata cannot make a candidate win.
    axes = [axis for axis in AXES if attrs[axis]["value"] is not None
            and attrs[axis]["visibility"] != "unknown"
            and all(a["attributes"][axis] is not None for a in candidates)]
    reasons = list(presentation_trace["reason_codes"])
    if any(attrs[k]["value"] is not None and k not in axes for k in AXES):
        reasons.append("incomplete_catalog_attributes_excluded")
    geometry = mask_geometry(observation)
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
    def terms(asset, clear_only=False):
        items = []
        for axis in axes:
            if clear_only and attrs[axis]["visibility"] != "visible":
                continue
            values = AXES[axis]
            q, c = attrs[axis]["value"], asset["attributes"][axis]
            loss = float(q != c) if axis in NOMINAL_AXES else abs(values.index(q) - values.index(c)) / (len(values) - 1)
            weight = VISIBILITY[attrs[axis]["visibility"]]
            if observation.get("foreshortening") and axis in {"head_ratio_class", "limb_proportion"}:
                weight *= 0.25
            items.append((loss, weight))
        return items
    by_id = {a["body_id"]: a for a in candidates}
    def tie_key(asset):
        return (presentation_tie_rank(asset, observation), -asset.get("quality_priority", 0), asset["body_id"] != default_body_id, asset["body_id"])
    def rank(clear_only=False):
        ranked, per_hypothesis = [], [{} for _ in hypotheses or [None]]
        for asset in candidates:
            scores = []
            for i, (_, records, common) in enumerate(hypotheses or [(None, None, [])]):
                items = terms(asset, clear_only)
                for axis in common:
                    q, c = geometry[axis], records[asset["body_id"]][axis]
                    items.append((abs(q - c) / max(abs(q), abs(c), 1e-8), 1.0))
                denom = sum(w for _, w in items)
                if denom:
                    score = sum(d * w for d, w in items) / denom
                    scores.append(score)
                    per_hypothesis[i][asset["body_id"]] = score
            if scores:
                ranked.append({"body_id": asset["body_id"], "body_version": asset["body_version"],
                               "rank_score": float(statistics.median(scores)),
                               "score_spread": float(max(scores) - min(scores)),
                               "acceptance_probability": None})
        ranked.sort(key=lambda r: (round(r["rank_score"], 12), *tie_key(by_id[r["body_id"]])))
        winners = []
        for scores in per_hypothesis:
            best = min(scores.values()) if scores else None
            tied = [body for body, score in scores.items() if abs(score - best) < 1e-12]
            winners.append(set(tied))
        return ranked, winners

    ranked, _ = rank()
    strong_ranked, winners = rank(clear_only=True)
    geometry_axes = sorted({a for _, _, common in hypotheses for a in common})
    stable = len(hypotheses) < 2 or bool(strong_ranked) and all(
        strong_ranked[0]["body_id"] in tied for tied in winners)
    policy = application_policy(observation, axes, geometry_axes, ranked, strong_ranked,
                                hypothesis_stable=stable)
    suggested = (by_id[ranked[0]["body_id"]] if ranked else
                 next((a for a in candidates if a["body_id"] == default_body_id), min(candidates, key=tie_key)))
    selected = suggested if policy["allowed"] else None
    source = "auto_best_effort" if ranked else "auto_default"
    # Preserve the BFF's explicit style-supported default path, never weak body
    # ranking. Ordinary defaults remain non-applied suggestions.
    from .presentation import asset_style
    presentation_default = (not ranked and not observation.get("provider_error")
        and not observation.get("ownership_ambiguous")
        and presentation_trace["mode"] == "compatible_candidates"
        and presentation_trace["visibility"] == "visible"
        and asset_style(suggested) == presentation_trace["observed"])
    if presentation_default:
        selected, source = suggested, "auto_presentation_default"
        reasons.append("presentation_supported_shape_default")
    diagnostic = ("uncertain" if selected and not presentation_default else "provider_error" if observation.get("provider_error")
                  else "insufficient_evidence")
    reasons.extend(["uncalibrated_rank_score", "library_coverage_unvalidated"] if ranked else ["no_comparable_evidence"])
    reasons.extend(policy["reason_codes"])
    def reference(asset):
        return {key: asset[key] for key in (
            "body_id", "body_version", "asset_sha256", "rig_version", "measurement_version")}
    base.update(auto_body_id=selected["body_id"] if selected else None,
                selection_source=source, diagnostic=diagnostic,
                decision="presentation_default" if presentation_default else "auto_apply" if selected else "suggest_only" if ranked else "insufficient_evidence",
                application_policy=policy, suggested_body_id=suggested["body_id"], suggested_asset=reference(suggested),
                tied_body_ids=[r["body_id"] for r in ranked if abs(r["rank_score"] - ranked[0]["rank_score"]) < 1e-12],
                selected_asset=reference(selected) if selected else None,
                candidates=[dict(r, rank=i + 1) for i, r in enumerate(ranked[:3])],
                evidence_axes=axes + geometry_axes,
                geometry_hypotheses=[dict(p) for p, _, _ in hypotheses], reason_codes=reasons)
    return base
