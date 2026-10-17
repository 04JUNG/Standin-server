"""Structural application gate; rank scores are not calibrated probabilities."""
from __future__ import annotations

from .evidence import independent_groups

POLICY_VERSION = "body-apply.structural-evidence.v2"


def application_policy(observation, axes, geometry_axes, ranked, strong_ranked,
                       *, hypothesis_stable=True):
    strong_axes = [a for a in axes if observation["attributes"][a]["visibility"] == "visible"]
    groups = independent_groups(observation, strong_axes, geometry_axes)
    reasons = []
    if observation.get("ownership_ambiguous"):
        reasons.append("ownership_ambiguous")
    if observation.get("provider_error"):
        reasons.append("provider_error")
    if not observation.get("coverage") or not observation.get("representation"):
        reasons.append("shape_coverage_or_representation_missing")
    if len(groups) < 2:
        reasons.append("insufficient_independent_shape_evidence")
    if not strong_axes and not geometry_axes:
        reasons.append("no_clear_shape_evidence")
    def unique_first(rows):
        return bool(rows) and (len(rows) == 1 or rows[1]["rank_score"] - rows[0]["rank_score"] > 1e-12)
    if not unique_first(ranked):
        reasons.append("rank_tied_or_missing")
    if (not unique_first(strong_ranked) or not ranked
            or strong_ranked[0]["body_id"] != ranked[0]["body_id"]):
        reasons.append("clear_evidence_does_not_resolve_winner")
    if not hypothesis_stable:
        reasons.append("projection_hypotheses_disagree")
    return dict(version=POLICY_VERSION, allowed=not reasons, strong_axes=strong_axes,
                evidence_groups=groups, reason_codes=reasons,
                calibrated=False, acceptance_probability=None)
