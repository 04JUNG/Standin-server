"""Experimental A: minimal stand/non-stand/unknown support gate.

The gate is intentionally narrower than a support taxonomy.  It does not try
to infer airborne, leaning, props, contact surfaces, or motion semantics.  An
automatic gate may only exclude library members that are confidently
``stand`` when the 2D query is confidently ``non_stand``.  Unknown library
members always pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..features import _ALL_JOINTS_VALID, _as_joint_mask
from ..search import PositionSearchIndex, knn_geometric
from .b1_pose_facts import extract_pose_facts_2d


A_POLICY_VERSION = "a-minimal-support-2d-v1"

_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16
_REQUIRED_LOWER = np.asarray(
    [_LEFT_HIP, _RIGHT_HIP, _LEFT_KNEE, _RIGHT_KNEE,
     _LEFT_ANKLE, _RIGHT_ANKLE],
    dtype=np.intp,
)
_REQUIRED_LOWER.setflags(write=False)


@dataclass(frozen=True)
class SupportObservation:
    value: str
    evidence: tuple[str, ...]

    def to_trace(self) -> dict:
        return {"value": self.value, "evidence": list(self.evidence)}


@dataclass(frozen=True)
class ASupportResult:
    suggested_candidates: tuple
    trace: dict


def classify_minimal_support_2d(feature, valid_joint_mask=None) -> SupportObservation:
    """Return a conservative ``stand|non_stand|unknown`` observation."""
    points = np.asarray(feature, dtype=np.float32).reshape(17, 2)
    if not np.isfinite(points).all():
        raise ValueError("A support feature must be finite with shape (17,2)")
    valid = _as_joint_mask(valid_joint_mask)
    if not bool(valid[_REQUIRED_LOWER].all()):
        return SupportObservation("unknown", ("lower_body_incomplete",))

    facts = {fact.name: fact for fact in extract_pose_facts_2d(points, valid)}
    knee_values = (
        facts.get("left_knee_flexion"),
        facts.get("right_knee_flexion"),
    )
    if any(fact is None for fact in knee_values):
        return SupportObservation("unknown", ("knee_flexion_dead_zone",))

    values = tuple(fact.value for fact in knee_values if fact is not None)
    bent_values = {"bent", "deeply_bent"}
    if all(value in bent_values for value in values):
        return SupportObservation(
            "non_stand", ("both_knees_confidently_flexed",)
        )

    torso_required = (
        _LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP,
    )
    if not all(bool(valid[index]) for index in torso_required):
        return SupportObservation("unknown", ("torso_incomplete",))
    shoulder_mid = (
        points[_LEFT_SHOULDER] + points[_RIGHT_SHOULDER]
    ) / 2.0
    hip_mid = (points[_LEFT_HIP] + points[_RIGHT_HIP]) / 2.0
    ankle_mid = (
        points[_LEFT_ANKLE] + points[_RIGHT_ANKLE]
    ) / 2.0
    torso = shoulder_mid - hip_mid
    torso_length = float(np.linalg.norm(torso))
    if torso_length < 1e-6:
        return SupportObservation("unknown", ("torso_degenerate",))
    upward_component = float(-torso[1] / torso_length)
    hip_to_ankle_drop = float(ankle_mid[1] - hip_mid[1])
    if (values == ("extended", "extended")
            and upward_component >= 0.85
            and hip_to_ankle_drop >= 1.25):
        return SupportObservation(
            "stand",
            ("both_knees_extended", "upright_torso", "long_leg_drop"),
        )
    return SupportObservation("unknown", ("support_rules_inconclusive",))


def _entry_key(entry) -> tuple[str, str]:
    view = entry.view.value if hasattr(entry.view, "value") else str(entry.view)
    return str(entry.pose_id), str(view)


def _candidate_ids(candidates: Sequence) -> list[tuple[str, str]]:
    return [_entry_key(candidate) for candidate in candidates]


@dataclass(frozen=True)
class MinimalSupportGate:
    entries: tuple
    support_by_pose_id: Mapping[str, SupportObservation]
    eligible_entries: tuple
    eligible_position_index: PositionSearchIndex

    @classmethod
    def build(cls, entries: Iterable) -> "MinimalSupportGate":
        frozen_entries = tuple(entries)
        votes: dict[str, list[SupportObservation]] = {}
        for entry in frozen_entries:
            votes.setdefault(str(entry.pose_id), []).append(
                classify_minimal_support_2d(
                    entry.feature, _ALL_JOINTS_VALID,
                )
            )

        support_by_pose_id: dict[str, SupportObservation] = {}
        for pose_id, observations in votes.items():
            stand_count = sum(item.value == "stand" for item in observations)
            non_stand_count = sum(
                item.value == "non_stand" for item in observations
            )
            if stand_count >= 2 and non_stand_count == 0:
                support = SupportObservation(
                    "stand", (f"projection_votes={stand_count}",)
                )
            elif non_stand_count >= 2 and stand_count == 0:
                support = SupportObservation(
                    "non_stand", (f"projection_votes={non_stand_count}",)
                )
            else:
                support = SupportObservation(
                    "unknown",
                    (f"stand_votes={stand_count}",
                     f"non_stand_votes={non_stand_count}"),
                )
            support_by_pose_id[pose_id] = support

        # Unknown passes deliberately.  The automatic gate only removes a
        # positive, multi-view ``stand`` determination.
        eligible_entries = tuple(
            entry for entry in frozen_entries
            if support_by_pose_id[str(entry.pose_id)].value != "stand"
        )
        return cls(
            entries=frozen_entries,
            support_by_pose_id=MappingProxyType(support_by_pose_id),
            eligible_entries=eligible_entries,
            eligible_position_index=PositionSearchIndex.build(eligible_entries),
        )

    def evaluate(self, query_feature, query_valid_mask,
                 baseline_candidates: Sequence, *, metric: str,
                 max_distance_ratio: float) -> ASupportResult:
        baseline = tuple(baseline_candidates)
        query = classify_minimal_support_2d(
            query_feature, query_valid_mask,
        )
        trace = {
            "policy_version": A_POLICY_VERSION,
            "query_support": query.to_trace(),
            "baseline_order": [
                {"pose_id": pose_id, "view": view}
                for pose_id, view in _candidate_ids(baseline)
            ],
            "library_entry_count": len(self.entries),
            "eligible_entry_count": len(self.eligible_entries),
            "gate_eligible": False,
            "rollback_reason": None,
        }
        if query.value != "non_stand":
            trace["rollback_reason"] = "query_not_confident_non_stand"
            trace["suggested_order"] = list(trace["baseline_order"])
            return ASupportResult(baseline, trace)
        if not baseline:
            trace["rollback_reason"] = "baseline_candidates_unavailable"
            trace["suggested_order"] = []
            return ASupportResult(baseline, trace)

        filtered = tuple(knn_geometric(
            self.eligible_entries,
            query_feature,
            top_k=len(baseline),
            query_valid_mask=query_valid_mask,
            search_index=self.eligible_position_index,
            metric=metric,
        ))
        if len(filtered) < len(baseline):
            trace["rollback_reason"] = "filtered_candidate_count_insufficient"
            trace["suggested_order"] = list(trace["baseline_order"])
            return ASupportResult(baseline, trace)

        baseline_distances = np.asarray(
            [float(candidate.distance) for candidate in baseline],
            dtype=np.float64,
        )
        filtered_distances = np.asarray(
            [float(candidate.distance) for candidate in filtered],
            dtype=np.float64,
        )
        if not np.isfinite(baseline_distances).all():
            trace["rollback_reason"] = "baseline_distance_not_finite"
            trace["suggested_order"] = list(trace["baseline_order"])
            return ASupportResult(baseline, trace)
        if not np.isfinite(filtered_distances).all():
            trace["rollback_reason"] = "filtered_distance_not_finite"
            trace["suggested_order"] = list(trace["baseline_order"])
            return ASupportResult(baseline, trace)

        baseline_distance = float(baseline_distances[0])
        filtered_distance = float(filtered_distances[0])
        baseline_mean = float(baseline_distances.mean())
        filtered_mean = float(filtered_distances.mean())
        distance_ratio = float(max_distance_ratio)
        top1_limit = max(
            baseline_distance * distance_ratio,
            baseline_distance + 1e-7,
        )
        mean_limit = max(
            baseline_mean * distance_ratio,
            baseline_mean + 1e-7,
        )
        trace.update({
            "baseline_top1_distance": baseline_distance,
            "filtered_top1_distance": filtered_distance,
            "top1_distance_limit": top1_limit,
            "baseline_topk_mean_distance": baseline_mean,
            "filtered_topk_mean_distance": filtered_mean,
            "topk_mean_distance_limit": mean_limit,
        })
        if filtered_distance > top1_limit or filtered_mean > mean_limit:
            trace["rollback_reason"] = "distance_guard_exceeded"
            trace["distance_guard_failed"] = (
                "top1" if filtered_distance > top1_limit else "topk_mean"
            )
            trace["suggested_order"] = list(trace["baseline_order"])
            return ASupportResult(baseline, trace)

        trace["gate_eligible"] = True
        trace["suggested_order"] = [
            {"pose_id": pose_id, "view": view}
            for pose_id, view in _candidate_ids(filtered)
        ]
        return ASupportResult(filtered, trace)
