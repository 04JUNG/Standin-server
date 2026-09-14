"""Experimental B1: deterministic 2D pose-fact reranking.

This module is deliberately isolated from ``src.search`` and from the
production semantic index.  It only reorders an already selected geometric
Top-K; it never adds/removes candidates, changes geometric distances, or makes
confidence/refine decisions.

The query and every library projection pass through the same extractor.  Facts
whose required query joints are missing, degenerate, or inside a threshold
dead-zone are omitted and therefore compare as ``unknown``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..features import _ALL_JOINTS_VALID, _as_joint_mask


B1_POLICY_VERSION = "b1-pose-facts-2d-v1"

_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16


@dataclass(frozen=True)
class PoseFact:
    """One conservative categorical observation from a 2D skeleton."""

    name: str
    value: str
    measure: float
    required_joints: tuple[int, ...]

    @property
    def token(self) -> str:
        return f"{self.name}={self.value}"

    def to_trace(self) -> dict:
        return {
            "value": self.value,
            "measure": round(float(self.measure), 6),
        }


@dataclass(frozen=True)
class FactComparison:
    matches: tuple[str, ...]
    violations: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def rank_key(self) -> tuple[int, int]:
        # Conservative lexicographic policy: do not promote a known
        # contradiction merely because several redundant facts matched.
        return len(self.violations), -len(self.matches)

    def to_trace(self) -> dict:
        return {
            "matches": list(self.matches),
            "violations": list(self.violations),
            "unknown": list(self.unknown),
        }


@dataclass(frozen=True)
class B1RerankResult:
    suggested_candidates: tuple
    trace: dict


def _finite_points(feature) -> np.ndarray:
    points = np.asarray(feature, dtype=np.float32).reshape(17, 2)
    if not np.isfinite(points).all():
        raise ValueError("B1 feature must be finite with shape (17,2)")
    return points


def _joint_angle(points: np.ndarray, a: int, b: int, c: int,
                 valid_mask: np.ndarray) -> float | None:
    required = (a, b, c)
    if not all(bool(valid_mask[index]) for index in required):
        return None
    first = points[a] - points[b]
    second = points[c] - points[b]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _flexion_value(angle: float) -> str | None:
    """Bucket flexion with explicit gaps between confident regions."""
    if angle <= 70.0:
        return "deeply_bent"
    if 85.0 <= angle <= 135.0:
        return "bent"
    if angle >= 155.0:
        return "extended"
    return None


def _append_flexion(facts: list[PoseFact], points: np.ndarray,
                     valid_mask: np.ndarray, *, name: str,
                     joints: tuple[int, int, int]) -> None:
    angle = _joint_angle(points, *joints, valid_mask)
    if angle is None:
        return
    value = _flexion_value(angle)
    if value is not None:
        facts.append(PoseFact(name, value, angle, joints))


def _append_wrist_height(facts: list[PoseFact], points: np.ndarray,
                         valid_mask: np.ndarray, *, side: str,
                         shoulder: int, wrist: int) -> None:
    required = (shoulder, wrist)
    if not all(bool(valid_mask[index]) for index in required):
        return
    # Features use image coordinates after torso normalization: smaller y is
    # visually higher.  The gap between 0.08 and 0.20 is the dead-zone.
    delta = float(points[wrist, 1] - points[shoulder, 1])
    if delta <= -0.20:
        value = "above"
    elif delta >= 0.20:
        value = "below"
    elif abs(delta) <= 0.08:
        value = "level"
    else:
        return
    facts.append(PoseFact(
        f"{side}_wrist_vs_shoulder", value, delta, required,
    ))


def _append_torso_lean(facts: list[PoseFact], points: np.ndarray,
                       valid_mask: np.ndarray) -> None:
    required = (
        _LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP,
    )
    if not all(bool(valid_mask[index]) for index in required):
        return
    shoulder_mid = (
        points[_LEFT_SHOULDER] + points[_RIGHT_SHOULDER]
    ) / 2.0
    hip_mid = (points[_LEFT_HIP] + points[_RIGHT_HIP]) / 2.0
    torso = shoulder_mid - hip_mid
    length = float(np.linalg.norm(torso))
    if length < 1e-6:
        return
    # A lying/upside-down projection is not labelled as lateral lean.  B1 v1
    # intentionally stays inside the common upright/reduced-body slice.
    upward_component = float(-torso[1] / length)
    if upward_component <= 0.25:
        return
    angle = math.degrees(math.atan2(float(torso[0]), float(-torso[1])))
    magnitude = abs(angle)
    if magnitude <= 7.0:
        value = "neutral"
    elif magnitude >= 15.0:
        value = "screen_right" if angle > 0.0 else "screen_left"
    else:
        return
    facts.append(PoseFact(
        "torso_lateral_lean", value, angle, required,
    ))


def extract_pose_facts_2d(feature, valid_joint_mask=None) -> tuple[PoseFact, ...]:
    """Extract the conservative B1-v1 subset from a normalized 2D feature."""
    points = _finite_points(feature)
    valid_mask = _as_joint_mask(valid_joint_mask)
    facts: list[PoseFact] = []
    for side, shoulder, elbow, wrist in (
        ("left", _LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST),
        ("right", _RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST),
    ):
        _append_flexion(
            facts, points, valid_mask,
            name=f"{side}_elbow_flexion",
            joints=(shoulder, elbow, wrist),
        )
        _append_wrist_height(
            facts, points, valid_mask, side=side,
            shoulder=shoulder, wrist=wrist,
        )
    for side, hip, knee, ankle in (
        ("left", _LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE),
        ("right", _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE),
    ):
        _append_flexion(
            facts, points, valid_mask,
            name=f"{side}_knee_flexion",
            joints=(hip, knee, ankle),
        )
    _append_torso_lean(facts, points, valid_mask)
    return tuple(facts)


def compare_pose_facts(query_facts: Sequence[PoseFact],
                       candidate_facts: Sequence[PoseFact]) -> FactComparison:
    candidate_by_name = {fact.name: fact for fact in candidate_facts}
    matches: list[str] = []
    violations: list[str] = []
    unknown: list[str] = []
    for query_fact in query_facts:
        candidate_fact = candidate_by_name.get(query_fact.name)
        if candidate_fact is None:
            unknown.append(query_fact.token)
        elif candidate_fact.value == query_fact.value:
            matches.append(query_fact.token)
        else:
            violations.append(
                f"{query_fact.name}:{query_fact.value}!={candidate_fact.value}"
            )
    return FactComparison(
        matches=tuple(matches),
        violations=tuple(violations),
        unknown=tuple(unknown),
    )


def _candidate_key(candidate) -> tuple[str, str]:
    view = candidate.view.value if hasattr(candidate.view, "value") else str(candidate.view)
    return str(candidate.pose_id), str(view)


@dataclass(frozen=True)
class PoseFactReranker:
    """Immutable fact lookup built only when the experimental flag is active."""

    facts_by_candidate: Mapping[tuple[str, str], tuple[PoseFact, ...]]

    @classmethod
    def build(cls, entries: Iterable) -> "PoseFactReranker":
        facts: dict[tuple[str, str], tuple[PoseFact, ...]] = {}
        for entry in entries:
            key = _candidate_key(entry)
            if key in facts:
                raise ValueError(f"duplicate B1 library entry: {key[0]}:{key[1]}")
            facts[key] = extract_pose_facts_2d(
                entry.feature, _ALL_JOINTS_VALID,
            )
        return cls(MappingProxyType(facts))

    def evaluate(self, query_feature, query_valid_mask,
                 candidates: Sequence) -> B1RerankResult:
        baseline = tuple(candidates)
        query_facts = extract_pose_facts_2d(
            query_feature, query_valid_mask,
        )
        comparisons: list[FactComparison] = []
        for candidate in baseline:
            candidate_facts = self.facts_by_candidate.get(
                _candidate_key(candidate), ()
            )
            comparisons.append(compare_pose_facts(query_facts, candidate_facts))

        order = sorted(
            range(len(baseline)),
            key=lambda index: (*comparisons[index].rank_key, index),
        )
        suggested = tuple(baseline[index] for index in order)
        suggested_rank = {baseline_index: rank for rank, baseline_index in enumerate(order)}
        candidate_trace = []
        for baseline_index, (candidate, comparison) in enumerate(
                zip(baseline, comparisons)):
            candidate_trace.append({
                "pose_id": str(candidate.pose_id),
                "view": _candidate_key(candidate)[1],
                "geometry_rank": baseline_index + 1,
                "suggested_rank": suggested_rank[baseline_index] + 1,
                **comparison.to_trace(),
            })
        baseline_ids = [_candidate_key(candidate) for candidate in baseline]
        suggested_ids = [_candidate_key(candidate) for candidate in suggested]
        return B1RerankResult(
            suggested_candidates=suggested,
            trace={
                "policy_version": B1_POLICY_VERSION,
                "scope": "existing_geometry_top_k",
                "query_facts": {
                    fact.name: fact.to_trace() for fact in query_facts
                },
                "candidate_comparisons": candidate_trace,
                "would_change_order": baseline_ids != suggested_ids,
            },
        )
