"""Per-person framing metadata, independent of search tags and refine visibility.

An occluded leg does not make a full-body composition a half-body shot. Missing
provider metadata therefore stays unknown instead of being inferred from scores.

The reverse is not symmetric: a knee the pose model actually observed means the
frame was not cut at the waist/hips/thighs, so a VLM half/bust label that would
crop those legs away is promoted to full (`promote_for_observed_legs`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np

if TYPE_CHECKING:
    from .schema import VLMAnalysis


class BodyScope(str, Enum):
    FULL = "full"
    HALF = "half"
    BUST = "bust"
    HEAD = "head"


@dataclass(frozen=True)
class ScopeDetection:
    detected: BodyScope | None = None
    source: Literal["vlm_person", "legacy_shot", "observed_legs", "unknown"] = "unknown"


# COCO17 hip→knee per leg. A visible knee is below every half-body cut line.
_THIGHS = {"left_leg": (11, 13), "right_leg": (12, 14)}
# Scopes whose output mesh drops the legs. Head keeps its own unsupported-search
# path, so it is deliberately not promoted here.
_LEG_CUTTING_SCOPES = frozenset({BodyScope.HALF, BodyScope.BUST})


def parse_person_scopes(raw: object, count: int) -> list[BodyScope | None]:
    """Reject misaligned arrays; never shift labels past an invalid entry."""
    if not isinstance(raw, list) or len(raw) != count:
        return [None] * count
    return [BodyScope(value) if isinstance(value, str)
            and value in BodyScope._value2member_map_ else None for value in raw]


def detect_scope(vlm: VLMAnalysis, person_index: int | None) -> ScopeDetection:
    # Only a VLM-owned slot has a trustworthy per-person semantic identity.
    if person_index is None:
        return ScopeDetection()
    if 0 <= person_index < len(vlm.body_scopes):
        scope = vlm.body_scopes[person_index]
        if scope is not None:
            return ScopeDetection(scope, "vlm_person")
    # A legacy cut-level label cannot describe each person in a mixed shot.
    if (vlm.num_people == 1 and person_index == 0
            and "body_scopes" not in vlm.raw):
        legacy = {"bust": BodyScope.BUST, "face": BodyScope.HEAD}
        scope = legacy.get(vlm.shot.value)
        if scope is not None:
            return ScopeDetection(scope, "legacy_shot")
    return ScopeDetection()


def observed_legs(keypoints, valid_mask, suspect_limbs: Iterable[str],
                  image_size: tuple[int, int]) -> tuple[str, ...]:
    """Legs whose hip and knee were actually observed inside the image.

    `valid_mask` must be the structural search mask (score threshold and length
    outliers already applied). Ownership-suspect legs never count, and a knee
    predicted outside the frame is a guess about an off-screen joint, not a leg.
    """
    width, height = image_size
    points = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    mask = np.asarray(valid_mask, dtype=bool).reshape(17)
    suspect = set(suspect_limbs)
    legs = []
    for limb, (hip, knee) in _THIGHS.items():
        if limb in suspect or not (mask[hip] and mask[knee]):
            continue
        x, y = points[knee]
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            legs.append(limb)
    return tuple(legs)


def promote_for_observed_legs(detection: ScopeDetection,
                              legs: tuple[str, ...]) -> ScopeDetection:
    """Never let a crop label remove legs the pose model saw. Only ever widens."""
    if not legs or detection.detected not in _LEG_CUTTING_SCOPES:
        return detection
    return ScopeDetection(BodyScope.FULL, "observed_legs")
