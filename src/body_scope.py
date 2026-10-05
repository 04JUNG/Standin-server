"""Per-person framing metadata, independent of search tags and refine visibility.

An occluded leg does not make a full-body composition a half-body shot. Missing
provider metadata therefore stays unknown instead of being inferred from scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

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
    source: Literal["vlm_person", "legacy_shot", "unknown"] = "unknown"


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
