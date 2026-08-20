"""Shared structural eligibility policy for pose refinement.

The production pipeline, HTTP boundary, and evaluation harness must make the
same decision from the same frozen skeleton lineage. Search confidence is a
separate v1-only gate and deliberately does not live in this helper.
"""
from __future__ import annotations

from collections.abc import Iterable


def structural_refine_allowed(
    *,
    skeleton_state: str | None,
    coverage_class: str | None,
    refinable_limbs: Iterable[str] | None,
    slot_origin: str | None,
    skeleton_source: str | None,
    ownership_valid: bool = True,
) -> bool:
    """Return the version-independent, fail-closed refine eligibility."""
    return bool(
        ownership_valid
        and skeleton_state in {"valid", "partial"}
        and coverage_class in {"full", "reduced"}
        and tuple(refinable_limbs or ())
        and slot_origin == "vlm"
        and skeleton_source == "full_image"
    )


__all__ = ["structural_refine_allowed"]
