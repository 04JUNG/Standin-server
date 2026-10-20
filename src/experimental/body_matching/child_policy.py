"""Child assets remain selectable by people, never inferred from rough proportions.

The image contracts have no validated developmental-design evidence. Even a VLM's
confident age word is not an authorization to recommend a child mannequin.
"""

import re

VERSION = "body-child-recommendation.v1.manual-only"
REASON = "child_requires_explicit_user_selection"
_CHILD_ID = re.compile(
    r"(?:^|[-_.\s])(child|kid|baby|infant|toddler)(?:$|[-_.\s])", re.I
)


def is_child_asset(asset):
    metadata = asset.get("metadata") or {}
    design = metadata.get("age_design")
    if isinstance(design, str) and design.lower() in {
        "child",
        "kid",
        "infant",
        "toddler",
        "prepubescent",
        "아동",
        "유아",
    }:
        return True
    return any(
        isinstance(asset.get(key), str) and _CHILD_ID.search(asset[key])
        for key in ("body_id", "character_id")
    )


def automatic_candidates(candidates):
    """Filter recommendations, not the catalog or manual asset eligibility."""
    return [a for a in candidates if not is_child_asset(a)]
