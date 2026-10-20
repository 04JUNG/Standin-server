"""Per-person action/view tags from the VLM, kept apart from search.

The cut-level action/view on ``PersonDescriptor`` stays what it always was. These
per-person values only describe the drawing for later analysis (the rough-data gap
loop, docs/POSE_GAP_LOOP.md). They are never inputs to matching, routing or refine
(CLAUDE.md invariant 1). Provider output that is missing or misaligned stays unknown
instead of being guessed, and one bad entry never shifts the people after it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional, TypeVar

if TYPE_CHECKING:
    from .schema import Action, View, VLMAnalysis

E = TypeVar("E", bound=Enum)

# VLM 응답에서 인물별 태그 배열의 키. 이 키가 응답에 있으면 프롬프트가 인물별로 물은 것이다.
PERSON_TAG_KEYS = ("person_actions", "person_views")


@dataclass(frozen=True)
class PersonTags:
    action: Optional["Action"] = None
    view: Optional["View"] = None
    source: Literal["vlm_person", "legacy_cut", "unknown"] = "unknown"


def parse_person_values(raw: object, count: int, enum_cls: type[E]) -> list[Optional[E]]:
    """Reject misaligned arrays; an invalid entry becomes None without shifting the rest."""
    if not isinstance(raw, list) or len(raw) != count:
        return [None] * count
    members = enum_cls._value2member_map_
    return [enum_cls(value) if isinstance(value, str) and value in members else None
            for value in raw]


def stated_value(raw: object, enum_cls: type[Enum]) -> Optional[str]:
    """The provider's own label if it is in the vocabulary, else None (no default)."""
    if isinstance(raw, str) and raw in enum_cls._value2member_map_:
        return raw
    return None


def apply_person_tags(vlm: "VLMAnalysis", payload: Optional[dict]) -> bool:
    """태그 전용 호출의 응답을 VLMAnalysis에 싣는다. 실은 값이 하나라도 있으면 True.

    배열 길이가 인원수와 다르면 통째로 버린다(`parse_person_values`). 한 칸이 밀리면
    다른 사람의 태그가 붙기 때문이다. 버린 경우에도 "물어보긴 했다"는 사실은 raw에
    남긴다 — 그래야 1인 컷에서 컷 값을 빌려 쓰는 legacy_cut 경로가 열리지 않는다.
    """
    from .schema import Action, View

    if not isinstance(payload, dict):
        return False
    actions = parse_person_values(payload.get("person_actions"), vlm.num_people, Action)
    views = parse_person_values(payload.get("person_views"), vlm.num_people, View)
    vlm.person_actions = actions
    vlm.person_views = views
    for key in PERSON_TAG_KEYS:
        vlm.raw[key] = payload.get(key)
    return any(value is not None for value in (*actions, *views))


def detect_person_tags(vlm: "VLMAnalysis", person_index: Optional[int]) -> PersonTags:
    # Only a VLM-owned slot has a per-person semantic identity.
    if person_index is None:
        return PersonTags()
    action = (vlm.person_actions[person_index]
              if 0 <= person_index < len(vlm.person_actions) else None)
    view = (vlm.person_views[person_index]
            if 0 <= person_index < len(vlm.person_views) else None)
    if action is not None or view is not None:
        return PersonTags(action, view, "vlm_person")
    # A prompt without per-person tags answers for the whole cut. That label can
    # describe the person only when the cut has exactly one, and only when the
    # provider stated it rather than the parser filling in a default.
    asked = any(key in vlm.raw for key in PERSON_TAG_KEYS)
    stated = vlm.stated_tags or {}
    if (not asked and vlm.num_people == 1 and person_index == 0
            and (stated.get("action") or stated.get("view"))):
        from .schema import Action, View

        return PersonTags(
            Action(stated["action"]) if stated.get("action") else None,
            View(stated["view"]) if stated.get("view") else None,
            "legacy_cut",
        )
    return PersonTags()
