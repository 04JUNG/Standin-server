"""포즈 라이브러리를 폴더처럼 훑어보기 위한 목록. 관리자 대시보드가 쓴다.

검색이 실제로 쓰는 메모리 목록(`Pipeline.entries`)에서 만든다. 그래서 격리(quarantine)된
포즈가 빠지는 것까지 검색과 똑같다 — 화면에 보이는데 검색에는 안 나오는 포즈가 생기지 않는다.

항목은 포즈 단위다. 라이브러리 행은 포즈 × 방향(front·three_quarter·side·back)이라,
방향은 한 포즈 안의 목록으로 묶는다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

#: 출처가 비어 있는 포즈는 대부분 초기 라이브러리다(`curation_group=existing`).
EXISTING_SOURCE = "existing"
UNKNOWN = "unknown"

#: 출처 키 → 화면 이름. 모르는 키는 키를 그대로 보여 준다.
SOURCE_LABELS = {
    EXISTING_SOURCE: "기존 라이브러리",
    "authored_scenario": "직접 제작 · 상황",
    "authored_combat": "직접 제작 · 전투",
    "cmu": "CMU",
    "quaternius": "Quaternius",
    "100style": "100STYLE",
    "accad": "ACCAD",
    UNKNOWN: "출처 미상",
}

VIEW_ORDER = ("front", "three_quarter", "side", "back")
MAX_LIMIT = 96


@dataclass(frozen=True)
class CatalogItem:
    pose_id: str
    source: str
    category: str
    action: str
    views: tuple[str, ...]
    review_status: Optional[str] = None
    extra: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        return {
            "pose_id": self.pose_id,
            "source": self.source,
            "source_label": SOURCE_LABELS.get(self.source, self.source),
            "category": self.category,
            "action": self.action,
            "views": list(self.views),
            "review_status": self.review_status,
        }


def _source(meta: dict) -> str:
    source = meta.get("source")
    if isinstance(source, str) and source:
        return source
    return EXISTING_SOURCE if meta.get("curation_group") == "existing" else UNKNOWN


def _category(meta: dict) -> str:
    for key in ("category_label", "category"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return "미분류"


def build_catalog(entries: Iterable) -> list[CatalogItem]:
    """포즈×방향 행들을 포즈 단위로 묶는다. pose_id 순으로 정렬한다."""
    grouped: dict[str, dict] = {}
    for entry in entries:
        slot = grouped.setdefault(entry.pose_id, {"entry": entry, "views": set()})
        view = getattr(entry.view, "value", entry.view)
        slot["views"].add(str(view))
    items = []
    for pose_id, slot in grouped.items():
        entry = slot["entry"]
        meta = entry.meta if isinstance(entry.meta, dict) else {}
        tags = entry.tags if isinstance(entry.tags, dict) else {}
        views = tuple(view for view in VIEW_ORDER if view in slot["views"]) + tuple(
            sorted(view for view in slot["views"] if view not in VIEW_ORDER))
        status = meta.get("review_status")
        items.append(CatalogItem(
            pose_id=pose_id,
            source=_source(meta),
            category=_category(meta),
            action=str(tags.get("action") or "other"),
            views=views,
            review_status=status if isinstance(status, str) else None,
        ))
    items.sort(key=lambda item: item.pose_id)
    return items


def folders(items: list[CatalogItem]) -> dict:
    """출처별·분류별 폴더와 그 안의 포즈 수. 많은 순."""
    def listing(counter: Counter, labels: Optional[dict] = None) -> list[dict]:
        return [
            {"key": key, "label": (labels or {}).get(key, key), "count": count}
            for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    return {
        "total": len(items),
        "sources": listing(Counter(item.source for item in items), SOURCE_LABELS),
        "categories": listing(Counter(item.category for item in items)),
    }


def page(items: list[CatalogItem], *, source: Optional[str] = None, category: Optional[str] = None,
         query: Optional[str] = None, cursor: Optional[str] = None, limit: int = 48) -> dict:
    """걸러서 한 페이지. 커서는 마지막 pose_id다(목록이 pose_id 순이라 안정적이다)."""
    limit = max(1, min(int(limit), MAX_LIMIT))
    needle = (query or "").strip().lower()
    selected = [
        item for item in items
        if (not source or item.source == source)
        and (not category or item.category == category)
        and (not needle or needle in item.pose_id.lower())
        and (not cursor or item.pose_id > cursor)
    ]
    chunk = selected[:limit]
    return {
        "items": [item.to_dict() for item in chunk],
        "next_cursor": chunk[-1].pose_id if len(selected) > limit and chunk else None,
        "matched": len(selected) if not cursor else None,
    }
