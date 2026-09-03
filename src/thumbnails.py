"""포즈 라이브러리 썸네일 경로와 API URL 구성."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote


THUMBNAIL_VIEWS = frozenset({"front", "back", "side", "three_quarter"})


def find_thumbnail(data_dir: str, pose_id: str, view: str) -> Path | None:
    """번들 ``thumbs/<pose_id>__<view>.jpg``를 안전하게 찾는다.

    포즈 라이브러리 번들이 2026-09-03부터 4시점 모두 JPEG로 바뀌었고 PNG는 빠졌다.
    번들과 이 확장자는 한 배를 탄다 — 옛 번들(PNG만 있는)로 되돌리면 썸네일이
    전부 사라지므로, 롤백할 때는 번들과 코드를 함께 되돌린다.
    """
    if view not in THUMBNAIL_VIEWS:
        return None

    root = (Path(data_dir) / "thumbs").resolve()
    candidate = (root / f"{pose_id}__{view}.jpg").resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def thumbnail_url(data_dir: str, pose_id: str, view: str) -> str | None:
    """파일이 실제로 있을 때만 내부 추론 API 경로를 반환한다."""
    if find_thumbnail(data_dir, pose_id, view) is None:
        return None
    encoded_pose = quote(pose_id, safe="")
    encoded_view = quote(view, safe="")
    return f"/pose/{encoded_pose}/thumbnail?view={encoded_view}"
