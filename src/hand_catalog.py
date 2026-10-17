"""손 카탈로그(04J-290): refine 화면의 손 칩이 쓰는 목록과 썸네일.

라이브러리 정의(converter/hand_library.v1.json)와 미리 그린 마네킹 손 썸네일
(converter/hand_thumbs/, scripts/render_hand_thumbnails.mjs)을 한 응답으로 묶는다.
썸네일 manifest의 library_sha256이 현재 라이브러리와 다르면 썸네일은 '없음'으로 다룬다 —
각도를 고친 뒤 옛 그림이 나가는 것을 막는다.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from converter.hand_library import SIDES, load_library
from src.hand_shapes import DEFAULT_PRESET

THUMBS_DIR = Path(__file__).resolve().parents[1] / "converter" / "hand_thumbs"
CATALOG_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _thumb_manifest(directory: str = str(THUMBS_DIR)) -> Optional[dict]:
    path = Path(directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("library_sha256") != load_library().sha256:
        return None
    # 이미지에 PNG가 빠진 채 manifest만 들어가면 카탈로그가 깨진 URL을 내보낸다 → 하나라도 없으면 '없음'
    files = manifest.get("files") or {}
    if not files or not all((Path(directory) / name).is_file() for name in files):
        return None
    return manifest


def thumbnail_path(preset_id: str, side: str, directory: str = str(THUMBS_DIR)) -> Optional[Path]:
    """검증된 썸네일 파일 경로. 없거나 라이브러리와 맞지 않거나 해시가 다르면 None."""
    library = load_library()
    if preset_id not in library.presets or side not in SIDES:
        return None
    manifest = _thumb_manifest(directory)
    if manifest is None:
        return None
    name = f"{preset_id}__{side}.png"
    expected = manifest.get("files", {}).get(name)
    path = Path(directory) / name
    if not expected or not path.is_file():
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        return None
    return path


def thumbnail_etag(preset_id: str, side: str, directory: str = str(THUMBS_DIR)) -> Optional[str]:
    manifest = _thumb_manifest(directory)
    if manifest is None:
        return None
    digest = manifest.get("files", {}).get(f"{preset_id}__{side}.png")
    return f'"{digest[:32]}"' if digest else None


def build_catalog(thumbnail_base: str = "/hands", directory: str = str(THUMBS_DIR)) -> dict:
    """클라이언트용 카탈로그. 묶음 순서·프리셋 순서는 라이브러리 정의 순서 그대로다."""
    library = load_library()
    manifest = _thumb_manifest(directory)
    groups = []
    for category_id, label in library.categories.items():
        presets = []
        for preset_id in library.in_category(category_id):
            preset = library.presets[preset_id]
            presets.append({
                "id": preset_id,
                "label": preset.label_ko,
                "description": preset.vlm_hint,
                "thumbnails": ({side: f"{thumbnail_base}/{preset_id}/thumbnail?side={side}"
                                for side in SIDES} if manifest else None),
            })
        groups.append({"id": category_id, "label": label, "presets": presets})
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "library_version": library.library_version,
        "library_sha256": library.sha256,
        "default_preset": DEFAULT_PRESET,
        "fallback_order": list(library.fallback_order),
        "thumbnails_available": manifest is not None,
        "categories": groups,
    }
