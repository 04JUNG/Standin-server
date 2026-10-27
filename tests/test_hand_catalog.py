"""손 카탈로그·썸네일(04J-290, src/hand_catalog.py)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converter.hand_library import load_library
from src.hand_catalog import THUMBS_DIR, _thumb_manifest, build_catalog, thumbnail_path


def test_catalog_lists_every_preset_once_in_library_order():
    library = load_library()
    catalog = build_catalog()
    ids = [p["id"] for group in catalog["categories"] for p in group["presets"]]
    assert sorted(ids) == sorted(library.preset_ids) and len(ids) == len(set(ids))
    assert [g["id"] for g in catalog["categories"]] == list(library.categories)
    assert catalog["library_sha256"] == library.sha256
    assert catalog["default_preset"] in library.presets
    assert all(p["label"] and p["description"] for g in catalog["categories"] for p in g["presets"])


def test_shipped_thumbnails_match_the_current_library():
    """라이브러리 각도를 바꾸고 썸네일을 다시 안 만들면 이 테스트가 실패한다."""
    catalog = build_catalog()
    assert catalog["thumbnails_available"], "run scripts/render_hand_mesh_thumbnails.py in Blender"
    for preset_id in load_library().preset_ids:
        for side in ("left", "right"):
            path = thumbnail_path(preset_id, side)
            assert path is not None and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", (preset_id, side)


def test_shipped_hand_mesh_previews_are_complete_and_identify_the_approved_character():
    import struct
    manifest = json.loads((THUMBS_DIR / "manifest.json").read_text())
    assert manifest["renderer"] == "default-human-hand-mesh-v1"
    assert manifest["style"] == "approved-reference-anatomy-detail-dark-v1"
    assert manifest["character_sha256"] == "7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d"
    expected = {f"{preset}__{side}.png" for preset in load_library().preset_ids
                for side in ("left", "right")}
    assert set(manifest["files"]) == expected
    pixels = manifest["size_css_px"] * manifest["scale"]
    for name in expected:
        assert struct.unpack(">II", (THUMBS_DIR / name).read_bytes()[16:24]) == (pixels, pixels)
    for preset in load_library().preset_ids:
        assert manifest["files"][f"{preset}__left.png"] != manifest["files"][f"{preset}__right.png"]


def test_stale_or_tampered_thumbnails_are_not_served(tmp_path):
    stale = tmp_path / "stale"
    shutil.copytree(THUMBS_DIR, stale)
    manifest = json.loads((stale / "manifest.json").read_text(encoding="utf-8"))
    manifest["library_sha256"] = "0" * 64
    (stale / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _thumb_manifest.cache_clear()
    assert thumbnail_path("fist", "left", str(stale)) is None
    assert build_catalog(directory=str(stale))["thumbnails_available"] is False

    tampered = tmp_path / "tampered"
    shutil.copytree(THUMBS_DIR, tampered)
    (tampered / "fist__left.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-the-same")
    _thumb_manifest.cache_clear()
    assert thumbnail_path("fist", "left", str(tampered)) is None
    assert thumbnail_path("fist", "right", str(tampered)) is not None
    _thumb_manifest.cache_clear()


def test_manifest_without_pngs_reports_no_thumbnails(tmp_path):
    """이미지에 manifest만 들어가고 PNG가 빠지면(.dockerignore *.png) 깨진 URL을 내보내지 않는다."""
    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copy(THUMBS_DIR / "manifest.json", bare / "manifest.json")
    _thumb_manifest.cache_clear()
    catalog = build_catalog(directory=str(bare))
    assert catalog["thumbnails_available"] is False
    assert all(p["thumbnails"] is None for c in catalog["categories"] for p in c["presets"])
    _thumb_manifest.cache_clear()


def test_dockerignore_keeps_hand_thumbnails_in_the_image():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "*.png" in lines
    assert lines.index("!converter/hand_thumbs/*.png") > lines.index("*.png")


def test_unknown_preset_or_side_has_no_thumbnail():
    assert thumbnail_path("tentacle", "left") is None
    assert thumbnail_path("fist", "middle") is None
