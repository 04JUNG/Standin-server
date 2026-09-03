"""Refined thumbnail and shared batch-renderer contract tests."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from PIL import Image, ImageChops

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.app as api_app
import scripts.render_bvh_thumbnails as batch_renderer
from src.thumbnail_renderer import (
    THUMBNAIL_RENDERER_VERSION,
    render_bvh_thumbnail,
)
from tests.test_smoke import _synthetic_bvh
from scripts.deploy_pose_library import make_archive


def _image_from_payload(payload, expected_view: str) -> Image.Image:
    assert payload.view == expected_view
    assert payload.media_type == "image/png"
    assert payload.encoding == "base64"
    assert payload.renderer_version == THUMBNAIL_RENDERER_VERSION
    return Image.open(io.BytesIO(base64.b64decode(payload.data))).convert("RGB")


def test_batch_and_refine_share_the_same_renderer():
    assert batch_renderer.render_bvh_thumbnail is render_bvh_thumbnail


def test_refine_thumbnail_uses_selected_view_for_path_and_inline_bvh():
    with tempfile.TemporaryDirectory() as directory:
        bvh = Path(_synthetic_bvh(directory, "pose.bvh"))
        front = _image_from_payload(
            api_app._refine_thumbnail(view="front", bvh_path=str(bvh)), "front"
        )
        from_path = _image_from_payload(
            api_app._refine_thumbnail(view="side", bvh_path=str(bvh)), "side"
        )
        from_text = _image_from_payload(
            api_app._refine_thumbnail(
                view="side", bvh_text=bvh.read_text(encoding="utf-8")
            ),
            "side",
        )
        assert from_path.size == from_text.size == (256, 256)
        assert ImageChops.difference(from_path, from_text).getbbox() is None
        assert ImageChops.difference(front, from_path).getbbox() is not None


def test_pose_bundle_includes_thumbnail_manifest():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data"
        thumbs = data / "thumbs"
        thumbs.mkdir(parents=True)
        (data / "poses.db").write_bytes(b"db")
        (thumbs / "thumbnail_manifest.json").write_text(
            '{"schema_version":1}\n', encoding="utf-8"
        )
        archive = root / "bundle.tar.gz"
        make_archive(data, archive)
        with tarfile.open(archive, "r:gz") as bundle:
            assert "thumbs/thumbnail_manifest.json" in bundle.getnames()


def test_batch_renderer_writes_portable_thumbnail_manifest():
    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bvh_dir = root / "bvh"
        output_dir = root / "thumbs"
        bvh_dir.mkdir()
        _synthetic_bvh(str(bvh_dir), "pose.bvh")
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/render_bvh_thumbnails.py"),
                str(bvh_dir),
                str(output_dir),
                "--views",
                "front",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(
            (output_dir / "thumbnail_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["renderer_version"] == THUMBNAIL_RENDERER_VERSION
        assert manifest["views"] == ["front"]
        assert manifest["jobs"][0]["bvh_path"] == "../bvh/pose.bvh"
        assert manifest["jobs"][0]["output"] == "pose__front.png"
        assert manifest["results"][0]["status"] == "rendered"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
