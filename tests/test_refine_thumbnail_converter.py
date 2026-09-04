"""/refine preview가 후보 썸네일과 같은 converter 렌더러를 타는지 고정한다.

배경: 라이브러리 썸네일은 V3.2.5 FBX 남성 모델 렌더인데 /refine preview는 옛 2D
마네킹이어서 두 그림이 달랐다. 여기서는 (1) 조정본은 converter ``/render-thumbnail``로
가고 (2) 베이스는 번들 썸네일을 재사용하며 (3) production에서 converter 주소 누락은
기동 실패, (4) converter 장애는 '그림 없음'으로 수렴하는 계약을 잠근다.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from dataclasses import replace
from email.message import Message
from pathlib import Path
from urllib import error as urllib_error

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.app as api_app
from api.models import RefineRequest
from src.config import CFG
from src.refine_thumbnail import (
    ConverterThumbnailRenderer,
    LIBRARY_RENDERER_VERSION,
    MannequinThumbnailRenderer,
    ThumbnailRenderFailed,
    ThumbnailRendererConfigError,
    ThumbnailSource,
    build_refine_thumbnail_renderer,
)
from tests.test_smoke import _synthetic_bvh


def _png(color=(51, 56, 66), size=256) -> bytes:
    image = Image.new("RGB", (size, size), (158, 158, 158))
    for x in range(size // 4, size * 3 // 4):
        for y in range(size // 8, size * 7 // 8):
            image.putpixel((x, y), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str], status: int = 200):
        self._data = data
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeConverter:
    """urlopen 대역. 받은 multipart를 풀어 두고 지정한 응답을 돌려준다."""

    def __init__(self, *, data: bytes | None = None, headers=None, error=None,
                 media_type="image/png"):
        self.requests = []
        self.data = _png() if data is None else data
        self.headers = {
            "Content-Type": media_type,
            "X-Standin-Thumbnail-Renderer": "fbx-anatomical-v1",
            "X-Standin-Thumbnail-Engine": "BLENDER_EEVEE",
            "X-Standin-Conversion-Id": "conv-1",
            **(headers or {}),
        }
        self.error = error

    def __call__(self, request, timeout=None):
        body = request.data
        boundary = request.get_header("Content-type").split("boundary=")[1]
        parts = {}
        file_bytes = None
        for chunk in body.split(f"--{boundary}".encode()):
            if b'name="' not in chunk:
                continue
            header, _, payload = chunk.partition(b"\r\n\r\n")
            name = header.split(b'name="')[1].split(b'"')[0].decode()
            payload = payload[:-2]  # trailing CRLF
            if b'filename="' in header:
                file_bytes = payload
                parts["__filename__"] = header.split(b'filename="')[1].split(b'"')[0].decode()
            else:
                parts[name] = payload.decode()
        self.requests.append({
            "url": request.full_url, "timeout": timeout, "fields": parts, "bvh": file_bytes,
        })
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.data, self.headers)


def _renderer(fake, **kwargs) -> ConverterThumbnailRenderer:
    defaults = dict(
        character_id="standin-master-v2", timeout_seconds=7.5,
        image_format="png", data_dir=None, reuse_library=True, opener=fake,
    )
    defaults.update(kwargs)
    return ConverterThumbnailRenderer("http://converter.internal:8001/", **defaults)


def test_refined_bvh_is_posted_to_converter_render_thumbnail():
    fake = _FakeConverter()
    renderer = _renderer(fake)
    bvh = "HIERARCHY\nROOT Hips\n{\n}\nMOTION\nFrames: 1\nFrame Time: 0.033\n"
    rendered = renderer.render(ThumbnailSource(
        view="three_quarter", bvh_text=bvh, pose_id="pose", refined=True,
    ))
    assert rendered.data == fake.data
    assert rendered.media_type == "image/png"
    assert rendered.renderer_version == "fbx-anatomical-v1/standin-master-v2"
    assert rendered.origin == "converter"
    request = fake.requests[0]
    assert request["url"] == "http://converter.internal:8001/render-thumbnail"
    assert request["timeout"] == 7.5
    assert request["bvh"] == bvh.encode("utf-8")
    assert request["fields"]["__filename__"] == "refined.bvh"
    assert request["fields"]["view"] == "three_quarter"
    assert request["fields"]["character_id"] == "standin-master-v2"
    assert request["fields"]["size"] == "256"
    assert request["fields"]["format"] == "png"


def test_base_reuses_the_library_thumbnail_without_calling_converter(tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    library = Image.new("RGB", (256, 256), (120, 120, 120))
    library.save(thumbs / "pose__side.jpg", format="JPEG", quality=78)
    fake = _FakeConverter()
    bvh = _synthetic_bvh(str(tmp_path), "pose.bvh")

    png_renderer = _renderer(fake, data_dir=str(tmp_path))
    rendered = png_renderer.render(ThumbnailSource(
        view="side", bvh_path=bvh, pose_id="pose", refined=False,
    ))
    assert fake.requests == []
    assert rendered.origin == "library"
    assert rendered.renderer_version == LIBRARY_RENDERER_VERSION
    assert rendered.media_type == "image/png"
    with Image.open(io.BytesIO(rendered.data)) as image:
        assert image.format == "PNG" and image.size == (256, 256)

    # 같은 포맷이면 번들 바이트를 그대로 돌려준다.
    jpeg_renderer = _renderer(fake, data_dir=str(tmp_path), image_format="jpeg")
    rendered = jpeg_renderer.render(ThumbnailSource(
        view="side", bvh_path=bvh, pose_id="pose", refined=False,
    ))
    assert rendered.data == (thumbs / "pose__side.jpg").read_bytes()
    assert rendered.media_type == "image/jpeg"

    # 번들에 그 view가 없으면 converter로 간다.
    rendered = png_renderer.render(ThumbnailSource(
        view="back", bvh_path=bvh, pose_id="pose", refined=False,
    ))
    assert rendered.origin == "converter"
    assert fake.requests[-1]["fields"]["__filename__"] == "base.bvh"
    assert fake.requests[-1]["bvh"] == Path(bvh).read_bytes()


def test_api_policy_fallback_reuses_the_library_thumbnail(monkeypatch, tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    Image.new("RGB", (256, 256), (120, 120, 120)).save(
        thumbs / "pose__front.jpg", format="JPEG", quality=78
    )
    base = _synthetic_bvh(str(tmp_path), "pose.bvh")
    fake = _FakeConverter()
    monkeypatch.setitem(
        api_app.STATE,
        "thumbnail_renderer",
        _renderer(fake, data_dir=str(tmp_path), image_format="jpeg"),
    )
    monkeypatch.setitem(api_app.STATE, "db_path", str(tmp_path / "unused.db"))
    monkeypatch.setattr(
        api_app,
        "get_pose_meta",
        lambda *_args: {"bvh_path": base, "set_id": None},
    )

    response = api_app.refine(RefineRequest(
        pose_id="pose",
        view="front",
        keypoints=np.zeros((17, 2)).tolist(),
        refine_allowed=False,
    ))

    assert response.refined is False
    assert response.thumbnail is not None
    assert response.thumbnail.renderer_version == LIBRARY_RENDERER_VERSION
    assert fake.requests == []


def test_refined_never_reuses_the_library_thumbnail(tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    Image.new("RGB", (256, 256)).save(thumbs / "pose__front.jpg", format="JPEG")
    fake = _FakeConverter()
    rendered = _renderer(fake, data_dir=str(tmp_path)).render(ThumbnailSource(
        view="front", bvh_text="HIERARCHY\nMOTION\n", pose_id="pose", refined=True,
    ))
    assert rendered.origin == "converter"
    assert len(fake.requests) == 1


def test_library_reuse_can_be_disabled(tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    Image.new("RGB", (256, 256)).save(thumbs / "pose__front.jpg", format="JPEG")
    fake = _FakeConverter()
    bvh = _synthetic_bvh(str(tmp_path), "pose.bvh")
    rendered = _renderer(fake, data_dir=str(tmp_path), reuse_library=False).render(
        ThumbnailSource(view="front", bvh_path=bvh, pose_id="pose", refined=False)
    )
    assert rendered.origin == "converter"


@pytest.mark.parametrize(
    "fake",
    [
        _FakeConverter(error=urllib_error.URLError("connection refused")),
        _FakeConverter(error=urllib_error.HTTPError(
            "http://converter.internal:8001/render-thumbnail", 500, "boom",
            Message(), io.BytesIO(b'{"error":{"code":"THUMBNAIL_RENDER_FAILED"}}'),
        )),
        _FakeConverter(error=TimeoutError("timed out")),
        _FakeConverter(data=b""),
        _FakeConverter(data=b"not an image"),
        _FakeConverter(data=_png(size=128)),
        _FakeConverter(media_type="text/html"),
    ],
)
def test_converter_failures_raise_render_failed(fake):
    with pytest.raises(ThumbnailRenderFailed):
        _renderer(fake).render(ThumbnailSource(
            view="front", bvh_text="HIERARCHY\nMOTION\n", refined=True,
        ))


def test_thumbnail_source_requires_exactly_one_bvh():
    with pytest.raises(ValueError):
        ThumbnailSource(view="front")
    with pytest.raises(ValueError):
        ThumbnailSource(view="front", bvh_path="a.bvh", bvh_text="HIERARCHY")


def test_factory_prefers_converter_and_fails_closed_in_production(monkeypatch):
    mannequin = lambda path, view: Image.new("RGB", (256, 256))  # noqa: E731

    monkeypatch.setattr(CFG, "thumbnail_renderer", "converter")
    monkeypatch.setattr(CFG, "thumbnail_converter_url", "http://converter.internal:8001")
    renderer = build_refine_thumbnail_renderer(CFG, mannequin_render=mannequin)
    assert isinstance(renderer, ConverterThumbnailRenderer)
    assert renderer.base_url == "http://converter.internal:8001"
    assert renderer.character_id == CFG.thumbnail_character_id
    assert renderer.data_dir == CFG.data_dir

    monkeypatch.setattr(CFG, "thumbnail_converter_url", "")
    monkeypatch.setattr(CFG, "app_env", "production")
    with pytest.raises(ThumbnailRendererConfigError, match="REFINE_THUMBNAIL_CONVERTER_URL"):
        build_refine_thumbnail_renderer(CFG, mannequin_render=mannequin)

    # 개발 모드는 오프라인에서도 돌아야 하므로 경고 후 마네킹으로 폴백한다.
    monkeypatch.setattr(CFG, "app_env", "development")
    renderer = build_refine_thumbnail_renderer(CFG, mannequin_render=mannequin)
    assert isinstance(renderer, MannequinThumbnailRenderer)

    # 명시적으로 옛 렌더러를 고르면 production에서도 그것을 쓴다(비상 복구).
    monkeypatch.setattr(CFG, "app_env", "production")
    monkeypatch.setattr(CFG, "thumbnail_renderer", "mannequin")
    renderer = build_refine_thumbnail_renderer(CFG, mannequin_render=mannequin)
    assert isinstance(renderer, MannequinThumbnailRenderer)


def test_api_refine_thumbnail_uses_converter_renderer_and_survives_failure(monkeypatch):
    fake = _FakeConverter()
    renderer = _renderer(fake)
    monkeypatch.setitem(api_app.STATE, "thumbnail_renderer", renderer)
    with tempfile.TemporaryDirectory() as directory:
        bvh = Path(_synthetic_bvh(directory, "pose.bvh"))
        payload = api_app._refine_thumbnail(
            view="front", bvh_text=bvh.read_text(encoding="utf-8"),
            pose_id="pose", refined=True,
        )
        assert payload is not None
        assert payload.media_type == "image/png"
        assert payload.renderer_version == "fbx-anatomical-v1/standin-master-v2"
        assert base64.b64decode(payload.data, validate=True) == fake.data
        assert fake.requests[0]["fields"]["__filename__"] == "refined.bvh"

        # refined를 생략하면 bvh_text 유무로 판단한다(기존 호출 호환).
        api_app._refine_thumbnail(view="front", bvh_path=str(bvh))
        assert fake.requests[-1]["fields"]["__filename__"] == "base.bvh"

        broken = _FakeConverter(error=urllib_error.URLError("down"))
        monkeypatch.setitem(api_app.STATE, "thumbnail_renderer", _renderer(broken))
        assert api_app._refine_thumbnail(view="front", bvh_path=str(bvh)) is None


def test_healthz_reports_the_thumbnail_renderer(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setitem(api_app.STATE, "pipeline", object())
    monkeypatch.setitem(api_app.STATE, "pose_count", 3)
    monkeypatch.setitem(api_app.STATE, "thumbnail_renderer", _renderer(_FakeConverter()))
    body = TestClient(api_app.app).get("/healthz").json()
    assert body["refine_thumbnail"]["renderer"] == "converter"
    assert body["refine_thumbnail"]["configured"] == CFG.thumbnail_renderer
    assert body["refine_thumbnail"]["format"] in {"png", "jpeg"}


def test_env_example_and_deploy_workflow_carry_the_converter_url():
    root = Path(__file__).resolve().parent.parent
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    assert "REFINE_THUMBNAIL_RENDERER=" in env_example
    assert "REFINE_THUMBNAIL_CONVERTER_URL=" in env_example
    workflow = (root / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    assert "Verify refine thumbnail converter URL" in workflow
    assert 'select(.name == "REFINE_THUMBNAIL_CONVERTER_URL"' in workflow
    # 주소는 infra 소유(task definition). 워크플로가 vars로 덮어쓰지 않는다.
    assert "REFINE_THUMBNAIL_CONVERTER_URL=${{" not in workflow
    assert "REFINE_THUMBNAIL_RENDERER=${{ vars.REFINE_THUMBNAIL_RENDERER || 'converter' }}" in workflow


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
