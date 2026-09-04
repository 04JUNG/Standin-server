"""/refine 결과 썸네일 렌더러 — 후보 썸네일과 같은 그림을 만드는 두 구현과 팩토리.

배경: 2026-09-03부터 포즈 라이브러리 썸네일은 V3.2.5 converter가 변환한 FBX(남성
마스터 모델)를 Blender로 렌더한 것이다. 그런데 /refine 응답의 preview는 옛 2D 마네킹
(`src/thumbnail_renderer.py`, warm-mannequin-v1)이 그려서, 작가가 고른 후보와 조정
결과가 다른 그림으로 보였다. 이 모듈은 preview도 converter 서비스의
``POST /render-thumbnail``(같은 캐릭터·solver·카메라·재질)로 그리게 한다.

인터페이스는 하나다 — ``render(source) -> RenderedThumbnail``. 구현 선택은
``build_refine_thumbnail_renderer``가 env로 한다(CLAUDE.md 기능 격리 규칙 2:
호출부에서 if 분기 금지). 실패 정책은 호출부(`api/app.py::_refine_thumbnail`)가
갖는다: 어떤 예외든 "그림 없음"으로 수렴하고 조정 결과는 유지한다.
"""
from __future__ import annotations

import io
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image

from .logging_setup import log_info, log_warn
from .thumbnails import find_thumbnail

MANNEQUIN_RENDERER_VERSION = "warm-mannequin-v1"
THUMBNAIL_SIZE = 256
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg"}
LIBRARY_JPEG_QUALITY = 78   # 번들 thumbnail_manifest.json의 quality


@dataclass(frozen=True)
class ThumbnailSource:
    """무엇을 그릴지. ``bvh_text``는 조정본, ``bvh_path``는 베이스 파일 경로다."""

    view: str
    bvh_path: Optional[str] = None
    bvh_text: Optional[str] = None
    pose_id: Optional[str] = None
    refined: bool = False

    def __post_init__(self) -> None:
        if (self.bvh_path is None) == (self.bvh_text is None):
            raise ValueError("provide exactly one thumbnail BVH source")

    def read_text(self) -> str:
        if self.bvh_text is not None:
            return self.bvh_text
        with open(self.bvh_path, "r", encoding="utf-8") as handle:  # type: ignore[arg-type]
            return handle.read()


@dataclass(frozen=True)
class RenderedThumbnail:
    data: bytes
    media_type: str
    renderer_version: str
    width: int = THUMBNAIL_SIZE
    height: int = THUMBNAIL_SIZE
    # 관측용. 응답 계약에는 싣지 않는다.
    origin: str = "render"


class RefineThumbnailRenderer(Protocol):
    name: str

    def render(self, source: ThumbnailSource) -> RenderedThumbnail: ...


class ThumbnailRenderFailed(RuntimeError):
    """렌더러가 그림을 못 만들었다. 호출부는 이것을 '그림 없음'으로 다룬다."""


# ─────────────────────────────────────────────────────────────────────────────
# 1) 옛 2D 마네킹 — 비상 복구/오프라인용
# ─────────────────────────────────────────────────────────────────────────────
class MannequinThumbnailRenderer:
    """`src/thumbnail_renderer.render_bvh_thumbnail`을 그대로 쓴다.

    ``render_fn``을 주입받는 이유: 기존 테스트가 ``api.app.render_bvh_thumbnail``을
    바꿔 끼워 실패를 흉내 낸다. 호출 시점에 그 이름을 찾도록 호출부가 람다를 넘긴다.
    """

    name = "mannequin"

    def __init__(self, render_fn: Callable[[str, str], Image.Image], image_format: str = "png"):
        if image_format not in MEDIA_TYPES:
            raise ValueError(f"unsupported thumbnail format: {image_format}")
        self._render = render_fn
        self._format = image_format

    def render(self, source: ThumbnailSource) -> RenderedThumbnail:
        if source.bvh_text is None:
            image = self._render(str(source.bvh_path), source.view)
        else:
            with tempfile.TemporaryDirectory(prefix="standin-refine-thumbnail-") as directory:
                temporary = os.path.join(directory, "refined.bvh")
                with open(temporary, "w", encoding="utf-8", newline="\n") as sink:
                    sink.write(source.bvh_text)
                image = self._render(temporary, source.view)
        return RenderedThumbnail(
            data=_encode(image, self._format),
            media_type=MEDIA_TYPES[self._format],
            renderer_version=MANNEQUIN_RENDERER_VERSION,
            origin="mannequin",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2) converter 서비스 — 라이브러리 썸네일과 같은 V3.2.5 FBX 렌더
# ─────────────────────────────────────────────────────────────────────────────
class ConverterThumbnailRenderer:
    """``POST {base_url}/render-thumbnail``로 그린다.

    - refined=false이고 번들에 같은 view의 후보 썸네일이 있으면 그 파일을 그대로
      돌려준다(``reuse_library``). 베이스 BVH를 같은 렌더러로 이미 구운 것이므로
      결과가 같고 converter 왕복(수 초)을 아낀다.
    - 그 외에는 converter를 부른다. 응답 헤더의 renderer 버전을 그대로 싣는다.
    """

    name = "converter"

    def __init__(
        self,
        base_url: str,
        *,
        character_id: str,
        timeout_seconds: float,
        image_format: str = "png",
        data_dir: Optional[str] = None,
        reuse_library: bool = True,
        size: int = THUMBNAIL_SIZE,
        opener: Optional[Callable[..., object]] = None,
    ):
        if not base_url:
            raise ValueError("converter base_url is required")
        if image_format not in MEDIA_TYPES:
            raise ValueError(f"unsupported thumbnail format: {image_format}")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.character_id = character_id
        self.timeout_seconds = float(timeout_seconds)
        self.image_format = image_format
        self.data_dir = data_dir
        self.reuse_library = reuse_library
        self.size = int(size)
        self._open = opener or urllib_request.urlopen

    # -- 라이브러리 재사용 --------------------------------------------------
    def _library_thumbnail(self, source: ThumbnailSource) -> Optional[RenderedThumbnail]:
        if source.refined or not self.reuse_library or not source.pose_id or not self.data_dir:
            return None
        path = find_thumbnail(self.data_dir, source.pose_id, source.view)
        if path is None:
            return None
        try:
            with Image.open(path) as image:
                bundled_format = (image.format or "").lower()
                bundled_size = image.size
                rgb = image.convert("RGB")
        except (OSError, ValueError) as exc:
            log_warn("refine_thumbnail_library_unreadable",
                     "번들 썸네일을 읽지 못해 converter로 넘어간다",
                     poseId=source.pose_id, view=source.view, error=str(exc)[:200])
            return None
        if bundled_format == self.image_format and bundled_size == (self.size, self.size):
            data = path.read_bytes()             # 번들 바이트 그대로(재인코딩 없음)
        else:
            if bundled_size != (self.size, self.size):
                rgb = rgb.resize((self.size, self.size), Image.Resampling.LANCZOS)
            data = _encode(rgb, self.image_format)
        return RenderedThumbnail(
            data=data,
            media_type=MEDIA_TYPES[self.image_format],
            renderer_version=LIBRARY_RENDERER_VERSION,
            origin="library",
        )

    # -- converter 호출 ----------------------------------------------------
    def render(self, source: ThumbnailSource) -> RenderedThumbnail:
        reused = self._library_thumbnail(source)
        if reused is not None:
            return reused
        bvh_text = source.read_text()
        body, content_type = _multipart(
            fields={
                "character_id": self.character_id,
                "view": source.view,
                "size": str(self.size),
                "format": self.image_format,
                "quality": str(LIBRARY_JPEG_QUALITY),
            },
            file_field="bvh",
            filename="refined.bvh" if source.refined else "base.bvh",
            file_bytes=bvh_text.encode("utf-8"),
        )
        request = urllib_request.Request(
            f"{self.base_url}/render-thumbnail",
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Accept": MEDIA_TYPES[self.image_format],
                "User-Agent": "standin-inference/refine-thumbnail",
            },
        )
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:  # type: ignore[call-arg]
                data = response.read()
                headers = response.headers
                status = getattr(response, "status", 200)
        except urllib_error.HTTPError as exc:
            detail = exc.read()[:300] if hasattr(exc, "read") else b""
            raise ThumbnailRenderFailed(
                f"converter returned HTTP {exc.code}: {detail!r}"
            ) from exc
        except (urllib_error.URLError, OSError, TimeoutError) as exc:
            raise ThumbnailRenderFailed(f"converter unreachable: {exc}") from exc
        if status != 200:
            raise ThumbnailRenderFailed(f"converter returned HTTP {status}")
        media_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if media_type != MEDIA_TYPES[self.image_format]:
            raise ThumbnailRenderFailed(f"converter returned {media_type!r}")
        if not data:
            raise ThumbnailRenderFailed("converter returned an empty body")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.size != (self.size, self.size):
                    raise ThumbnailRenderFailed(f"converter returned {image.size} thumbnail")
        except (OSError, ValueError) as exc:
            raise ThumbnailRenderFailed("converter returned an undecodable image") from exc
        renderer = headers.get("X-Standin-Thumbnail-Renderer") or "fbx-anatomical"
        engine = headers.get("X-Standin-Thumbnail-Engine") or "unknown"
        log_info("refine_thumbnail_rendered", "converter가 preview를 그렸다",
                 view=source.view, refined=source.refined, engine=engine,
                 renderer=renderer, bytes=len(data),
                 conversionId=headers.get("X-Standin-Conversion-Id"))
        return RenderedThumbnail(
            data=data,
            media_type=MEDIA_TYPES[self.image_format],
            renderer_version=f"{renderer}/{self.character_id}",
            origin="converter",
        )


LIBRARY_RENDERER_VERSION = "fbx-anatomical-v1/library"


def _encode(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    if image_format == "jpeg":
        rgb.save(buffer, format="JPEG", quality=LIBRARY_JPEG_QUALITY, optimize=True)
    else:
        rgb.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _multipart(
    *, fields: dict[str, str], file_field: str, filename: str, file_bytes: bytes,
) -> tuple[bytes, str]:
    boundary = f"standin-{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(value.encode("utf-8") + b"\r\n")
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    lines.append(file_bytes + b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리
# ─────────────────────────────────────────────────────────────────────────────
class ThumbnailRendererConfigError(RuntimeError):
    """production에서 converter 렌더러를 골랐는데 주소가 없다 — 기동 실패."""


def build_refine_thumbnail_renderer(
    cfg, *, mannequin_render: Callable[[str, str], Image.Image],
) -> RefineThumbnailRenderer:
    """env로 구현을 고른다. mannequin은 명시했을 때만 쓴다.

    converter인데 URL이 비어 있으면: production은 기동 실패(옛 마네킹을 몰래 서빙하면
    이 변경의 목적 자체가 조용히 사라진다), 개발은 경고 후 마네킹으로 폴백(오프라인
    개발·테스트는 converter 없이 돌아야 한다).
    """
    choice = cfg.thumbnail_renderer
    if choice == "mannequin":
        return MannequinThumbnailRenderer(mannequin_render, cfg.thumbnail_format)
    if choice != "converter":
        raise ThumbnailRendererConfigError(f"unknown thumbnail renderer: {choice!r}")
    if not cfg.thumbnail_converter_url:
        if cfg.is_production:
            raise ThumbnailRendererConfigError(
                "REFINE_THUMBNAIL_CONVERTER_URL이 비어 있습니다. converter 서비스 주소"
                "(BFF가 쓰는 것과 같은 내부 주소, 예: http://standin-converter.internal:8001)"
                "를 지정하거나, 옛 마네킹을 의도적으로 쓰려면 "
                "REFINE_THUMBNAIL_RENDERER=mannequin을 명시하세요."
            )
        log_warn("thumbnail_renderer_fallback",
                 "REFINE_THUMBNAIL_CONVERTER_URL 없음 → 개발 모드라 옛 마네킹 렌더러로 폴백",
                 errorCode="THUMBNAIL_CONVERTER_URL_MISSING")
        return MannequinThumbnailRenderer(mannequin_render, cfg.thumbnail_format)
    return ConverterThumbnailRenderer(
        cfg.thumbnail_converter_url,
        character_id=cfg.thumbnail_character_id,
        timeout_seconds=cfg.thumbnail_timeout_seconds,
        image_format=cfg.thumbnail_format,
        data_dir=cfg.data_dir,
        reuse_library=cfg.thumbnail_reuse_library,
    )


__all__ = [
    "ConverterThumbnailRenderer",
    "MannequinThumbnailRenderer",
    "RefineThumbnailRenderer",
    "RenderedThumbnail",
    "ThumbnailRenderFailed",
    "ThumbnailRendererConfigError",
    "ThumbnailSource",
    "build_refine_thumbnail_renderer",
]
