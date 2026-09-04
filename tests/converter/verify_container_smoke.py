#!/usr/bin/env python3
"""Verify the real container HTTP smoke response without test-only packages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import uuid
import zipfile
import zlib


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--bundle-headers", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--thumbnail-headers", type=Path)
    parser.add_argument("--thumbnail", type=Path)
    return parser.parse_args()


def _png_pixels(data: bytes) -> tuple[int, int, set[bytes]]:
    """표준 라이브러리만으로 8-bit RGB PNG를 풀어 크기와 색 집합을 돌려준다."""
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), "thumbnail is not a PNG"
    offset = 8
    width = height = 0
    channels = 0
    idat = bytearray()
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            assert depth == 8, f"unexpected PNG bit depth {depth}"
            assert color in (2, 6), f"unexpected PNG color type {color}"
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            idat.extend(body)
        elif kind == b"IEND":
            break
        offset += 12 + length
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    colors: set[bytes] = set()
    previous = bytearray(stride)
    position = 0
    for _row in range(height):
        filter_type = raw[position]
        line = bytearray(raw[position + 1:position + 1 + stride])
        position += 1 + stride
        for index in range(stride):
            left = line[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                line[index] = (line[index] + left) & 0xFF
            elif filter_type == 2:
                line[index] = (line[index] + up) & 0xFF
            elif filter_type == 3:
                line[index] = (line[index] + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:
                paeth = left + up - upper_left
                pa, pb, pc = abs(paeth - left), abs(paeth - up), abs(paeth - upper_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else upper_left
                line[index] = (line[index] + predictor) & 0xFF
        for index in range(0, stride, channels):
            colors.add(bytes(line[index:index + 3]))
        previous = line
    return width, height, colors


def _headers(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="iso-8859-1").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        rows[name.strip().lower()] = value.strip()
    return rows


def main() -> int:
    args = _args()
    health = json.loads(args.health.read_text(encoding="utf-8"))
    assert health["ok"] is True
    assert health["solver_version"] == "chain-transport-v3.2.5"
    assert health["checks"]["blender"] == {
        "ok": True,
        "version": "5.2.0",
        "build_hash": "fbe6228777e7",
    }
    assert health["checks"]["tempdir"]["ok"] is True
    assert health["checks"]["default_character"]["ok"] is True

    headers = _headers(args.headers)
    artifact = args.fbx.read_bytes()
    bvh = args.bvh.read_bytes()
    assert len(artifact) > 1_000
    assert artifact.startswith(b"Kaydara FBX Binary")
    assert headers["content-type"].startswith("application/octet-stream")
    assert headers["x-standin-solver-version"] == "chain-transport-v3.2.5"
    assert headers["x-standin-source-bvh-sha256"] == hashlib.sha256(bvh).hexdigest()
    assert headers["x-standin-artifact-sha256"] == hashlib.sha256(artifact).hexdigest()
    assert headers["x-standin-source-profile"] == "mixamo_noprefix"
    assert headers["x-standin-target-profile"] == "mixamo"
    assert headers["x-standin-mapped-bones"] == "22"
    assert int(headers["x-standin-warning-count"]) >= 0
    uuid.UUID(headers["x-standin-conversion-id"])
    assert int(headers["content-length"]) == len(artifact)

    bundle_headers = _headers(args.bundle_headers)
    bundle_bytes = args.bundle.read_bytes()
    assert bundle_headers["content-type"].startswith("application/zip")
    assert bundle_headers["x-standin-artifact-kind"] == "base"
    assert bundle_headers["x-standin-artifact-sha256"] == hashlib.sha256(
        bundle_bytes
    ).hexdigest()
    assert bundle_headers["x-standin-bundle-sha256"] == bundle_headers[
        "x-standin-artifact-sha256"
    ]
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
        assert bundle.namelist() == ["final.bvh", "final.fbx", "manifest.json"]
        bundled_bvh = bundle.read("final.bvh")
        bundled_fbx = bundle.read("final.fbx")
        manifest = json.loads(bundle.read("manifest.json"))
    assert bundled_bvh == bvh
    assert len(bundled_fbx) > 1_000
    assert bundled_fbx.startswith(b"Kaydara FBX Binary")
    assert manifest["artifact_kind"] == "base"
    assert manifest["artifacts"]["bvh"]["sha256"] == hashlib.sha256(
        bundled_bvh
    ).hexdigest()
    assert manifest["artifacts"]["fbx"]["sha256"] == hashlib.sha256(
        bundled_fbx
    ).hexdigest()
    assert bundle_headers["x-standin-source-bvh-sha256"] == manifest[
        "artifacts"
    ]["bvh"]["sha256"]
    assert bundle_headers["x-standin-fbx-artifact-sha256"] == manifest[
        "artifacts"
    ]["fbx"]["sha256"]
    uuid.UUID(bundle_headers["x-standin-conversion-id"])
    assert int(bundle_headers["content-length"]) == len(bundle_bytes)

    if args.thumbnail is not None or args.thumbnail_headers is not None:
        assert args.thumbnail is not None and args.thumbnail_headers is not None
        thumbnail_headers = _headers(args.thumbnail_headers)
        thumbnail = args.thumbnail.read_bytes()
        assert thumbnail_headers["content-type"].startswith("image/png")
        assert thumbnail_headers["x-standin-thumbnail-renderer"] == "fbx-anatomical-v1"
        assert thumbnail_headers["x-standin-thumbnail-view"] == "front"
        assert thumbnail_headers["x-standin-thumbnail-size"] == "256"
        assert thumbnail_headers["x-standin-thumbnail-engine"] in {"BLENDER_EEVEE", "CYCLES"}
        assert thumbnail_headers["x-standin-thumbnail-sha256"] == hashlib.sha256(
            thumbnail
        ).hexdigest()
        assert thumbnail_headers["x-standin-source-bvh-sha256"] == hashlib.sha256(
            bvh
        ).hexdigest()
        uuid.UUID(thumbnail_headers["x-standin-conversion-id"])
        assert int(thumbnail_headers["content-length"]) == len(thumbnail)
        width, height, colors = _png_pixels(thumbnail)
        assert (width, height) == (256, 256), (width, height)
        # 현재 라이브러리와 같은 밝은 남성 모델 + 회색 배경이어야 한다. 캐릭터가
        # 반드시 절대 명도 90 미만이어야 한다는 옛 실루엣 가정은 합성 CI 리그와
        # 실제 라이브러리 렌더 모두에 맞지 않으므로, 충분한 명도 범위를 검증한다.
        assert len(colors) >= 32, f"thumbnail is nearly uniform ({len(colors)} colors)"
        luminances = [sum(color) / 3 for color in colors]
        assert max(luminances) > 120, "no background/model tone"
        assert max(luminances) - min(luminances) >= 24, "no visible character contrast"
        print(
            "[PASS] converter container thumbnail HTTP smoke "
            f"(engine={thumbnail_headers['x-standin-thumbnail-engine']}, colors={len(colors)})"
        )

    print("[PASS] converter container FBX and BVH+FBX bundle HTTP smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
