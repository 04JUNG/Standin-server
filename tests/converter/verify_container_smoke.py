#!/usr/bin/env python3
"""Verify the real container HTTP smoke response without test-only packages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import uuid
import zipfile


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--bundle-headers", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    return parser.parse_args()


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
    print("[PASS] converter container FBX and BVH+FBX bundle HTTP smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
