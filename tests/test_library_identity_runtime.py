"""기동 시 번들 manifest → 응답의 pose_library_version (api/app.py::_resolve_library_identity)."""
from __future__ import annotations

from datetime import datetime, timezone
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.app as api_app
from api.models import InferenceMetadataOut
from src.config import CFG
from src.library_manifest import build_manifest, write_manifest
from tests.test_library_manifest import make_bundle

FIXED = datetime(2026, 10, 6, tzinfo=timezone.utc)


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    root = make_bundle(tmp_path / "bundle")
    monkeypatch.setattr(api_app, "DB_PATH", str(root / "poses.db"))
    monkeypatch.setattr(CFG, "pose_library_version", "v1")
    monkeypatch.setattr(CFG, "app_env", "production")
    return root


def test_manifest_version_replaces_env_everywhere(bundle):
    manifest = build_manifest(bundle, created_at=FIXED)
    write_manifest(bundle, manifest)

    identity = api_app._resolve_library_identity()

    assert identity["source"] == "manifest"
    assert identity["version"] == manifest["library_version"] == CFG.pose_library_version
    assert identity["content_sha256"] == manifest["content_sha256"]


def test_bundle_without_manifest_keeps_env_version(bundle):
    identity = api_app._resolve_library_identity()
    assert identity == {"version": "v1", "source": "env", "content_sha256": None,
                        "db_sha256": None}
    assert CFG.pose_library_version == "v1"


def test_mismatched_manifest_blocks_production_startup(bundle):
    write_manifest(bundle, build_manifest(bundle, created_at=FIXED))
    (bundle / "poses.db").write_bytes((bundle / "poses.db").read_bytes() + b"\0")

    with pytest.raises(api_app.StartupError, match="library_manifest.json"):
        api_app._resolve_library_identity()
    assert CFG.pose_library_version == "v1"


def test_mismatched_manifest_falls_back_to_env_in_development(bundle, monkeypatch):
    monkeypatch.setattr(CFG, "app_env", "development")
    write_manifest(bundle, build_manifest(bundle, created_at=FIXED))
    (bundle / "poses.db").write_bytes((bundle / "poses.db").read_bytes() + b"\0")

    assert api_app._resolve_library_identity()["source"] == "env"
    assert CFG.pose_library_version == "v1"


def test_healthz_reports_the_loaded_library(monkeypatch):
    from fastapi.testclient import TestClient

    library = {"version": "lib-20261006-abcdef12", "source": "manifest",
               "content_sha256": "abcdef12" + "0" * 56, "db_sha256": "1" * 64}
    monkeypatch.setitem(api_app.STATE, "pipeline", object())
    monkeypatch.setitem(api_app.STATE, "pose_count", 3)
    monkeypatch.setitem(api_app.STATE, "pose_library", library)
    body = TestClient(api_app.app).get("/healthz").json()
    assert body["pose_library"] == library


def test_inference_metadata_carries_the_bundle_hash():
    common = dict(deployment_version="x", vlm_provider="mock", vlm_model="mock",
                  pose_backend="mock", pose_model_version="m", feature_version=1)
    legacy = InferenceMetadataOut(pose_library_version="v1", **common)
    assert legacy.pose_library_sha256 is None
    current = InferenceMetadataOut(pose_library_version="lib-20261006-abcdef12",
                                   pose_library_sha256="abcdef12" + "0" * 56, **common)
    assert current.model_dump()["pose_library_sha256"].startswith("abcdef12")
