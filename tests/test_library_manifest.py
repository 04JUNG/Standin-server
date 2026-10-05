"""library_manifest.json 계약: 내용 해시 → 버전, 변조 감지, 기동 시 식별자 폴백."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library_manifest import (
    MANIFEST_NAME,
    LibraryManifestError,
    build_manifest,
    content_sha256,
    gate_allows_deploy,
    library_version,
    read_manifest,
    resolve_library_identity,
    verify_manifest,
    write_manifest,
)
from src.repo import FEATURE_VERSION, build_db
from src.schema import LibraryEntry, View
from src.thumbnails import THUMBNAIL_EXTENSION, find_thumbnail, thumbnail_filename

FIXED = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def make_bundle(root: Path, pose_ids=("pose_a", "pose_b")) -> Path:
    (root / "bvh").mkdir(parents=True)
    (root / "thumbs").mkdir()
    entries = []
    for index, pose_id in enumerate(pose_ids):
        (root / "bvh" / f"{pose_id}.bvh").write_text(f"HIERARCHY {pose_id}\n", encoding="utf-8")
        for view in View:
            feature = np.full(34, index + 1, dtype=np.float32)
            entries.append(LibraryEntry(pose_id=pose_id, view=view, feature=feature,
                                        tags={"shot": "full_half"},
                                        bvh_path=f"data/bvh/{pose_id}.bvh"))
            (root / "thumbs" / thumbnail_filename(pose_id, view.value)).write_bytes(
                f"{pose_id}-{view.value}".encode())
    build_db(entries, str(root / "poses.db"))
    return root


def test_version_is_derived_from_content_hash(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    manifest = build_manifest(bundle, created_at=FIXED)

    assert manifest["library_version"] == f"lib-20261002-{manifest['content_sha256'][:8]}"
    assert manifest["feature_version"] == FEATURE_VERSION
    assert manifest["counts"]["poses"] == 2
    assert manifest["counts"]["projections"] == 2 * len(View)
    assert manifest["counts"]["thumbs"] == 2 * len(View)
    assert manifest["parent"] == {"library_version": "v1", "content_sha256": None}


def test_same_content_gives_same_hash_and_irrelevant_files_are_ignored(tmp_path):
    first = make_bundle(tmp_path / "first")
    second = make_bundle(tmp_path / "second")
    assert content_sha256(first) == content_sha256(second)

    before = content_sha256(first)
    (first / "index.pkl").write_bytes(b"legacy pickle")
    (first / "thumbs" / "Thumbs.db").write_bytes(b"windows cache")
    write_manifest(first, build_manifest(first, created_at=FIXED))
    assert content_sha256(first) == before


def test_verify_detects_changed_thumbnail_and_db(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    write_manifest(bundle, build_manifest(bundle, created_at=FIXED))
    verify_manifest(bundle)

    thumb = bundle / "thumbs" / thumbnail_filename("pose_a", "front")
    thumb.write_bytes(b"replaced after the manifest was written")
    with pytest.raises(LibraryManifestError, match="content_sha256"):
        verify_manifest(bundle)
    verify_manifest(bundle, check_content=False)  # 형식·DB만 볼 때는 통과

    (bundle / "poses.db").write_bytes((bundle / "poses.db").read_bytes() + b"\0")
    with pytest.raises(LibraryManifestError, match="db_sha256"):
        verify_manifest(bundle, check_content=False)


def test_verify_rejects_malformed_manifest(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    good = build_manifest(bundle, created_at=FIXED)

    for change, message in (
        ({"schema_version": 2}, "schema_version"),
        ({"library_version": "v1"}, "library_version 형식"),
        ({"library_version": "lib-20261002-00000000"}, "해시 부분"),
        ({"feature_version": FEATURE_VERSION + 1}, "feature_version"),
    ):
        with pytest.raises(LibraryManifestError, match=message):
            verify_manifest(bundle, {**good, **change})

    with pytest.raises(LibraryManifestError, match="없습니다"):
        verify_manifest(bundle)  # 아직 파일로 쓰지 않았다
    (bundle / MANIFEST_NAME).write_text("not json", encoding="utf-8")
    with pytest.raises(LibraryManifestError, match="JSON"):
        read_manifest(bundle)


def test_library_version_requires_full_sha():
    with pytest.raises(LibraryManifestError):
        library_version("abc", FIXED)
    assert library_version("f" * 64, FIXED) == "lib-20261002-ffffffff"


def test_identity_falls_back_to_env_without_or_with_bad_manifest(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    assert resolve_library_identity(bundle, fallback_version="v1").to_dict() == {
        "version": "v1", "source": "env", "content_sha256": None, "db_sha256": None}

    manifest = build_manifest(bundle, created_at=FIXED)
    write_manifest(bundle, manifest)
    identity = resolve_library_identity(bundle, fallback_version="v1", check_content=True)
    assert (identity.version, identity.source) == (manifest["library_version"], "manifest")
    assert identity.content_sha256 == manifest["content_sha256"]

    (bundle / "poses.db").write_bytes((bundle / "poses.db").read_bytes() + b"\0")
    assert resolve_library_identity(bundle, fallback_version="v1").source == "env"
    with pytest.raises(LibraryManifestError):
        resolve_library_identity(bundle, fallback_version="v1", strict=True)


def test_gate_status_controls_deploy():
    assert gate_allows_deploy({"replay_gate": {"status": "passed"}})
    assert gate_allows_deploy({"replay_gate": {"status": "not_required"}})
    for status in ("failed", "not_run", None):
        assert not gate_allows_deploy({"replay_gate": {"status": status}})
    assert not gate_allows_deploy({})


def test_thumbnail_filename_is_the_single_source(tmp_path):
    assert THUMBNAIL_EXTENSION == ".jpg"
    assert thumbnail_filename("2hand Idle_01", "front") == "2hand Idle_01__front.jpg"
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "thumbs" / thumbnail_filename("p", "side")).write_bytes(b"jpg")
    (tmp_path / "thumbs" / "p__front.png").write_bytes(b"png")
    assert find_thumbnail(str(tmp_path), "p", "side") is not None
    assert find_thumbnail(str(tmp_path), "p", "front") is None  # PNG는 더 이상 읽지 않는다


def test_manifest_file_round_trip(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    manifest = build_manifest(bundle, created_at=FIXED,
                              replay_gate={"status": "passed", "report_sha256": "x"})
    path = write_manifest(bundle, manifest)
    assert path.name == MANIFEST_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == read_manifest(bundle) == manifest
