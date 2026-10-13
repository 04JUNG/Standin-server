"""시맨틱 태그의 번들 포함·배포 검증 계약(04J-285).

- attach-semantic은 library_version·content_sha256을 바꾸지 않는다.
- 배포 검증기는 semantic/ 변조·다른 DB 기준 태그를 막는다.
- 서버 기동 식별자(strict)는 시맨틱 불일치로 실패하지 않는다(기하 검색 보호).
- 서버 기본 경로(DATA_DIR/semantic/)로 색인이 올라간다.

    python tests/test_semantic_bundle.py
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_pose_bundle import BundleError, attach_semantic
from scripts.deploy_pose_library import DeployError, make_archive, validate
from src.library_manifest import (LibraryManifestError, build_manifest, content_sha256,
                                  read_manifest, resolve_library_identity, sha256_file,
                                  verify_manifest, write_manifest)
from src.repo import build_db
from src.schema import LibraryEntry, View
from src.semantic_search import SemanticSearchService
from src.thumbnails import thumbnail_filename
from tests.test_smoke import _synthetic_bvh

POSES = {
    "pitch_a": ([("action", "throw", "던지기"), ("activity", "baseball", "야구")],
                "팔을 머리 위로 휘둘러 공을 던지는 자세"),
    "rest_b": ([("action", "rest", "쉬기"), ("posture", "sitting", "앉기")],
               "의자에 앉아 쉬는 자세"),
}
TAXONOMY = {
    "version": "2.0",
    "groups": [{"key": "pose_action", "fields": ["action", "posture"]},
               {"key": "scene", "fields": ["domain", "activity"]},
               {"key": "objects", "fields": ["object"]},
               {"key": "meaning", "fields": ["emotion", "intent"]}],
    "vocabulary": {"action": {"throw": "던지기", "rest": "쉬기"},
                   "posture": {"sitting": "앉기"},
                   "activity": {"baseball": "야구"}},
}
GROUP = {"action": "pose_action", "posture": "pose_action", "activity": "scene"}


def make_bundle(root: Path) -> Path:
    (root / "bvh").mkdir(parents=True)
    (root / "thumbs").mkdir()
    entries = []
    for i, pose_id in enumerate(POSES):
        _synthetic_bvh(str(root / "bvh"), f"{pose_id}.bvh")  # 운영 변환기가 아는 리그
        for view in View:
            entries.append(LibraryEntry(pose_id=pose_id, view=view,
                                        feature=np.full(34, i + 1, dtype=np.float32),
                                        tags={"shot": "full_half"},
                                        bvh_path=f"data/bvh/{pose_id}.bvh"))
            (root / "thumbs" / thumbnail_filename(pose_id, view.value)).write_bytes(b"x")
    build_db(entries, str(root / "poses.db"))
    manifest = build_manifest(root, replay_gate={"status": "passed"},
                              privacy_scan={"status": "passed"}, compat={"status": "passed"})
    write_manifest(root, manifest)
    return root


def make_tags(tmp: Path, db: Path, *, db_sha: str | None = None) -> tuple[Path, Path]:
    entries = []
    for pose_id, (tags, desc) in POSES.items():
        groups = {"pose_action": [], "scene": [], "objects": [], "meaning": []}
        for f, v, label in tags:
            groups[GROUP[f]].append({"field": f, "value": v, "label": label})
        entries.append({"pose_id": pose_id, "groups": groups, "description": {"text": desc}})
    tags_path = tmp / "search-v2.json"
    tags_path.write_text(json.dumps({"version": "2.0", "source_db_sha256": db_sha or sha256_file(db),
                                     "poses": entries}, ensure_ascii=False), encoding="utf-8")
    tax_path = tmp / "search-taxonomy-v2.json"
    tax_path.write_text(json.dumps(TAXONOMY, ensure_ascii=False), encoding="utf-8")
    return tags_path, tax_path


def test_attach_keeps_library_version_and_content_hash():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = make_bundle(tmp / "bundle")
        before = read_manifest(bundle)
        tags, tax = make_tags(tmp, bundle / "poses.db")
        manifest = attach_semantic(bundle, tags, tax)
        assert manifest["library_version"] == before["library_version"]
        assert content_sha256(bundle) == before["content_sha256"]
        assert manifest["replay_gate"] == before["replay_gate"]
        sem = manifest["semantic"]
        assert sem["status"] == "passed" and sem["tagged_poses"] == 2 and sem["untagged_poses"] == 0
        verify_manifest(bundle)  # 해시·semantic 절 모두 통과


def test_tag_tampering_is_caught_by_deploy_but_not_by_server_startup():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = make_bundle(tmp / "bundle")
        tags, tax = make_tags(tmp, bundle / "poses.db")
        attach_semantic(bundle, tags, tax)
        with (bundle / "semantic" / "search-v2.json").open("a", encoding="utf-8") as fh:
            fh.write(" ")
        try:
            verify_manifest(bundle)
        except LibraryManifestError as exc:
            assert "semantic" in str(exc)
        else:
            raise AssertionError("tampered semantic tags must fail deploy verification")
        try:
            validate(bundle)
        except DeployError:
            pass
        else:
            raise AssertionError("deploy validator must refuse tampered tags")
        identity = resolve_library_identity(bundle, fallback_version="v1", strict=True)
        assert identity.source == "manifest"  # 기하 검색 기동은 막지 않는다


def test_tags_built_for_another_db_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = make_bundle(tmp / "bundle")
        tags, tax = make_tags(tmp, bundle / "poses.db", db_sha="0" * 64)
        try:
            attach_semantic(bundle, tags, tax)
        except BundleError as exc:
            assert "index_version_mismatch" in str(exc)
        else:
            raise AssertionError("tags for another poses.db must be refused")
        assert not (bundle / "semantic").exists()


def test_unrecorded_semantic_folder_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = make_bundle(tmp / "bundle")
        tags, tax = make_tags(tmp, bundle / "poses.db")
        (bundle / "semantic").mkdir()
        for src in (tags, tax):
            (bundle / "semantic" / src.name).write_bytes(src.read_bytes())
        try:
            verify_manifest(bundle)
        except LibraryManifestError as exc:
            assert "attach-semantic" in str(exc)
        else:
            raise AssertionError("semantic/ without manifest section must fail")


def test_deploy_validate_and_archive_include_semantic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = make_bundle(tmp / "bundle")
        try:
            validate(bundle, require_semantic=True)
        except DeployError:
            pass
        else:
            raise AssertionError("--require-semantic must fail without tags")
        assert validate(bundle)["semantic"] is None  # 없으면 SKIP
        tags, tax = make_tags(tmp, bundle / "poses.db")
        attach_semantic(bundle, tags, tax)
        summary = validate(bundle, require_semantic=True)
        assert summary["semantic"]["pose_count"] == 2
        archive = tmp / "bundle.tar.gz"
        make_archive(bundle, archive)
        with tarfile.open(archive) as tar:
            names = set(tar.getnames())
        assert {"semantic/search-v2.json", "semantic/search-taxonomy-v2.json"} <= names


def test_server_default_paths_load_from_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = make_bundle(tmp / "bundle")
        tags, tax = make_tags(tmp, bundle / "poses.db")
        attach_semantic(bundle, tags, tax)
        root = Path(__file__).resolve().parents[1]
        from dataclasses import replace
        from src.config import CFG
        cfg = replace(CFG, data_dir=str(bundle), semantic_enabled=True, semantic_tags_path="",
                      semantic_taxonomy_path="", semantic_verify_db_sha256=True,
                      semantic_dense_enabled=False,
                      semantic_lexicon_path=str(root / "config" / "semantic_lexicon.v1.json"))
        service = SemanticSearchService.from_config(cfg, db_path=str(bundle / "poses.db"),
                                                    allowed_pose_ids={"pitch_a", "rest_b"})
        assert service.ready, service.health()
        r = service.search("야구공을 던지는 자세")
        assert r["status"] == "ok" and r["candidates"][0]["pose_id"] == "pitch_a"
        assert r["candidates"][0]["pose"]["thumbnail_urls"]["front"].startswith("/pose/pitch_a/")


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
