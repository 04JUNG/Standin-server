"""배포 번들 규칙: 개인정보·meta 허용 목록·리그 호환·manifest·재생 게이트, 그리고 번들 빌더."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import tarfile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_pose_bundle import (
    BundleError,
    build_baseline,
    build_curated,
    record_gate,
)
from scripts.deploy_pose_library import DeployError, make_archive, validate
from scripts.pose_bundle_policy import (
    bvh_path_ok,
    converter_profile,
    disallowed_meta_keys,
    privacy_findings,
    sanitize_meta,
)
from src.library_manifest import MANIFEST_NAME, content_sha256, read_manifest, verify_manifest
from src.repo import build_db
from src.schema import LibraryEntry, View
from src.thumbnails import thumbnail_filename
from tests.test_smoke import _synthetic_bvh

USER_HASH = "user_0123456789ab"
INSTALLATION = "inst_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"
JOB = "job_7c9e6679-7425-40de-944b-e07fc1f90ae7"
VIEWS = ("front", "three_quarter", "side", "back")


def _jpeg(path: Path, size: int, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (shade, shade, shade)).save(path, "JPEG", quality=90)


def _unresolved_bvh(path: Path) -> None:
    joints = "".join(f"  JOINT Bone{i:03d}\n  {{ OFFSET 0 1 0\n    CHANNELS 3 Zrotation Xrotation Yrotation\n"
                     for i in range(10))
    closing = "  }\n" * 10
    path.write_text(
        "HIERARCHY\nROOT Root\n{ OFFSET 0 0 0\n  CHANNELS 6 Xposition Yposition Zposition "
        "Zrotation Xrotation Yrotation\n" + joints + "  End Site { OFFSET 0 1 0 }\n" + closing
        + "}\nMOTION\nFrames: 1\nFrame Time: 0.033\n" + " ".join(["0"] * 36) + "\n",
        encoding="utf-8")


def make_base(root: Path, pose_ids=("existing_a", "existing_b")) -> Path:
    """운영 번들과 같은 모양: data/bvh 상대경로, 256px jpg 썸네일, thumbnail_manifest, ATTRIBUTION."""
    (root / "bvh").mkdir(parents=True)
    entries, results = [], []
    for index, pose_id in enumerate(pose_ids):
        _synthetic_bvh(str(root / "bvh"), f"{pose_id}.bvh")
        for view in View:
            entries.append(LibraryEntry(
                pose_id=pose_id, view=view,
                feature=np.full(34, index + 1, dtype=np.float32), tags={"shot": "full_half"},
                bvh_path=f"data/bvh/{pose_id}.bvh"))
            thumb = root / "thumbs" / thumbnail_filename(pose_id, view.value)
            _jpeg(thumb, 256, 40 + index)
            results.append({"pose_id": pose_id, "view": view.value, "filename": thumb.name,
                            "sha256": "x", "artifact_source": "convert_safe"})
    build_db(entries, str(root / "poses.db"))
    (root / "thumbs" / "thumbnail_manifest.json").write_text(
        json.dumps({"schema_version": 1, "source_build_manifest_sha256": "abc",
                    "results": results}), encoding="utf-8")
    (root / "ATTRIBUTION.md").write_text("# Attribution — base\n", encoding="utf-8")
    (root / "index.pkl").write_bytes(b"legacy")
    return root


def make_curation(tmp: Path, base: Path, *, new_bvh=None, extra_meta=None,
                  new_pose_id="gap_preset_sit_01") -> tuple[Path, Path]:
    """publication.py가 만드는 정리 DB와 배치 manifest를 흉내 낸다."""
    curation = tmp / "curation"
    batch = curation / "batches" / "gap-20261002"
    (batch / "bvh").mkdir(parents=True)
    if new_bvh is None:
        _synthetic_bvh(str(batch / "bvh"), f"{new_pose_id}.bvh")
    else:
        new_bvh(batch / "bvh" / f"{new_pose_id}.bvh")
    new_bvh_path = batch / "bvh" / f"{new_pose_id}.bvh"
    from src.library_manifest import sha256_file

    thumbs = {}
    for offset, view in enumerate(VIEWS):
        path = batch / "character-thumbs" / "fp" / f"{new_pose_id}__{view}.jpg"
        _jpeg(path, 512, 120 + offset)
        thumbs[view] = {"path": path.relative_to(batch).as_posix(), "sha256": sha256_file(path)}
    (batch / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "batch_id": "gap-20261002", "status": "complete",
        "poses": [{"pose_id": new_pose_id, "bvh": f"bvh/{new_pose_id}.bvh",
                   "bvh_sha256": sha256_file(new_bvh_path), "thumbnails": thumbs}],
    }), encoding="utf-8")

    entries = []
    for pose_id in ("existing_a", "existing_b"):
        for view in View:
            entries.append(LibraryEntry(
                pose_id=pose_id, view=view, feature=np.ones(34, dtype=np.float32),
                tags={"shot": "full_half"}, bvh_path=str((base / "bvh" / f"{pose_id}.bvh").resolve()),
                meta={"curation_group": "existing", "batch_id": "s3-v1", "review_status": "pending",
                      "bvh_sha256": sha256_file(base / "bvh" / f"{pose_id}.bvh")}))
    meta = {"curation_group": "new", "batch_id": "gap-20261002", "review_status": "accepted",
            "bvh_sha256": sha256_file(new_bvh_path), "source": "authored_scenario",
            "author": "Standin", "license": "CC0-1.0", "source_url": "https://quaternius.com",
            "batch_directory": str(batch.resolve()),
            "recipe": {"reference_roughs": [USER_HASH]},
            "hand_augmentation": {"captured_from_source": False, "left": "relaxed",
                                  "preset_path": str(batch.resolve())},
            "gap": {"schema": 1, "method": "preset_search", "cluster_key": "g-abc",
                    "support": {"installations": 5}}}
    meta.update(extra_meta or {})
    for view in View:
        entries.append(LibraryEntry(
            pose_id=new_pose_id, view=view, feature=np.full(34, 3, dtype=np.float32),
            tags={"shot": "full_half"}, bvh_path=str(new_bvh_path.resolve()), meta=meta))
    library = curation / "library"
    library.mkdir(parents=True)
    build_db(entries, str(library / "poses.db"))
    (library / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")
    return curation, library / "poses.db"


# ── 정책 ──────────────────────────────────────────────────────────────
def test_privacy_findings_catch_user_identifiers_but_not_normal_names():
    assert privacy_findings(f"combat_composition_{USER_HASH}_p0") == ["user_input_hash:pose_id"]
    assert "installation_id:meta" in privacy_findings("p", {"note": f"installations/{INSTALLATION}"})
    assert "beta_data_key:meta" in privacy_findings("p", {"note": f"installations/{INSTALLATION}"})
    assert privacy_findings("p", {"x": JOB}) == ["job_id:meta"]
    assert privacy_findings("p", {"calibration_image_sha256": "f" * 64}) == ["calibration_image:meta"]
    assert privacy_findings("p", {"batch_directory": r"C:\workspaces\data"}) == ["local_path:meta"]
    assert privacy_findings("p", bvh_path="/Users/me/x.bvh") == ["local_path:bvh_path"]
    for name in ("job_interview_01", "instrument_pose", "2hand Idle_01", "webtoon_daily_01"):
        assert privacy_findings(name, {"source": "100style"}, f"data/bvh/{name}.bvh") == []


def test_meta_allowlist_keeps_only_reviewed_keys():
    meta = {"source": "100style", "license": "CC-BY-4.0", "batch_directory": "C:/x",
            "recipe": {"reference_roughs": [USER_HASH]},
            "hand_augmentation": {"captured_from_source": False, "preset_path": "C:/x"},
            "gap": {"schema": 1, "method": "pool_mining", "cluster_key": "g-1", "support": {}}}
    assert disallowed_meta_keys(meta) == ["batch_directory", "gap.support",
                                          "hand_augmentation.preset_path", "recipe"]
    clean = sanitize_meta(meta)
    assert clean == {"source": "100style", "license": "CC-BY-4.0",
                     "hand_augmentation": {"captured_from_source": False},
                     "gap": {"schema": 1, "method": "pool_mining", "cluster_key": "g-1"}}
    assert disallowed_meta_keys(clean) == []


def test_bvh_path_format():
    assert bvh_path_ok("data/bvh/2hand Idle_01.bvh")
    for bad in ("C:/data/bvh/x.bvh", "/app/data/bvh/x.bvh", "data/bvh/sub/x.bvh", "bvh/x.bvh", None):
        assert not bvh_path_ok(bad)


def test_converter_profile_resolves_known_rig_and_rejects_unknown(tmp_path):
    known = Path(_synthetic_bvh(str(tmp_path), "known.bvh"))
    assert converter_profile(known) == "mixamo_noprefix"
    unknown = tmp_path / "unknown.bvh"
    _unresolved_bvh(unknown)
    with pytest.raises(ValueError):
        converter_profile(unknown)


# ── baseline + 배포 검증 ───────────────────────────────────────────────
def test_baseline_from_directory_keeps_content_and_passes_deploy_validation(tmp_path):
    base = make_base(tmp_path / "base")
    out = tmp_path / "bundle"
    manifest = build_baseline(base, out)

    assert manifest["parent"]["content_sha256"] == content_sha256(base) == manifest["content_sha256"]
    assert manifest["replay_gate"]["status"] == "not_required"
    assert not (out / "index.pkl").exists()
    summary = validate(out)
    assert summary["manifest"]["library_version"] == manifest["library_version"]


def test_baseline_from_tarball_matches_directory_build(tmp_path):
    base = make_base(tmp_path / "base")
    archive = tmp_path / "v1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("poses.db", "index.pkl", "ATTRIBUTION.md", "bvh", "thumbs"):
            tar.add(base / name, arcname=name)
    from_tar = build_baseline(archive, tmp_path / "from_tar")
    from_dir = build_baseline(base, tmp_path / "from_dir")
    assert from_tar["content_sha256"] == from_dir["content_sha256"]
    assert from_tar["curation"]["source_sha256"]
    with pytest.raises(BundleError, match="비어 있지 않습니다"):
        build_baseline(archive, tmp_path / "from_tar")


def test_deploy_validation_requires_manifest_and_jpeg_thumbnails(tmp_path):
    base = make_base(tmp_path / "base")
    with pytest.raises(DeployError, match="library_manifest.json"):
        validate(base)
    assert validate(base, allow_no_manifest=True)["manifest"] is None

    for path in (base / "thumbs").glob("*.jpg"):
        path.rename(path.with_suffix(".png"))
    with pytest.raises(DeployError, match="썸네일"):
        validate(base, allow_no_manifest=True)


def test_deploy_validation_blocks_privacy_leaks_and_unknown_rigs(tmp_path):
    base = make_base(tmp_path / "base")
    con = sqlite3.connect(base / "poses.db")
    con.execute("UPDATE poses SET meta_json=? WHERE pose_id='existing_a'",
                (json.dumps({"reference_roughs": [USER_HASH]}),))
    con.commit()
    con.close()
    with pytest.raises(DeployError) as excinfo:
        validate(base, allow_no_manifest=True)
    message = str(excinfo.value)
    assert "user_input_hash" in message and "허용 목록 밖 meta 키" in message
    assert USER_HASH not in message  # 값 자체는 출력하지 않는다

    clean = make_base(tmp_path / "clean")
    _unresolved_bvh(clean / "bvh" / "existing_b.bvh")
    with pytest.raises(DeployError, match="모르는 리그"):
        validate(clean, allow_no_manifest=True)


def test_deploy_validation_requires_gate_or_explicit_reason(tmp_path):
    base = make_base(tmp_path / "base")
    out = tmp_path / "bundle"
    build_baseline(base, out)
    manifest = read_manifest(out)
    manifest["replay_gate"] = {"status": "not_run"}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DeployError, match="재생 게이트"):
        validate(out)
    assert validate(out, skip_gate_reason="hotfix rollback rehearsal")["manifest"]


def test_archive_carries_manifest_and_attribution_but_not_index_pickle(tmp_path):
    base = make_base(tmp_path / "base")
    out = tmp_path / "bundle"
    build_baseline(base, out)
    (out / "index.pkl").write_bytes(b"stale")
    archive = tmp_path / "bundle.tar.gz"
    make_archive(out, archive)
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert {MANIFEST_NAME, "ATTRIBUTION.md", "poses.db", "thumbs/thumbnail_manifest.json"} <= names
    assert "index.pkl" not in names


# ── curated ───────────────────────────────────────────────────────────
def test_curated_bundle_is_relative_sanitized_and_resized(tmp_path):
    base = make_base(tmp_path / "base")
    curation, curated_db = make_curation(tmp_path, base)
    out = tmp_path / "bundle"
    manifest = build_curated(curated_db, curation, base, out)

    con = sqlite3.connect(out / "poses.db")
    rows = dict(con.execute("SELECT pose_id, bvh_path FROM poses"))
    metas = {pid: json.loads(meta) for pid, meta in con.execute("SELECT pose_id, meta_json FROM poses")}
    con.close()
    assert rows == {"existing_a": "data/bvh/existing_a.bvh", "existing_b": "data/bvh/existing_b.bvh",
                    "gap_preset_sit_01": "data/bvh/gap_preset_sit_01.bvh"}
    new_meta = metas["gap_preset_sit_01"]
    assert new_meta["gap"] == {"schema": 1, "method": "preset_search", "cluster_key": "g-abc"}
    assert new_meta["hand_augmentation"] == {"captured_from_source": False, "left": "relaxed"}
    assert "batch_directory" not in new_meta and "recipe" not in new_meta

    raw = (out / "poses.db").read_bytes()
    assert USER_HASH.encode() not in raw and b"batch_directory" not in raw  # VACUUM으로 흔적까지 제거

    for view in VIEWS:
        with Image.open(out / "thumbs" / thumbnail_filename("gap_preset_sit_01", view)) as image:
            assert image.size == (256, 256) and image.format == "JPEG"
    thumbs = json.loads((out / "thumbs" / "thumbnail_manifest.json").read_text(encoding="utf-8"))
    assert thumbs["counts"]["by_artifact_source"] == {"convert_safe": 8, "curation_character_v1": 4}
    attribution = (out / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert attribution.startswith("# Attribution — base") and "authored_scenario" in attribution

    assert manifest["counts"]["poses"] == 3
    assert manifest["parent"]["content_sha256"] == content_sha256(base)
    assert manifest["replay_gate"]["status"] == "not_run"
    verify_manifest(out)
    with pytest.raises(DeployError, match="재생 게이트"):
        validate(out)


def test_record_gate_unblocks_deploy_without_changing_version(tmp_path):
    base = make_base(tmp_path / "base")
    curation, curated_db = make_curation(tmp_path, base)
    out = tmp_path / "bundle"
    version = build_curated(curated_db, curation, base, out)["library_version"]

    report = tmp_path / "gate.json"
    report.write_text(json.dumps({"schema_version": 1, "status": "passed",
                                  "thresholds_version": "pose_gaps-test.1",
                                  "summary": {"regressed_share": 0.0}}), encoding="utf-8")
    manifest = record_gate(out, report)
    assert manifest["library_version"] == version
    assert manifest["replay_gate"]["status"] == "passed"
    assert validate(out)["manifest"]["replay_gate"]["thresholds_version"] == "pose_gaps-test.1"

    report.write_text(json.dumps({"status": "maybe"}), encoding="utf-8")
    with pytest.raises(BundleError):
        record_gate(out, report)


def test_curated_unknown_rig_fails_or_is_excluded_on_request(tmp_path):
    base = make_base(tmp_path / "base")
    curation, curated_db = make_curation(tmp_path, base, new_bvh=_unresolved_bvh)
    with pytest.raises(BundleError, match="모르는 리그"):
        build_curated(curated_db, curation, base, tmp_path / "strict")

    manifest = build_curated(curated_db, curation, base, tmp_path / "lenient",
                             exclude_unresolved_rigs=True)
    assert [e["pose_id"] for e in manifest["compat"]["excluded"]] == ["gap_preset_sit_01"]
    assert manifest["counts"]["poses"] == 2
    assert not list((tmp_path / "lenient" / "thumbs").glob("gap_preset_sit_01__*"))


def test_curated_refuses_user_derived_pose_ids_and_changed_bvh(tmp_path):
    base = make_base(tmp_path / "base")
    curation, curated_db = make_curation(tmp_path, base,
                                         new_pose_id=f"combat_composition_{USER_HASH}_p0")
    with pytest.raises(BundleError, match="사용자 식별자"):
        build_curated(curated_db, curation, base, tmp_path / "out")

    other = tmp_path / "other"
    other.mkdir()
    base2 = make_base(other / "base")
    curation2, curated_db2 = make_curation(other, base2,
                                           extra_meta={"bvh_sha256": "0" * 64})
    with pytest.raises(BundleError, match="BVH가 바뀌었습니다"):
        build_curated(curated_db2, curation2, base2, other / "out")
