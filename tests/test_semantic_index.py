"""Regression checks for the pinned E5 staging semantic index."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_embedding import (
    OnnxE5Encoder,
    embedding_profile_fingerprint,
    load_embedding_profile,
    model_directory,
)
from src.semantic_index import (
    SEMANTIC_INDEX_BUILDER_VERSION,
    build_semantic_index,
    validate_semantic_index,
)


PROFILE_PATH = REPO_ROOT / "config/semantic_embedding.e5-small.v1.json"
DOCUMENTS_PATH = REPO_ROOT / "data/semantic/search_documents.v2.jsonl"
DOCUMENT_SUMMARY_PATH = REPO_ROOT / "data/semantic/search-document-summary.v2.json"
INVENTORY_PATH = REPO_ROOT / "data/semantic/inventory.v1.jsonl"
PROPOSALS_PATH = REPO_ROOT / "data/semantic/proposals.v1.jsonl"
GEOMETRY_DB_PATH = REPO_ROOT / "data/poses.db"
MODELS_ROOT = REPO_ROOT / "data/models"
BUILDS_ROOT = REPO_ROOT / "data/semantic/builds"


def _current_build() -> tuple[Path, dict]:
    profile = load_embedding_profile(PROFILE_PATH)
    fingerprint = embedding_profile_fingerprint(profile)
    candidates = []
    for manifest_path in BUILDS_ROOT.glob("*/semantic-build.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["embedding"]["profile_fingerprint"] == fingerprint
            and manifest.get("semantic_db_schema_version") == 2
            and manifest.get("semantic_index_builder_version")
            == SEMANTIC_INDEX_BUILDER_VERSION
        ):
            candidates.append((manifest_path.parent, manifest))
    assert len(candidates) == 1
    return candidates[0]


def test_embedding_profile_pins_the_complete_e5_contract() -> None:
    profile = load_embedding_profile(PROFILE_PATH)

    assert profile["model"]["id"] == "intfloat/multilingual-e5-small"
    assert profile["model"]["revision"] == (
        "fd1525a9fd15316a2d503bf26ab031a61d056e98"
    )
    assert profile["runtime"] == {
        "onnxruntime_version": "1.28.0",
        "tokenizers_version": "0.23.1",
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "providers": ["CPUExecutionProvider"],
    }
    assert profile["encoding"] == {
        "dimension": 384,
        "dtype": "float32",
        "pooling": "attention_mask_mean",
        "l2_normalized": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_length": 128,
        "truncation": "fail_on_truncation",
        "padding": "longest_in_batch",
        "batch_size": 32,
    }
    assert len(profile["model"]["artifacts"]) == 6
    assert all(len(row["sha256"]) == 64 for row in profile["model"]["artifacts"])


def test_staging_semantic_index_validates_against_current_inputs() -> None:
    build_dir, manifest = _current_build()
    result = validate_semantic_index(
        build_dir / "pose_semantics.db",
        build_dir / "semantic-build.json",
        profile_path=PROFILE_PATH,
        documents_path=DOCUMENTS_PATH,
        inventory_path=INVENTORY_PATH,
        geometry_db_path=GEOMETRY_DB_PATH,
        proposals_path=PROPOSALS_PATH,
    )

    assert result["status"] == "pass"
    assert result["semantic_build_id"] == manifest["semantic_build_id"]
    assert result["counts"] == {
        "semantic_units": 616,
        "pose_semantic_members": 1232,
        "semantic_text_documents": 2892,
        "semantic_text_documents_fts": 2892,
        "semantic_embeddings": 2892,
        "semantic_atoms": 5044,
    }
    assert manifest["staging_ready"] is True
    assert manifest["production_ready"] is False
    assert "semantic_holdout_not_passed" in manifest["promotion_blockers"]
    assert "semantic_release_bundle_not_promoted" in manifest["promotion_blockers"]
    assert manifest["embedding"]["encoding_stats"]["truncated"] == 0
    assert manifest["embedding"]["encoding_stats"]["max_tokens"] <= 128


def test_every_vector_is_float32_384_finite_and_normalized() -> None:
    build_dir, manifest = _current_build()
    db_path = build_dir / "pose_semantics.db"
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT embedding_blob, dimension, dtype, l2_norm FROM semantic_embeddings"
        ).fetchall()

    assert len(rows) == 2892
    for blob, dimension, dtype, stored_norm in rows:
        vector = np.frombuffer(blob, dtype="<f4")
        assert dimension == manifest["embedding"]["dimension"] == 384
        assert dtype == "float32_le"
        assert vector.shape == (384,)
        assert np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector)) - 1.0) <= 1e-5
        assert abs(stored_norm - 1.0) <= 1e-5


def test_unknown_action_and_context_policy_survive_sqlite_build() -> None:
    build_dir, _ = _current_build()
    with sqlite3.connect(build_dir / "pose_semantics.db") as connection:
        unit = connection.execute(
            """SELECT mapping_status, source_mapping_json, retrieval_policy_json
               FROM semantic_units
               WHERE semantic_unit_id='pose:rokoko_FootTapping_mixamo_00040'"""
        ).fetchone()
        documents = connection.execute(
            """SELECT document_type, evidence_state, candidate_only,
                      hard_filter_eligible
               FROM semantic_text_documents
               WHERE semantic_unit_id='pose:rokoko_FootTapping_mixamo_00040'
               ORDER BY document_type, language"""
        ).fetchall()

    mapping_status, mapping_json, policy_json = unit
    mapping = json.loads(mapping_json)
    policy = json.loads(policy_json)
    assert mapping_status == "unknown"
    assert mapping["canonical"]["source_action_ids"] == []
    assert policy["action_id_absence_excludes_from_search"] is False
    assert [row[0] for row in documents] == [
        "posecode_render",
        "posecode_render",
        "source_context",
    ]
    assert all(row[3] == 0 for row in documents)
    assert documents[-1][1:] == ("contextual", 1, 0)


def test_all_semantic_members_exist_in_geometry_db_and_exclusions_do_not_leak() -> None:
    build_dir, _ = _current_build()
    exclusions = json.loads(
        (REPO_ROOT / "config/library_exclusions.v1.json").read_text(encoding="utf-8")
    )
    with sqlite3.connect(build_dir / "pose_semantics.db") as semantic:
        members = {row[0] for row in semantic.execute("SELECT pose_id FROM pose_semantic_members")}
        sources = {row[0] for row in semantic.execute("SELECT source_clip_id FROM semantic_units")}
    with sqlite3.connect(GEOMETRY_DB_PATH) as geometry:
        geometry_members = {row[0] for row in geometry.execute("SELECT pose_id FROM poses")}

    assert len(members) == 1232
    assert members.issubset(geometry_members)
    assert sources.isdisjoint(exclusions["source_clip_ids"])


def test_member_posecode_measurements_are_pinned_for_original_and_mirror() -> None:
    build_dir, manifest = _current_build()
    with sqlite3.connect(build_dir / "pose_semantics.db") as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        rows = connection.execute(
            """SELECT posecode_version,coordinate_profile,posecode_measurements_json
               FROM pose_semantic_members"""
        ).fetchall()

    assert version == manifest["semantic_db_schema_version"] == 2
    assert len(rows) == 1232
    assert manifest["posecode"]["measurements_per_member"] == 27
    expected_keys = set(manifest["posecode"]["measurement_keys"])
    for posecode_version, coordinate_profile, payload in rows:
        measurements = json.loads(payload)
        assert posecode_version == manifest["posecode"]["posecode_version"] == 2
        assert coordinate_profile == manifest["posecode"]["coordinate_profile"]
        assert set(measurements) == expected_keys


def test_fts_channel_contains_the_same_documents() -> None:
    build_dir, _ = _current_build()
    with sqlite3.connect(build_dir / "pose_semantics.db") as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM semantic_text_documents_fts"
        ).fetchone()[0]
        matches = connection.execute(
            """SELECT semantic_unit_id FROM semantic_text_documents_fts
               WHERE semantic_text_documents_fts MATCH 'Foot'"""
        ).fetchall()

    assert count == 2892
    assert ("pose:rokoko_FootTapping_mixamo_00040",) in matches


def test_query_encoder_is_deterministic_and_normalized() -> None:
    profile = load_embedding_profile(PROFILE_PATH)
    encoder = OnnxE5Encoder(profile, model_directory(profile, MODELS_ROOT))
    texts = [
        "왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세",
        "옛 전통 춤을 추는 자세",
    ]
    first, stats = encoder.encode(texts, kind="query")
    second, _ = encoder.encode(texts, kind="query")

    assert first.shape == (2, 384)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert stats["truncated"] == 0


def test_same_inputs_reuse_the_immutable_build() -> None:
    expected_dir, expected_manifest = _current_build()
    actual_dir, actual_manifest, reused = build_semantic_index(
        documents_path=DOCUMENTS_PATH,
        document_summary_path=DOCUMENT_SUMMARY_PATH,
        inventory_path=INVENTORY_PATH,
        geometry_db_path=GEOMETRY_DB_PATH,
        profile_path=PROFILE_PATH,
        models_root=MODELS_ROOT,
        output_root=BUILDS_ROOT,
        proposals_path=PROPOSALS_PATH,
    )

    assert reused is True
    assert actual_dir == expected_dir
    assert actual_manifest["semantic_build_id"] == expected_manifest["semantic_build_id"]
    assert (
        actual_manifest["artifacts"]["semantic_db_sha256"]
        == expected_manifest["artifacts"]["semantic_db_sha256"]
    )


def test_changed_profile_is_rejected_by_validator() -> None:
    build_dir, _ = _current_build()
    profile = load_embedding_profile(PROFILE_PATH)
    profile["retrieval"]["rrf_k"] += 1
    with tempfile.TemporaryDirectory() as directory:
        changed_path = Path(directory) / "changed-profile.json"
        changed_path.write_text(json.dumps(profile), encoding="utf-8")
        try:
            validate_semantic_index(
                build_dir / "pose_semantics.db",
                build_dir / "semantic-build.json",
                profile_path=changed_path,
                proposals_path=PROPOSALS_PATH,
            )
        except ValueError as exc:
            assert "profile mismatch" in str(exc)
        else:
            raise AssertionError("changed embedding profile was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
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
