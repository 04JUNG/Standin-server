"""Regression checks for the final pre-embedding semantic document sets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_documents import build_search_documents


DOCUMENTS_PATH = REPO_ROOT / "data/semantic/search_documents.v2.jsonl"
SUMMARY_PATH = REPO_ROOT / "data/semantic/search-document-summary.v2.json"
SIDE_WORD = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rows_by_unit() -> dict[str, dict]:
    rows = _read_jsonl(DOCUMENTS_PATH)
    return {row["semantic_unit_id"]: row for row in rows}


def _text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_final_document_artifact_has_complete_active_coverage() -> None:
    rows = _read_jsonl(DOCUMENTS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    unit_ids = [row["semantic_unit_id"] for row in rows]
    document_ids = [
        document["document_id"]
        for row in rows
        for document in row["text_documents"]
    ]

    assert len(rows) == len(set(unit_ids)) == summary["semantic_units"] == 616
    assert sum(len(row["members"]) for row in rows) == summary["pose_members"] == 1232
    assert len(document_ids) == len(set(document_ids)) == summary["text_documents"] == 2892
    assert summary["observed_unit_atoms"] == 5044
    assert summary["document_type_counts"] == {
        "canonical_context": 1044,
        "posecode_render": 1232,
        "source_context": 616,
    }
    assert summary["mapping_status_unit_counts"] == {
        "facets_only": 89,
        "mapped": 433,
        "unknown": 94,
    }
    assert summary["unknown_action_units_searchable"] == 94
    assert summary["searchable_units"] == 616
    assert summary["unsearchable_units"] == 0
    assert summary["production_ready"] is False
    assert summary["embedding_status"] == "not_built_pending_model_pin"
    assert all(row["searchable"] is True for row in rows)


def test_documents_keep_observed_and_contextual_evidence_separate() -> None:
    for row in _read_jsonl(DOCUMENTS_PATH):
        posecode = [
            document
            for document in row["text_documents"]
            if document["document_type"] == "posecode_render"
        ]
        assert {document["language"] for document in posecode} == {"ko", "en"}
        assert len(posecode) == 2
        assert row["observed_unit_atoms"]
        assert all(
            atom["evidence_state"] == "observed"
            for atom in row["observed_unit_atoms"]
        )

        for document in row["text_documents"]:
            retrieval = document["retrieval"]
            assert document["text_sha256"] == _text_hash(document["text"])
            assert retrieval["hard_filter_eligible"] is False
            if document["document_type"] == "posecode_render":
                assert document["evidence_state"] == "observed"
                assert retrieval["candidate_only"] is False
                assert retrieval["weight"] == 1.0
            else:
                assert document["evidence_state"] == "contextual"
                assert retrieval["candidate_only"] is True


def test_unknown_action_uses_posecode_and_raw_context_fallback() -> None:
    row = _rows_by_unit()["pose:rokoko_FootTapping_mixamo_00040"]
    types = [document["document_type"] for document in row["text_documents"]]

    assert row["source_mapping"]["status"] == "unknown"
    assert row["source_mapping"]["canonical"]["source_action_ids"] == []
    assert types.count("posecode_render") == 2
    assert types.count("source_context") == 1
    assert "canonical_context" not in types
    assert row["retrieval_policy"]["action_id_absence_excludes_from_search"] is False
    assert row["retrieval_policy"]["unknown_is_not_negative"] is True


def test_shared_documents_are_direction_neutral_and_preserve_raw_label() -> None:
    rows = _read_jsonl(DOCUMENTS_PATH)
    for row in rows:
        for document in row["text_documents"]:
            assert SIDE_WORD.search(document["text"]) is None
            assert "왼쪽" not in document["text"]
            assert "오른쪽" not in document["text"]
        assert row["mirror"]["shared_passage_direction_neutral"] is True

    sample = _rows_by_unit()["pose:cmu_144_10_02831"]
    source = sample["source_mapping"]
    assert "Left" in source["raw_action_label"]
    assert "one side" in source["raw_search_text"]
    assert SIDE_WORD.search(source["raw_search_text"]) is None
    assert source["raw_direction_neutralized"] is True


def test_every_document_set_is_one_complete_mirror_pair() -> None:
    rows = _read_jsonl(DOCUMENTS_PATH)
    statuses: dict[str, int] = {}
    for row in rows:
        variants = {member["variant_kind"] for member in row["members"]}
        member_ids = {member["pose_id"] for member in row["members"]}
        statuses[row["mirror"]["validation_status"]] = (
            statuses.get(row["mirror"]["validation_status"], 0) + 1
        )

        assert len(row["members"]) == 2
        assert variants == {"original", "mirrored"}
        assert row["canonical_pose_id"] in member_ids
        assert row["mirrored_pose_id"] in member_ids
        assert row["canonical_pose_id"] != row["mirrored_pose_id"]

    assert statuses == {"pass": 561, "canonicalized": 55}


def test_excluded_cmu_sources_do_not_leak_into_documents() -> None:
    rows = _read_jsonl(DOCUMENTS_PATH)
    exclusions = json.loads(
        (REPO_ROOT / "config/library_exclusions.v1.json").read_text(encoding="utf-8")
    )
    output_sources = {row["source_clip_id"] for row in rows}

    assert output_sources.isdisjoint(exclusions["source_clip_ids"])
    assert len(exclusions["source_clip_ids"]) == 35


def test_committed_artifact_is_reproducible_from_current_inputs() -> None:
    rebuilt, rebuilt_summary = build_search_documents(
        inventory_path=REPO_ROOT / "data/semantic/inventory.v1.jsonl",
        proposals_path=REPO_ROOT / "data/semantic/proposals.v1.jsonl",
        mappings_path=REPO_ROOT / "data/semantic/action_mapping.v2.jsonl",
        exclusions_path=REPO_ROOT / "config/library_exclusions.v1.json",
        vocab_path=REPO_ROOT / "config/semantic_vocab.v2.json",
    )
    committed = _read_jsonl(DOCUMENTS_PATH)
    committed_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert rebuilt_summary == committed_summary
    assert rebuilt == committed


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
