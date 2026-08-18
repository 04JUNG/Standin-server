"""Focused checks for deterministic BVH semantic tagging artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bvh import coco17_from_fk, fk, load_coco17, parse_bvh
from src.posecode import (
    common_neutral_atoms,
    measure_posecode,
    mirror_atom_report,
    render_posecode_documents,
)
from src.semantic_catalog import _proposal_priority, _source_display, parse_cmu_catalog
from src.schema import Action
from src.semantic_vocab import (
    SEMANTIC_VOCAB_VERSION,
    load_semantic_vocab,
    resolve_exact_alias,
    validate_semantic_annotation,
)
from scripts.build_semantic_action_mapping import (
    _compile_and_validate_rules,
    attach_search_coverage,
    map_source,
)
from scripts.mirror_bvh import mirror_frame_values


def test_cmu_catalog_parser_preserves_official_title_and_ids() -> None:
    html = """
    <table><tr><td></td><td>3</td>
    <td>dance - sideways arabesque, turn step, folding arms</td>
    <td><a href="/subjects/05/05_03.amc">amc</a></td><td>120</td></tr></table>
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cmu.html"
        path.write_text(html, encoding="utf-8")
        records = parse_cmu_catalog(path)
    record = records[("5", "3")]
    assert record["subject_id"] == "05"
    assert record["clip_id"] == "03"
    assert record["title"] == "dance - sideways arabesque, turn step, folding arms"
    assert record["fps"] == 120


def test_posecode_contains_only_observed_atoms() -> None:
    path = REPO_ROOT / "data/bvh/Waving_02.bvh"
    joints, _ = load_coco17(str(path))
    posecode = measure_posecode(joints, provenance_ref="data/bvh/Waving_02.bvh")
    assert posecode["measurements"]["left_elbow_flexion_deg"] > 150.0
    assert posecode["observed_atoms"]
    for atom in posecode["observed_atoms"]:
        assert atom["evidence_state"] == "observed"
        assert atom["provenance"]["kind"] == "bvh_rule"
        assert atom["predicate"] != "action_intent"


def test_mirror_atoms_swap_left_and_right() -> None:
    original_path = REPO_ROOT / "data/bvh/Waving_02.bvh"
    mirror_path = REPO_ROOT / "data/bvh/Waving_02_mirror.bvh"
    original_joints, _ = load_coco17(str(original_path))
    mirror_joints, _ = load_coco17(str(mirror_path))
    original = measure_posecode(original_joints, provenance_ref=original_path.name)
    mirrored = measure_posecode(mirror_joints, provenance_ref=mirror_path.name)
    report = mirror_atom_report(original, mirrored)
    assert report["status"] == "pass", json.dumps(report, ensure_ascii=False)


def test_group_passage_is_direction_neutral() -> None:
    original_path = REPO_ROOT / "data/bvh/cmu_05_03_00150.bvh"
    mirror_path = REPO_ROOT / "data/bvh/cmu_05_03_00150_mirror.bvh"
    original_joints, _ = load_coco17(str(original_path))
    mirror_joints, _ = load_coco17(str(mirror_path))
    original = measure_posecode(original_joints, provenance_ref=original_path.name)
    mirrored = measure_posecode(mirror_joints, provenance_ref=mirror_path.name)
    atoms = common_neutral_atoms([original["observed_atoms"], mirrored["observed_atoms"]])
    documents = render_posecode_documents(atoms)
    assert "왼쪽" not in documents["ko"]
    assert "오른쪽" not in documents["ko"]
    assert "양팔을 넓게 벌림" in documents["ko"]


def test_missing_action_name_keeps_observed_posecode_searchable() -> None:
    source = {
        "provider": "cmu_graphics_lab",
        "original": {"title": "walk and turn", "local_label": None},
    }
    assert _source_display(source) == "CMU"
    assert _proposal_priority([], None, source) == "P2"
    source["original"]["title"] = None
    assert _proposal_priority([], None, source) == "P2"


def test_generated_review_policy_canonicalizes_mirrors_and_auto_verifies_p2() -> None:
    proposals_path = REPO_ROOT / "data/semantic/proposals.v1.jsonl"
    current: dict[str, dict] = {}
    for line in proposals_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        unit_id = row["semantic_unit_id"]
        if unit_id not in current or row["content_revision"] > current[unit_id]["content_revision"]:
            current[unit_id] = row
    assert all(
        proposal["workflow_status"] == "auto_verified_observed_tags"
        for proposal in current.values()
        if proposal["validation"]["review_priority"] == "P2"
    )
    canonicalized = [
        proposal
        for proposal in current.values()
        if (proposal.get("mirror_validation") or {}).get("resolution", {}).get("status") == "canonicalized"
    ]
    assert canonicalized
    assert all(
        "mirror_posecode_canonicalized" in proposal["validation"]["warnings"]
        for proposal in canonicalized
    )


def test_action_name_proposals_are_superseded_by_excluded_units() -> None:
    proposal_path = REPO_ROOT / "data/semantic/action_name_review/action_name_proposals.v1.json"
    review_path = REPO_ROOT / "data/semantic/review_queue.csv"
    proposal_document = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposals = proposal_document["proposals"]
    with review_path.open(encoding="utf-8", newline="") as stream:
        review_rows = list(csv.DictReader(stream))

    expected = {
        row["semantic_unit_id"]
        for row in review_rows
        if row["action_name_status"] == "excluded"
    }
    actual = {row["semantic_unit_id"] for row in proposals}
    assert proposal_document["status"] == "superseded_by_library_exclusion"
    assert len(proposals) == len(actual) == 38
    assert actual == expected
    assert all(row["confidence"] in {"high", "medium", "low"} for row in proposals)
    assert all(row["action_name_proposed_ko"].strip() for row in proposals)
    assert all(row["action_name_proposed_en"].strip() for row in proposals)
    assert all(row["visual_evidence"].strip() for row in proposals)
    assert all(row["ambiguity"].strip() for row in proposals)


def test_explicit_exclusion_policy_covers_all_rejected_sources() -> None:
    policy = json.loads(
        (REPO_ROOT / "config/library_exclusions.v1.json").read_text(encoding="utf-8")
    )
    source_rows = [
        json.loads(line)
        for line in (REPO_ROOT / "data/semantic/source_clips.v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    excluded = {
        row["source_clip_id"]
        for row in source_rows
        if (row.get("library_policy") or {}).get("state") == "pending_removal"
    }
    assert len(excluded) == 35
    assert excluded == set(policy["source_clip_ids"])
    assert all(
        (row["library_policy"][field] is False)
        for row in source_rows
        if row["source_clip_id"] in excluded
        for field in ("semantic_index", "geometry_index", "release")
    )


def test_typing_mouse_orphan_is_now_a_valid_mirror_pair() -> None:
    base = REPO_ROOT / "data/bvh/rokoko_Typing_UsingMouse_mixamo_00882"
    assert base.with_suffix(".bvh").is_file()
    assert Path(str(base) + "_mirror.bvh").is_file()
    original, _ = load_coco17(str(base.with_suffix(".bvh")))
    mirrored, _ = load_coco17(str(base) + "_mirror.bvh")
    report = mirror_atom_report(
        measure_posecode(original, provenance_ref=base.name),
        measure_posecode(mirrored, provenance_ref=base.name + "_mirror"),
    )
    assert report["status"] == "pass", json.dumps(report, ensure_ascii=False)


def test_semantic_vocab_v2_is_separate_from_runtime_action_contract() -> None:
    document = load_semantic_vocab()
    assert document["semantic_vocab_version"] == SEMANTIC_VOCAB_VERSION == 2
    assert resolve_exact_alias("action_ids", "Walking", document) == "walk"
    assert {item.value for item in Action} == {
        "standing", "sitting", "walking", "running", "reaching", "lying", "other"
    }


def test_semantic_vocab_v2_validates_domains_and_unknown_policy() -> None:
    valid = {
        "action_domain": ["dance"],
        "action_ids": ["traditional_dance"],
        "posture": ["one_leg_balance"],
        "motion_phase": "unknown",
        "style_context": ["traditional_indian"],
        "intended_props": [],
        "interaction": {"kind": "solo"},
    }
    assert validate_semantic_annotation(valid) == []
    invalid = {**valid, "action_domain": ["combat"], "posture": ["unknown", "standing"]}
    errors = validate_semantic_annotation(invalid)
    assert any("domain 'dance' absent" in error for error in errors)
    assert any("unknown cannot coexist" in error for error in errors)


def test_bvh_mirror_transform_matches_existing_pairs() -> None:
    for pose_id in (
        "rokoko_Typing_UsingMouse_mixamo_00390",
        "Waving_02",
        "cmu_05_03_00150",
    ):
        original_path = REPO_ROOT / f"data/bvh/{pose_id}.bvh"
        mirror_path = REPO_ROOT / f"data/bvh/{pose_id}_mirror.bvh"
        original_joints, original_frames = parse_bvh(str(original_path))
        mirror_joints, mirror_frames = parse_bvh(str(mirror_path))
        assert [joint[0] for joint in original_joints] == [joint[0] for joint in mirror_joints]
        transformed = mirror_frame_values(original_joints, original_frames[0])
        expected, _ = coco17_from_fk(mirror_joints, fk(mirror_joints, mirror_frames[0]))
        actual, _ = coco17_from_fk(original_joints, fk(original_joints, transformed))
        assert np.allclose(actual, expected, atol=1e-5)


def test_semantic_action_mapping_v2_covers_328_named_sources() -> None:
    mapping_path = REPO_ROOT / "data/semantic/action_mapping.v2.jsonl"
    rows = [
        json.loads(line)
        for line in mapping_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(
        (REPO_ROOT / "data/semantic/action-mapping-summary.v2.json").read_text(
            encoding="utf-8"
        )
    )
    exclusions = json.loads(
        (REPO_ROOT / "config/library_exclusions.v1.json").read_text(encoding="utf-8")
    )
    source_ids = {row["source_clip_id"] for row in rows}

    assert len(rows) == len(source_ids) == 328
    assert source_ids.isdisjoint(exclusions["source_clip_ids"])
    assert summary["source_clips"] == 328
    assert summary["status_counts"] == {
        "facets_only": 40,
        "mapped": 242,
        "unknown": 46,
    }
    assert summary["validation_errors"] == 0
    assert summary["source_context_only"] is True
    assert summary["pose_action_assignment"] == "not_performed"
    assert summary["search_coverage"] == {
        "observed_unit_atoms": 5044,
        "semantic_units_covered": 616,
        "source_clips_searchable": 328,
        "source_clips_unsearchable": 0,
        "source_clips_with_canonical_context": 282,
        "source_clips_with_observed_posecode": 328,
        "source_clips_with_raw_context": 328,
        "unknown_action_but_searchable": 46,
        "unknown_action_semantic_units": 94,
        "no_label_but_searchable": 0,
    }

    for row in rows:
        canonical = row["canonical"]
        annotation = {
            "action_domain": canonical["action_domain"],
            "action_ids": canonical["source_action_ids"],
            "posture": canonical["posture_hints"],
            "motion_phase": canonical["motion_phase"],
            "style_context": canonical["style_context"],
            "intended_props": canonical["intended_props"],
            "interaction": canonical["interaction"],
        }
        assert row["source_context_only"] is True
        assert "pose_action_ids" not in canonical
        assert canonical["motion_phase"] == "unknown"
        assert validate_semantic_annotation(annotation) == []
        coverage = row["search_coverage"]
        assert coverage["searchable"] is True
        assert coverage["semantic_unit_ids"]
        assert coverage["channels"]["observed_posecode"]["enabled"] is True
        assert coverage["channels"]["raw_source_context"]["enabled"] is True
        assert coverage["channels"]["raw_source_context"]["candidate_only"] is True
        assert coverage["channels"]["raw_source_context"]["hard_filter_eligible"] is False
        assert coverage["policy"]["action_id_absence_excludes_from_search"] is False
        assert coverage["policy"]["unknown_is_not_negative"] is True

    unknown_rows = [row for row in rows if row["mapping"]["status"] == "unknown"]
    assert len(unknown_rows) == 46
    assert all(not row["canonical"]["source_action_ids"] for row in unknown_rows)
    assert all(row["search_coverage"]["searchable"] for row in unknown_rows)


def test_semantic_action_mapping_v2_regression_samples() -> None:
    rows = {
        row["source_clip_id"]: row
        for row in (
            json.loads(line)
            for line in (REPO_ROOT / "data/semantic/action_mapping.v2.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    expected_actions = {
        "cmu:144_06": ["kick"],
        "cmu:81_09": ["pull"],
        "local_action_raw:Big Side Hit": ["punch"],
        "local_named:Hands Forward Gesture": ["reach"],
        "rokoko:BurstThroughDoor": ["push", "run"],
        "rokoko:MiddleFingers": ["dismiss_gesture"],
        "rokoko:ShoulderShimmy": ["dance_step"],
    }
    for source_id, action_ids in expected_actions.items():
        assert rows[source_id]["canonical"]["source_action_ids"] == action_ids

    duck = rows["cmu:131_06"]
    assert duck["mapping"]["status"] == "facets_only"
    assert duck["canonical"]["action_domain"] == ["transition"]
    assert duck["canonical"]["posture_hints"] == ["crouching"]

    unknown = rows["rokoko:FootTapping"]
    assert unknown["mapping"]["status"] == "unknown"
    assert unknown["mapping"]["requires_human_review"] is True
    assert unknown["search_coverage"]["semantic_unit_count"] == 4
    assert unknown["search_coverage"]["channels"]["observed_posecode"]["unit_atom_count"] > 0
    assert unknown["search_coverage"]["channels"]["canonical_context"]["enabled"] is False


def test_future_source_without_name_remains_posecode_searchable() -> None:
    vocab = load_semantic_vocab()
    rules = json.loads(
        (REPO_ROOT / "config/semantic_action_mapping_rules.v2.json").read_text(
            encoding="utf-8"
        )
    )
    source = {
        "source_clip_id": "future:unnamed",
        "provider": None,
        "collection": {"id": "future_import"},
        "original": {"title": None, "local_label": None},
    }
    proposal = {
        "semantic_unit_id": "pose:future_unnamed_0001",
        "workflow_status": "auto_verified_observed_tags",
        "validation": {"review_priority": "P2"},
        "semantic": {
            "caption_ko": "상체를 세우고 양팔을 넓게 벌림",
            "caption_en": "upright torso with both arms spread wide",
            "unit_atoms": [{"predicate": "limb_configuration"}],
        },
    }
    mapping = map_source(
        source,
        vocab=vocab,
        compiled_rules=_compile_and_validate_rules(rules, vocab),
        vocab_hash="sha256:test-vocab",
        rules_hash="sha256:test-rules",
    )
    attach_search_coverage(mapping, [proposal])

    assert mapping["mapping"]["status"] == "no_label"
    assert mapping["canonical"]["source_action_ids"] == []
    assert mapping["search_coverage"]["searchable"] is True
    assert mapping["search_coverage"]["channels"]["observed_posecode"]["enabled"] is True
    assert mapping["search_coverage"]["channels"]["raw_source_context"]["enabled"] is False
    assert mapping["search_coverage"]["policy"]["action_id_absence_excludes_from_search"] is False


def test_unknown_action_units_have_fallback_documents_in_review_index() -> None:
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "data/semantic/action_mapping.v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    unknown_units = sorted(
        {
            unit_id
            for row in rows
            if row["mapping"]["status"] == "unknown"
            for unit_id in row["search_coverage"]["semantic_unit_ids"]
        }
    )
    placeholders = ",".join("?" for _ in unknown_units)
    with sqlite3.connect(REPO_ROOT / "data/semantic/tagging_review.v1.db") as connection:
        counts = dict(
            connection.execute(
                f"""SELECT document_type, COUNT(DISTINCT semantic_unit_id)
                FROM text_documents
                WHERE semantic_unit_id IN ({placeholders})
                GROUP BY document_type""",
                unknown_units,
            )
        )

    assert len(unknown_units) == 94
    assert counts["posecode_render"] == 94
    assert counts["source_context"] == 94


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
