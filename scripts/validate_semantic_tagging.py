"""Validate generated semantic tagging ledgers and the review-only SQLite index."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _latest_proposals(rows: list[dict]) -> dict[str, dict]:
    current: dict[str, dict] = {}
    for row in rows:
        unit_id = row["semantic_unit_id"]
        if unit_id not in current or int(row.get("content_revision", 0)) > int(current[unit_id].get("content_revision", 0)):
            current[unit_id] = row
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate(output_dir: Path, geometry_db: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    inventory_rows = _jsonl(output_dir / "inventory.v1.jsonl")
    if not inventory_rows or inventory_rows[0].get("record_type") != "inventory_header":
        raise ValueError("inventory header missing")
    header = inventory_rows[0]
    inventory = inventory_rows[1:]
    sources = _jsonl(output_dir / "source_clips.v1.jsonl")
    lineage = _jsonl(output_dir / "pose_lineage.v1.jsonl")
    registry = json.loads((output_dir / "library_numbers.v1.json").read_text(encoding="utf-8"))
    proposal_history = _jsonl(output_dir / "proposals.v1.jsonl")
    decisions = _jsonl(output_dir / "decisions.v1.jsonl")
    proposals = _latest_proposals(proposal_history)

    inventory_ids = [row["pose_id"] for row in inventory]
    source_ids = [row["source_clip_id"] for row in sources]
    lineage_ids = [row["pose_id"] for row in lineage]
    library_numbers = [row["library_no"] for row in lineage]
    registry_assignments = registry.get("assignments", {})
    for label, values in (
        ("inventory pose_id", inventory_ids),
        ("source_clip_id", source_ids),
        ("lineage pose_id", lineage_ids),
        ("library_no", library_numbers),
    ):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label}")

    if set(inventory_ids) != set(lineage_ids):
        errors.append("inventory and lineage pose coverage differ")
    if any(registry_assignments.get(row["pose_id"]) != row["library_no"] for row in lineage):
        errors.append("library number registry and active lineage differ")
    registry_values = list(registry_assignments.values())
    if len(registry_values) != len(set(registry_values)):
        errors.append("library number registry reuses an internal number")
    source_id_set = set(source_ids)
    if any(row["source_clip_id"] not in source_id_set for row in lineage):
        errors.append("lineage contains orphan source_clip_id")

    proposal_members: list[str] = []
    mirror_review = 0
    mirror_canonicalized = 0
    atom_count = 0
    unit_atom_count = 0
    for unit_id, proposal in proposals.items():
        if proposal.get("workflow_status") not in {
            "needs_review", "auto_verified_observed_tags", "accepted",
            "accepted_with_edits", "rejected", "blocked",
        }:
            errors.append(f"{unit_id}: invalid workflow_status")
        priority = proposal.get("validation", {}).get("review_priority")
        if priority == "P2" and proposal.get("workflow_status") != "auto_verified_observed_tags":
            errors.append(f"{unit_id}: P2 observed tags must be auto-verified")
        if priority in {"P0", "P1"} and proposal.get("workflow_status") != "needs_review":
            errors.append(f"{unit_id}: priority review item must remain needs_review")
        if priority == "PX" and proposal.get("workflow_status") != "rejected":
            errors.append(f"{unit_id}: excluded source must be rejected")
        members = proposal.get("member_pose_ids", [])
        proposal_members.extend(members)
        if set(members) != set(proposal.get("member_posecodes", {})):
            errors.append(f"{unit_id}: member_pose_ids and member_posecodes differ")
        report = proposal.get("mirror_validation")
        if report and report.get("status") != "pass":
            resolution = report.get("resolution", {})
            if resolution.get("status") == "canonicalized":
                mirror_canonicalized += 1
                if resolution.get("method") != "original_posecode_side_neutralization":
                    errors.append(f"{unit_id}: unknown mirror canonicalization method")
            else:
                mirror_review += 1
        for atom in proposal.get("semantic", {}).get("unit_atoms", []):
            unit_atom_count += 1
            if atom.get("evidence_state") != "observed":
                errors.append(f"{unit_id}: non-observed generated unit atom")
            provenance = atom.get("provenance")
            if not isinstance(provenance, dict) or not {"kind", "ref", "version", "review_status"} <= set(provenance):
                errors.append(f"{unit_id}: invalid unit atom provenance")
        for pose_id, posecode in proposal.get("member_posecodes", {}).items():
            measurements = posecode.get("measurements", {})
            if not measurements or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in measurements.values()):
                errors.append(f"{unit_id}/{pose_id}: invalid posecode measurements")
            for atom in posecode.get("observed_atoms", []):
                atom_count += 1
                if atom.get("evidence_state") != "observed":
                    errors.append(f"{unit_id}/{pose_id}: non-observed member atom")
                provenance = atom.get("provenance")
                if not isinstance(provenance, dict) or not {"kind", "ref", "version", "review_status"} <= set(provenance):
                    errors.append(f"{unit_id}/{pose_id}: invalid atom provenance")
    member_counts = Counter(proposal_members)
    missing_members = sorted(set(inventory_ids) - set(member_counts))
    duplicate_members = sorted(pose_id for pose_id, count in member_counts.items() if count != 1)
    if missing_members:
        errors.append(f"proposal member coverage missing {len(missing_members)} poses")
    if duplicate_members:
        errors.append(f"proposal member coverage duplicates {len(duplicate_members)} poses")

    decision_ids = {row.get("proposal_id") for row in decisions}
    accepted_without_decision = [
        unit_id
        for unit_id, proposal in proposals.items()
        if proposal.get("workflow_status") in {"accepted", "accepted_with_edits"}
        and proposal.get("proposal_id") not in decision_ids
    ]
    if accepted_without_decision:
        errors.append(f"accepted proposals without decisions: {len(accepted_without_decision)}")

    db_path = output_dir / "tagging_review.v1.db"
    connection = sqlite3.connect(db_path)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    db_counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "source_clips", "pose_lineage", "semantic_units", "pose_members",
            "semantic_atoms", "text_documents", "semantic_embeddings",
        )
    }
    meta = {key: json.loads(value) for key, value in connection.execute("SELECT key,value_json FROM meta")}
    golden_a_sql = """
        WITH flags AS (
          SELECT pose_id,
            MAX(predicate='limb_state' AND subject='left_leg' AND value='raised') a,
            MAX(predicate='relative_direction' AND subject='left_ankle' AND relation='behind' AND object='pelvis') b,
            MAX(predicate='limb_configuration' AND subject='both_arms' AND value='widely_spread') c,
            MAX(predicate='joint_flexion' AND subject='left_elbow' AND bucket='extended') d,
            MAX(predicate='joint_flexion' AND subject='right_elbow' AND bucket='extended') e
          FROM semantic_atoms WHERE pose_id IS NOT NULL GROUP BY pose_id
        ) SELECT pose_id FROM flags WHERE a AND b AND c AND d AND e ORDER BY pose_id
    """
    golden_a_candidates = [row[0] for row in connection.execute(golden_a_sql)]
    dance_context_units = connection.execute(
        """SELECT COUNT(DISTINCT f.semantic_unit_id)
        FROM text_documents_fts f JOIN text_documents d USING(document_id)
        WHERE text_documents_fts MATCH 'dance' AND d.document_type='source_context'"""
    ).fetchone()[0]
    traditional_historical_atoms = connection.execute(
        """SELECT COUNT(*) FROM semantic_atoms
        WHERE predicate IN ('style','style_context','period_context')
           OR value IN ('traditional','historical')"""
    ).fetchone()[0]
    connection.close()
    if integrity != "ok":
        errors.append(f"sqlite integrity_check={integrity}")
    expected_counts = {
        "source_clips": len(sources),
        "pose_lineage": len(lineage),
        "semantic_units": len(proposals),
        "pose_members": len(inventory),
        "semantic_atoms": atom_count + unit_atom_count,
    }
    for table, expected in expected_counts.items():
        if db_counts[table] != expected:
            errors.append(f"{table} count mismatch: db={db_counts[table]}, expected={expected}")
    if meta.get("production_ready") is not False:
        errors.append("review index must not claim production readiness")
    if db_counts["semantic_embeddings"] != 0:
        errors.append("review index must not contain unapproved embeddings")
    if int(header["counts"]["pose_members"]) != len(inventory):
        errors.append("inventory header pose count mismatch")
    if int(header["counts"]["semantic_units"]) != len(proposals):
        errors.append("inventory header semantic unit count mismatch")

    geometry_connection = sqlite3.connect(geometry_db)
    geometry_pose_rows = geometry_connection.execute("SELECT pose_id,bvh_path FROM poses").fetchall()
    geometry_projection_counts = dict(
        geometry_connection.execute("SELECT pose_id,COUNT(*) FROM pose_projections GROUP BY pose_id")
    )
    geometry_feature_versions = [
        row[0] for row in geometry_connection.execute("SELECT DISTINCT feature_version FROM pose_projections")
    ]
    geometry_connection.close()
    geometry_pose_ids = {row[0] for row in geometry_pose_rows}
    if geometry_pose_ids != set(inventory_ids):
        errors.append("geometry DB and inventory pose coverage differ")
    invalid_projection_counts = [pose_id for pose_id, count in geometry_projection_counts.items() if count != 4]
    if invalid_projection_counts:
        errors.append(f"geometry DB has {len(invalid_projection_counts)} poses without four projections")
    missing_geometry_paths = [path for _, path in geometry_pose_rows if not path or not Path(path).is_file()]
    if missing_geometry_paths:
        errors.append(f"geometry DB has {len(missing_geometry_paths)} missing BVH paths")

    cmu = [row for row in sources if row.get("provider") == "cmu_graphics_lab"]
    unmatched_cmu = [row["source_clip_id"] for row in cmu if row["catalog_evidence"]["status"] != "verified"]
    excluded_sources = sorted(
        row["source_clip_id"]
        for row in sources
        if (row.get("library_policy") or {}).get("state") == "pending_removal"
    )
    excluded_pose_ids = sorted(
        row["pose_id"] for row in lineage if row["source_clip_id"] in excluded_sources
    )
    for source in sources:
        policy = source.get("library_policy") or {}
        if policy.get("state") == "pending_removal" and any(
            policy.get(field) is not False
            for field in ("semantic_index", "geometry_index", "release")
        ):
            errors.append(f"{source['source_clip_id']}: exclusion eligibility must be false")
    missing_action_sources = sorted(
        row["source_clip_id"]
        for row in sources
        if row["source_clip_id"] not in excluded_sources
        if not (row.get("original", {}).get("title") or row.get("original", {}).get("local_label"))
    )
    orphan_mirrors = [
        row["pose_id"]
        for row in inventory
        if row["grouping"]["variant"]["kind"] == "mirrored"
        and not row["grouping"]["variant"]["paired_member_present"]
    ]
    if mirror_review:
        warnings.append(f"mirror atom boundary review required for {mirror_review} semantic units")
    if orphan_mirrors:
        warnings.append(f"orphan mirror review required for {len(orphan_mirrors)} poses")
    priority_review_units = sum(
        proposal.get("validation", {}).get("review_priority") in {"P0", "P1"}
        for proposal in proposals.values()
    )
    if priority_review_units:
        warnings.append(f"P0/P1 review required for {priority_review_units} semantic units")
    if excluded_pose_ids:
        warnings.append(
            f"{len(excluded_pose_ids)} excluded pose members remain in geometry DB pending physical deletion"
        )

    return {
        "artifact_type": "semantic_tagging_validation",
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "pose_members": len(inventory),
            "semantic_units": len(proposals),
            "source_clips": len(sources),
            "proposal_history_rows": len(proposal_history),
            "decisions": len(decisions),
            "observed_atoms": atom_count,
            "unit_observed_atoms": unit_atom_count,
            "mirror_units_needing_review": mirror_review,
            "mirror_units_canonicalized": mirror_canonicalized,
            "cmu_catalog_unmatched": len(unmatched_cmu),
            "missing_action_name_sources": len(missing_action_sources),
            "excluded_source_clips": len(excluded_sources),
            "excluded_pose_members": len(excluded_pose_ids),
            "excluded_semantic_units": sum(
                proposal.get("validation", {}).get("review_priority") == "PX"
                for proposal in proposals.values()
            ),
            "priority_review_units": priority_review_units,
            "auto_verified_observed_tag_units": sum(
                proposal.get("workflow_status") == "auto_verified_observed_tags"
                for proposal in proposals.values()
            ),
            "orphan_mirrors": len(orphan_mirrors),
            "geometry_poses": len(geometry_pose_rows),
            "geometry_projections": sum(geometry_projection_counts.values()),
            "geometry_feature_versions": geometry_feature_versions,
            **{f"db_{key}": value for key, value in db_counts.items()},
        },
        "review_items": {
            "cmu_catalog_unmatched": unmatched_cmu,
            "missing_action_name_sources": missing_action_sources,
            "excluded_source_clips": excluded_sources,
            "orphan_mirrors": orphan_mirrors,
        },
        "diagnostic_probes": {
            "left_leg_behind_both_arms_wide_exact_atom_candidates": golden_a_candidates,
            "dance_source_context_units": dance_context_units,
            "traditional_or_historical_atoms": traditional_historical_atoms,
            "traditional_dance_exact_status": (
                "library_gap" if traditional_historical_atoms == 0 else "requires_constraint_evaluation"
            ),
        },
        "pose_library_version": header["pose_library_version"],
        "geometry_db": {"path": str(geometry_db), "sha256": _sha256(geometry_db)},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated semantic tagging artifacts.")
    parser.add_argument("--output-dir", default="data/semantic")
    parser.add_argument("--report", default="data/semantic/tagging-validation.v1.json")
    parser.add_argument("--geometry-db", default="data/poses.db")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate(Path(args.output_dir), Path(args.geometry_db))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
