from __future__ import annotations

from pathlib import Path
from typing import Any

from .labels import resolve_run
from .report import write_run_report
from .util import (
    atomic_write_text,
    hash_json,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)


SEMANTIC_ARTIFACTS = {
    "cut_results.jsonl": {
        "keys": ("cut_id",),
        "exact": (
            "status", "route", "vlm_count", "detector_count",
            "count_confidence", "people_count", "error_kind",
        ),
        "numeric": (),
    },
    "predictions.jsonl": {
        "keys": ("prediction_id",),
        "exact": (
            "cut_id", "person_index", "skeleton_state", "skeleton_source",
            "coverage_class", "valid_joint_count", "confidence", "candidate_count",
        ),
        "numeric": ("box_xyxy",),
    },
    "matches.jsonl": {
        "keys": ("cut_id", "person_id", "prediction_id", "match_status"),
        "exact": ("expected_route", "predicted_route"),
        "numeric": ("iou", "normalized_center_distance"),
    },
    "candidates.jsonl": {
        "keys": ("prediction_id", "rank"),
        "exact": (
            "cut_id", "person_id", "pose_id", "view", "family_id",
            "candidate_artifact_id", "display_filter_status", "surfaced",
        ),
        "numeric": ("distance",),
    },
}


def _row_key(row: dict, fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _numeric_equal(first: Any, second: Any, tolerance: float) -> bool:
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return (
            isinstance(second, (int, float))
            and not isinstance(second, bool)
            and abs(float(first) - float(second)) <= tolerance
        )
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        return all(_numeric_equal(left, right, tolerance) for left, right in zip(first, second))
    if first is None or second is None:
        return first is second
    return first == second


def semantic_compare(
    first_dir: str | Path,
    second_dir: str | Path,
    *,
    numeric_tolerance: float = 1e-6,
    sample_limit: int = 50,
) -> dict:
    """Compare deterministic output semantics while excluding wall-clock fields."""
    first_dir, second_dir = Path(first_dir), Path(second_dir)
    artifacts: dict[str, dict] = {}
    all_exact = True
    for filename, spec in SEMANTIC_ARTIFACTS.items():
        left_rows = read_jsonl(first_dir / filename)
        right_rows = read_jsonl(second_dir / filename)
        left = {_row_key(row, spec["keys"]): row for row in left_rows}
        right = {_row_key(row, spec["keys"]): row for row in right_rows}
        duplicate_keys = (
            len(left) != len(left_rows) or len(right) != len(right_rows)
        )
        missing_left = sorted(set(right) - set(left), key=repr)
        missing_right = sorted(set(left) - set(right), key=repr)
        field_mismatches: list[dict] = []
        for key in sorted(set(left) & set(right), key=repr):
            for field in spec["exact"]:
                if left[key].get(field) != right[key].get(field):
                    field_mismatches.append({
                        "key": list(key), "field": field,
                        "first": left[key].get(field), "second": right[key].get(field),
                        "comparison": "exact",
                    })
            for field in spec["numeric"]:
                if not _numeric_equal(
                    left[key].get(field), right[key].get(field), numeric_tolerance
                ):
                    field_mismatches.append({
                        "key": list(key), "field": field,
                        "first": left[key].get(field), "second": right[key].get(field),
                        "comparison": "numeric_tolerance",
                    })
        exact = not (
            duplicate_keys or missing_left or missing_right or field_mismatches
        )
        all_exact = all_exact and exact
        artifacts[filename] = {
            "equal": exact,
            "first_rows": len(left_rows),
            "second_rows": len(right_rows),
            "duplicate_keys": duplicate_keys,
            "missing_from_first": [list(key) for key in missing_left[:sample_limit]],
            "missing_from_second": [list(key) for key in missing_right[:sample_limit]],
            "field_mismatch_count": len(field_mismatches),
            "field_mismatches": field_mismatches[:sample_limit],
        }
    return {
        "status": "equal" if all_exact else "different",
        "numeric_tolerance": numeric_tolerance,
        "wall_clock_fields_excluded": ["latency_ms", "created_at", "completed_at"],
        "artifacts": artifacts,
    }


def _artifact_hash(manifest: dict, name: str):
    value = manifest.get("artifacts", {}).get(name)
    return value.get("sha256") if isinstance(value, dict) else value


def compatibility_errors(first: dict, second: dict, changed: set[str]) -> list[str]:
    checks = [
        ("dataset_id", first.get("dataset", {}).get("dataset_id"), second.get("dataset", {}).get("dataset_id"), None),
        ("cut_manifest", first.get("dataset", {}).get("cut_manifest_sha256"), second.get("dataset", {}).get("cut_manifest_sha256"), None),
        ("gt", first.get("dataset", {}).get("gt_sha256"), second.get("dataset", {}).get("gt_sha256"), None),
        ("rubric", first.get("dataset", {}).get("rubric_version"), second.get("dataset", {}).get("rubric_version"), None),
        ("metric_schema", first.get("metric_schema_version"), second.get("metric_schema_version"), "metric"),
        ("db", _artifact_hash(first, "db"), _artifact_hash(second, "db"), "library"),
        ("bvh", _artifact_hash(first, "bvh"), _artifact_hash(second, "bvh"), "library"),
        ("thumbnails", _artifact_hash(first, "thumbnails"), _artifact_hash(second, "thumbnails"), "renderer"),
        ("renderer", first.get("artifacts", {}).get("renderer_version"), second.get("artifacts", {}).get("renderer_version"), "renderer"),
        ("fixture", first.get("fixture_id"), second.get("fixture_id"), "fixture"),
        ("fixture_content", first.get("fixture_content_sha256"), second.get("fixture_content_sha256"), "fixture"),
        ("requested_backend", first.get("requested_backend"), second.get("requested_backend"), "backend"),
        ("actual_backend", first.get("actual_backend"), second.get("actual_backend"), "backend"),
        ("surface_policy", first.get("config", {}).get("surface_policy"), second.get("config", {}).get("surface_policy"), "policy"),
        ("match_policy", first.get("config", {}).get("match_policy"), second.get("config", {}).get("match_policy"), "matching"),
        ("time_budget_ms", first.get("config", {}).get("time_budget_ms"), second.get("config", {}).get("time_budget_ms"), "latency_budget"),
    ]
    errors = []
    for name, left, right, allowed_change in checks:
        if left != right and (allowed_change is None or allowed_change not in changed):
            errors.append(f"{name} differs: {left!r} != {right!r}")
    return errors


def _paired(first: dict, second: dict, field: str) -> dict:
    left = {row["person_id"]: row for row in first["person_outcomes"]}
    right = {row["person_id"]: row for row in second["person_outcomes"]}
    ids = sorted(set(left) | set(right))
    improved: list[str] = []
    regressed: list[str] = []
    equal: list[str] = []
    unresolved: list[str] = []
    for person_id in ids:
        a = left.get(person_id, {}).get(field)
        b = right.get(person_id, {}).get(field)
        if a is None or b is None:
            unresolved.append(person_id)
        elif not a and b:
            improved.append(person_id)
        elif a and not b:
            regressed.append(person_id)
        else:
            equal.append(person_id)
    return {
        "field": field,
        "improved": len(improved),
        "regressed": len(regressed),
        "equal": len(equal),
        "unresolved": len(unresolved),
        "improved_person_ids": improved,
        "regressed_person_ids": regressed,
        "unresolved_person_ids": unresolved,
    }


def compare_runs(
    first_run: str | Path,
    second_run: str | Path,
    *,
    labels_path: str | Path,
    changed: set[str] | None = None,
    output_root: str | Path = "out/eval/comparisons",
    numeric_tolerance: float = 1e-6,
) -> Path:
    changed = set(changed or ())
    first_dir, second_dir = resolve_run(first_run), resolve_run(second_run)
    first_manifest = read_json(first_dir / "manifest.json")
    second_manifest = read_json(second_dir / "manifest.json")
    errors = compatibility_errors(first_manifest, second_manifest, changed)
    if errors:
        raise ValueError("incomparable runs: " + "; ".join(errors))

    first_report = write_run_report(first_dir, labels_path)
    second_report = write_run_report(second_dir, labels_path)
    coverage = _paired(first_report, second_report, "candidate_coverage_at_5")
    assist = _paired(first_report, second_report, "assist_success_at_5")
    semantic = semantic_compare(
        first_dir, second_dir, numeric_tolerance=numeric_tolerance
    )
    status = (
        "complete"
        if first_report["status"] == second_report["status"] == "complete"
        else "incomplete"
    )
    comparison_id = (
        f"compare-{hash_json({'a': first_manifest['run_id'], 'b': second_manifest['run_id'], 'labels': sha256_file(labels_path)})[:12]}"
    )
    output = Path(output_root).resolve() / comparison_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "created_at": utc_now(),
        "status": status,
        "first_run": first_manifest["run_id"],
        "second_run": second_manifest["run_id"],
        "labels_sha256": sha256_file(labels_path),
        "changed": sorted(changed),
        "candidate_coverage_at_5": coverage,
        "assist_success_at_5": assist,
        "semantic_replay": semantic,
        "rates": {
            "first": {
                "candidate_coverage_at_5": first_report["search"]["candidate_coverage_at_5"],
                "assist_success_at_5": first_report["product"]["assist_success_at_5"],
            },
            "second": {
                "candidate_coverage_at_5": second_report["search"]["candidate_coverage_at_5"],
                "assist_success_at_5": second_report["product"]["assist_success_at_5"],
            },
        },
    }
    write_json(output / "comparison.json", result)
    lines = [
        f"# Comparison `{first_manifest['run_id']}` → `{second_manifest['run_id']}`",
        "",
        f"- status: **{status.upper()}**",
        f"- declared changes: {', '.join(sorted(changed)) or 'code only'}",
        f"- deterministic semantic outputs: **{semantic['status'].upper()}** "
        f"(numeric tolerance `{numeric_tolerance:g}`)",
        "",
        "| metric | improved | regressed | equal | unresolved |",
        "|---|---:|---:|---:|---:|",
        f"| candidate_coverage@5 | {coverage['improved']} | {coverage['regressed']} | {coverage['equal']} | {coverage['unresolved']} |",
        f"| assist_success@5 | {assist['improved']} | {assist['regressed']} | {assist['equal']} | {assist['unresolved']} |",
        "",
        "## Regressions",
        "",
    ]
    lines.extend(f"- `{person_id}`" for person_id in coverage["regressed_person_ids"])
    if not coverage["regressed_person_ids"]:
        lines.append("- none")
    lines.extend(["", "## Semantic replay differences", ""])
    different = [
        (name, value) for name, value in semantic["artifacts"].items()
        if not value["equal"]
    ]
    if different:
        lines.extend(
            f"- `{name}`: {value['field_mismatch_count']} field mismatches, "
            f"{len(value['missing_from_first'])} only in second, "
            f"{len(value['missing_from_second'])} only in first"
            for name, value in different
        )
    else:
        lines.append("- all compared deterministic fields match")
    atomic_write_text(output / "comparison.md", "\n".join(lines) + "\n")
    return output
