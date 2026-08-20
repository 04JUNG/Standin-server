"""Three-arm, intent-to-treat reporting for refine evaluation.

The report in this module deliberately compares only final artifacts.  Internal
v1/v2 losses are not comparable, so callers must put measurements produced by
the common external evaluator in ``automatic_metrics``.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .util import (
    atomic_write_text,
    hash_json,
    hash_jsonl,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from .refine_mesh import (
    MESH_CHECK_CONTRACT_VERSION,
    MESH_REQUIRED_CHECKS,
    validate_mesh_evidence_bundle,
)


ARMS = ("B0_no_refine", "B1_v1", "B2_v24_aggressive")
REPORT_VERSION = "refine-report-v1"
CONTRASTS = {
    "B1_vs_B0": ("B1_v1", "B0_no_refine"),
    "B2_vs_B0": ("B2_v24_aggressive", "B0_no_refine"),
    "B2_vs_B1": ("B2_v24_aggressive", "B1_v1"),
}
USABILITY = {"direct", "reference", "unusable"}
WINNERS = {"left", "right", "tie", "both_bad"}
REJECT_REASONS = {
    "pose_mismatch", "anatomy", "collision", "contact", "feet_ground",
    "balance", "silhouette", "ownership", "other",
}
PAIR_SEVERITIES = {"minor", "major"}
PAIR_BODY_PARTS = {"overall", "arm", "hand", "leg", "foot", "torso"}
SAFETY_LABELS = {"none", "anatomy", "collision", "contact", "ground", "other"}

_ARM_ALIASES = {
    "b0": ARMS[0], "b0_no_refine": ARMS[0], "no_refine": ARMS[0],
    "base": ARMS[0], ARMS[0].lower(): ARMS[0],
    "b1": ARMS[1], "b1_v1": ARMS[1], "v1": ARMS[1],
    ARMS[1].lower(): ARMS[1],
    "b2": ARMS[2], "b2_v24": ARMS[2], "v2": ARMS[2], "v2.4": ARMS[2],
    "v24": ARMS[2], "aggressive": ARMS[2], ARMS[2].lower(): ARMS[2],
}
_CONTRAST_ALIASES = {
    "b1_b0": "B1_vs_B0", "b1-vs-b0": "B1_vs_B0", "b1_vs_b0": "B1_vs_B0",
    "b2_b0": "B2_vs_B0", "b2-vs-b0": "B2_vs_B0", "b2_vs_b0": "B2_vs_B0",
    "b2_b1": "B2_vs_B1", "b2-vs-b1": "B2_vs_B1", "b2_vs_b1": "B2_vs_B1",
}
_FALSE_STRINGS = {"", "0", "false", "none", "no", "ok", "pass"}
_NO_VIOLATION_STRINGS = _FALSE_STRINGS | {"unknown", "n/a", "na"}


def _unit_id(row: dict) -> str | None:
    value = row.get("unit_id") or row.get("evaluation_unit_id") or row.get("person_id")
    return str(value) if value not in (None, "") else None


def _arm(value: Any) -> str | None:
    if value is None:
        return None
    return _ARM_ALIASES.get(str(value).strip().lower())


def _contrast(value: Any) -> str | None:
    if value in CONTRASTS:
        return str(value)
    if value is None:
        return None
    return _CONTRAST_ALIASES.get(str(value).strip().lower())


def _artifact_id(row: dict) -> str | None:
    value = (
        row.get("blind_artifact_id") or row.get("artifact_id")
        or row.get("candidate_artifact_id")
        or row.get("result_artifact_id") or row.get("base_artifact_id")
    )
    return str(value) if value not in (None, "") else None


def _delivery_failure(row: dict) -> bool:
    """True only when timeout/error left no usable final artifact to judge."""
    failed = _truthy_failure(row.get("timeout")) or _truthy_failure(row.get("error"))
    has_identity = bool(
        _artifact_id(row) or row.get("geometry_sha256") or row.get("bvh_sha256")
    )
    has_artifact = bool(row.get("artifact_path") or row.get("artifact_available"))
    return bool(failed and not (has_identity and has_artifact))


def _truthy_failure(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    """Expose nested evaluator scalars without treating bools/counts as errors."""
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_numeric_metrics(child, name))
    else:
        number = _finite_number(value)
        if number is not None and prefix:
            output[prefix] = number
    return output


def _ratio(numerator: int, denominator: int, *, complete: bool = True,
           observed: int | None = None) -> dict:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "observed": int(denominator if observed is None else observed),
        "rate": (numerator / denominator) if denominator and complete else None,
        "complete": bool(complete),
    }


def _cluster_key(unit: dict) -> tuple[str, ...]:
    artist = unit.get("artist_id")
    project = unit.get("project_id")
    if artist not in (None, "") or project not in (None, ""):
        return ("artist_project", str(artist or "unknown"), str(project or "unknown"))
    scene = unit.get("scene_group_id")
    if scene not in (None, ""):
        return ("scene_group", str(scene))
    return ("unit", str(_unit_id(unit)))


def _cluster_bootstrap(
    values: list[tuple[dict, float]], *, repetitions: int, seed: int,
) -> dict | None:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for unit, value in values:
        grouped[_cluster_key(unit)].append(float(value))
    groups = sorted(grouped)
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repetitions):
        selected = [rng.choice(groups) for _ in groups]
        sample = [value for group in selected for value in grouped[group]]
        samples.append(sum(sample) / len(sample))
    return {
        "method": "artist_project_or_scene_cluster_bootstrap",
        "clusters": len(groups),
        "repetitions": repetitions,
        "seed": seed,
        "low": percentile(samples, 0.025),
        "high": percentile(samples, 0.975),
    }


def _safety_from_label(row: dict) -> bool | None:
    if isinstance(row.get("hard_safety_violation"), bool):
        return bool(row["hard_safety_violation"])
    value = row.get("safety_violation")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _NO_VIOLATION_STRINGS:
            return False if normalized != "unknown" else None
        return True
    return None


def _new_violation(row: dict) -> bool:
    for field in ("new_hard_safety_violation", "new_hard_violation"):
        if isinstance(row.get(field), bool):
            return bool(row[field])
    for field in ("new_hard_safety_violations", "new_hard_violations"):
        if field in row:
            return _truthy_failure(row.get(field))
    evaluation = row.get("external_evaluation")
    if isinstance(evaluation, dict):
        safety = evaluation.get("safety") or {}
        if isinstance(safety.get("new_hard_violation"), bool):
            return bool(safety["new_hard_violation"])
    # Legacy rows used one list for both concepts. Preserve compatibility, but
    # new runs always write ``new_hard_safety_violations`` explicitly.
    return _truthy_failure(row.get("hard_safety_violations"))


def _absolute_violation(row: dict) -> bool:
    for field in ("hard_safety_violation", "hard_safety_violations"):
        if field in row:
            return _truthy_failure(row.get(field))
    evaluation = row.get("external_evaluation")
    if isinstance(evaluation, dict):
        safety = evaluation.get("safety") or {}
        if isinstance(safety.get("hard_safety_violation"), bool):
            return bool(safety["hard_safety_violation"])
    return _new_violation(row)


def _hard_violation(row: dict, human_value: bool) -> bool:
    return human_value or _absolute_violation(row)


def _normalize_inputs(units: list[dict], arm_rows: list[dict]) -> tuple[
    dict[str, dict], dict[tuple[str, str], dict], list[str]
]:
    errors: list[str] = []
    units_by_id: dict[str, dict] = {}
    for index, unit in enumerate(units, 1):
        key = _unit_id(unit)
        if key is None:
            errors.append(f"unit row {index}: unit_id is required")
        elif key in units_by_id:
            errors.append(f"unit row {index}: duplicate unit_id {key}")
        else:
            units_by_id[key] = unit

    rows_by_key: dict[tuple[str, str], dict] = {}
    for index, row in enumerate(arm_rows, 1):
        unit_id, arm = _unit_id(row), _arm(row.get("arm"))
        if unit_id is None:
            errors.append(f"arm row {index}: unit_id is required")
            continue
        if unit_id not in units_by_id:
            errors.append(f"arm row {index}: unknown unit_id {unit_id}")
        if arm is None:
            errors.append(f"arm row {index}: unknown arm {row.get('arm')!r}")
            continue
        key = (unit_id, arm)
        if key in rows_by_key:
            errors.append(f"arm row {index}: duplicate arm row {unit_id}/{arm}")
        else:
            rows_by_key[key] = row
    for unit_id in units_by_id:
        for arm in ARMS:
            if (unit_id, arm) not in rows_by_key:
                errors.append(f"missing arm row {unit_id}/{arm}")
                continue
            row = rows_by_key[(unit_id, arm)]
            has_artifact = bool(
                _artifact_id(row) or row.get("artifact_path") or row.get("bvh_sha256")
                or row.get("geometry_sha256")
            )
            operational_failure = bool(
                _truthy_failure(row.get("timeout")) or _truthy_failure(row.get("error"))
            )
            if _truthy_failure(row.get("contract_error")):
                errors.append(
                    f"arm row {unit_id}/{arm}: strict response contract or lineage failure"
                )
            if not has_artifact and not operational_failure:
                errors.append(f"arm row {unit_id}/{arm}: final artifact identity is required")
            if arm == ARMS[0]:
                expected = units_by_id[unit_id].get("selected_base_sha256")
                actual = row.get("bvh_sha256") or row.get("artifact_sha256")
                if expected and actual and expected != actual:
                    errors.append(f"arm row {unit_id}/{arm}: selected base hash mismatch")
    return units_by_id, rows_by_key, errors


def _resolve_independent_labels(
    labels: list[dict], units: dict[str, dict], arms: dict[tuple[str, str], dict],
    provenance: list[dict] | None = None,
    usable_categories: set[str] | None = None,
) -> tuple[dict[tuple[str, str], tuple[bool, bool]], dict, list[str]]:
    errors: list[str] = []
    usable_categories = usable_categories or set()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    provenance_by_item: dict[str, dict] = {}
    for index, raw in enumerate(provenance or [], 1):
        item_id = raw.get("item_id")
        unit_id = _unit_id(raw)
        raw_arms = raw.get("arms")
        if not isinstance(raw_arms, list):
            raw_arms = [raw.get("arm")]
        normalized_arms = [_arm(value) for value in raw_arms]
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"independent provenance row {index}: item_id is required")
            continue
        if item_id in provenance_by_item:
            errors.append(f"independent provenance row {index}: duplicate item_id {item_id}")
            continue
        if unit_id not in units:
            errors.append(
                f"independent provenance row {index}: unknown unit_id {unit_id!r}"
            )
            continue
        if not normalized_arms or any(value is None for value in normalized_arms):
            errors.append(f"independent provenance row {index}: arms are required")
            continue
        artifact = _artifact_id(raw)
        mismatch = [
            arm for arm in normalized_arms
            if artifact and _artifact_id(arms.get((unit_id, arm), {})) not in (None, artifact)
        ]
        if mismatch:
            errors.append(
                f"independent provenance row {index}: artifact does not match {unit_id}/{mismatch[0]}"
            )
            continue
        provenance_by_item[item_id] = {
            **raw, "unit_id": unit_id, "arms": normalized_arms,
        }

    nonfinal = 0
    for index, row in enumerate(labels, 1):
        item_id = row.get("item_id")
        item_source = provenance_by_item.get(str(item_id))
        if not isinstance(item_id, str) or not item_id or item_source is None:
            errors.append(
                f"independent label row {index}: a known provenance item_id is required"
            )
            continue
        unit_id = item_source["unit_id"]
        if unit_id not in units:
            errors.append(f"independent label row {index}: unknown unit_id {unit_id!r}")
            continue
        usability = row.get("overall_usability") or row.get("usefulness")
        safety = _safety_from_label(row)
        if usability == "unknown" or safety is None and row.get("safety_violation") == "unknown":
            nonfinal += 1
            continue
        if usability not in USABILITY:
            errors.append(
                f"independent label row {index}: overall_usability must be one of {sorted(USABILITY)}"
            )
            continue
        if usability == "unusable" and row.get("reject_reason") not in REJECT_REASONS:
            errors.append(
                f"independent label row {index}: unusable requires a valid reject_reason"
            )
            continue
        if not str(row.get("labeler_id") or "").strip():
            errors.append(f"independent label row {index}: labeler_id is required")
            continue
        if safety is None:
            errors.append(f"independent label row {index}: final safety label is required")
            continue
        targets = item_source["arms"]
        for target in targets:
            target_row = arms.get((unit_id, target), {})
            operational_failure = _delivery_failure(target_row)
            if operational_failure:
                expected = (False, _new_violation(target_row))
                actual = (usability in usable_categories, bool(safety))
                if actual != expected:
                    errors.append(
                        f"independent label row {index}: operational-failure arm "
                        f"{unit_id}/{target} must be unusable with safety={expected[1]}"
                    )
                # Operational state is authoritative.  Never let a human label
                # turn a timeout/error into a successful ITT outcome.
                continue
            grouped[(unit_id, target)].append({
                **row,
                "_judgment": (usability in usable_categories, bool(safety)),
            })

    resolved: dict[tuple[str, str], tuple[bool, bool]] = {}
    agreement_pairs = agreement_matches = conflicts = 0
    for key, rows in grouped.items():
        judgments = [row["_judgment"] for row in rows]
        for left in range(len(judgments)):
            for right in range(left + 1, len(judgments)):
                agreement_pairs += 1
                agreement_matches += judgments[left] == judgments[right]
        adjudicated = [
            row for row in rows
            if row.get("consensus") is True or row.get("adjudicated") is True
        ]
        selected = adjudicated if adjudicated else rows
        values = {row["_judgment"] for row in selected}
        if len(values) == 1:
            resolved[key] = next(iter(values))
        else:
            conflicts += 1
            errors.append(f"conflicting independent labels for {key[0]}/{key[1]}")

    # A timed-out or errored arm has no artifact to rate and is an ITT failure,
    # not a silently dropped denominator.  Operational state is authoritative
    # even when a contradictory human row was supplied above.
    automatic_failures = 0
    for key, row in arms.items():
        if _delivery_failure(row):
            resolved[key] = (False, _new_violation(row))
            automatic_failures += 1

    expected = len(units) * len(ARMS)
    return resolved, {
        "expected_arm_judgments": expected,
        "resolved_arm_judgments": len(resolved),
        "raw_label_rows": len(labels),
        "automatic_operational_failures": automatic_failures,
        "nonfinal_label_rows": nonfinal,
        "conflicts": conflicts,
        "agreement": _ratio(agreement_matches, agreement_pairs),
    }, errors


def _derive_contrast(left: str | None, right: str | None) -> str | None:
    for name, (arm_a, arm_b) in CONTRASTS.items():
        if {left, right} == {arm_a, arm_b}:
            return name
    return None


def _resolve_pair_labels(
    labels: list[dict], provenance: list[dict], units: dict[str, dict],
    arms: dict[tuple[str, str], dict],
) -> tuple[dict[tuple[str, str], dict], dict, list[str]]:
    errors: list[str] = []
    provenance_by_id: dict[str, dict] = {}
    expected_keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(provenance, 1):
        pair_id = raw.get("pair_id")
        unit_id = _unit_id(raw)
        left_arm = _arm(raw.get("left_arm") or raw.get("left_variant"))
        right_arm = _arm(raw.get("right_arm") or raw.get("right_variant"))
        contrast = _contrast(raw.get("contrast")) or _derive_contrast(left_arm, right_arm)
        if not isinstance(pair_id, str) or not pair_id:
            errors.append(f"pair provenance row {index}: pair_id is required")
            continue
        if pair_id in provenance_by_id:
            errors.append(f"pair provenance row {index}: duplicate pair_id {pair_id}")
            continue
        if unit_id not in units:
            errors.append(f"pair provenance row {index}: unknown unit_id {unit_id!r}")
            continue
        if contrast is None or left_arm is None or right_arm is None:
            errors.append(f"pair provenance row {index}: contrast and left/right arms are required")
            continue
        if {left_arm, right_arm} != set(CONTRASTS[contrast]):
            errors.append(f"pair provenance row {index}: arms do not match {contrast}")
            continue
        key = (unit_id, contrast)
        if key in expected_keys:
            errors.append(f"pair provenance row {index}: duplicate contrast {unit_id}/{contrast}")
            continue
        expected_keys.add(key)
        provenance_by_id[pair_id] = {
            **raw, "unit_id": unit_id, "contrast": contrast,
            "left_arm": left_arm, "right_arm": right_arm,
        }
    for unit_id in units:
        for contrast in CONTRASTS:
            if (unit_id, contrast) not in expected_keys:
                errors.append(f"missing pair provenance {unit_id}/{contrast}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    nonfinal = 0
    for index, row in enumerate(labels, 1):
        pair_id = row.get("pair_id")
        if pair_id not in provenance_by_id:
            errors.append(f"pair label row {index}: unknown pair_id {pair_id!r}")
            continue
        winner = row.get("winner") or row.get("preference")
        if winner == "unknown":
            nonfinal += 1
            continue
        if winner not in WINNERS:
            errors.append(f"pair label row {index}: winner must be one of {sorted(WINNERS)}")
            continue
        label_source = row.get("label_source")
        automatic = label_source in {
            "automatic_exact_geometry", "automatic_operational_failure",
        }
        if row.get("severity") not in PAIR_SEVERITIES:
            errors.append(f"pair label row {index}: final severity is required")
            continue
        if row.get("body_part") not in PAIR_BODY_PARTS:
            errors.append(f"pair label row {index}: final body_part is required")
            continue
        if row.get("safety_violation") not in SAFETY_LABELS:
            errors.append(f"pair label row {index}: final safety_violation is required")
            continue
        if not automatic and not str(row.get("labeler_id") or "").strip():
            errors.append(f"pair label row {index}: labeler_id is required")
            continue
        if label_source == "automatic_operational_failure":
            source = provenance_by_id[str(pair_id)]
            left_row = arms.get((source["unit_id"], source["left_arm"]), {})
            right_row = arms.get((source["unit_id"], source["right_arm"]), {})
            left_failed = _delivery_failure(left_row)
            right_failed = _delivery_failure(right_row)
            expected_winner = (
                "both_bad" if left_failed and right_failed
                else "right" if left_failed
                else "left" if right_failed
                else None
            )
            expected = (expected_winner, "major", "overall", "other")
            actual = (
                winner, row.get("severity"), row.get("body_part"),
                row.get("safety_violation"),
            )
            if expected_winner is None or actual != expected:
                errors.append(
                    f"pair label row {index}: automatic operational-failure judgment "
                    f"must match failed arms ({expected!r})"
                )
                continue
        grouped[str(pair_id)].append({**row, "_winner": winner})

    resolved: dict[tuple[str, str], dict] = {}
    agreement_pairs = agreement_matches = conflicts = auto_ties = 0
    automatic_operational_failures = 0
    for pair_id, source in provenance_by_id.items():
        key = (source["unit_id"], source["contrast"])
        rows = grouped.get(pair_id, [])
        left_row = arms.get((source["unit_id"], source["left_arm"]), {})
        right_row = arms.get((source["unit_id"], source["right_arm"]), {})
        left_artifact = source.get("left_artifact_id") or _artifact_id(left_row)
        right_artifact = source.get("right_artifact_id") or _artifact_id(right_row)
        left_geometry = left_row.get("geometry_sha256") or left_row.get("bvh_sha256")
        right_geometry = right_row.get("geometry_sha256") or right_row.get("bvh_sha256")
        left_failed = _delivery_failure(left_row)
        right_failed = _delivery_failure(right_row)
        if left_failed or right_failed:
            winner = (
                "both_bad" if left_failed and right_failed
                else "right" if left_failed
                else "left"
            )
            expected = (winner, "major", "overall", "other")
            for row in rows:
                actual = (
                    row.get("_winner"), row.get("severity"),
                    row.get("body_part"), row.get("safety_violation"),
                )
                if actual != expected:
                    errors.append(
                        f"pair label {pair_id}: human/automatic judgment contradicts "
                        f"operational-failure outcome {expected!r}"
                    )
            resolved[key] = {
                "pair_id": pair_id,
                "winner": winner,
                "severity": "major",
                "body_part": "overall",
                "safety_violation": "other",
                "left_arm": source["left_arm"],
                "right_arm": source["right_arm"],
                "automatic_tie": False,
                "label_source": "automatic_operational_failure",
            }
            automatic_operational_failures += 1
            continue
        identical = bool(
            (left_geometry and right_geometry and left_geometry == right_geometry)
            or (left_artifact and right_artifact and left_artifact == right_artifact)
        )
        if identical:
            if any(row.get("_winner") != "tie" for row in rows):
                errors.append(f"exact geometry pair {pair_id} must be an automatic tie")
            resolved[key] = {
                "pair_id": pair_id, "winner": "tie", "severity": "minor",
                "body_part": "overall", "safety_violation": "none",
                "left_arm": source["left_arm"], "right_arm": source["right_arm"],
                "automatic_tie": True,
                "label_source": "automatic_exact_geometry",
            }
            auto_ties += 1
            continue
        if any(row.get("label_source") == "automatic_exact_geometry" for row in rows):
            errors.append(f"non-identical pair {pair_id} cannot claim automatic tie")
            continue
        if rows:
            judgments = [
                (
                    row["_winner"], row.get("severity"), row.get("body_part"),
                    row.get("safety_violation"),
                )
                for row in rows
            ]
            for left in range(len(judgments)):
                for right in range(left + 1, len(judgments)):
                    agreement_pairs += 1
                    agreement_matches += judgments[left] == judgments[right]
            adjudicated = [
                row for row in rows
                if row.get("consensus") is True or row.get("adjudicated") is True
            ]
            selected = adjudicated if adjudicated else rows
            values = {
                (
                    row["_winner"], row.get("severity"), row.get("body_part"),
                    row.get("safety_violation"),
                )
                for row in selected
            }
            if len(values) != 1:
                conflicts += 1
                errors.append(f"conflicting pair labels for {pair_id}")
                continue
            winner, severity, body_part, safety_violation = next(iter(values))
            resolved[key] = {
                "pair_id": pair_id,
                "winner": winner,
                "severity": severity,
                "body_part": body_part,
                "safety_violation": safety_violation,
                "left_arm": source["left_arm"],
                "right_arm": source["right_arm"],
                "automatic_tie": False,
                "label_source": selected[0].get("label_source") or (
                    "human_adjudicated" if adjudicated else "human"
                ),
            }
            automatic_operational_failures += (
                resolved[key]["label_source"] == "automatic_operational_failure"
            )
            continue

    expected = len(units) * len(CONTRASTS)
    return resolved, {
        "expected_pair_judgments": expected,
        "resolved_pair_judgments": len(resolved),
        "raw_label_rows": len(labels),
        "nonfinal_label_rows": nonfinal,
        "automatic_exact_geometry_ties": auto_ties,
        "automatic_operational_failures": automatic_operational_failures,
        "conflicts": conflicts,
        "agreement": _ratio(agreement_matches, agreement_pairs),
    }, errors


def validate_refine_evaluation(
    units: list[dict], arm_rows: list[dict], independent_labels: list[dict] | None = None,
    pair_labels: list[dict] | None = None, pair_provenance: list[dict] | None = None,
    independent_provenance: list[dict] | None = None,
    usability_rubric: dict | None = None,
    mesh_evidence: list[dict] | None = None,
) -> dict:
    """Validate keys, all three ITT arms, labels, and blind-pair provenance."""
    units_by_id, arms, errors = _normalize_inputs(units, arm_rows)
    rubric = usability_rubric if isinstance(usability_rubric, dict) else {}
    usable = set(rubric.get("human_usable_categories") or [])
    rubric_valid = bool(
        (rubric.get("version") or rubric.get("rubric_version"))
        and usable and usable <= USABILITY and "unusable" not in usable
    )
    if not rubric_valid:
        errors.append("versioned usability rubric and human_usable_categories are required")
    _, independent, independent_errors = _resolve_independent_labels(
        independent_labels or [], units_by_id, arms, independent_provenance, usable
    )
    _, pairs, pair_errors = _resolve_pair_labels(
        pair_labels or [], pair_provenance or [], units_by_id, arms
    )
    mesh_validation, mesh_errors = _mesh_safety_evidence(
        units_by_id, arms, mesh_evidence
    )
    errors.extend(independent_errors)
    errors.extend(pair_errors)
    errors.extend(mesh_errors)
    complete = bool(units_by_id) and not errors and (
        independent["resolved_arm_judgments"] == independent["expected_arm_judgments"]
        and pairs["resolved_pair_judgments"] == pairs["expected_pair_judgments"]
    )
    return {
        "complete": complete,
        "units": len(units_by_id),
        "arm_rows": len(arms),
        "independent_labels": independent,
        "pair_labels": pairs,
        "mesh_safety": mesh_validation,
        "errors": errors,
    }


def _arm_summary(
    arm: str, units: dict[str, dict], rows: dict[tuple[str, str], dict],
    judgments: dict[tuple[str, str], tuple[bool, bool]],
) -> tuple[dict, dict[str, bool | None]]:
    outcomes: dict[str, bool | None] = {}
    safe = 0
    observed = 0
    changed = attempted = eligible = fallback_required = exact_fallback = 0
    new_violations = changed_violations = timeouts = errors = 0
    latencies: list[float] = []
    automatic: dict[str, list[float]] = defaultdict(list)
    automatic_missing: Counter = Counter()
    metric_names: set[str] = set()

    for unit_id in units:
        row = rows.get((unit_id, arm), {})
        judgment = judgments.get((unit_id, arm))
        if judgment is None:
            outcomes[unit_id] = None
        else:
            observed += 1
            usable, human_safety = judgment
            value = bool(usable and not _hard_violation(row, human_safety))
            outcomes[unit_id] = value
            safe += value
        is_changed = bool(row.get("geometry_changed"))
        is_new_violation = _new_violation(row)
        changed += is_changed
        attempted += bool(row.get("attempted"))
        eligible += bool(row.get("eligible"))
        fallback_required += bool(row.get("fallback_required"))
        exact_fallback += bool(row.get("fallback_required") and row.get("exact_base"))
        new_violations += is_new_violation
        changed_violations += is_changed and is_new_violation
        timeouts += _truthy_failure(row.get("timeout"))
        errors += _truthy_failure(row.get("error"))
        latency = _finite_number(row.get("latency_ms", row.get("post_click_latency_ms")))
        if (
            arm != ARMS[0] and latency is not None
            and not _truthy_failure(row.get("timeout"))
            and not _truthy_failure(row.get("error"))
        ):
            latencies.append(latency)
        values = row.get("automatic_metrics")
        if isinstance(values, dict):
            flat_values = _flatten_numeric_metrics(values)
            metric_names.update(flat_values)
            for name, number in flat_values.items():
                automatic[name].append(number)

    for name in metric_names:
        automatic_missing[name] = len(units) - len(automatic[name])
    complete = observed == len(units)
    artists: dict[str, list[bool | None]] = defaultdict(list)
    for unit_id, unit in units.items():
        artists[str(unit.get("artist_id") or "unknown")].append(outcomes[unit_id])
    artist_rates: dict[str, dict] = {}
    for artist, values in sorted(artists.items()):
        known = [value for value in values if value is not None]
        artist_rates[artist] = _ratio(
            sum(value is True for value in known), len(values),
            complete=len(known) == len(values), observed=len(known),
        )
    macro_values = [value["rate"] for value in artist_rates.values() if value["rate"] is not None]
    macro_complete = len(macro_values) == len(artist_rates) and bool(artist_rates)
    n_eval = len(units)
    summary = {
        "n_eval": n_eval,
        "safe_usable_rate": _ratio(safe, n_eval, complete=complete, observed=observed),
        "artist_macro_safe_usable_rate": {
            "artists": len(artist_rates),
            "complete_artists": len(macro_values),
            "rate": (sum(macro_values) / len(macro_values)) if macro_complete else None,
            "by_artist": artist_rates,
        },
        "eligible": _ratio(eligible, n_eval),
        "attempted": _ratio(attempted, n_eval),
        "geometry_changed": _ratio(changed, n_eval),
        "new_violation_rate": {
            **_ratio(new_violations, n_eval),
            "zero_observed_rule_of_three_upper_95": (
                min(1.0, 3.0 / n_eval) if n_eval and new_violations == 0 else None
            ),
        },
        "changed_violation_rate": _ratio(changed_violations, changed),
        "fallback_required": _ratio(fallback_required, n_eval),
        "exact_fallback_rate": _ratio(exact_fallback, fallback_required),
        "latency": {
            "basis": "immediate_selected_base" if arm == ARMS[0] else "cache_off_post_click",
            "completed_count": len(latencies),
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
            "max_ms": max(latencies) if latencies else None,
            "timeout_rate": _ratio(timeouts, n_eval),
            "error_rate": _ratio(errors, n_eval),
            "timeout_or_error_rate": _ratio(
                sum(
                    bool(_truthy_failure(rows.get((unit_id, arm), {}).get("timeout"))
                         or _truthy_failure(rows.get((unit_id, arm), {}).get("error")))
                    for unit_id in units
                ),
                n_eval,
            ),
        },
        "automatic_metrics": {
            name: {
                "mean": sum(values) / len(values), "count": len(values),
                "missing": automatic_missing[name],
            }
            for name, values in sorted(automatic.items())
        },
    }
    return summary, outcomes


def _paired_effect(
    arm_a: str, arm_b: str, units: dict[str, dict],
    outcomes: dict[str, dict[str, bool | None]], *, repetitions: int, seed: int,
) -> dict:
    pairs: list[tuple[dict, float]] = []
    discordant_a_only = discordant_b_only = 0
    for unit_id, unit in units.items():
        value_a = outcomes[arm_a][unit_id]
        value_b = outcomes[arm_b][unit_id]
        if value_a is not None and value_b is not None:
            pairs.append((unit, float(value_a) - float(value_b)))
            discordant_a_only += bool(value_a and not value_b)
            discordant_b_only += bool(value_b and not value_a)
    complete = len(pairs) == len(units) and bool(units)
    effect = sum(value for _, value in pairs) / len(pairs) if complete else None
    discordant = discordant_a_only + discordant_b_only
    mcnemar_p = None
    if complete and discordant:
        tail = sum(
            math.comb(discordant, index) for index in range(min(discordant_a_only, discordant_b_only) + 1)
        ) / (2 ** discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    return {
        "difference": effect,
        "difference_percentage_points": effect * 100.0 if effect is not None else None,
        "paired_n": len(pairs),
        "n_eval": len(units),
        "complete": complete,
        "mcnemar_exact_two_sided": {
            "a_only": discordant_a_only,
            "b_only": discordant_b_only,
            "discordant": discordant,
            "p_value": mcnemar_p,
        },
        "clustered_bootstrap_95_ci": (
            _cluster_bootstrap(pairs, repetitions=repetitions, seed=seed) if complete else None
        ),
    }


def _preference_summary(
    contrast: str, units: dict[str, dict], pairs: dict[tuple[str, str], dict],
    rows: dict[tuple[str, str], dict], *, repetitions: int, seed: int,
) -> dict:
    arm_a, arm_b = CONTRASTS[contrast]
    counts = Counter()
    net_values: list[tuple[dict, float]] = []
    safe_better = 0
    major_a_regression = 0
    label_sources = Counter()
    observed = 0
    for unit_id, unit in units.items():
        pair = pairs.get((unit_id, contrast))
        if pair is None:
            continue
        observed += 1
        label_sources[str(pair.get("label_source") or "unknown")] += 1
        winner = pair["winner"]
        if winner in {"tie", "both_bad"}:
            outcome = winner
        else:
            winning_arm = pair[f"{winner}_arm"]
            outcome = "win" if winning_arm == arm_a else "loss"
        counts[outcome] += 1
        net = 1.0 if outcome == "win" else -1.0 if outcome == "loss" else 0.0
        net_values.append((unit, net))
        pair_safety = _safety_from_label(pair)
        if (
            outcome == "win"
            and pair_safety is False
            and not _new_violation(rows.get((unit_id, arm_a), {}))
        ):
            safe_better += 1
        if outcome == "loss" and str(pair.get("severity", "")).lower() == "major":
            major_a_regression += 1
    n_eval = len(units)
    complete = observed == n_eval and bool(units)
    net_preference = (counts["win"] - counts["loss"]) / n_eval if complete else None
    decisive = counts["win"] + counts["loss"]
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "observed": observed,
        "n_eval": n_eval,
        "complete": complete,
        "raw": {
            "win": counts["win"], "tie": counts["tie"],
            "loss": counts["loss"], "both_bad": counts["both_bad"],
        },
        "win_rate_all": _ratio(counts["win"], n_eval, complete=complete, observed=observed),
        "loss_rate": _ratio(counts["loss"], n_eval, complete=complete, observed=observed),
        "tie_rate": _ratio(counts["tie"], n_eval, complete=complete, observed=observed),
        "both_bad_rate": _ratio(counts["both_bad"], n_eval, complete=complete, observed=observed),
        "net_preference": net_preference,
        "safe_better_rate": _ratio(safe_better, n_eval, complete=complete, observed=observed),
        "wins_among_decisive": _ratio(counts["win"], decisive),
        "major_arm_a_regressions": major_a_regression,
        "label_sources": dict(label_sources),
        "net_preference_clustered_bootstrap_95_ci": (
            _cluster_bootstrap(net_values, repetitions=repetitions, seed=seed)
            if complete else None
        ),
    }


def _automatic_contrasts(
    units: dict[str, dict], rows: dict[tuple[str, str], dict], *,
    repetitions: int, seed: int,
) -> dict:
    output: dict[str, dict] = {}
    for contrast_index, (contrast, (arm_a, arm_b)) in enumerate(CONTRASTS.items()):
        names: set[str] = set()
        for unit_id in units:
            for arm in (arm_a, arm_b):
                values = rows.get((unit_id, arm), {}).get("automatic_metrics")
                if isinstance(values, dict):
                    names.update(_flatten_numeric_metrics(values))
        metrics: dict[str, dict] = {}
        for metric_index, name in enumerate(sorted(names)):
            values: list[tuple[dict, float]] = []
            a_values: list[float] = []
            b_values: list[float] = []
            for unit_id, unit in units.items():
                row_a = _flatten_numeric_metrics(
                    rows.get((unit_id, arm_a), {}).get("automatic_metrics") or {}
                )
                row_b = _flatten_numeric_metrics(
                    rows.get((unit_id, arm_b), {}).get("automatic_metrics") or {}
                )
                value_a, value_b = _finite_number(row_a.get(name)), _finite_number(row_b.get(name))
                if value_a is not None and value_b is not None:
                    values.append((unit, value_a - value_b))
                    a_values.append(value_a)
                    b_values.append(value_b)
            complete = len(values) == len(units) and bool(units)
            mean_a = sum(a_values) / len(a_values) if a_values else None
            mean_b = sum(b_values) / len(b_values) if b_values else None
            metrics[name] = {
                "paired_n": len(values),
                "n_eval": len(units),
                "complete": complete,
                "mean_arm_a": mean_a,
                "mean_arm_b": mean_b,
                "mean_difference_a_minus_b": (
                    sum(value for _, value in values) / len(values) if values else None
                ),
                "error_reduction_percent": (
                    (mean_b - mean_a) / mean_b * 100.0
                    if mean_a is not None and mean_b not in (None, 0.0) else None
                ),
                "clustered_bootstrap_95_ci": (
                    _cluster_bootstrap(
                        values, repetitions=repetitions,
                        seed=seed + contrast_index * 1009 + metric_index,
                    ) if complete else None
                ),
            }
        output[contrast] = metrics
    return output


def _b2_funnel(units: dict[str, dict], rows: dict[tuple[str, str], dict]) -> dict:
    counts = Counter()
    gate_reasons: Counter = Counter()
    adopted_blocks: Counter = Counter()
    for unit_id in units:
        row = rows.get((unit_id, ARMS[2]), {})
        counts["eligible"] += bool(row.get("eligible"))
        counts["attempted"] += bool(row.get("attempted"))
        counts["geometry_changed"] += bool(row.get("geometry_changed"))
        if _truthy_failure(row.get("timeout")):
            counts["timeout"] += 1
            continue
        if _truthy_failure(row.get("error")):
            counts["error"] += 1
            continue
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        blocks = row.get("adopted_blocks") or row.get("limbs") or []
        if isinstance(blocks, list):
            for block in blocks:
                adopted_blocks[str(block)] += 1
            counts["adopted_block_total"] += len(blocks)
            counts["units_with_adopted_blocks"] += bool(blocks)
        counts["partial_rollback"] += bool(row.get("partial_rollback"))
        reason = row.get("aggressive_reason") or row.get("reason")
        if reason not in (None, "", "none", "unknown"):
            gate_reasons[str(reason)] += 1
        applied = row.get("mode_applied") or diagnostics.get("mode_applied")
        outcome = str(row.get("outcome") or row.get("refine_outcome") or "").lower()
        if applied == "aggressive" or outcome == "aggressive_applied":
            counts["aggressive_applied"] += 1
        elif applied == "conservative" or "conservative" in outcome:
            counts["conservative_fallback"] += 1
        elif row.get("exact_base") or "base_fallback" in outcome:
            counts["exact_base_fallback"] += 1
        else:
            counts["other_or_unchanged"] += 1
    denominator = len(units)
    return {
        "n_eval": denominator,
        **dict(counts),
        "rates": {
            name: _ratio(int(counts[name]), denominator)
            for name in (
                "eligible", "attempted", "geometry_changed",
                "aggressive_applied", "conservative_fallback",
                "exact_base_fallback", "partial_rollback",
            )
        },
        "adopted_blocks": dict(adopted_blocks),
        "gate_reason_distribution": dict(gate_reasons),
    }


def _unit_cohort_tags(unit: dict) -> set[str]:
    tags: set[str] = {"all"}
    explicit = unit.get("cohorts")
    if isinstance(explicit, list):
        tags.update(str(value) for value in explicit if value not in (None, ""))
    elif isinstance(explicit, dict):
        tags.update(str(name) for name, value in explicit.items() if value is True)
    evidence = unit.get("evidence")
    if isinstance(evidence, dict):
        tags.update(f"evidence:{name}" for name, value in evidence.items() if value is True)
    query_evidence = unit.get("query_evidence")
    if isinstance(query_evidence, dict):
        for name in ("hand_pair", "lower_pair"):
            value = query_evidence.get(name)
            if isinstance(value, dict) and value.get("feature_active") is True:
                tags.add(f"evidence:{name}")
        lap = query_evidence.get("lap_contact")
        if isinstance(lap, dict) and lap.get("active") is True:
            tags.add("evidence:lap_contact")
    for field in (
        "pose_type", "view", "coverage_class", "foreshortening_ambiguity",
        "selected_rank", "search_distance_band", "gap_type",
    ):
        value = unit.get(field)
        if value not in (None, ""):
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            tags.add(f"{field}:{rendered}")
    refinable = unit.get("refinable_limbs") or []
    if isinstance(refinable, list):
        lowered = {str(value).lower() for value in refinable}
        if any("arm" in value or "hand" in value for value in lowered):
            tags.add("block:arm")
        if any("leg" in value or "foot" in value for value in lowered):
            tags.add("block:leg")
        if any("torso" in value or "spine" in value for value in lowered):
            tags.add("block:torso")
    return tags


def _cohort_summaries(
    units: dict[str, dict], rows: dict[tuple[str, str], dict],
    outcomes: dict[str, dict[str, bool | None]], pairs: dict[tuple[str, str], dict],
) -> dict:
    members: dict[str, list[str]] = defaultdict(list)
    for unit_id, unit in units.items():
        tags = _unit_cohort_tags(unit)
        b2 = rows.get((unit_id, ARMS[2]), {})
        mode = str(b2.get("mode_applied") or "base").lower()
        if mode == "aggressive" and b2.get("geometry_changed"):
            tags.add("b2_path:aggressive")
        elif mode == "conservative":
            tags.add("b2_path:conservative_fallback")
        elif b2.get("exact_base") or mode == "base":
            tags.add("b2_path:base_fallback")
        for tag in tags:
            members[tag].append(unit_id)
    result: dict[str, dict] = {}
    for tag, unit_ids in sorted(members.items()):
        arm_values: dict[str, dict] = {}
        for arm in ARMS:
            known = [outcomes[arm][unit_id] for unit_id in unit_ids if outcomes[arm][unit_id] is not None]
            arm_values[arm] = {
                "safe_usable_rate": _ratio(
                    sum(value is True for value in known), len(unit_ids),
                    complete=len(known) == len(unit_ids), observed=len(known),
                ),
                "geometry_changed": _ratio(
                    sum(bool(rows.get((unit_id, arm), {}).get("geometry_changed")) for unit_id in unit_ids),
                    len(unit_ids),
                ),
                "new_violation_rate": _ratio(
                    sum(_new_violation(rows.get((unit_id, arm), {})) for unit_id in unit_ids),
                    len(unit_ids),
                ),
            }
        preferences: dict[str, dict] = {}
        for contrast in CONTRASTS:
            raw = Counter()
            for unit_id in unit_ids:
                pair = pairs.get((unit_id, contrast))
                if pair is None:
                    continue
                winner = pair["winner"]
                if winner in {"tie", "both_bad"}:
                    raw[winner] += 1
                else:
                    arm_a = CONTRASTS[contrast][0]
                    raw["win" if pair[f"{winner}_arm"] == arm_a else "loss"] += 1
            preferences[contrast] = {
                "observed": sum(raw.values()),
                "win": raw["win"], "tie": raw["tie"],
                "loss": raw["loss"], "both_bad": raw["both_bad"],
            }
        result[tag] = {"n": len(unit_ids), "arms": arm_values, "preferences": preferences}
    return result


def _cluster_count(units: Iterable[dict]) -> int:
    return len({_cluster_key(unit) for unit in units})


def _holdout_evidence(units: dict[str, dict], evidence: dict | None) -> dict:
    """Validate that the decision uses a sealed, immutable holdout snapshot."""
    evidence = evidence if isinstance(evidence, dict) else {}
    purpose = str(evidence.get("purpose") or evidence.get("split") or "").lower()
    sealed = bool(evidence.get("sealed") is True or evidence.get("sealed_at"))
    integrity_valid = evidence.get("integrity_valid") is True
    snapshot_keys = (
        "manifest_sha256", "dataset_manifest_sha256", "units_sha256",
        "cut_manifest_sha256", "gt_sha256", "person_gt_sha256",
    )
    snapshot_hashes = {
        key: evidence.get(key) for key in snapshot_keys if evidence.get(key)
    }
    unit_splits = {
        str(unit.get("split")).lower()
        for unit in units.values() if unit.get("split") not in (None, "")
    }
    purpose_holdout = purpose in {"holdout", "sealed_holdout", "d2"}
    units_holdout = bool(unit_splits) and unit_splits <= {
        "holdout", "sealed_holdout", "d2"
    }
    return {
        "provided": bool(evidence),
        "purpose": purpose or None,
        "sealed": sealed,
        "snapshot_hashes": snapshot_hashes,
        "integrity_checks": dict(evidence.get("integrity_checks") or {}),
        "integrity_valid": integrity_valid,
        "unit_splits": sorted(unit_splits),
        "is_sealed_holdout": bool(
            sealed and snapshot_hashes and integrity_valid
            and (purpose_holdout or units_holdout)
        ),
    }


def _labeling_audit(
    independent_labels: list[dict], pair_labels: list[dict],
    assignment_plan: list[dict] | None = None,
    independent_provenance: list[dict] | None = None,
    pair_provenance: list[dict] | None = None,
) -> dict:
    """Measure rater independence and hidden-repeat coverage from raw labels."""
    human_rows: list[tuple[tuple[str, str], dict]] = []
    for index, row in enumerate(independent_labels, 1):
        if str(row.get("label_source") or "").startswith("automatic_"):
            continue
        usability = row.get("overall_usability") or row.get("usefulness")
        if usability not in USABILITY:
            continue
        item_id = row.get("item_id")
        key = ("independent", str(item_id) if item_id not in (None, "") else f"missing:{index}")
        human_rows.append((key, row))
    for index, row in enumerate(pair_labels, 1):
        if str(row.get("label_source") or "").startswith("automatic_"):
            continue
        winner = row.get("winner") or row.get("preference")
        if winner not in WINNERS:
            continue
        pair_id = row.get("pair_id")
        key = ("pair", str(pair_id) if pair_id not in (None, "") else f"missing:{index}")
        human_rows.append((key, row))

    labelers_by_item: dict[tuple[str, str], set[str]] = defaultdict(set)
    distinct_labelers: set[str] = set()
    labels_by_assignment: dict[str, tuple[tuple[str, str], dict]] = {}
    for key, row in human_rows:
        labelers_by_item[key]
        labeler = str(row.get("labeler_id") or "").strip()
        if labeler:
            labelers_by_item[key].add(labeler)
            distinct_labelers.add(labeler)
        assignment_id = row.get("assignment_id")
        if isinstance(assignment_id, str) and assignment_id:
            labels_by_assignment[assignment_id] = (key, row)

    audit_errors: list[str] = []
    plan_by_id: dict[str, dict] = {}
    independent_sources = {
        str(row.get("item_id")): _artifact_id(row)
        for row in (independent_provenance or [])
        if row.get("item_id") not in (None, "")
    }
    pair_sources = {
        str(row.get("pair_id")): [
            str(row.get("left_artifact_id") or row.get("left_arm") or row.get("left_variant")),
            str(row.get("right_artifact_id") or row.get("right_arm") or row.get("right_variant")),
        ]
        for row in (pair_provenance or [])
        if row.get("pair_id") not in (None, "")
    }
    for index, row in enumerate(assignment_plan or [], 1):
        assignment_id = row.get("assignment_id")
        if not isinstance(assignment_id, str) or not assignment_id:
            audit_errors.append(f"assignment row {index}: assignment_id is required")
            continue
        if assignment_id in plan_by_id:
            audit_errors.append(f"assignment row {index}: duplicate assignment_id")
            continue
        task_type = row.get("task_type")
        source_id = str(row.get("source_id") or "")
        expected_identity = (
            independent_sources.get(source_id) if task_type == "independent"
            else pair_sources.get(source_id) if task_type == "pair"
            else None
        )
        if expected_identity is None:
            audit_errors.append(
                f"assignment row {index}: unknown task source {task_type}/{source_id}"
            )
        elif row.get("artifact_identity") != expected_identity:
            audit_errors.append(
                f"assignment row {index}: artifact identity does not match provenance"
            )
        plan_by_id[assignment_id] = row
    for key, row in human_rows:
        assignment_id = row.get("assignment_id")
        if not isinstance(assignment_id, str) or assignment_id not in plan_by_id:
            audit_errors.append(
                f"human label {key[0]}/{key[1]} lacks a known assignment_id"
            )
    duplicate_sources: set[tuple[str, str]] = set()
    valid_hidden_repeats = 0
    hidden_repeat_matches = 0
    for assignment_id, plan in plan_by_id.items():
        assigned = labels_by_assignment.get(assignment_id)
        if assigned is None:
            continue
        key, label = assigned
        kind = plan.get("assignment_kind")
        artifact_identity = plan.get("artifact_identity")
        same_source = str(plan.get("source_id")) == key[1]
        if kind == "duplicate" and same_source:
            for other_id, other_plan in plan_by_id.items():
                other = labels_by_assignment.get(other_id)
                if other_id == assignment_id or other is None:
                    continue
                other_key, other_label = other
                if (
                    other_key == key
                    and other_plan.get("artifact_identity") == artifact_identity
                    and str(other_label.get("labeler_id") or "").strip()
                    != str(label.get("labeler_id") or "").strip()
                ):
                    duplicate_sources.add(key)
                    break
        if kind == "hidden_repeat" and same_source:
            original_id = plan.get("repeat_of_assignment_id")
            original_plan = plan_by_id.get(str(original_id))
            original = labels_by_assignment.get(str(original_id))
            if original_plan is None or original is None or original_id == assignment_id:
                continue
            original_key, original_label = original
            same_identity = (
                original_key == key
                and original_plan.get("artifact_identity") == artifact_identity
            )
            same_labeler = (
                str(original_label.get("labeler_id") or "").strip()
                == str(label.get("labeler_id") or "").strip()
                != ""
            )
            if same_identity and same_labeler:
                valid_hidden_repeats += 1
                judgment_fields = (
                    ("overall_usability", "safety_violation")
                    if key[0] == "independent" else
                    ("winner", "severity", "body_part", "safety_violation")
                )
                hidden_repeat_matches += all(
                    label.get(field) == original_label.get(field)
                    for field in judgment_fields
                )

    item_count = len(labelers_by_item)
    duplicated_items = len(duplicate_sources)
    raw_count = len(human_rows)
    return {
        "complete": bool(assignment_plan) and not audit_errors,
        "errors": audit_errors,
        "human_label_rows": raw_count,
        "distinct_labelers": len(distinct_labelers),
        "distinct_labeler_ids": sorted(distinct_labelers),
        "human_labeled_items": item_count,
        "items_with_two_or_more_independent_raters": duplicated_items,
        "duplicate_fraction": duplicated_items / item_count if item_count else 0.0,
        "hidden_repeat_rows": valid_hidden_repeats,
        "hidden_repeat_fraction": valid_hidden_repeats / raw_count if raw_count else 0.0,
        "hidden_repeat_agreement": _ratio(
            hidden_repeat_matches, valid_hidden_repeats
        ),
        "assignment_plan_rows": len(assignment_plan or []),
    }


def _mesh_safety_evidence(
    units: dict[str, dict], arms: dict[tuple[str, str], dict],
    evidence_rows: list[dict] | None,
) -> tuple[dict, list[str]]:
    """Require independent, versioned mesh evidence for every evaluable arm."""
    expected = [
        {
            "unit_id": unit_id,
            "arm": arm,
            "artifact_id": str(_artifact_id(row) or ""),
            "geometry_sha256": str(row.get("geometry_sha256") or ""),
        }
        for (unit_id, arm), row in sorted(arms.items())
        if unit_id in units and not _delivery_failure(row)
    ]
    validation = validate_mesh_evidence_bundle(
        evidence_rows or [], expected_rows=expected,
        require_complete=True, allow_placeholders=False,
    )
    errors = list(validation["errors"])
    complete = bool(expected) and validation["valid"]
    return {
        "evaluable_arm_rows": len(expected),
        "mesh_evidence_rows": validation["row_count"],
        "mesh_arm_rows": validation["row_count"],
        "mesh_safety_available": complete,
        "mesh_safety_complete": complete,
        "proxy_only": False,
        "check_contract_version": validation["check_contract_version"],
        "required_checks": validation["required_checks"],
        "versions": validation["versions"],
        "evidence_errors": errors,
    }, errors


def _worst_slice_regression(
    report: dict, minimum_n: int, required_cohorts: Iterable[str],
) -> dict:
    """Find the worst pre-fixed cohort's B2−B1 SUR difference."""
    required = [str(name) for name in required_cohorts if str(name).strip()]
    missing: list[str] = []
    candidates: list[dict] = []
    for name in required:
        cohort = report.get("cohorts", {}).get(name)
        if not isinstance(cohort, dict):
            missing.append(name)
            continue
        if int(cohort.get("n", 0)) < minimum_n:
            missing.append(name)
            continue
        b2 = cohort.get("arms", {}).get(ARMS[2], {}).get("safe_usable_rate", {})
        b1 = cohort.get("arms", {}).get(ARMS[1], {}).get("safe_usable_rate", {})
        if b2.get("rate") is None or b1.get("rate") is None:
            continue
        candidates.append({
            "cohort": name,
            "n": int(cohort["n"]),
            "difference": float(b2["rate"]) - float(b1["rate"]),
        })
    if not required or missing or not candidates:
        return {
            "available": False, "minimum_n": minimum_n,
            "required_cohorts": required, "missing_or_too_small": missing,
            "cohorts_tested": len(candidates),
        }
    worst = min(candidates, key=lambda item: (item["difference"], item["cohort"]))
    return {
        "available": True,
        "minimum_n": minimum_n,
        "required_cohorts": required,
        "missing_or_too_small": [],
        "cohorts_tested": len(candidates),
        "cohort": worst["cohort"],
        "n": worst["n"],
        "difference": worst["difference"],
        "difference_percentage_points": worst["difference"] * 100.0,
    }


def _criteria_status(report: dict, criteria: dict | None) -> tuple[str, list[str], list[dict]]:
    reasons: list[str] = []
    checks: list[dict] = []
    if report["validation"]["errors"]:
        reasons.append("validation errors must be resolved")
    if not report["labels"]["complete"]:
        reasons.append("human labels are incomplete or conflicted")
    if not report.get("labeling_audit", {}).get("complete"):
        reasons.append("sealed label assignment plan is missing or invalid")
    required = {
        "primary_mcid", "primary_ci_low_min", "major_worse_max",
        "changed_major_worse_vs_b0_max", "worst_slice_regression_max",
        "worst_slice_min_n", "minimum_n_eval", "minimum_clusters",
        "worst_slice_cohorts", "usability_rubric_version",
        "human_usable_categories",
        "minimum_distinct_labelers", "minimum_duplicate_fraction",
        "minimum_hidden_repeat_fraction",
        "analysis_seed", "bootstrap_repetitions", "report_version",
        "new_violation_rate_max", "exact_fallback_rate_min",
        "p95_latency_ms_max", "timeout_error_rate_max",
    }
    if criteria is None:
        reasons.append("pre-registered promotion criteria were not provided")
        return "INCONCLUSIVE", reasons, checks
    missing = sorted(required - set(criteria))
    if missing:
        reasons.append("missing promotion criteria: " + ", ".join(missing))

    expected_rubric_version = criteria.get("usability_rubric_version")
    expected_usable = set(criteria.get("human_usable_categories") or [])
    actual_rubric = report.get("usability_rubric", {})
    rubric_matches = bool(
        expected_rubric_version
        and expected_rubric_version == actual_rubric.get("version")
        and expected_usable == set(actual_rubric.get("human_usable_categories") or [])
        and actual_rubric.get("valid") is True
    )
    checks.append({
        "name": "usability_rubric_snapshot",
        "status": "pass" if rubric_matches else "inconclusive",
        "expected_version": expected_rubric_version,
        "actual_version": actual_rubric.get("version"),
    })
    if not rubric_matches:
        reasons.append("applied usability rubric does not match pre-registered criteria")

    analysis = report.get("analysis", {})
    expected_seed = criteria.get("analysis_seed")
    actual_seed = analysis.get("seed")
    seed_matches = expected_seed is not None and actual_seed == int(expected_seed)
    checks.append({
        "name": "analysis_seed",
        "status": "pass" if seed_matches else "inconclusive",
        "value": actual_seed,
        "expected": expected_seed,
    })
    if not seed_matches:
        reasons.append("analysis seed does not match pre-registered criteria")
    expected_repetitions = criteria.get("bootstrap_repetitions")
    actual_repetitions = analysis.get("bootstrap_repetitions")
    repetitions_match = bool(
        expected_repetitions is not None
        and actual_repetitions == int(expected_repetitions)
        and int(expected_repetitions) >= 10_000
    )
    checks.append({
        "name": "bootstrap_repetitions",
        "status": "pass" if repetitions_match else "inconclusive",
        "value": actual_repetitions,
        "expected": expected_repetitions,
        "minimum_required": 10_000,
    })
    if not repetitions_match:
        reasons.append(
            "bootstrap repetitions must match pre-registered criteria and be at least 10000"
        )
    expected_report_version = criteria.get("report_version")
    report_version_matches = (
        expected_report_version == REPORT_VERSION
        and report.get("report_version") == REPORT_VERSION
    )
    checks.append({
        "name": "report_version",
        "status": "pass" if report_version_matches else "inconclusive",
        "value": report.get("report_version"),
        "expected": expected_report_version,
    })
    if not report_version_matches:
        reasons.append("report version does not match pre-registered criteria")

    labeling_audit = report.get("labeling_audit", {})
    for name, value, cast in (
        (
            "minimum_distinct_labelers",
            labeling_audit.get("distinct_labelers", 0), int,
        ),
        (
            "minimum_duplicate_fraction",
            labeling_audit.get("duplicate_fraction", 0.0), float,
        ),
        (
            "minimum_hidden_repeat_fraction",
            labeling_audit.get("hidden_repeat_fraction", 0.0), float,
        ),
    ):
        if name not in criteria:
            continue
        hard_floor = {
            "minimum_distinct_labelers": 2,
            "minimum_duplicate_fraction": 0.15,
            "minimum_hidden_repeat_fraction": 0.05,
        }[name]
        threshold = max(cast(criteria[name]), cast(hard_floor))
        passed = value >= threshold
        checks.append({
            "name": name,
            "status": "pass" if passed else "inconclusive",
            "value": value,
            "threshold": threshold,
        })
        if not passed:
            reasons.append(
                f"{name} audit value {value} is below pre-registered minimum {threshold}"
            )

    n_eval = int(report.get("n_eval", 0))
    clusters = int(report.get("sample", {}).get("clusters", 0))
    if "minimum_n_eval" in criteria:
        threshold = int(criteria["minimum_n_eval"])
        passed = n_eval >= threshold
        checks.append({"name": "minimum_n_eval", "status": "pass" if passed else "inconclusive",
                       "value": n_eval, "threshold": threshold})
        if not passed:
            reasons.append(f"n_eval {n_eval} is below pre-registered minimum {threshold}")
    if "minimum_clusters" in criteria:
        threshold = int(criteria["minimum_clusters"])
        passed = clusters >= threshold
        checks.append({"name": "minimum_clusters", "status": "pass" if passed else "inconclusive",
                       "value": clusters, "threshold": threshold})
        if not passed:
            reasons.append(f"cluster count {clusters} is below pre-registered minimum {threshold}")
    holdout = report.get("holdout", {})
    if not holdout.get("is_sealed_holdout"):
        checks.append({"name": "sealed_holdout", "status": "inconclusive", "value": holdout})
        reasons.append("sealed holdout purpose and immutable snapshot evidence are required")
    else:
        checks.append({"name": "sealed_holdout", "status": "pass"})

    primary = report["sur_contrasts"]["B2_vs_B1"]
    ci = primary.get("clustered_bootstrap_95_ci") or {}
    effect, low, high = primary.get("difference"), ci.get("low"), ci.get("high")
    if effect is None or low is None or high is None:
        reasons.append("primary paired SUR effect or clustered CI is unavailable")
    elif "primary_mcid" in criteria and "primary_ci_low_min" in criteria:
        effect_min = float(criteria["primary_mcid"])
        low_min = float(criteria["primary_ci_low_min"])
        if effect >= effect_min and low >= low_min:
            checks.append({"name": "primary_SUR", "status": "pass", "value": effect})
        elif high < effect_min or high < low_min:
            checks.append({"name": "primary_SUR", "status": "fail", "value": effect})
        else:
            checks.append({"name": "primary_SUR", "status": "inconclusive", "value": effect})
            reasons.append("primary SUR CI crosses a pre-registered decision boundary")

    preference = report["preferences"]["B2_vs_B1"]
    if "major_worse_max" in criteria and preference["complete"]:
        allowed = min(0, int(criteria["major_worse_max"]))
        status = "pass" if preference["major_arm_a_regressions"] <= allowed else "fail"
        checks.append({"name": "major_B2_vs_B1_regression", "status": status,
                       "value": preference["major_arm_a_regressions"],
                       "maximum_allowed": allowed})

    changed_major = report["human_guardrails"]["B2_changed_major_worse_vs_B0"]
    if "changed_major_worse_vs_b0_max" in criteria:
        if changed_major["complete"]:
            allowed = min(0, int(criteria["changed_major_worse_vs_b0_max"]))
            passed = changed_major["numerator"] <= allowed
            checks.append({
                "name": "B2_changed_major_worse_vs_B0",
                "status": "pass" if passed else "fail",
                "value": changed_major["numerator"],
                "denominator": changed_major["denominator"],
                "maximum_allowed": allowed,
            })
        else:
            checks.append({"name": "B2_changed_major_worse_vs_B0", "status": "inconclusive"})
            reasons.append("B2 changed major-worse vs B0 evidence is incomplete")

    if "worst_slice_min_n" in criteria and "worst_slice_regression_max" in criteria:
        slice_result = report["worst_slice"]
        if not slice_result["available"]:
            checks.append({"name": "worst_slice_regression", "status": "inconclusive"})
            reasons.append("no complete pre-fixed cohort meets worst_slice_min_n")
        else:
            maximum_regression = max(0.0, float(criteria["worst_slice_regression_max"]))
            floor = -maximum_regression
            passed = slice_result["difference"] >= floor
            checks.append({
                "name": "worst_slice_regression",
                "status": "pass" if passed else "fail",
                "value": slice_result["difference"],
                "cohort": slice_result["cohort"],
                "minimum_allowed": floor,
            })

    b2 = report["arms"][ARMS[2]]
    for name, value, threshold, operator in (
        ("new_violation_rate", b2["new_violation_rate"]["rate"],
         min(0.0, float(criteria.get("new_violation_rate_max", 0.0))), "max"),
        ("exact_fallback_rate", b2["exact_fallback_rate"]["rate"],
         max(1.0, float(criteria.get("exact_fallback_rate_min", 1.0))), "min"),
        ("p95_latency_ms", b2["latency"]["p95_ms"],
         criteria.get("p95_latency_ms_max"), "max"),
        ("timeout_error_rate", b2["latency"]["timeout_or_error_rate"]["rate"],
         criteria.get("timeout_error_rate_max"), "max"),
    ):
        if threshold is None:
            continue
        if name == "exact_fallback_rate" and value is None and b2["fallback_required"]["denominator"] == 0:
            checks.append({"name": name, "status": "pass", "value": None,
                           "note": "no fallback was required"})
        elif value is None:
            checks.append({"name": name, "status": "inconclusive", "value": None})
            reasons.append(f"{name} is unavailable")
        else:
            passed = value <= float(threshold) if operator == "max" else value >= float(threshold)
            checks.append({"name": name, "status": "pass" if passed else "fail", "value": value})

    absolute_better = report["preferences"]["B2_vs_B0"]
    better_required = max(1, int(criteria.get("b2_vs_b0_better_min", 1)))
    if absolute_better["complete"]:
        passed = absolute_better["raw"]["win"] >= better_required
        checks.append({
            "name": "B2_vs_B0_better",
            "status": "pass" if passed else "fail",
            "value": absolute_better["raw"]["win"],
            "minimum_required": better_required,
        })
    else:
        checks.append({"name": "B2_vs_B0_better", "status": "inconclusive"})
        reasons.append("B2 vs B0 better evidence is incomplete")

    if reasons or missing or any(check["status"] == "inconclusive" for check in checks):
        return "INCONCLUSIVE", reasons, checks
    if any(check["status"] == "fail" for check in checks):
        return "FAIL", reasons, checks
    return "PASS", reasons, checks


def compute_refine_report(
    units: list[dict], arm_rows: list[dict], independent_labels: list[dict] | None = None,
    pair_labels: list[dict] | None = None, pair_provenance: list[dict] | None = None,
    *, bootstrap_repetitions: int = 2000, seed: int = 20260805,
    promotion_criteria: dict | None = None,
    independent_provenance: list[dict] | None = None,
    holdout_evidence: dict | None = None,
    usability_rubric: dict | None = None,
    cache_policy: dict | None = None,
    mesh_evidence: list[dict] | None = None,
    label_assignments: list[dict] | None = None,
) -> dict:
    """Aggregate the frozen B0/B1/B2 evaluation without changing its denominator.

    Missing or conflicting human labels leave the affected rate as ``None`` and
    force the promotion status to ``INCONCLUSIVE``.
    """
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    if not isinstance(usability_rubric, dict):
        usability_rubric = {
            "version": (promotion_criteria or {}).get("usability_rubric_version"),
            "human_usable_categories": (
                (promotion_criteria or {}).get("human_usable_categories") or []
            ),
        }
    rubric_version = usability_rubric.get("version") or usability_rubric.get("rubric_version")
    usable_categories = set(usability_rubric.get("human_usable_categories") or [])
    rubric_valid = bool(
        rubric_version
        and usable_categories
        and usable_categories <= USABILITY
        and "unusable" not in usable_categories
    )
    units_by_id, rows, shape_errors = _normalize_inputs(units, arm_rows)
    judgments, independent_info, independent_errors = _resolve_independent_labels(
        independent_labels or [], units_by_id, rows, independent_provenance,
        usable_categories,
    )
    independent_info["rubric_applicable"] = rubric_valid
    if not rubric_valid:
        judgments = {}
    pair_results, pair_info, pair_errors = _resolve_pair_labels(
        pair_labels or [], pair_provenance or [], units_by_id, rows
    )
    mesh_validation, mesh_errors = _mesh_safety_evidence(
        units_by_id, rows, mesh_evidence
    )
    # The independent mesh evaluator is authoritative for absolute safety.
    # Copy only its frozen violations into the working rows so every existing
    # SUR and guardrail calculation sees them.  For refine arms, a violation is
    # "new" only when it was not already present on that unit's B0 evidence.
    mesh_by_key = {
        (_unit_id(row), _arm(row.get("arm"))): row
        for row in (mesh_evidence or [])
        if _unit_id(row) is not None and _arm(row.get("arm")) is not None
    }
    for unit_id in units_by_id:
        for arm in ARMS:
            arm_row = rows.get((unit_id, arm))
            evidence = mesh_by_key.get((unit_id, arm))
            if arm_row is None or evidence is None:
                continue
            common_hard = list(arm_row.get("hard_safety_violations") or [])
            common_new = list(arm_row.get("new_hard_safety_violations") or [])
            mesh_hard = list(evidence.get("hard_violations") or [])
            mesh_new = list(evidence.get("new_hard_violations") or [])
            arm_row["hard_safety_violations"] = list(dict.fromkeys(
                [*common_hard, *mesh_hard]
            ))
            arm_row["new_hard_safety_violations"] = list(dict.fromkeys(
                [*common_new, *mesh_new]
            ))
    validation_errors = (
        shape_errors + independent_errors + pair_errors + mesh_errors
    )

    arm_summaries: dict[str, dict] = {}
    outcomes: dict[str, dict[str, bool | None]] = {}
    for arm in ARMS:
        arm_summaries[arm], outcomes[arm] = _arm_summary(
            arm, units_by_id, rows, judgments
        )
    sur_contrasts = {
        contrast: _paired_effect(
            arm_a, arm_b, units_by_id, outcomes,
            repetitions=bootstrap_repetitions, seed=seed + index,
        )
        for index, (contrast, (arm_a, arm_b)) in enumerate(CONTRASTS.items())
    }
    preferences = {
        contrast: _preference_summary(
            contrast, units_by_id, pair_results, rows,
            repetitions=bootstrap_repetitions, seed=seed + 101 + index,
        )
        for index, contrast in enumerate(CONTRASTS)
    }
    labels_complete = bool(units_by_id) and rubric_valid and not independent_errors and not pair_errors and (
        independent_info["resolved_arm_judgments"] == independent_info["expected_arm_judgments"]
        and pair_info["resolved_pair_judgments"] == pair_info["expected_pair_judgments"]
    )
    ownership_complete = bool(rows) and all(
        bool(row.get("ownership_validated"))
        for (unit_id, arm), row in rows.items()
        if arm != ARMS[0]
    )
    evaluable_rows = [
        rows[(unit_id, arm)]
        for unit_id in units_by_id for arm in ARMS
        if (unit_id, arm) in rows
        and not _delivery_failure(rows[(unit_id, arm)])
    ]
    cache_policy = cache_policy if isinstance(cache_policy, dict) else {}
    cache_off_declared = bool(
        cache_policy.get("expected_cache_hit") is False
        and cache_policy.get("latency_basis") == "cache_off_post_click"
    )
    cache_rows = [
        row for (unit_id, arm), row in rows.items()
        if arm != ARMS[0]
        and row.get("endpoint_called") is True
        and not _truthy_failure(row.get("timeout"))
        and not _truthy_failure(row.get("error"))
    ]
    not_called_rows = [
        row for (unit_id, arm), row in rows.items()
        if arm != ARMS[0] and row.get("endpoint_called") is not True
    ]
    cache_hits = [row for row in cache_rows if row.get("cache_hit") is True]
    cache_misses = [row for row in cache_rows if row.get("cache_hit") is False]
    cache_policy_complete = bool(cache_rows) and cache_off_declared and (
        len(cache_misses) == len(cache_rows)
    )
    for arm in (ARMS[1], ARMS[2]):
        arm_summaries[arm]["latency"]["basis"] = (
            "cache_off_post_click" if cache_policy_complete else "mixed_or_unverified"
        )
    mesh_safety_complete = mesh_validation["mesh_safety_complete"]
    cohort_summaries = _cohort_summaries(
        units_by_id, rows, outcomes, pair_results
    )
    changed_unit_ids = {
        unit_id for unit_id in units_by_id
        if bool(rows.get((unit_id, ARMS[2]), {}).get("geometry_changed"))
    }
    b2_b0_pairs = [
        pair_results.get((unit_id, "B2_vs_B0")) for unit_id in sorted(changed_unit_ids)
    ]
    changed_major_worse = sum(
        bool(
            pair
            and pair.get("winner") in {"left", "right"}
            and pair.get(f"{pair['winner']}_arm") == ARMS[0]
            and pair.get("severity") == "major"
        )
        for pair in b2_b0_pairs
    )
    changed_major_complete = all(pair is not None for pair in b2_b0_pairs)
    minimum_slice_n = int((promotion_criteria or {}).get("worst_slice_min_n", 0) or 0)
    required_cohorts = (promotion_criteria or {}).get("worst_slice_cohorts") or []
    report = {
        "schema_version": 1,
        "report_version": REPORT_VERSION,
        "metric_contract": "REFINE_V2_DESIGN.md#10",
        "status": "INCONCLUSIVE",
        "n_eval": len(units_by_id),
        "sample": {
            "n_eval": len(units_by_id),
            "clusters": _cluster_count(units_by_id.values()),
        },
        "analysis": {
            "seed": seed,
            "bootstrap_repetitions": bootstrap_repetitions,
        },
        "holdout": _holdout_evidence(units_by_id, holdout_evidence),
        "validation": {"complete": not validation_errors, "errors": validation_errors},
        "usability_rubric": {
            "version": rubric_version,
            "human_usable_categories": sorted(usable_categories),
            "valid": rubric_valid,
        },
        "cache_validation": {
            "policy": cache_policy,
            "cache_off_declared": cache_off_declared,
            "evaluable_refine_arm_rows": len(cache_rows),
            "endpoint_called_evaluable_rows": len(cache_rows),
            "not_called_not_applicable_rows": len(not_called_rows),
            "cache_hit_arm_rows": len(cache_hits),
            "cache_miss_arm_rows": len(cache_misses),
            "cache_off_complete": cache_policy_complete,
        },
        "safety_validation": {
            "ownership_complete": ownership_complete,
            **mesh_validation,
        },
        "labels": {
            "complete": labels_complete,
            "independent": independent_info,
            "pairs": pair_info,
        },
        "labeling_audit": _labeling_audit(
            independent_labels or [], pair_labels or [], label_assignments,
            independent_provenance, pair_provenance,
        ),
        "arms": arm_summaries,
        "sur_contrasts": sur_contrasts,
        "preferences": preferences,
        "automatic_metric_contrasts": _automatic_contrasts(
            units_by_id, rows, repetitions=bootstrap_repetitions, seed=seed + 1000
        ),
        "b2_funnel": _b2_funnel(units_by_id, rows),
        "cohorts": cohort_summaries,
        "human_guardrails": {
            "B2_changed_major_worse_vs_B0": _ratio(
                changed_major_worse, len(changed_unit_ids),
                complete=changed_major_complete,
                observed=sum(pair is not None for pair in b2_b0_pairs),
            ),
        },
        "promotion": {
            "confirmatory_primary": "B2_vs_B1",
            "criteria": promotion_criteria,
            "checks": [],
            "inconclusive_reasons": [],
        },
    }
    report["worst_slice"] = _worst_slice_regression(
        report, minimum_slice_n, required_cohorts
    )
    status, reasons, checks = _criteria_status(report, promotion_criteria)
    if not ownership_complete:
        status = "INCONCLUSIVE"
        reasons.append("person ownership was not independently validated for every refine arm")
    if not mesh_safety_complete:
        status = "INCONCLUSIVE"
        reasons.append(
            "complete mesh safety coverage is required for every evaluable arm; proxy-only or partial coverage cannot promote"
        )
    if not rubric_valid:
        status = "INCONCLUSIVE"
        reasons.append(
            "a versioned pre-registered usability rubric with explicit human_usable_categories is required"
        )
    if not cache_policy_complete:
        status = "INCONCLUSIVE"
        reasons.append(
            "cache-off policy must be declared and every refine arm must be an observed cache miss"
        )
    report["status"] = status
    report["promotion"]["checks"] = checks
    report["promotion"]["inconclusive_reasons"] = reasons
    return report


def _format_ratio(value: dict) -> str:
    if value.get("rate") is None:
        return f"{value.get('numerator', 0)}/{value.get('denominator', 0)} (INCOMPLETE)"
    return f"{value['numerator']}/{value['denominator']} ({value['rate'] * 100:.1f}%)"


def render_refine_markdown(report: dict) -> str:
    """Render the minimum decision table required by the design document."""
    arms = report["arms"]
    lines = [
        "# Refine B0/B1/B2 evaluation",
        "",
        f"- status: **{report['status']}**",
        f"- ITT denominator: `{report['n_eval']}` frozen units",
        "- confirmatory primary: `B2_vs_B1`",
        "",
        "## Headline",
        "",
    ]
    primary = report["sur_contrasts"]["B2_vs_B1"]
    ci = primary.get("clustered_bootstrap_95_ci") or {}
    diff = primary.get("difference_percentage_points")
    if diff is None:
        lines.append("B2−B1 SUR: INCOMPLETE")
    else:
        ci_text = "CI unavailable" if ci.get("low") is None else (
            f"95% clustered CI [{ci['low'] * 100:.1f}, {ci['high'] * 100:.1f}]%p"
        )
        lines.append(f"B2−B1 SUR: {diff:+.1f}%p; {ci_text}")
    preference = report["preferences"]["B2_vs_B1"]
    raw = preference["raw"]
    lines.append(
        "B2 vs B1 direct W/T/L/BB: "
        f"{raw['win']}/{raw['tie']}/{raw['loss']}/{raw['both_bad']}"
    )
    lines.extend([
        "",
        "## Three-arm ITT table",
        "",
        "| Arm | SUR x/n | Δ vs B0 | W/T/L/BB vs B0 | 2D joint NME | new violation | changed | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for index, arm in enumerate(ARMS):
        summary = arms[arm]
        if index == 0:
            delta = "baseline"
            pref = "baseline"
        else:
            contrast = "B1_vs_B0" if index == 1 else "B2_vs_B0"
            effect = report["sur_contrasts"][contrast]["difference_percentage_points"]
            delta = "INCOMPLETE" if effect is None else f"{effect:+.1f}%p"
            raw_vs_base = report["preferences"][contrast]["raw"]
            pref = f"{raw_vs_base['win']}/{raw_vs_base['tie']}/{raw_vs_base['loss']}/{raw_vs_base['both_bad']}"
        nme = summary["automatic_metrics"].get("joint_nme", {}).get("mean")
        if nme is None:
            nme = summary["automatic_metrics"].get("metrics.joint_nme", {}).get("mean")
        nme_text = "n/a" if nme is None else f"{nme:.4f}"
        p95 = summary["latency"]["p95_ms"]
        p95_text = "immediate base" if index == 0 else ("n/a" if p95 is None else f"{p95:.1f} ms")
        lines.append(
            f"| {arm} | {_format_ratio(summary['safe_usable_rate'])} | {delta} | {pref} | "
            f"{nme_text} | {_format_ratio(summary['new_violation_rate'])} | "
            f"{_format_ratio(summary['geometry_changed'])} | {p95_text} |"
        )
    if report["validation"]["errors"] or report["promotion"]["inconclusive_reasons"]:
        lines.extend(["", "## Incomplete / validation", ""])
        for reason in report["validation"]["errors"] + report["promotion"]["inconclusive_reasons"]:
            lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def _force_inconclusive(report: dict, reason: str) -> None:
    report["status"] = "INCONCLUSIVE"
    reasons = report.setdefault("promotion", {}).setdefault(
        "inconclusive_reasons", []
    )
    if reason not in reasons:
        reasons.append(reason)


def _criteria_snapshot(
    run_dir: Path, manifest: dict, frozen_manifest: dict,
    override: dict | str | Path | None,
) -> tuple[dict | None, dict]:
    """Load only the pre-server frozen criteria and prove its hash lineage."""
    frozen_file = run_dir / "promotion_criteria.frozen.json"
    errors: list[str] = []
    frozen_value: dict | None = None
    actual_sha256: str | None = None
    if not frozen_file.exists():
        errors.append("promotion_criteria.frozen.json is missing")
    else:
        actual_sha256 = sha256_file(frozen_file)
        try:
            candidate = read_json(frozen_file)
            if isinstance(candidate, dict) and candidate:
                frozen_value = candidate
            else:
                errors.append("frozen promotion criteria must be a non-empty object")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"frozen promotion criteria cannot be read: {type(exc).__name__}")

    metadata = manifest.get("promotion_criteria")
    metadata = metadata if isinstance(metadata, dict) else {}
    manifest_preregistered = metadata.get("preregistered") is True
    manifest_frozen_before = metadata.get("frozen_before_server_contact") is True
    declared_path = metadata.get("frozen_path")
    manifest_path_matches = False
    if declared_path not in (None, ""):
        try:
            manifest_path_matches = (
                Path(declared_path).expanduser().resolve() == frozen_file.resolve()
            )
        except (OSError, TypeError, ValueError):
            manifest_path_matches = False
    manifest_hash_matches = bool(
        actual_sha256 and metadata.get("sha256") == actual_sha256
    )
    frozen_manifest_hash_matches = bool(
        actual_sha256
        and frozen_manifest.get("promotion_criteria_sha256") == actual_sha256
    )
    frozen_manifest_preregistered = (
        frozen_manifest.get("promotion_criteria_preregistered") is True
    )

    for passed, message in (
        (manifest_preregistered, "manifest does not mark promotion criteria pre-registered"),
        (
            manifest_frozen_before,
            "manifest does not prove criteria were frozen before server contact",
        ),
        (manifest_path_matches, "manifest frozen criteria path does not match run snapshot"),
        (manifest_hash_matches, "manifest promotion criteria hash mismatch"),
        (frozen_manifest_hash_matches, "frozen_manifest promotion criteria hash mismatch"),
        (
            frozen_manifest_preregistered,
            "frozen_manifest does not mark promotion criteria pre-registered",
        ),
    ):
        if not passed:
            errors.append(message)

    override_provided = override is not None
    override_content_matches = True
    override_hash_matches = True
    override_sha256: str | None = None
    override_content_sha256: str | None = None
    if override_provided:
        override_value: Any = None
        if isinstance(override, (str, Path)):
            override_file = Path(override).expanduser().resolve()
            if not override_file.exists():
                errors.append("caller promotion criteria override is missing")
                override_content_matches = False
                override_hash_matches = False
            else:
                override_sha256 = sha256_file(override_file)
                override_hash_matches = bool(
                    actual_sha256 and override_sha256 == actual_sha256
                )
                try:
                    override_value = read_json(override_file)
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(
                        f"caller promotion criteria override cannot be read: {type(exc).__name__}"
                    )
        else:
            override_value = override
        if isinstance(override_value, dict):
            override_content_sha256 = hash_json(override_value)
            override_content_matches = bool(
                frozen_value is not None and override_value == frozen_value
                and override_content_sha256 == hash_json(frozen_value)
            )
        else:
            override_content_matches = False
        if not override_content_matches:
            errors.append("caller promotion criteria content differs from frozen snapshot")
        if not override_hash_matches:
            errors.append("caller promotion criteria file hash differs from frozen snapshot")

    complete = bool(frozen_value) and not errors
    return frozen_value, {
        "complete": complete,
        "frozen_path": str(frozen_file),
        "actual_sha256": actual_sha256,
        "manifest_sha256": metadata.get("sha256"),
        "frozen_manifest_sha256": frozen_manifest.get(
            "promotion_criteria_sha256"
        ),
        "manifest_preregistered": manifest_preregistered,
        "manifest_frozen_before_server_contact": manifest_frozen_before,
        "manifest_path_matches": manifest_path_matches,
        "manifest_hash_matches": manifest_hash_matches,
        "frozen_manifest_hash_matches": frozen_manifest_hash_matches,
        "frozen_manifest_preregistered": frozen_manifest_preregistered,
        "caller_override_provided": override_provided,
        "caller_override_sha256": override_sha256,
        "caller_override_content_sha256": override_content_sha256,
        "caller_override_content_matches": override_content_matches,
        "caller_override_hash_matches": override_hash_matches,
        "errors": errors,
    }


def _label_artifact_validation(
    run_dir: Path, manifest: dict, arm_rows: list[dict],
    independent_labels: list[dict], independent_provenance: list[dict],
    pair_labels: list[dict], pair_provenance: list[dict],
) -> dict:
    """Bind every human judgment to the exact rendered, blinded artifact."""
    errors: list[str] = []
    arms = {
        (_unit_id(row), _arm(row.get("arm"))): row
        for row in arm_rows
        if _unit_id(row) is not None and _arm(row.get("arm")) is not None
    }
    independent_by_id = {
        str(row.get("item_id")): row
        for row in independent_provenance
        if row.get("item_id") not in (None, "")
    }
    pair_by_id = {
        str(row.get("pair_id")): row
        for row in pair_provenance
        if row.get("pair_id") not in (None, "")
    }
    renderer = manifest.get("renderer")
    expected_renderer = (
        renderer.get("version") if isinstance(renderer, dict) else None
    )
    if not isinstance(expected_renderer, str) or not expected_renderer:
        errors.append("manifest renderer.version is required for human labels")

    checked_files: dict[tuple[str, str], bool] = {}

    def resolved_path(value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else run_dir / candidate).resolve()

    def check_render(
        *, context: str, source_path: Any, source_sha: Any,
        source_version: Any, arm_row: dict,
    ) -> None:
        if arm_row.get("render_error") not in (None, "", False):
            errors.append(f"{context}: arm has render_error")
        arm_path = resolved_path(arm_row.get("render_path"))
        provenance_path = resolved_path(source_path)
        if provenance_path is None or arm_path is None:
            errors.append(f"{context}: render_path is required in provenance and arm row")
        elif provenance_path != arm_path:
            errors.append(f"{context}: provenance render_path does not match arm row")
        if not isinstance(source_sha, str) or not source_sha:
            errors.append(f"{context}: provenance render_sha256 is required")
        if arm_row.get("render_sha256") != source_sha:
            errors.append(f"{context}: provenance render_sha256 does not match arm row")
        if not isinstance(source_version, str) or not source_version:
            errors.append(f"{context}: provenance renderer_version is required")
        if arm_row.get("renderer_version") != source_version:
            errors.append(f"{context}: provenance renderer_version does not match arm row")
        if source_version != expected_renderer:
            errors.append(f"{context}: renderer_version does not match manifest")
        if provenance_path is not None and isinstance(source_sha, str) and source_sha:
            cache_key = (str(provenance_path), source_sha)
            if cache_key not in checked_files:
                checked_files[cache_key] = bool(
                    provenance_path.is_file()
                    and sha256_file(provenance_path) == source_sha
                )
            if not checked_files[cache_key]:
                errors.append(f"{context}: render file is missing or its hash differs")

    human_independent = 0
    for index, label in enumerate(independent_labels, 1):
        if str(label.get("label_source") or "").startswith("automatic_"):
            continue
        usability = label.get("overall_usability") or label.get("usefulness")
        if usability not in USABILITY:
            continue
        human_independent += 1
        item_id = label.get("item_id")
        source = independent_by_id.get(str(item_id))
        if not isinstance(item_id, str) or not item_id or source is None:
            errors.append(
                f"independent human label row {index}: known provenance item_id is required"
            )
            continue
        unit_id = _unit_id(source)
        raw_arms = source.get("arms")
        if not isinstance(raw_arms, list):
            raw_arms = [source.get("arm")]
        source_arms = [_arm(value) for value in raw_arms]
        if unit_id is None or not source_arms or any(value is None for value in source_arms):
            errors.append(f"independent human label row {index}: invalid provenance arms")
            continue
        source_artifact = _artifact_id(source)
        if not source_artifact:
            errors.append(f"independent human label row {index}: provenance artifact_id is required")
        for arm in source_arms:
            arm_row = arms.get((unit_id, arm))
            context = f"independent human label row {index} ({unit_id}/{arm})"
            if arm_row is None:
                errors.append(f"{context}: referenced arm row is missing")
                continue
            if source_artifact != _artifact_id(arm_row):
                errors.append(f"{context}: provenance artifact_id does not match arm row")
            check_render(
                context=context,
                source_path=source.get("render_path"),
                source_sha=source.get("render_sha256"),
                source_version=source.get("renderer_version"),
                arm_row=arm_row,
            )

    human_pairs = 0
    for index, label in enumerate(pair_labels, 1):
        if str(label.get("label_source") or "").startswith("automatic_"):
            continue
        winner = label.get("winner") or label.get("preference")
        if winner not in WINNERS:
            continue
        human_pairs += 1
        pair_id = label.get("pair_id")
        source = pair_by_id.get(str(pair_id))
        if not isinstance(pair_id, str) or not pair_id or source is None:
            errors.append(
                f"pair human label row {index}: known provenance pair_id is required"
            )
            continue
        if source.get("rateable") is not True:
            errors.append(f"pair human label row {index}: provenance is not rateable")
        if source.get("operational_failure") is True:
            errors.append(
                f"pair human label row {index}: operational-failure pair cannot be human-rated"
            )
        unit_id = _unit_id(source)
        for side in ("left", "right"):
            arm = _arm(source.get(f"{side}_arm") or source.get(f"{side}_variant"))
            context = f"pair human label row {index} ({pair_id}/{side})"
            arm_row = arms.get((unit_id, arm))
            if unit_id is None or arm is None or arm_row is None:
                errors.append(f"{context}: referenced arm row is missing")
                continue
            if source.get(f"{side}_artifact_id") != _artifact_id(arm_row):
                errors.append(f"{context}: provenance artifact_id does not match arm row")
            check_render(
                context=context,
                source_path=source.get(f"{side}_render_path"),
                source_sha=source.get(f"{side}_render_sha256"),
                source_version=source.get("renderer_version"),
                arm_row=arm_row,
            )

    return {
        "complete": not errors,
        "expected_renderer_version": expected_renderer,
        "human_independent_label_rows": human_independent,
        "human_pair_label_rows": human_pairs,
        "verified_render_files": sum(checked_files.values()),
        "errors": errors,
    }


def _result_seal_validation(
    run_dir: Path, manifest: dict,
    independent_provenance: list[dict], pair_provenance: list[dict],
) -> dict:
    """Verify the complete pre-label result bundle and private blind seed."""
    errors: list[str] = []
    seal = manifest.get("result_seal")
    seal = seal if isinstance(seal, dict) else {}
    seal_path_value = seal.get("path")
    seal_path = (
        (run_dir / seal_path_value).resolve()
        if isinstance(seal_path_value, str) and seal_path_value
        else run_dir / "result_manifest.json"
    )
    result: dict = {}
    actual_seal_sha: str | None = None
    if not seal_path.is_file():
        errors.append("sealed result_manifest.json is missing")
    else:
        actual_seal_sha = sha256_file(seal_path)
        try:
            candidate = read_json(seal_path)
            if isinstance(candidate, dict):
                result = candidate
            else:
                errors.append("result manifest must be a JSON object")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"result manifest cannot be read: {type(exc).__name__}")
    if actual_seal_sha is None or seal.get("sha256") != actual_seal_sha:
        errors.append("manifest result_seal hash mismatch")

    files = result.get("files")
    if not isinstance(files, dict) or not files:
        errors.append("result manifest file inventory is missing")
        files = {}
    required_files = {
        "frozen_units.jsonl", "frozen_manifest.json", "refine_arms.jsonl",
        "refine_independent_provenance.private.jsonl",
        "refine_pair_provenance.private.jsonl",
        "refine_label_assignments.private.jsonl",
        "blind_randomization.private.json", "label_assignment.private.json",
        "mesh_safety_evidence.template.jsonl",
    }
    promotion_metadata = manifest.get("promotion_criteria")
    if (
        isinstance(promotion_metadata, dict)
        and promotion_metadata.get("preregistered") is True
    ):
        required_files.add("promotion_criteria.frozen.json")
    missing_required_files = sorted(required_files - set(files))
    if missing_required_files:
        errors.append(
            "result seal is missing required files: "
            + ", ".join(missing_required_files)
        )
    verified_files = 0
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            errors.append("result manifest has an invalid file inventory entry")
            continue
        target = (run_dir / relative).resolve()
        try:
            target.relative_to(run_dir)
        except ValueError:
            errors.append(f"result manifest path escapes run directory: {relative}")
            continue
        if not target.is_file():
            errors.append(f"sealed result file is missing: {relative}")
            continue
        if (
            sha256_file(target) != metadata.get("sha256")
            or target.stat().st_size != metadata.get("bytes")
        ):
            errors.append(f"sealed result file hash/size mismatch: {relative}")
            continue
        verified_files += 1

    identity_fields = (
        "run_id", "dataset", "servers", "renderer", "cache_policy",
        "promotion_criteria", "blind_randomization", "strict_capabilities",
        "label_assignment", "capability_warnings",
    )
    expected_identity = {field: manifest.get(field) for field in identity_fields}
    if result.get("run_identity") != expected_identity:
        errors.append("result manifest run_identity differs from current manifest")

    private_relative = "blind_randomization.private.json"
    private_file = run_dir / private_relative
    blind_seed: Any = None
    if private_relative not in files or not private_file.is_file():
        errors.append("private blind randomization snapshot is not sealed")
    else:
        private: dict = {}
        try:
            private = read_json(private_file)
            blind_seed = private.get("blind_seed") if isinstance(private, dict) else None
        except (OSError, ValueError, TypeError):
            blind_seed = None
        expected_commitment = hash_json({
            "run_id": manifest.get("run_id"), "blind_seed": blind_seed,
        }) if blind_seed is not None else None
        declared_commitment = (
            (manifest.get("blind_randomization") or {}).get("seed_commitment")
            if isinstance(manifest.get("blind_randomization"), dict) else None
        )
        if expected_commitment is None or declared_commitment != expected_commitment:
            errors.append("private blind seed does not match public commitment")
        if isinstance(private, dict) and private.get("seed_commitment") != expected_commitment:
            errors.append("private blind seed snapshot commitment mismatch")
        if isinstance(private, dict) and private.get("run_id") != manifest.get("run_id"):
            errors.append("private blind seed snapshot run_id mismatch")
    for kind, rows in (
        ("independent", independent_provenance), ("pair", pair_provenance),
    ):
        if not rows or any(row.get("blind_seed") != blind_seed for row in rows):
            errors.append(f"{kind} provenance blind seeds do not match sealed seed")

    assignment_private_relative = "label_assignment.private.json"
    assignment_private_file = run_dir / assignment_private_relative
    if (
        assignment_private_relative not in files
        or not assignment_private_file.is_file()
    ):
        errors.append("private label assignment seed snapshot is not sealed")
    else:
        try:
            assignment_private = read_json(assignment_private_file)
        except (OSError, ValueError, TypeError):
            assignment_private = {}
        assignment_seed = (
            assignment_private.get("assignment_seed")
            if isinstance(assignment_private, dict) else None
        )
        assignment_commitment = hash_json({
            "run_id": manifest.get("run_id"),
            "assignment_seed": assignment_seed,
        }) if assignment_seed is not None else None
        declared_assignment = manifest.get("label_assignment")
        declared_assignment = (
            declared_assignment if isinstance(declared_assignment, dict) else {}
        )
        if assignment_commitment is None or declared_assignment.get(
            "seed_commitment"
        ) != assignment_commitment:
            errors.append("private label assignment seed does not match commitment")
        if assignment_private.get("seed_commitment") != assignment_commitment:
            errors.append("private label assignment snapshot commitment mismatch")
        if assignment_private.get("run_id") != manifest.get("run_id"):
            errors.append("private label assignment snapshot run_id mismatch")

    return {
        "complete": not errors,
        "path": str(seal_path),
        "expected_sha256": seal.get("sha256"),
        "actual_sha256": actual_seal_sha,
        "sealed_files": len(files),
        "verified_files": verified_files,
        "errors": errors,
    }


def _mesh_evidence_seal_validation(
    run_dir: Path, manifest: dict, mesh_file: Path,
    result_seal_validation: dict,
) -> dict:
    """Chain post-run mesh evidence to the immutable pre-label result seal."""
    errors: list[str] = []
    pointer = manifest.get("evidence_seal")
    pointer = pointer if isinstance(pointer, dict) else {}
    pointer_path = pointer.get("path")
    evidence_manifest = (
        (run_dir / pointer_path).resolve()
        if isinstance(pointer_path, str) and pointer_path
        else run_dir / "evidence_manifest.json"
    )
    evidence: dict = {}
    actual_manifest_sha: str | None = None
    if not evidence_manifest.is_file():
        errors.append("post-run evidence_manifest.json is missing")
    else:
        actual_manifest_sha = sha256_file(evidence_manifest)
        try:
            candidate = read_json(evidence_manifest)
            if isinstance(candidate, dict):
                evidence = candidate
            else:
                errors.append("evidence manifest must be a JSON object")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"evidence manifest cannot be read: {type(exc).__name__}")
    if actual_manifest_sha is None or pointer.get("sha256") != actual_manifest_sha:
        errors.append("manifest evidence_seal hash mismatch")
    if evidence.get("run_id") != manifest.get("run_id"):
        errors.append("evidence manifest run_id mismatch")
    if evidence.get("result_manifest_sha256") != result_seal_validation.get(
        "actual_sha256"
    ):
        errors.append("evidence manifest is not chained to the sealed result")
    if evidence.get("mesh_check_contract_version") != MESH_CHECK_CONTRACT_VERSION:
        errors.append("evidence manifest mesh check contract version mismatch")
    if evidence.get("required_checks") != list(MESH_REQUIRED_CHECKS):
        errors.append("evidence manifest required mesh checks mismatch")
    files = evidence.get("files")
    files = files if isinstance(files, dict) else {}
    try:
        mesh_relative = mesh_file.resolve().relative_to(run_dir).as_posix()
    except ValueError:
        mesh_relative = ""
        errors.append("mesh evidence file must live inside the run directory")
    mesh_metadata = files.get(mesh_relative)
    mesh_sha: str | None = None
    if not mesh_file.is_file():
        errors.append("mesh safety evidence file is missing")
    else:
        mesh_sha = sha256_file(mesh_file)
        if not isinstance(mesh_metadata, dict) or (
            mesh_metadata.get("sha256") != mesh_sha
            or mesh_metadata.get("bytes") != mesh_file.stat().st_size
        ):
            errors.append("mesh safety evidence hash/size is not sealed")
        if evidence.get("row_count") != len(read_jsonl(mesh_file)):
            errors.append("evidence manifest mesh row_count mismatch")
    result_manifest_path = Path(str(result_seal_validation.get("path") or ""))
    template_name = "mesh_safety_evidence.template.jsonl"
    template_path = run_dir / template_name
    result_files: dict = {}
    if result_manifest_path.is_file():
        result_value = read_json(result_manifest_path)
        result_files = (
            result_value.get("files")
            if isinstance(result_value.get("files"), dict) else {}
        )
    template_metadata = result_files.get(template_name)
    template_sha = (
        sha256_file(template_path) if template_path.is_file() else None
    )
    if not isinstance(template_metadata, dict) or (
        template_sha is None
        or template_metadata.get("sha256") != template_sha
        or evidence.get("template_sha256") != template_sha
    ):
        errors.append("mesh evidence template is missing or not chained to both seals")
    return {
        "complete": not errors,
        "path": str(evidence_manifest),
        "expected_sha256": pointer.get("sha256"),
        "actual_sha256": actual_manifest_sha,
        "mesh_path": str(mesh_file),
        "mesh_sha256": mesh_sha,
        "errors": errors,
    }


def write_refine_report(
    run: str | Path, *, units_path: str | Path | None = None,
    arms_path: str | Path | None = None,
    independent_labels_path: str | Path | None = None,
    independent_provenance_path: str | Path | None = None,
    pair_labels_path: str | Path | None = None,
    pair_provenance_path: str | Path | None = None,
    mesh_evidence_path: str | Path | None = None,
    label_assignments_path: str | Path | None = None,
    output_json: str | Path | None = None, output_markdown: str | Path | None = None,
    bootstrap_repetitions: int | None = None, seed: int | None = None,
    promotion_criteria: dict | str | Path | None = None,
) -> dict:
    """Read a run directory, compute the report, and write JSON plus Markdown."""
    run_dir = Path(run).expanduser().resolve()
    manifest_file = run_dir / "manifest.json"
    manifest = read_json(manifest_file) if manifest_file.exists() else {}

    def path(value: str | Path | None, default: str) -> Path:
        return Path(value).expanduser().resolve() if value else run_dir / default

    units_file = path(units_path, "frozen_units.jsonl")
    arms_file = path(arms_path, "refine_arms.jsonl")
    independent_file = path(independent_labels_path, "refine_independent_labels.jsonl")
    independent_provenance_file = path(
        independent_provenance_path, "refine_independent_provenance.private.jsonl"
    )
    pair_file = path(pair_labels_path, "refine_pair_labels.jsonl")
    provenance_file = path(
        pair_provenance_path, "refine_pair_provenance.private.jsonl"
    )
    mesh_file = path(mesh_evidence_path, "mesh_safety_evidence.jsonl")
    assignments_file = path(
        label_assignments_path, "refine_label_assignments.private.jsonl"
    )
    frozen_manifest_file = run_dir / "frozen_manifest.json"
    frozen_manifest = (
        read_json(frozen_manifest_file) if frozen_manifest_file.exists() else {}
    )
    criteria_value, criteria_validation = _criteria_snapshot(
        run_dir, manifest, frozen_manifest, promotion_criteria
    )
    frozen_seed = (criteria_value or {}).get("analysis_seed")
    frozen_repetitions = (criteria_value or {}).get("bootstrap_repetitions")
    analysis_seed = int(frozen_seed) if frozen_seed is not None else int(
        seed if seed is not None else 20260805
    )
    analysis_repetitions = (
        int(frozen_repetitions) if frozen_repetitions is not None
        else int(bootstrap_repetitions if bootstrap_repetitions is not None else 2000)
    )
    invocation_matches = bool(
        (seed is None or int(seed) == analysis_seed)
        and (
            bootstrap_repetitions is None
            or int(bootstrap_repetitions) == analysis_repetitions
        )
    )
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    dataset_manifest = {}
    live_integrity: dict[str, bool] = {
        "dataset_json": False,
        "cuts_jsonl": False,
        "persons_jsonl": False,
        "frozen_units_jsonl": False,
    }
    dataset_root = dataset.get("root")
    if dataset_root:
        dataset_manifest_file = Path(dataset_root).expanduser() / "dataset.json"
        if dataset_manifest_file.exists():
            dataset_manifest = read_json(dataset_manifest_file)
            live_integrity["dataset_json"] = (
                sha256_file(dataset_manifest_file) == dataset.get("manifest_sha256")
            )
        cuts_file = Path(dataset_root).expanduser() / "cuts.jsonl"
        if cuts_file.exists():
            live_integrity["cuts_jsonl"] = (
                hash_jsonl(read_jsonl(cuts_file)) == dataset.get("cut_manifest_sha256")
            )
        persons_file = Path(dataset_root).expanduser() / "persons.jsonl"
        if persons_file.exists():
            live_integrity["persons_jsonl"] = (
                hash_jsonl(read_jsonl(persons_file)) == dataset.get("gt_sha256")
            )
    live_integrity["frozen_units_jsonl"] = bool(
        units_file.exists()
        and frozen_manifest.get("units_sha256")
        and sha256_file(units_file) == frozen_manifest.get("units_sha256")
    )
    snapshot_integrity = all(live_integrity.values())
    holdout_evidence = {
        # Never retrofit an old engineering run into a holdout by reading a
        # later-edited live dataset manifest.  Purpose/seal must be frozen in
        # the source run itself.
        "purpose": dataset.get("purpose"),
        "sealed_at": dataset.get("sealed_at"),
        "manifest_sha256": dataset.get("manifest_sha256"),
        "cut_manifest_sha256": dataset.get("cut_manifest_sha256"),
        "gt_sha256": dataset.get("gt_sha256"),
        "units_sha256": frozen_manifest.get("units_sha256"),
        "integrity_checks": live_integrity,
        "integrity_valid": snapshot_integrity,
    }
    units_rows = read_jsonl(units_file)
    arm_rows = read_jsonl(arms_file)
    independent_labels = (
        read_jsonl(independent_file) if independent_file.exists() else []
    )
    independent_provenance = (
        read_jsonl(independent_provenance_file)
        if independent_provenance_file.exists() else []
    )
    pair_labels = read_jsonl(pair_file) if pair_file.exists() else []
    pair_provenance = (
        read_jsonl(provenance_file) if provenance_file.exists() else []
    )
    mesh_evidence = read_jsonl(mesh_file) if mesh_file.exists() else []
    label_assignments = (
        read_jsonl(assignments_file) if assignments_file.exists() else []
    )
    report = compute_refine_report(
        units_rows, arm_rows, independent_labels, pair_labels, pair_provenance,
        bootstrap_repetitions=analysis_repetitions, seed=analysis_seed,
        promotion_criteria=criteria_value,
        holdout_evidence=holdout_evidence,
        usability_rubric={
            "version": (criteria_value or {}).get("usability_rubric_version"),
            "human_usable_categories": (
                (criteria_value or {}).get("human_usable_categories") or []
            ),
        },
        cache_policy=manifest.get("cache_policy"),
        independent_provenance=independent_provenance,
        mesh_evidence=mesh_evidence,
        label_assignments=label_assignments,
    )
    report["run_id"] = manifest.get("run_id") or run_dir.name
    report["promotion_criteria_validation"] = criteria_validation
    report["analysis_invocation_validation"] = {
        "complete": invocation_matches,
        "requested_seed": seed,
        "requested_bootstrap_repetitions": bootstrap_repetitions,
        "applied_seed": analysis_seed,
        "applied_bootstrap_repetitions": analysis_repetitions,
    }
    result_seal_validation = _result_seal_validation(
        run_dir, manifest, independent_provenance, pair_provenance
    )
    report["result_seal_validation"] = result_seal_validation
    report["mesh_evidence_seal_validation"] = _mesh_evidence_seal_validation(
        run_dir, manifest, mesh_file, result_seal_validation
    )
    report["label_artifact_validation"] = _label_artifact_validation(
        run_dir, manifest, arm_rows, independent_labels,
        independent_provenance, pair_labels, pair_provenance,
    )
    invalid_raw_score_units = []
    for unit in units_rows:
        raw_scores = unit.get("raw_scores")
        target_keypoints = unit.get("target_keypoints")
        target_scores = unit.get("target_scores")
        target_mask = unit.get("target_valid_mask")
        raw_valid = bool(
            unit.get("raw_scores_available") is True
            and isinstance(raw_scores, list) and len(raw_scores) == 17
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) >= 0.0
                for value in raw_scores
            )
        )
        target_valid = bool(
            isinstance(target_keypoints, list) and len(target_keypoints) == 17
            and all(
                isinstance(point, list) and len(point) == 2
                and all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) for value in point
                ) for point in target_keypoints
            )
            and isinstance(target_scores, list) and len(target_scores) == 17
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) >= 0.0
                for value in target_scores
            )
            and isinstance(target_mask, list) and len(target_mask) == 17
            and all(isinstance(value, bool) for value in target_mask)
            and target_mask == [float(value) >= 0.3 for value in target_scores]
        )
        expected_query_hash = hash_json({
            "keypoints": target_keypoints,
            "scores": target_scores,
            "skeleton_state": unit.get("skeleton_state"),
            "coverage_class": unit.get("coverage_class"),
            "slot_origin": unit.get("slot_origin"),
            "skeleton_source": unit.get("skeleton_source"),
        }) if target_valid else None
        evidence = unit.get("query_evidence")
        evidence_payload = (
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
            if isinstance(evidence, dict) else None
        )
        evidence_digest = (
            hash_json(evidence_payload) if evidence_payload is not None else None
        )
        declared_unit_evidence_hash = unit.get("query_evidence_sha256")
        declared_payload_evidence_hash = (
            evidence.get("evidence_sha256") if isinstance(evidence, dict) else None
        )
        # refine-external-v1.1 writes an explicit ``sha256:`` algorithm
        # prefix; early frozen fixtures used the bare digest. Both are
        # unambiguous, but the unit and embedded evidence must agree exactly.
        allowed_evidence_hashes = (
            {evidence_digest, f"sha256:{evidence_digest}"}
            if evidence_digest is not None else set()
        )
        lineage_valid = bool(
            expected_query_hash == unit.get("query_preprocess_sha256")
            and declared_unit_evidence_hash == declared_payload_evidence_hash
            and declared_unit_evidence_hash in allowed_evidence_hashes
        )
        if not (raw_valid and target_valid and lineage_valid):
            invalid_raw_score_units.append(_unit_id(unit))
    report["raw_query_evidence_validation"] = {
        "complete": bool(units_rows) and not invalid_raw_score_units
        and manifest.get("raw_query_evidence_complete") is True,
        "manifest_value": manifest.get("raw_query_evidence_complete"),
        "invalid_unit_ids": invalid_raw_score_units,
    }
    capability_warnings = manifest.get("capability_warnings") or []
    strict_capabilities = manifest.get("strict_capabilities") is True
    capability_complete = strict_capabilities and not capability_warnings
    report["capability_validation"] = {
        "strict_capabilities": strict_capabilities,
        "warnings": capability_warnings,
        "complete": capability_complete,
    }
    if not capability_complete:
        reason = (
            "strict capability validation is required and capability_warnings must be empty"
        )
        _force_inconclusive(report, reason)
    for complete, reason in (
        (
            criteria_validation["complete"],
            "pre-registered frozen promotion criteria lineage is incomplete or mismatched",
        ),
        (
            invocation_matches,
            "requested analysis seed/repetitions differ from frozen criteria",
        ),
        (
            result_seal_validation["complete"],
            "pre-label result seal is incomplete or mismatched",
        ),
        (
            report["mesh_evidence_seal_validation"]["complete"],
            "post-run mesh evidence seal is incomplete or mismatched",
        ),
        (
            report["label_artifact_validation"]["complete"],
            "human labels are not bound to verified blinded render artifacts",
        ),
        (
            report["raw_query_evidence_validation"]["complete"],
            "complete raw query score evidence is required",
        ),
    ):
        if not complete:
            _force_inconclusive(report, reason)
    report["input_snapshots"] = {
        "frozen_units": {"path": str(units_file), "sha256": sha256_file(units_file)},
        "refine_arms": {"path": str(arms_file), "sha256": sha256_file(arms_file)},
        "independent_labels": {
            "path": str(independent_file),
            "sha256": sha256_file(independent_file) if independent_file.exists() else None,
        },
        "independent_provenance": {
            "path": str(independent_provenance_file),
            "sha256": (
                sha256_file(independent_provenance_file)
                if independent_provenance_file.exists() else None
            ),
        },
        "pair_labels": {
            "path": str(pair_file),
            "sha256": sha256_file(pair_file) if pair_file.exists() else None,
        },
        "pair_provenance": {
            "path": str(provenance_file),
            "sha256": sha256_file(provenance_file) if provenance_file.exists() else None,
        },
        "promotion_criteria": {
            "path": str(run_dir / "promotion_criteria.frozen.json"),
            "sha256": criteria_validation.get("actual_sha256"),
        },
        "mesh_safety_evidence": {
            "path": str(mesh_file),
            "sha256": sha256_file(mesh_file) if mesh_file.exists() else None,
        },
        "label_assignments": {
            "path": str(assignments_file),
            "sha256": (
                sha256_file(assignments_file) if assignments_file.exists() else None
            ),
        },
        "result_manifest": {
            "path": result_seal_validation.get("path"),
            "sha256": result_seal_validation.get("actual_sha256"),
        },
        "evidence_manifest": {
            "path": report["mesh_evidence_seal_validation"].get("path"),
            "sha256": report["mesh_evidence_seal_validation"].get("actual_sha256"),
        },
    }
    json_path = path(output_json, "refine_evaluation_report.json")
    markdown_path = path(output_markdown, "refine_evaluation_report.md")
    write_json(json_path, report)
    atomic_write_text(markdown_path, render_refine_markdown(report))
    return report
