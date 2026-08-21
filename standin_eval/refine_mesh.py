from __future__ import annotations

"""Shared, fail-closed contract for independent CSP/avatar mesh evidence.

The runner only creates a worksheet; an external mesh evaluator fills it and
``refine_evidence`` seals it.  Both paths use this module so a field cannot be
interpreted differently at generation, sealing, and reporting time.

Violation IDs are stable strings of the form ``<check>:<detail>``.  The first
segment binds every violation to one required boolean check.  A failed check
without a matching hard violation is invalid evidence, never an implicit pass.
"""

import re
from typing import Iterable, Mapping


MESH_CHECK_CONTRACT_VERSION = "mesh-checks-v2"
MESH_REQUIRED_CHECKS = (
    "parse_fk",
    "ownership",
    "anatomy",
    "collision",
    "contact",
    "ground",
    "foot_direction",
)
MESH_ARMS = (
    "B0_no_refine",
    "B1_v1",
    "B2_v24_aggressive",
)
MESH_BASELINE_ARM = MESH_ARMS[0]
MESH_IDENTITY_FIELDS = (
    "unit_id",
    "arm",
    "artifact_id",
    "geometry_sha256",
)

_PLACEHOLDER = "REQUIRED"
_VIOLATION_ID = re.compile(
    r"^[a-z][a-z0-9_]*(?::[a-z0-9][a-z0-9_.-]*)+$"
)


def mesh_contract_metadata() -> dict:
    """Return JSON-serializable contract metadata for manifests/reporters."""
    return {
        "check_contract_version": MESH_CHECK_CONTRACT_VERSION,
        "required_checks": list(MESH_REQUIRED_CHECKS),
        "identity_fields": list(MESH_IDENTITY_FIELDS),
        "violation_id_format": "<required_check>:<stable_detail>",
        "baseline_arm": MESH_BASELINE_ARM,
    }


def mesh_failure_id(check: str, detail: str = "failed") -> str:
    """Build a stable violation ID belonging to a required mesh check."""
    check = str(check)
    detail = str(detail)
    if check not in MESH_REQUIRED_CHECKS:
        raise ValueError(f"unknown mesh check: {check}")
    value = f"{check}:{detail}"
    if _VIOLATION_ID.fullmatch(value) is None:
        raise ValueError(f"invalid mesh violation detail: {detail!r}")
    return value


def build_mesh_evidence_template_row(
    *,
    unit_id: str,
    arm: str,
    artifact_id: str,
    geometry_sha256: str,
) -> dict:
    """Create one deliberately non-promotable but contract-valid worksheet row."""
    row = {
        "unit_id": unit_id,
        "arm": arm,
        "artifact_id": artifact_id,
        "geometry_sha256": geometry_sha256,
        "evaluator_kind": "mesh",
        "evaluator_version": _PLACEHOLDER,
        "body_version": _PLACEHOLDER,
        "check_contract_version": MESH_CHECK_CONTRACT_VERSION,
        "checks_complete": False,
        "checks": {name: False for name in MESH_REQUIRED_CHECKS},
        "hard_violations": [
            mesh_failure_id(name, "not_evaluated")
            for name in MESH_REQUIRED_CHECKS
        ],
        "new_hard_violations": [],
    }
    errors = validate_mesh_evidence_row(
        row, require_complete=False, allow_placeholders=True,
    )
    if errors:  # Programmer error: never write a malformed frozen template.
        raise ValueError("invalid generated mesh template: " + "; ".join(errors))
    return row


def _stable_ids(value, field: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{field} must be a list of stable string IDs")
        return []
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or _VIOLATION_ID.fullmatch(item) is None:
            errors.append(
                f"{field}[{index}] must match <required_check>:<stable_detail>"
            )
            continue
        check = item.split(":", 1)[0]
        if check not in MESH_REQUIRED_CHECKS:
            errors.append(f"{field}[{index}] has unknown check prefix {check!r}")
            continue
        values.append(item)
    if len(values) != len(set(values)):
        errors.append(f"{field} must not contain duplicate IDs")
    return values


def validate_mesh_evidence_row(
    row: Mapping | object,
    *,
    expected_identity: Mapping | None = None,
    require_complete: bool = True,
    allow_placeholders: bool = False,
) -> list[str]:
    """Return all structural/semantic errors for one mesh evidence row.

    This function never mutates ``row`` and is suitable for direct reuse by the
    reporter.  An empty list is the only valid result.
    """
    if not isinstance(row, Mapping):
        return ["row must be an object"]
    errors: list[str] = []

    for field in MESH_IDENTITY_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} is required and must be a non-empty string")
    arm = row.get("arm")
    if isinstance(arm, str) and arm not in MESH_ARMS:
        errors.append(f"arm must be one of {', '.join(MESH_ARMS)}")

    if expected_identity is not None:
        for field in MESH_IDENTITY_FIELDS:
            if row.get(field) != expected_identity.get(field):
                errors.append(f"{field} does not match frozen artifact identity")

    if row.get("evaluator_kind") != "mesh":
        errors.append("evaluator_kind must be 'mesh'")
    for field in ("evaluator_version", "body_version"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} is required")
        elif not allow_placeholders and value == _PLACEHOLDER:
            errors.append(f"{field} must replace the template placeholder")
    if row.get("check_contract_version") != MESH_CHECK_CONTRACT_VERSION:
        errors.append(
            f"check_contract_version must be {MESH_CHECK_CONTRACT_VERSION!r}"
        )

    checks = row.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("checks must be an object with named boolean results")
        checks = {}
    missing_checks = [name for name in MESH_REQUIRED_CHECKS if name not in checks]
    if missing_checks:
        errors.append("checks missing required names: " + ", ".join(missing_checks))
    extra_checks = [name for name in checks if name not in MESH_REQUIRED_CHECKS]
    if extra_checks:
        errors.append(
            "checks contains names outside this contract version: "
            + ", ".join(sorted(str(name) for name in extra_checks))
        )
    for name, value in checks.items():
        if not isinstance(name, str) or not name:
            errors.append("check names must be non-empty strings")
        if not isinstance(value, bool):
            errors.append(f"check {name!r} must be boolean")

    if not isinstance(row.get("checks_complete"), bool):
        errors.append("checks_complete must be boolean")
    elif require_complete and row.get("checks_complete") is not True:
        errors.append("checks_complete must be true")

    hard = _stable_ids(row.get("hard_violations"), "hard_violations", errors)
    new = _stable_ids(
        row.get("new_hard_violations"), "new_hard_violations", errors,
    )
    hard_set, new_set = set(hard), set(new)
    if not new_set.issubset(hard_set):
        errors.append("new_hard_violations must be a subset of hard_violations")
    if arm == MESH_BASELINE_ARM and new:
        errors.append("B0 new_hard_violations must be empty")

    hard_checks = {value.split(":", 1)[0] for value in hard}
    for name in MESH_REQUIRED_CHECKS:
        value = checks.get(name)
        if value is False and name not in hard_checks:
            errors.append(
                f"failed check {name!r} requires a matching hard violation ID"
            )
        elif value is True and name in hard_checks:
            errors.append(
                f"passed check {name!r} conflicts with a hard violation ID"
            )
    return errors


def validate_mesh_evidence_bundle(
    rows: Iterable[Mapping] | object,
    *,
    expected_rows: Iterable[Mapping] | None = None,
    require_complete: bool = True,
    allow_placeholders: bool = False,
) -> dict:
    """Validate row semantics, exact coverage, identity, and version consistency."""
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        return {
            "valid": False,
            "errors": ["mesh evidence must be an iterable of row objects"],
            "row_count": 0,
            **mesh_contract_metadata(),
        }
    values = list(rows)
    errors: list[str] = []
    expected: dict[tuple[object, object], Mapping] | None = None
    if expected_rows is not None:
        expected = {}
        for index, item in enumerate(expected_rows, 1):
            if not isinstance(item, Mapping):
                errors.append(f"expected row {index}: row must be an object")
                continue
            key = (item.get("unit_id"), item.get("arm"))
            if key in expected:
                errors.append(f"expected row {index}: duplicate identity {key[0]}/{key[1]}")
            expected[key] = item

    seen: dict[tuple[object, object], Mapping] = {}
    versions: set[tuple[str, str, str]] = set()
    for index, row in enumerate(values, 1):
        key = (
            row.get("unit_id") if isinstance(row, Mapping) else None,
            row.get("arm") if isinstance(row, Mapping) else None,
        )
        expected_identity = expected.get(key) if expected is not None else None
        if key in seen:
            errors.append(f"row {index}: duplicate identity {key[0]}/{key[1]}")
        else:
            seen[key] = row
        if expected is not None and expected_identity is None:
            errors.append(f"row {index}: unexpected identity {key[0]}/{key[1]}")
        row_errors = validate_mesh_evidence_row(
            row,
            expected_identity=expected_identity,
            require_complete=require_complete,
            allow_placeholders=allow_placeholders,
        )
        errors.extend(f"row {index}: {message}" for message in row_errors)
        if isinstance(row, Mapping):
            versions.add((
                str(row.get("evaluator_version") or ""),
                str(row.get("body_version") or ""),
                str(row.get("check_contract_version") or ""),
            ))

    if not values:
        errors.append("mesh evidence must contain at least one row")
    if len(versions) != 1:
        errors.append(
            "mesh evidence must use one evaluator/body/check-contract version"
        )
    if expected is not None:
        missing = sorted(
            set(expected) - set(seen), key=lambda key: (str(key[0]), str(key[1]))
        )
        for unit_id, arm in missing:
            errors.append(f"missing mesh evidence {unit_id}/{arm}")

    # New violations are defined against the same unit's frozen B0 artifact,
    # so row-local validation is insufficient. If B0 passed a required check
    # and a result arm fails it, that result must name at least one new
    # violation with the same check prefix. Existing B0 failures may remain
    # failures without being new; a distinct/worsened failure can still be
    # marked new by the external evaluator.
    by_unit: dict[object, dict[object, Mapping]] = {}
    for (unit_id, arm), row in seen.items():
        if isinstance(row, Mapping):
            by_unit.setdefault(unit_id, {})[arm] = row
    for unit_id, arm_rows in sorted(by_unit.items(), key=lambda item: str(item[0])):
        baseline = arm_rows.get(MESH_BASELINE_ARM)
        if not isinstance(baseline, Mapping):
            if any(arm != MESH_BASELINE_ARM for arm in arm_rows):
                errors.append(
                    f"unit {unit_id}: B0 mesh evidence is required for new-violation comparison"
                )
            continue
        baseline_checks = baseline.get("checks")
        if not isinstance(baseline_checks, Mapping):
            continue
        for arm, row in sorted(arm_rows.items(), key=lambda item: str(item[0])):
            if arm == MESH_BASELINE_ARM or not isinstance(row, Mapping):
                continue
            checks = row.get("checks")
            new = row.get("new_hard_violations")
            if not isinstance(checks, Mapping) or not isinstance(new, list):
                continue
            new_checks = {
                value.split(":", 1)[0]
                for value in new
                if isinstance(value, str) and ":" in value
            }
            for check in MESH_REQUIRED_CHECKS:
                if (
                    baseline_checks.get(check) is True
                    and checks.get(check) is False
                    and check not in new_checks
                ):
                    errors.append(
                        f"unit {unit_id}/{arm}: newly failed check {check!r} "
                        "requires a matching new_hard_violations ID"
                    )

    return {
        "valid": not errors,
        "errors": errors,
        "row_count": len(values),
        "expected_row_count": None if expected is None else len(expected),
        "versions": [list(value) for value in sorted(versions)],
        **mesh_contract_metadata(),
    }


def require_valid_mesh_evidence_bundle(*args, **kwargs) -> dict:
    """Validate a bundle and raise one deterministic error on any defect."""
    result = validate_mesh_evidence_bundle(*args, **kwargs)
    if not result["valid"]:
        raise ValueError("invalid mesh evidence: " + "; ".join(result["errors"]))
    return result


__all__ = [
    "MESH_ARMS",
    "MESH_BASELINE_ARM",
    "MESH_CHECK_CONTRACT_VERSION",
    "MESH_IDENTITY_FIELDS",
    "MESH_REQUIRED_CHECKS",
    "build_mesh_evidence_template_row",
    "mesh_contract_metadata",
    "mesh_failure_id",
    "require_valid_mesh_evidence_bundle",
    "validate_mesh_evidence_bundle",
    "validate_mesh_evidence_row",
]
