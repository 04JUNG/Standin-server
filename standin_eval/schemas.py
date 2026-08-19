from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1

ROUTES = {"core", "bust", "skip"}
SPLITS = {"engineering", "calibration", "validation", "holdout", "pilot"}
SCALE_CLASSES = {"near", "far"}
USEFULNESS = {"direct", "reference", "unusable", "unknown"}
APPEARANCE = {"allow", "reject", "unknown"}
SKELETON_STATES = {"valid", "partial", "suspect", "missing", "invalid"}
COVERAGE_CLASSES = {"full", "reduced", "sparse", "insufficient"}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    record_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "record_id": self.record_id,
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def bbox(value: Any, field: str = "bbox_xyxy") -> list[float]:
    if not isinstance(value, list) or len(value) != 4 or not all(
        _is_number(item) for item in value
    ):
        raise ValueError(f"{field} must contain four numbers")
    out = [float(item) for item in value]
    if out[2] <= out[0] or out[3] <= out[1]:
        raise ValueError(f"{field} must satisfy x2>x1 and y2>y1")
    return out


def validate_cut_shape(row: dict) -> list[str]:
    errors: list[str] = []
    for field in ("cut_id", "image_path", "image_sha256", "scene_group_id",
                  "artist_id", "project_id", "split", "expected_route"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{field} is required")
    if row.get("expected_route") not in ROUTES:
        errors.append(f"expected_route must be one of {sorted(ROUTES)}")
    if row.get("split") not in SPLITS:
        errors.append(f"split must be one of {sorted(SPLITS)}")
    count = row.get("num_people_gt")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        errors.append("num_people_gt must be a non-negative integer")
    return errors


def validate_person_shape(row: dict) -> list[str]:
    errors: list[str] = []
    for field in ("person_id", "cut_id", "bbox_source"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{field} is required")
    try:
        bbox(row.get("bbox_xyxy"))
    except ValueError as exc:
        errors.append(str(exc))
    for field in ("eligible", "out_of_scope"):
        if not isinstance(row.get(field), bool):
            errors.append(f"{field} must be boolean")
    if row.get("scale_class") not in SCALE_CLASSES:
        errors.append(f"scale_class must be one of {sorted(SCALE_CLASSES)}")
    return errors


def validate_label_shape(row: dict) -> list[str]:
    errors: list[str] = []
    for field in ("dataset_id", "person_id", "candidate_artifact_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{field} is required")
    if row.get("usefulness") not in USEFULNESS:
        errors.append(f"usefulness must be one of {sorted(USEFULNESS)}")
    if row.get("appearance") not in APPEARANCE:
        errors.append(f"appearance must be one of {sorted(APPEARANCE)}")
    rubric = row.get("rubric_version")
    if not isinstance(rubric, int) or isinstance(rubric, bool) or rubric < 1:
        errors.append("rubric_version must be a positive integer")
    return errors


def is_target_person(person: dict, cut: dict) -> bool:
    return bool(
        person.get("eligible") is True
        and person.get("scale_class") == "near"
        and cut.get("expected_route") == "core"
        and person.get("out_of_scope") is False
    )


def accepted_label(label: dict | None) -> bool:
    if not label:
        return False
    return (
        label.get("usefulness") in {"direct", "reference"}
        and label.get("appearance") == "allow"
    )


def label_is_final(label: dict | None) -> bool:
    return bool(
        label
        and label.get("usefulness") in USEFULNESS - {"unknown"}
        and label.get("appearance") in APPEARANCE - {"unknown"}
    )
