"""Deterministic BVH pose measurements and typed observed atoms.

This module deliberately describes only facts visible in a single-frame COCO-17
skeleton.  It never infers an action, prop, emotion, culture, or motion phase.
The output is used by the offline semantic tagging workflow; it is not injected
into the existing geometric search feature space.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


POSECODE_VERSION = 2
ATOM_SCHEMA_VERSION = 1
COORDINATE_PROFILE = "coco17_body_local_y_up_v1"

_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16
_HEAD = 0


@dataclass(frozen=True)
class BodyFrame:
    origin: np.ndarray
    lateral: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    torso_length: float


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"cannot build body frame: degenerate {label}")
    return np.asarray(vector, dtype=np.float64) / norm


def build_body_frame(joints3d: np.ndarray) -> BodyFrame:
    """Build a right-handed frame from labelled hips and the global BVH Y-up axis."""
    points = np.asarray(joints3d, dtype=np.float64)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        raise ValueError(f"joints3d must be finite (17,3), got {points.shape}")

    hip_mid = (points[_LEFT_HIP] + points[_RIGHT_HIP]) / 2.0
    shoulder_mid = (points[_LEFT_SHOULDER] + points[_RIGHT_SHOULDER]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_length < 1e-6:
        raise ValueError("cannot build body frame: degenerate torso")

    # Project the labelled left->right hip vector onto the ground plane.  This
    # tracks a turned body while retaining the repository's Y-up BVH convention.
    lateral_raw = points[_RIGHT_HIP] - points[_LEFT_HIP]
    lateral_raw[1] = 0.0
    if np.linalg.norm(lateral_raw) < 1e-6:
        lateral_raw = points[_RIGHT_SHOULDER] - points[_LEFT_SHOULDER]
        lateral_raw[1] = 0.0
    lateral = _normalize(lateral_raw, "lateral axis")
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = _normalize(np.cross(lateral, up), "forward axis")
    return BodyFrame(hip_mid, lateral, up, forward, torso_length)


def _local(point: np.ndarray, frame: BodyFrame, origin: np.ndarray | None = None) -> np.ndarray:
    delta = (np.asarray(point, dtype=np.float64) - (frame.origin if origin is None else origin))
    delta = delta / frame.torso_length
    return np.array(
        [
            float(np.dot(delta, frame.lateral)),
            float(np.dot(delta, frame.up)),
            float(np.dot(delta, frame.forward)),
        ],
        dtype=np.float64,
    )


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    second = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < 1e-9:
        raise ValueError("cannot measure joint angle with zero-length limb")
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _flexion_bucket(angle: float) -> str:
    if angle >= 155.0:
        return "extended"
    if angle < 75.0:
        return "deeply_bent"
    return "bent"


def _degree_bucket(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 7.0:
        return "neutral"
    if magnitude < 13.0:
        return "unknown_boundary"
    if magnitude < 22.0:
        return "slight"
    if magnitude < 28.0:
        return "unknown_boundary"
    return "deep"


def _atom(
    *,
    predicate: str,
    provenance: dict[str, Any],
    subject: str | None = None,
    relation: str | None = None,
    object_: str | None = None,
    axis: str | None = None,
    value: str | None = None,
    measure: float | None = None,
    measure_unit: str | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    atom: dict[str, Any] = {"predicate": predicate}
    for key, item in (
        ("subject", subject),
        ("relation", relation),
        ("object", object_),
        ("axis", axis),
        ("value", value),
        ("measure", round(float(measure), 6) if measure is not None else None),
        ("measure_unit", measure_unit),
        ("bucket", bucket),
    ):
        if item is not None:
            atom[key] = item
    atom.update(
        {
            "polarity": "positive",
            "evidence_state": "observed",
            "provenance": dict(provenance),
        }
    )
    return atom


def measure_posecode(
    joints3d: np.ndarray,
    *,
    provenance_ref: str,
    provenance_version: int = POSECODE_VERSION,
) -> dict[str, Any]:
    """Return continuous measurements and conservative typed observed atoms."""
    points = np.asarray(joints3d, dtype=np.float64)
    frame = build_body_frame(points)
    provenance = {
        "kind": "bvh_rule",
        "ref": provenance_ref,
        "version": provenance_version,
        "review_status": "generated",
    }

    shoulder_mid = (points[_LEFT_SHOULDER] + points[_RIGHT_SHOULDER]) / 2.0
    torso = shoulder_mid - frame.origin
    torso_unit = _normalize(torso, "torso direction")
    torso_forward_lean = math.degrees(
        math.atan2(float(np.dot(torso_unit, frame.forward)), float(np.dot(torso_unit, frame.up)))
    )
    torso_lateral_lean = math.degrees(
        math.atan2(float(np.dot(torso_unit, frame.lateral)), float(np.dot(torso_unit, frame.up)))
    )

    local = {index: _local(points[index], frame) for index in range(17)}
    elbow_angles = {
        "left": _joint_angle(points[_LEFT_SHOULDER], points[_LEFT_ELBOW], points[_LEFT_WRIST]),
        "right": _joint_angle(points[_RIGHT_SHOULDER], points[_RIGHT_ELBOW], points[_RIGHT_WRIST]),
    }
    knee_angles = {
        "left": _joint_angle(points[_LEFT_HIP], points[_LEFT_KNEE], points[_LEFT_ANKLE]),
        "right": _joint_angle(points[_RIGHT_HIP], points[_RIGHT_KNEE], points[_RIGHT_ANKLE]),
    }

    wrist_height = {
        "left": local[_LEFT_WRIST][1] - local[_LEFT_SHOULDER][1],
        "right": local[_RIGHT_WRIST][1] - local[_RIGHT_SHOULDER][1],
    }
    ankle_height = {"left": local[_LEFT_ANKLE][1], "right": local[_RIGHT_ANKLE][1]}
    ankle_forward = {"left": local[_LEFT_ANKLE][2], "right": local[_RIGHT_ANKLE][2]}
    ankle_lateral = {"left": local[_LEFT_ANKLE][0], "right": local[_RIGHT_ANKLE][0]}
    wrist_lateral = {"left": local[_LEFT_WRIST][0], "right": local[_RIGHT_WRIST][0]}
    wrist_forward = {"left": local[_LEFT_WRIST][2], "right": local[_RIGHT_WRIST][2]}

    foot_delta = points[_RIGHT_ANKLE] - points[_LEFT_ANKLE]
    foot_delta[1] = 0.0
    foot_spacing = float(np.linalg.norm(foot_delta) / frame.torso_length)
    wrist_span = abs(wrist_lateral["right"] - wrist_lateral["left"])
    shoulder_lateral = {
        "left": local[_LEFT_SHOULDER][0],
        "right": local[_RIGHT_SHOULDER][0],
    }
    arms_outward = (
        shoulder_lateral["left"] - wrist_lateral["left"] > 0.35
        and wrist_lateral["right"] - shoulder_lateral["right"] > 0.35
    )

    distances: dict[str, float] = {}
    for side, wrist_index in (("left", _LEFT_WRIST), ("right", _RIGHT_WRIST)):
        for label, target_index in (
            ("head", _HEAD),
            ("hip", _LEFT_HIP if side == "left" else _RIGHT_HIP),
            ("thigh", _LEFT_KNEE if side == "left" else _RIGHT_KNEE),
        ):
            distances[f"{side}_hand_to_{label}_torso_units"] = float(
                np.linalg.norm(points[wrist_index] - points[target_index]) / frame.torso_length
            )

    measurements: dict[str, float] = {
        "torso_length_bvh_units": frame.torso_length,
        "torso_forward_lean_deg": torso_forward_lean,
        "torso_lateral_lean_deg": torso_lateral_lean,
        "left_elbow_flexion_deg": elbow_angles["left"],
        "right_elbow_flexion_deg": elbow_angles["right"],
        "left_knee_flexion_deg": knee_angles["left"],
        "right_knee_flexion_deg": knee_angles["right"],
        "left_wrist_height_from_shoulder_torso_units": wrist_height["left"],
        "right_wrist_height_from_shoulder_torso_units": wrist_height["right"],
        "left_wrist_lateral_from_pelvis_torso_units": wrist_lateral["left"],
        "right_wrist_lateral_from_pelvis_torso_units": wrist_lateral["right"],
        "left_wrist_forward_from_pelvis_torso_units": wrist_forward["left"],
        "right_wrist_forward_from_pelvis_torso_units": wrist_forward["right"],
        "left_ankle_height_from_pelvis_torso_units": ankle_height["left"],
        "right_ankle_height_from_pelvis_torso_units": ankle_height["right"],
        "left_ankle_lateral_from_pelvis_torso_units": ankle_lateral["left"],
        "right_ankle_lateral_from_pelvis_torso_units": ankle_lateral["right"],
        "left_ankle_forward_from_pelvis_torso_units": ankle_forward["left"],
        "right_ankle_forward_from_pelvis_torso_units": ankle_forward["right"],
        "foot_spacing_torso_units": foot_spacing,
        "wrist_span_torso_units": wrist_span,
        **distances,
    }
    measurements = {key: round(float(value), 6) for key, value in measurements.items()}

    atoms: list[dict[str, Any]] = []
    for side, angle in elbow_angles.items():
        atoms.append(
            _atom(
                predicate="joint_flexion",
                subject=f"{side}_elbow",
                measure=angle,
                measure_unit="degree",
                bucket=_flexion_bucket(angle),
                provenance=provenance,
            )
        )
    for side, angle in knee_angles.items():
        atoms.append(
            _atom(
                predicate="joint_flexion",
                subject=f"{side}_knee",
                measure=angle,
                measure_unit="degree",
                bucket=_flexion_bucket(angle),
                provenance=provenance,
            )
        )

    for side in ("left", "right"):
        height = wrist_height[side]
        if abs(height) >= 0.16:
            relation = "above" if height > 0.0 else "below"
            atoms.append(
                _atom(
                    predicate="relative_position",
                    subject=f"{side}_wrist",
                    relation=relation,
                    object_=f"{side}_shoulder",
                    axis="vertical",
                    measure=abs(height),
                    measure_unit="torso_length",
                    bucket="clear",
                    provenance=provenance,
                )
            )
            atoms.append(
                _atom(
                    predicate="limb_state",
                    subject=f"{side}_arm",
                    value="raised" if height > 0.0 else "lowered",
                    measure=height,
                    measure_unit="torso_length",
                    bucket="clear",
                    provenance=provenance,
                )
            )

        forward_value = ankle_forward[side]
        if abs(forward_value) >= 0.30:
            atoms.append(
                _atom(
                    predicate="relative_direction",
                    subject=f"{side}_ankle",
                    relation="forward" if forward_value > 0.0 else "behind",
                    object_="pelvis",
                    axis="body_forward",
                    measure=abs(forward_value),
                    measure_unit="torso_length",
                    bucket="clear",
                    provenance=provenance,
                )
            )

    height_difference = ankle_height["left"] - ankle_height["right"]
    if abs(height_difference) >= 0.37:
        raised_side = "left" if height_difference > 0.0 else "right"
        support_side = "right" if raised_side == "left" else "left"
        atoms.append(
            _atom(
                predicate="limb_state",
                subject=f"{raised_side}_leg",
                value="raised",
                measure=abs(height_difference),
                measure_unit="torso_length",
                bucket="clear",
                provenance=provenance,
            )
        )
        atoms.append(
            _atom(
                predicate="support_pattern",
                subject=f"{support_side}_leg",
                value="lower_foot_support_candidate",
                measure=abs(height_difference),
                measure_unit="torso_length",
                bucket="candidate",
                provenance=provenance,
            )
        )

    if wrist_span >= 1.70 and arms_outward:
        atoms.append(
            _atom(
                predicate="limb_configuration",
                subject="both_arms",
                value="widely_spread",
                measure=wrist_span,
                measure_unit="torso_length",
                bucket="wide",
                provenance=provenance,
            )
        )

    foot_bucket = (
        "wide"
        if foot_spacing >= 1.40
        else "normal"
        if 0.50 <= foot_spacing < 1.20
        else "narrow"
        if foot_spacing < 0.40
        else None
    )
    if foot_bucket is not None:
        atoms.append(
            _atom(
                predicate="foot_spacing",
                subject="both_feet",
                measure=foot_spacing,
                measure_unit="torso_length",
                bucket=foot_bucket,
                provenance=provenance,
            )
        )

    for axis, value in (("body_forward", torso_forward_lean), ("body_lateral", torso_lateral_lean)):
        bucket = _degree_bucket(value)
        if bucket not in {"neutral", "unknown_boundary"}:
            if axis == "body_forward":
                relation = "forward" if value > 0.0 else "backward"
            else:
                relation = "right" if value > 0.0 else "left"
            atoms.append(
                _atom(
                    predicate="torso_lean",
                    subject="torso",
                    relation=relation,
                    axis=axis,
                    measure=abs(value),
                    measure_unit="degree",
                    bucket=bucket,
                    provenance=provenance,
                )
            )

    torso_tilt = math.hypot(torso_forward_lean, torso_lateral_lean)
    if torso_tilt < 17.0:
        atoms.append(
            _atom(
                predicate="torso_orientation",
                subject="torso",
                value="upright",
                measure=torso_tilt,
                measure_unit="degree",
                bucket="upright",
                provenance=provenance,
            )
        )

    for key, distance in distances.items():
        if distance >= 0.35:
            continue
        side, _, _, target, _ = key.split("_", 4)
        atoms.append(
            _atom(
                predicate="proximity",
                subject=f"{side}_hand",
                relation="near",
                object_=f"{side}_{target}" if target != "head" else "head",
                measure=distance,
                measure_unit="torso_length",
                bucket="near",
                provenance=provenance,
            )
        )

    return {
        "posecode_version": POSECODE_VERSION,
        "atom_schema_version": ATOM_SCHEMA_VERSION,
        "coordinate_profile": COORDINATE_PROFILE,
        "measurements": measurements,
        "observed_atoms": atoms,
        "warnings": [],
    }


def _swap_side(value: str) -> str:
    return value.replace("left", "__side__").replace("right", "left").replace("__side__", "right")


def mirror_measurement_report(original: dict[str, float], mirrored: dict[str, float]) -> dict[str, Any]:
    """Validate that scalar posecode measurements swap/reflect across a mirror pair."""
    failures: list[dict[str, Any]] = []
    compared = 0
    for key, original_value in sorted(original.items()):
        if key == "torso_length_bvh_units":
            continue
        mirror_key = _swap_side(key)
        if mirror_key not in mirrored:
            failures.append({"measurement": key, "reason": "missing_mirror_measurement"})
            continue
        expected = float(original_value)
        if "_lateral_" in key or key == "torso_lateral_lean_deg":
            expected = -expected
        actual = float(mirrored[mirror_key])
        tolerance = 1.0 if key.endswith("_deg") else 0.05
        compared += 1
        if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=tolerance):
            failures.append(
                {
                    "measurement": key,
                    "mirror_measurement": mirror_key,
                    "expected": round(expected, 6),
                    "actual": round(actual, 6),
                    "tolerance": tolerance,
                }
            )
    return {
        "status": "pass" if not failures else "needs_review",
        "compared_measurements": compared,
        "failure_count": len(failures),
        "failures": failures,
        "posecode_version": POSECODE_VERSION,
        "coordinate_profile": COORDINATE_PROFILE,
    }


def _mirror_atom_signature(atom: dict[str, Any], *, swap_sides: bool) -> tuple[Any, ...]:
    subject = atom.get("subject")
    object_ = atom.get("object")
    relation = atom.get("relation")
    if swap_sides:
        subject = _swap_side(subject) if isinstance(subject, str) else subject
        object_ = _swap_side(object_) if isinstance(object_, str) else object_
        if atom.get("axis") == "body_lateral" and relation in {"left", "right"}:
            relation = "right" if relation == "left" else "left"
    return (
        atom.get("predicate"),
        subject,
        relation,
        object_,
        atom.get("axis"),
        atom.get("value"),
        atom.get("bucket"),
        atom.get("polarity"),
    )


def mirror_atom_report(original: dict[str, Any], mirrored: dict[str, Any]) -> dict[str, Any]:
    """Validate search-relevant atom buckets after swapping left and right."""
    expected = {
        _mirror_atom_signature(atom, swap_sides=True)
        for atom in original.get("observed_atoms", [])
    }
    actual = {
        _mirror_atom_signature(atom, swap_sides=False)
        for atom in mirrored.get("observed_atoms", [])
    }
    missing = sorted(expected - actual, key=str)
    unexpected = sorted(actual - expected, key=str)
    measurement_report = mirror_measurement_report(
        original.get("measurements", {}), mirrored.get("measurements", {})
    )
    return {
        "status": "pass" if not missing and not unexpected else "needs_review",
        "expected_atom_count": len(expected),
        "actual_atom_count": len(actual),
        "missing_after_mirror": [list(item) for item in missing],
        "unexpected_after_mirror": [list(item) for item in unexpected],
        "measurement_diagnostics": measurement_report,
        "posecode_version": POSECODE_VERSION,
        "coordinate_profile": COORDINATE_PROFILE,
    }


def neutral_atom_key(atom: dict[str, Any]) -> tuple[Any, ...]:
    """Return a direction-neutral key used to intersect original/mirror facts."""
    def neutral(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.replace("left", "side").replace("right", "side")

    relation = atom.get("relation")
    if atom.get("axis") == "body_lateral" and relation in {"left", "right"}:
        relation = "sideways"
    return (
        atom.get("predicate"),
        neutral(atom.get("subject")),
        relation,
        neutral(atom.get("object")),
        atom.get("axis"),
        atom.get("value"),
        atom.get("bucket"),
        atom.get("polarity"),
    )


def common_neutral_atoms(member_atoms: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Intersect member atoms and return side-neutral, measurement-free group facts."""
    atom_lists = list(member_atoms)
    if not atom_lists:
        return []
    maps = [{neutral_atom_key(atom): atom for atom in atoms} for atoms in atom_lists]
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)

    output: list[dict[str, Any]] = []
    for key in sorted(common, key=str):
        atom = dict(maps[0][key])
        for field in ("subject", "object"):
            if field in atom and isinstance(atom[field], str):
                atom[field] = atom[field].replace("left", "side").replace("right", "side")
        if atom.get("axis") == "body_lateral" and atom.get("relation") in {"left", "right"}:
            atom["relation"] = "sideways"
        atom.pop("measure", None)
        atom.pop("measure_unit", None)
        atom["scope"] = "unit"
        output.append(atom)
    return output


def render_posecode_documents(atoms: list[dict[str, Any]]) -> dict[str, str]:
    """Render short deterministic Korean/English review passages from observed atoms."""
    signatures = {(a.get("predicate"), a.get("subject"), a.get("value"), a.get("relation"), a.get("bucket")) for a in atoms}
    ko: list[str] = []
    en: list[str] = []

    if any(p == "torso_lean" and r == "forward" for p, _, _, r, _ in signatures):
        ko.append("상체를 앞으로 숙임")
        en.append("torso leaning forward")
    elif any(p == "torso_lean" and r == "backward" for p, _, _, r, _ in signatures):
        ko.append("상체를 뒤로 젖힘")
        en.append("torso leaning backward")
    elif any(p == "torso_orientation" and v == "upright" for p, _, v, _, _ in signatures):
        ko.append("상체를 세움")
        en.append("upright torso")

    if any(p == "limb_configuration" and s == "both_arms" and v == "widely_spread" for p, s, v, _, _ in signatures):
        ko.append("양팔을 넓게 벌림")
        en.append("both arms spread wide")

    raised_arms = sum(1 for p, s, v, _, _ in signatures if p == "limb_state" and s == "side_arm" and v == "raised")
    if raised_arms:
        ko.append("한쪽 팔을 어깨 위로 듦")
        en.append("one arm raised above the shoulder")

    extended_elbows = sum(1 for p, s, _, _, b in signatures if p == "joint_flexion" and s == "side_elbow" and b == "extended")
    if extended_elbows:
        ko.append("팔꿈치를 폄")
        en.append("extended elbow")

    if any(p == "limb_state" and s == "side_leg" and v == "raised" for p, s, v, _, _ in signatures):
        ko.append("한쪽 다리를 듦")
        en.append("one leg raised")
    if any(p == "relative_direction" and s == "side_ankle" and r == "behind" for p, s, _, r, _ in signatures):
        ko.append("한쪽 발을 골반 뒤에 둠")
        en.append("one foot behind the pelvis")
    if any(p == "foot_spacing" and b == "wide" for p, _, _, _, b in signatures):
        ko.append("두 발을 넓게 벌림")
        en.append("wide foot spacing")

    if not ko:
        return {
            "ko": "BVH 관절 관계로 만든 정적인 전신 자세",
            "en": "a static full-body pose derived from BVH joint relations",
        }
    ko_text = "정적 자세 특징: " + ", ".join(ko)
    en_text = "static pose features: " + ", ".join(en)
    return {"ko": ko_text, "en": en_text}
