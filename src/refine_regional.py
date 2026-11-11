"""Fail-closed local-arm capability, independent of the full-body refine gate.

This module accepts trusted server observations/camera lookup only. Its hashes
bind immutable input; they are not signatures or evidence of human visibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .refine_projection import query_sha256, search_rotation

POLICY_VERSION = "regional-observed-arms-v1"
CAMERA_VERSION = "regional-search-camera-v1"
CHAINS = {"left_arm": (5, 7, 9), "right_arm": (6, 8, 10)}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _records(provenance):
    return [p.to_serializable() if hasattr(p, "to_serializable") else
            asdict(p) if is_dataclass(p) else dict(p) for p in provenance or ()]


@dataclass(frozen=True)
class RegionalDecision:
    allowed: bool
    reason: str
    binding: dict | None = None
    limbs: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def regional_search_camera(*, pose_id, source_bvh_sha256, keypoints, rotation,
                           projection="orthographic", distance_torso=None,
                           camera_index=None):
    """Create a camera from a server-verified original hit's index camera.

    Caller must compare this lookup with the selected original library hit.
    No query pelvis or invented hidden point is required for this new contract.
    """
    r = np.asarray(rotation, float)
    if (r.shape != (3, 3) or not np.isfinite(r).all() or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(r), 1.0, atol=1e-7)):
        raise ValueError("proper camera rotation required")
    if projection not in {"orthographic", "perspective"}:
        raise ValueError("unsupported projection")
    if projection == "perspective" and (distance_torso is None or
            not math.isfinite(distance_torso) or distance_torso < .8):
        raise ValueError("invalid fixed perspective distance")
    if projection == "orthographic" and distance_torso is not None:
        raise ValueError("orthographic camera cannot carry perspective distance")
    return {"version": CAMERA_VERSION, "pose_id": pose_id,
            "source_bvh_sha256": source_bvh_sha256,
            "query_sha256": query_sha256(keypoints), "rotation": r.tolist(),
            "projection": projection, "distance_torso": distance_torso,
            "camera_index": camera_index, "reference": "verified-library-index",
            "labels_swapped": False}


def camera_from_verified_hit(*, candidate, geometry, indexed_camera,
                             source_bvh_sha256, keypoints):
    """Check trusted retrieval lineage before emitting the new camera contract."""
    if (candidate.get("kind") != "library_original" or
            candidate.get("review_required") is True or
            candidate.get("constraints_satisfied") is False or
            candidate.get("match_level") in {"near_match", "geometry_fallback"} or
            candidate.get("pose_id") != geometry.get("pose_id") or
            candidate.get("camera") != indexed_camera or
            geometry.get("camera") != indexed_camera or
            type(geometry.get("camera_index")) is not int or
            geometry["camera_index"] < 0 or not source_bvh_sha256):
        raise ValueError("unverified original library camera lineage")
    distance = indexed_camera.get("distance")
    return regional_search_camera(
        pose_id=candidate["pose_id"], source_bvh_sha256=source_bvh_sha256,
        keypoints=keypoints,
        rotation=search_rotation(indexed_camera["yaw"], indexed_camera["pitch"]),
        projection="orthographic" if distance is None else "perspective",
        distance_torso=distance, camera_index=geometry["camera_index"])


def build_regional_binding(*, pose_id, source_bvh_sha256, keypoints, provenance,
                           camera, ownership, enabled=False):
    """Authorize only complete, directly observed same-person arm chains.

    The internal caller must supply server-owned lineage. Do not expose this
    constructor on a public request accepting arbitrary provenance/ownership.
    """
    if not enabled:
        return RegionalDecision(False, "regional_disabled")
    try:
        points = np.asarray(keypoints, float)
        query_hash = query_sha256(points)
        rows = _records(provenance)
        if (not isinstance(source_bvh_sha256, str) or len(source_bvh_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in source_bvh_sha256)):
            return RegionalDecision(False, "missing_source_digest")
        if (not isinstance(ownership, Mapping) or
                ownership.get("slot_origin") != "vlm" or
                ownership.get("ownership_valid") is not True or
                ownership.get("side_labels_known") is not True or
                not ownership.get("person_id")):
            return RegionalDecision(False, "unverified_ownership")
        if (camera.get("version") != CAMERA_VERSION or
                camera.get("reference") != "verified-library-index" or
                camera.get("pose_id") != pose_id or
                camera.get("source_bvh_sha256") != source_bvh_sha256 or
                camera.get("query_sha256") != query_hash or
                camera.get("labels_swapped") is not False):
            return RegionalDecision(False, "camera_lineage_mismatch")
        # Reconstruct to independently validate numeric camera fields.
        checked = regional_search_camera(pose_id=pose_id,
            source_bvh_sha256=source_bvh_sha256, keypoints=points,
            rotation=camera["rotation"], projection=camera["projection"],
            distance_torso=camera.get("distance_torso"),
            camera_index=camera.get("camera_index"))
        if camera != checked:
            return RegionalDecision(False, "camera_contract_mismatch")
        by_index = {r["index"]: r for r in rows}
        if len(rows) != 17 or len(by_index) != 17 or set(by_index) != set(range(17)):
            return RegionalDecision(False, "incomplete_provenance")
        direct = set()
        for i, row in by_index.items():
            if (row.get("source") == "full_image" and
                    row.get("visibility") in {"visible", "observed", "direct_observed"} and
                    row.get("person_id") == ownership["person_id"] and
                    np.array_equal(np.asarray(row.get("current")), points[i]) and
                    np.array_equal(np.asarray(row.get("original")), points[i]) and
                    math.isfinite(float(row.get("current_score", 0))) and
                    float(row.get("current_score", 0)) > 0):
                direct.add(i)
        if not {5, 6}.issubset(direct):
            return RegionalDecision(False, "shoulders_not_directly_observed")
        width = float(np.linalg.norm(points[5] - points[6]))
        extent = float(np.linalg.norm(np.ptp(points[list(direct)], axis=0)))
        if width < 1e-6 or width < .03 * extent:
            return RegionalDecision(False, "degenerate_observed_shoulders")
        limbs = tuple(name for name, chain in CHAINS.items() if set(chain).issubset(direct))
        missing = tuple("endpoint_only_unsupported:" + name for name, chain in CHAINS.items()
                        if chain[0] in direct and chain[2] in direct and chain[1] not in direct)
        if not limbs:
            return RegionalDecision(False, "no_complete_direct_arm", diagnostics=missing)
        observed = sorted({5, 6}.union(*(set(CHAINS[n]) for n in limbs)))
        payload = {"version": POLICY_VERSION, "pose_id": pose_id,
                   "source_bvh_sha256": source_bvh_sha256,
                   "query_sha256": query_hash,
                   "provenance_sha256": canonical_sha256(rows),
                   "ownership_sha256": canonical_sha256(dict(ownership)),
                   "camera": checked, "limbs": list(limbs),
                   "observed_joint_mask": [i in observed for i in range(17)],
                   "query_shoulder_center": points[[5, 6]].mean(axis=0).tolist(),
                   "query_shoulder_width": width,
                   "normalization": "fixed-baseline-shoulder-v1"}
        payload["binding_sha256"] = canonical_sha256(payload)
        return RegionalDecision(True, "ok", payload, limbs, missing)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return RegionalDecision(False, "malformed_regional_input")


def validate_regional_binding(binding, *, pose_id, source_bvh_sha256,
                              keypoints, provenance, ownership, enabled=False):
    if not isinstance(binding, dict):
        return RegionalDecision(False, "missing_regional_binding")
    decision = build_regional_binding(pose_id=pose_id,
        source_bvh_sha256=source_bvh_sha256, keypoints=keypoints,
        provenance=provenance, camera=binding.get("camera", {}),
        ownership=ownership, enabled=enabled)
    if decision.allowed and decision.binding != binding:
        return RegionalDecision(False, "regional_binding_mismatch")
    return decision
