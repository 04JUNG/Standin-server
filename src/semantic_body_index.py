"""Offline, versioned BVH observations. Never estimates a rough's coordinates.

Coordinates are the existing library's Y-up COCO17 mapping. Thresholds are
experimental, with dead bands: ambiguous measurements stay unknown. We do not
guess forward/backward lean from a BVH's exporter-dependent facing direction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .body_language import validated_observations

BODY_INDEX_VERSION = "bvh-body-v2-torso-relative-arms"
THRESHOLDS = {
    "knee_bent_max_deg": 140.0, "knee_straight_min_deg": 160.0,
    "legs_apart_min_hip_widths": 2.0, "legs_together_max_hip_widths": 1.35,
    "stride_min_torso_lengths": 0.7,
    "hand_raised_min_torso_lengths": 0.15,
    "hand_lowered_max_torso_lengths": -0.55,
    "elbow_lowered_max_torso_lengths": -0.25,
    "arm_spread_min_torso_lengths": 0.8,
    "torso_upright_max_deg": 12.0,
}


def _angle(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1))))


def describe_library_body(joints, scores) -> dict:
    p = np.asarray(joints, dtype=float)
    s = np.asarray(scores, dtype=float)
    if p.shape != (17, 3) or s.shape != (17,):
        raise ValueError("invalid COCO17 shape")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("nonfinite COCO17")
    observations = {k: "unknown" for k in ("knees", "legs", "arms", "torso")}
    metrics = {}

    def valid(*indices):
        # These are deterministic BVH mapping-presence scores, NOT RTMPose scores.
        return all(s[i] >= 0.3 for i in indices)

    knees = []
    for label, hip, knee, ankle in (("left", 11, 13, 15), ("right", 12, 14, 16)):
        angle = _angle(p[hip] - p[knee], p[ankle] - p[knee]) if valid(hip, knee, ankle) else None
        metrics[label + "_knee_deg"] = angle
        knees.append("unknown" if angle is None else
                     "bent" if angle <= THRESHOLDS["knee_bent_max_deg"] else
                     "straight" if angle >= THRESHOLDS["knee_straight_min_deg"] else "unknown")
    if "unknown" not in knees:
        observations["knees"] = knees[0] if knees[0] == knees[1] else "mixed"

    if valid(5, 6, 11, 12):
        shoulder = (p[5] + p[6]) / 2
        hip = (p[11] + p[12]) / 2
        torso = shoulder - hip
        length = float(np.linalg.norm(torso))
        lateral = p[12] - p[11]
        lateral[1] = 0
        width = float(np.linalg.norm(lateral))
        tilt = _angle(torso, np.array([0., 1., 0.]))
        metrics["torso_tilt_deg"] = tilt
        if tilt is not None and tilt <= THRESHOLDS["torso_upright_max_deg"]:
            observations["torso"] = "upright"
        if length > 1e-8 and width > 1e-8:
            lateral /= width
            forward = np.cross(lateral, [0., 1., 0.])
            if valid(13, 14, 15, 16):
                separation = max(abs(float(np.dot(p[r] - p[l], lateral)))
                                 for l, r in ((13, 14), (15, 16))) / width
                stride = abs(float(np.dot(p[16] - p[15], forward))) / length
                metrics.update(leg_separation_hip_widths=separation,
                               stride_torso_lengths=stride)
                if separation >= THRESHOLDS["legs_apart_min_hip_widths"]:
                    observations["legs"] = "apart"
                elif stride >= THRESHOLDS["stride_min_torso_lengths"]:
                    observations["legs"] = "staggered"
                elif separation <= THRESHOLDS["legs_together_max_hip_widths"]:
                    observations["legs"] = "together"
        # Arms use the body's own hip-to-shoulder axis, not world gravity.
        # A prone person's hands below world-Y shoulders are not necessarily
        # lowered along the torso. Dead-band positions remain unknown.
        if length > 1e-8:
            up = torso / length
            arm_lateral = p[6] - p[5]
            arm_lateral -= np.dot(arm_lateral, up) * up
            arm_width = float(np.linalg.norm(arm_lateral))
            if arm_width <= 1e-8:
                return {"observations": observations,
                        "metrics": {k: None if v is None else round(v, 6) for k, v in metrics.items()}}
            arm_lateral /= arm_width
            arm_states = []
            for label, sh, elbow, hand in (("left", 5, 7, 9), ("right", 6, 8, 10)):
                state = "unknown"
                if valid(sh, elbow, hand):
                    height = float(np.dot(p[hand] - p[sh], up) / length)
                    eh = float(np.dot(p[elbow] - p[sh], up) / length)
                    spread = abs(float(np.dot(p[hand] - p[sh], arm_lateral))) / length
                    metrics.update({label + "_hand_height": height,
                                    label + "_elbow_height": eh,
                                    label + "_hand_lateral": spread})
                    if height >= THRESHOLDS["hand_raised_min_torso_lengths"]:
                        state = "raised"
                    elif (height <= THRESHOLDS["hand_lowered_max_torso_lengths"]
                          and eh <= THRESHOLDS["elbow_lowered_max_torso_lengths"]):
                        state = "lowered"
                    elif spread >= THRESHOLDS["arm_spread_min_torso_lengths"] and abs(height) <= 0.25:
                        state = "spread"
                arm_states.append(state)
            if "unknown" not in arm_states:
                observations["arms"] = (arm_states[0] if arm_states[0] == arm_states[1] else "mixed")
    return {"observations": observations,
            "metrics": {k: None if v is None else round(v, 6) for k, v in metrics.items()}}


def records_digest(records) -> str:
    """Bind the cache to descriptions/tags as well as pose IDs; no stale corpus mix."""
    payload = json.dumps(sorted(records, key=lambda r: r["pose_id"]),
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_body_index(path, records) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != BODY_INDEX_VERSION or data.get("thresholds") != THRESHOLDS:
        raise ValueError("body index version/threshold mismatch")
    if data.get("frame") != 0:
        raise ValueError("body index representative frame mismatch")
    if data.get("tags_sha256") != records_digest(records):
        raise ValueError("body index corpus mismatch")
    poses = data.get("poses")
    if not isinstance(poses, dict) or not poses:
        raise ValueError("body index empty")
    if not set(poses).issubset({r["pose_id"] for r in records}):
        raise ValueError("body index contains unrelated pose IDs")
    for pose_id, value in poses.items():
        if not isinstance(pose_id, str) or not isinstance(value, dict):
            raise ValueError("invalid body record")
        obs = value.get("observations")
        if (not isinstance(obs, dict) or len(obs) != 4
                or len(validated_observations(obs)) != 4):
            raise ValueError("invalid body observations")
        if not isinstance(value.get("metrics"), dict):
            raise ValueError("invalid body metrics")
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not np.isfinite(v)) for v in value["metrics"].values()):
            raise ValueError("invalid body metric value")
        digest = value.get("source_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid BVH provenance")
    return poses
