"""Bounded observed-arm source/target experiment with byte-exact rollback.

No VLM, camera fitting, profile mutation, root or hidden-chain optimization.
A source improvement is explicitly not final FBX or skin-mesh certification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .bvh import coco17_from_fk, find_joint, fk, parse_bvh, rotation_channel_indices
from .refine_projection import FixedProjection, ProjectionDomainError
from .refine_regional import CHAINS, validate_regional_binding
from .collision import arm_torso_penetration, arm_leg_penetration

SOLVER_VERSION = "refine-v3-observed-arm-experimental-v1"
ARM_JOINTS = {"left_arm": (("LeftArm",), ("LeftForeArm", "LeftElbow")),
              "right_arm": (("RightArm",), ("RightForeArm", "RightElbow"))}


class _Deadline(Exception):
    pass


@dataclass
class RegionalRefineResult:
    refined: bool = False
    reason: str = "regional_disabled"
    bvh_path: str | None = None
    source_bvh_sha256: str | None = None
    result_bvh_sha256: str | None = None
    mesh_status: str = "mesh_unverified"
    target_status: str = "unsupported"
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(vars(self))


class _MotionDocument:
    """Patch only first-frame numeric token spans; copy every other byte."""
    def __init__(self, raw, nchannels):
        self.raw = raw
        header = re.search(rb"(?<![A-Za-z0-9_:])MOTION\s+Frames:\s+(\d+)\s+Frame\s+Time:\s+[^\s]+", raw)
        if not header:
            raise ValueError("invalid BVH motion header")
        self.tokens = list(re.finditer(rb"\S+", raw[header.end():]))
        self.offset = header.end()
        if len(self.tokens) != int(header.group(1)) * nchannels:
            raise ValueError("BVH motion token count mismatch")
        self.nchannels = nchannels

    def render(self, values, allowed):
        updates = {i: (format(float(values[i]), ".12g")).encode("ascii") for i in allowed}
        parts, start = [], 0
        for i in sorted(updates):
            token = self.tokens[i]
            left, right = self.offset + token.start(), self.offset + token.end()
            parts.extend((self.raw[start:left], updates[i]))
            start = right
        parts.append(self.raw[start:])
        return b"".join(parts)

    def frozen_bytes(self, allowed):
        # Includes hierarchy, frame headers, original whitespace and every
        # inactive channel across all frames, including translations.
        parts, start = [], 0
        for i in sorted(allowed):
            token = self.tokens[i]
            left, right = self.offset + token.start(), self.offset + token.end()
            parts.extend((self.raw[start:left], b"<ACTIVE>"))
            start = right
        parts.append(self.raw[start:])
        return b"".join(parts)


def _joint_rotation(joint, values, starts):
    channels = joint[3]
    angles = [values[starts + i] for i, channel in enumerate(channels) if channel.endswith("rotation")]
    axes = "".join(channel[0] for channel in channels if channel.endswith("rotation"))
    return Rotation.from_euler(axes, angles, degrees=True)


def _arm_map(joints, limbs):
    indices, joint_indices = [], []
    for limb in limbs:
        for names in ARM_JOINTS[limb]:
            found = [find_joint(joints, name) for name in names]
            joint = next((i for i in found if i >= 0), -1)
            if joint < 0:
                raise ValueError("unsupported arm rig")
            rotations = rotation_channel_indices(joints, joint)
            if len(rotations) != 3:
                raise ValueError("unsupported arm rotation channels")
            indices.extend(rotations)
            joint_indices.append(joint)
    if len(indices) != len(set(indices)):
        raise ValueError("overlapping arm rig")
    return indices, joint_indices


def _source_points(joints, values):
    points, scores = coco17_from_fk(joints, fk(joints, values))
    if not np.all(scores[[5, 6, 7, 8, 9, 10]] > 0):
        raise ValueError("incomplete source arm mapping")
    return points.astype(float)


def _branch_preserved(base, candidate, limbs):
    for limb in limbs:
        shoulder, elbow, wrist = CHAINS[limb]
        old = np.cross(base[elbow] - base[shoulder], base[wrist] - base[elbow])
        new = np.cross(candidate[elbow] - candidate[shoulder], candidate[wrist] - candidate[elbow])
        # A straight baseline has no pole to preserve. Cumulative rotation and
        # endpoint nonregression still bound the newly introduced bend.
        if np.linalg.norm(old) > 1e-8 and float(np.dot(old, new)) < -1e-10:
            return False
    return True


def refine_regional_bvh(source_path, keypoints, binding, provenance, ownership, *,
                        out_path=None, enabled=False, target_adapter=None,
                        target_profile_sha256=None, deadline_seconds=1.0,
                        max_rotation_degrees=12.0):
    """Compute a source candidate or an exactly certified target candidate.

    Caller supplies the selected B0/P file, never a different retrieval result.
    An unsupported/failed target adapter returns that exact input, rather than
    silently reporting source-only improvement as target success. A target-only
    experiment may use any binding-authorized complete arms, still preserving
    source geometry and all inactive bytes. Final mesh/FBX validation is separate.
    """
    started = time.monotonic()
    raw = Path(source_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    result = RegionalRefineResult(source_bvh_sha256=digest, result_bvh_sha256=digest)
    result.diagnostics = {"solver_version": SOLVER_VERSION, "vlm_calls": 0,
                          "source_improvement_is_fbx_certification": False}

    def finish(reason, payload=raw, refined=False):
        result.reason, result.refined = reason, refined
        result.result_bvh_sha256 = hashlib.sha256(payload).hexdigest()
        result.diagnostics["elapsed_ms"] = (time.monotonic() - started) * 1000
        if out_path is not None:
            destination = Path(out_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replacement also breaks a pre-existing hard link to B0;
            # opening that link for writes would mutate the selected source.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent,
                        prefix=".refine-v3-", delete=False) as handle:
                    temporary = handle.name
                    handle.write(payload)
                os.replace(temporary, destination)
                temporary = None
            finally:
                if temporary is not None:
                    os.unlink(temporary)
            result.bvh_path = str(out_path)
        else:
            result.bvh_path = str(source_path) if not refined else None
        return result

    if out_path is not None and Path(out_path).resolve() == Path(source_path).resolve():
        result.reason, result.bvh_path = "output_overwrites_source", str(source_path)
        return result
    decision = validate_regional_binding(binding,
        pose_id=binding.get("pose_id") if isinstance(binding, dict) else None, source_bvh_sha256=digest,
        keypoints=keypoints, provenance=provenance, ownership=ownership, enabled=enabled)
    if not decision.allowed:
        return finish(decision.reason)
    if (not np.isfinite(deadline_seconds) or deadline_seconds <= 0 or
            not np.isfinite(max_rotation_degrees) or not 0 < max_rotation_degrees <= 12):
        return finish("invalid_solver_budget")
    deadline = started + deadline_seconds

    def check_time():
        if time.monotonic() >= deadline:
            raise _Deadline()

    try:
        check_time()
        joints, frames = parse_bvh(str(source_path))
        if not len(frames) or not np.isfinite(frames).all():
            return finish("invalid_source_frame")
        base = frames[0].copy()
        indices, arm_joints = _arm_map(joints, decision.limbs)
        doc = _MotionDocument(raw, len(base))
        base_source = _source_points(joints, base)
        rotation = np.asarray(binding["camera"]["rotation"], float)
        target_mode = target_adapter is not None
        if target_mode:
            from .target_profile import certified_adapter
            if not certified_adapter(target_adapter, target_profile_sha256, digest):
                return finish("target_adapter_uncertified")
            if getattr(target_adapter, "coordinate_frame", None) != "source-canonical-m":
                return finish("target_coordinate_frame_unsupported")
            result.target_status = "certified_baseline"

        def points_for(values):
            check_time()
            if target_mode:
                data = doc.render(values, indices)
                evaluated = target_adapter.evaluate(data.decode("utf-8-sig"))
                points = np.asarray(evaluated.coco17, float)
                if points.shape != (17, 3) or not np.isfinite(points).all():
                    raise ValueError("invalid target FK")
                # Adapter coordinates have an explicitly declared frame. Only
                # source-canonical meters are accepted by this projection.
                if getattr(target_adapter, "coordinate_frame", None) != "source-canonical-m":
                    raise ValueError("unsupported target coordinate frame")
            else:
                points = _source_points(joints, values)
            return points @ rotation.T

        base_points = points_for(base)
        projection = FixedProjection.from_aligned_base(base_points,
            binding["camera"].get("distance_torso"))
        base_xy = projection.project(base_points).astype(float)
        width = float(np.linalg.norm(base_xy[5] - base_xy[6]))
        physical_width = float(np.linalg.norm(base_points[5] - base_points[6]))
        if width < 1e-7 or np.linalg.norm((base_points[5] - base_points[6])[:2]) < .03 * physical_width:
            return finish("degenerate_projected_shoulders")
        center = base_xy[[5, 6]].mean(axis=0)
        query = np.asarray(keypoints, float)
        normalized_query = (query - np.asarray(binding["query_shoulder_center"])) / binding["query_shoulder_width"]
        observed = [i for limb in decision.limbs for i in CHAINS[limb][1:]]

        def projected(values):
            return (projection.project(points_for(values)).astype(float) - center) / width

        base_projected = projected(base)
        base_errors = np.linalg.norm(base_projected - normalized_query, axis=1)
        result.diagnostics.update({"limbs": list(decision.limbs),
            "frozen_lower_body": True, "fixed_camera": True,
            "baseline_shoulder_center": center.tolist(), "baseline_shoulder_width": width,
            "baseline_observed_error": float(np.mean(base_errors[observed]))})
        # Only numerically observable axes are opened. This does not certify
        # anatomy; it avoids optimizing null-space twist from 2D evidence.
        active = []
        for i in indices:
            probe = base.copy(); probe[i] += .1
            if np.linalg.norm(projected(probe)[observed] - base_projected[observed]) > 1e-5:
                active.append(i)
        if not active:
            return finish("no_observable_arm_dof")
        def values_for(delta):
            values = base.copy(); values[active] += delta
            return values
        def residual(delta):
            values = values_for(delta)
            fitted = projected(values)[observed] - normalized_query[observed]
            return np.r_[fitted.ravel(), .0005 * delta]
        def jacobian(delta):
            # Source COCO/projection uses float32. SciPy's default epsilon at
            # zero rounds away all movement; use a fixed physical angle step.
            columns = []
            for parameter in range(len(delta)):
                plus, minus = delta.copy(), delta.copy()
                plus[parameter] += .05; minus[parameter] -= .05
                columns.append((residual(plus) - residual(minus)) / .1)
            return np.column_stack(columns)
        component_bound = max_rotation_degrees / np.sqrt(3)
        solved = least_squares(residual, np.zeros(len(active)),
            bounds=(-component_bound, component_bound), jac=jacobian, max_nfev=32,
            ftol=1e-5, xtol=1e-5, gtol=1e-6, diff_step=.01)
        check_time()
        candidate = values_for(solved.x)
        # Independent serialized-file parser and FK audit: no reliance on the
        # numerical solver's success flag or cached candidate metric.
        payload = doc.render(candidate, active)
        with tempfile.TemporaryDirectory(prefix="refine-v3-audit-") as directory:
            audit_path = Path(directory) / "candidate.bvh"
            audit_path.write_bytes(payload)
            audited_joints, audited_frames = parse_bvh(str(audit_path))
        check_time()
        audit_doc = _MotionDocument(payload, len(base))
        if doc.frozen_bytes(active) != audit_doc.frozen_bytes(active):
            return finish("serialized_frozen_bytes_changed")
        if (not np.isfinite(audited_frames).all() or
                not np.array_equal(frames[:, [i for i in range(len(base)) if i not in active]],
                                   audited_frames[:, [i for i in range(len(base)) if i not in active]]) or
                not np.array_equal(frames[1:], audited_frames[1:])):
            return finish("serialized_frozen_channels_changed")
        starts = np.cumsum([0] + [len(j[3]) for j in joints])[:-1]
        for index in arm_joints:
            before = _joint_rotation(joints[index], base, starts[index])
            after = _joint_rotation(audited_joints[index], audited_frames[0], starts[index])
            if np.rad2deg((before.inv() * after).magnitude()) > max_rotation_degrees + 1e-6:
                return finish("rotation_budget_exceeded")
        source_candidate = _source_points(audited_joints, audited_frames[0])
        proxy_checks = {}
        for limb in decision.limbs:
            measures = [("torso", arm_torso_penetration, (limb,)),
                        ("left_leg", arm_leg_penetration, (limb, "left_leg")),
                        ("right_leg", arm_leg_penetration, (limb, "right_leg"))]
            for name, measure, arguments in measures:
                old, new = measure(base_source, *arguments), measure(source_candidate, *arguments)
                proxy_checks[limb + ":" + name] = {"available": old.available and new.available,
                    "base_depth": old.depth, "candidate_depth": new.depth}
                if not old.available or not new.available or new.depth > old.depth + 1e-5:
                    result.diagnostics["source_collision_proxy"] = proxy_checks
                    return finish("source_collision_proxy_regression")
        result.diagnostics["source_collision_proxy"] = proxy_checks
        if not _branch_preserved(base_source, source_candidate, decision.limbs):
            return finish("source_pole_branch_changed")
        if target_mode:
            if not certified_adapter(target_adapter, target_profile_sha256,
                                     hashlib.sha256(payload).hexdigest()):
                return finish("target_candidate_uncertified")
        final_projected = projected(audited_frames[0])
        errors = np.linalg.norm(final_projected - normalized_query, axis=1)
        # Per-chain wrist and total error must not regress; an average cannot
        # hide a worse wrist on the other observed arm.
        for limb in decision.limbs:
            _, elbow, wrist = CHAINS[limb]
            if (errors[wrist] > base_errors[wrist] + .001 or
                    np.mean(errors[[elbow, wrist]]) > np.mean(base_errors[[elbow, wrist]]) + .001):
                return finish("observed_chain_regression")
        gain = float(np.mean(base_errors[observed]) - np.mean(errors[observed]))
        result.diagnostics.update({"final_observed_error": float(np.mean(errors[observed])),
                                   "gain": gain, "nfev": solved.nfev,
                                   "active_channels": active,
                                   "serialized_audit_passed": True})
        if gain <= .0001:
            return finish("insufficient_gain")
        if target_mode:
            # Source nonregression is independent of the target objective.
            source_projection = FixedProjection.from_aligned_base(
                base_source @ rotation.T, binding["camera"].get("distance_torso"))
            source_base_xy = source_projection.project(base_source @ rotation.T)
            source_new_xy = source_projection.project(source_candidate @ rotation.T)
            source_center = source_base_xy[[5, 6]].mean(axis=0)
            source_width = np.linalg.norm(source_base_xy[5] - source_base_xy[6])
            if source_width < 1e-7:
                return finish("source_projection_degenerate")
            old_errors = np.linalg.norm((source_base_xy - source_center) / source_width - normalized_query, axis=1)
            new_errors = np.linalg.norm((source_new_xy - source_center) / source_width - normalized_query, axis=1)
            if any(new_errors[CHAINS[n][2]] > old_errors[CHAINS[n][2]] + .001 or
                   np.mean(new_errors[list(CHAINS[n][1:])]) >
                   np.mean(old_errors[list(CHAINS[n][1:])]) + .001 for n in decision.limbs):
                return finish("source_endpoint_regression")
            result.target_status = "certified_candidate_geometry"
        check_time()
        return finish("target_geometry_improved" if target_mode else "source_candidate_improved",
                      payload, True)
    except _Deadline:
        return finish("deadline_exceeded")
    except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError,
            FloatingPointError, ImportError, ProjectionDomainError) as exc:
        result.diagnostics["failure_type"] = type(exc).__name__
        result.diagnostics["failure_reason"] = str(exc)
        return finish("solver_validation_failed")
