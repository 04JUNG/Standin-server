from __future__ import annotations

"""Frozen, three-arm intent-to-treat evaluation for refine v1/v2.4.

This module implements the executable contract in REFINE_V2_DESIGN.md §10.
It deliberately lives beside the legacy ``refine-pairs`` runner so old probes
remain reproducible while the decision-grade protocol can fail closed.
"""

import json
import random
import secrets
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.refine_policy import structural_refine_allowed

from .http_runner import _fetch_binary, _request_json
from .labels import resolve_run
from .refine_mesh import (
    build_mesh_evidence_template_row,
    mesh_contract_metadata,
    require_valid_mesh_evidence_bundle,
)
from .refine_render import (
    RENDERER_VERSION,
    normalize_pose,
    project_bvh,
    render_blind_artifact,
    shared_bounds,
)
from .schemas import METRIC_SCHEMA_VERSION, SCHEMA_VERSION
from .util import (
    hash_json,
    read_json,
    read_jsonl,
    resolve_path,
    sha256_bytes,
    sha256_file,
    slug,
    utc_now,
    write_json,
    write_jsonl,
)


B0 = "B0_no_refine"
B1 = "B1_v1"
B2 = "B2_v24_aggressive"
ARMS = (B0, B1, B2)
CONTRASTS = (
    ("B1_vs_B0", B1, B0),
    ("B2_vs_B0", B2, B0),
    ("B2_vs_B1", B2, B1),
)
_ALLOWED_JOINTS = {
    "left_arm": ("LeftArm", "LeftForeArm"),
    "right_arm": ("RightArm", "RightForeArm"),
    "left_leg": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
    "right_leg": ("RightUpLeg", "RightLeg", "RightFoot"),
}
_LIMB_KEYPOINTS = {
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
}
_PROMOTION_CRITERIA_REQUIRED = {
    "primary_mcid", "primary_ci_low_min", "major_worse_max",
    "changed_major_worse_vs_b0_max", "worst_slice_regression_max",
    "worst_slice_min_n", "minimum_n_eval", "minimum_clusters",
    "worst_slice_cohorts", "usability_rubric_version",
    "human_usable_categories", "minimum_distinct_labelers",
    "minimum_duplicate_fraction", "minimum_hidden_repeat_fraction",
    "analysis_seed", "bootstrap_repetitions", "report_version",
    "new_violation_rate_max", "exact_fallback_rate_min",
    "p95_latency_ms_max", "timeout_error_rate_max",
}


def _url(target: str, path: str | None) -> str | None:
    if not path:
        return None
    return urllib.parse.urljoin(target.rstrip("/") + "/", path.lstrip("/"))


def _validate_preregistered_criteria(criteria: dict) -> None:
    from .refine_report import REPORT_VERSION

    missing = sorted(_PROMOTION_CRITERIA_REQUIRED - set(criteria))
    if missing:
        raise ValueError("missing promotion criteria: " + ", ".join(missing))
    if criteria.get("report_version") != REPORT_VERSION:
        raise ValueError(
            f"report_version must be {REPORT_VERSION!r}"
        )
    if int(criteria["bootstrap_repetitions"]) < 10_000:
        raise ValueError("bootstrap_repetitions must be at least 10000")
    if int(criteria["minimum_distinct_labelers"]) < 2:
        raise ValueError("minimum_distinct_labelers cannot be below 2")
    if float(criteria["minimum_duplicate_fraction"]) < 0.15:
        raise ValueError("minimum_duplicate_fraction cannot be below 0.15")
    if float(criteria["minimum_hidden_repeat_fraction"]) < 0.05:
        raise ValueError("minimum_hidden_repeat_fraction cannot be below 0.05")
    if float(criteria["new_violation_rate_max"]) > 0.0:
        raise ValueError("new_violation_rate_max cannot exceed 0")
    if float(criteria["exact_fallback_rate_min"]) < 1.0:
        raise ValueError("exact_fallback_rate_min cannot be below 1")


def _artifact_id(content_sha256: str | None) -> str | None:
    return f"sha256:{content_sha256}" if content_sha256 else None


def _blind_artifact_id(row: dict) -> str | None:
    geometry = row.get("geometry_sha256")
    return f"geometry:{geometry}" if geometry else row.get("artifact_id")


def _delivery_status(row: dict) -> str:
    """Classify final-artifact delivery without turning failures into visual ties."""
    has_artifact = bool(_blind_artifact_id(row) and row.get("artifact_path"))
    if row.get("timeout") and not has_artifact:
        return "timeout"
    if row.get("error") and not has_artifact:
        return "error"
    if not has_artifact:
        return "artifact_unavailable"
    return "ready"


def _pair_artifact_id(row: dict) -> str:
    return _blind_artifact_id(row) or (
        f"operational:{row.get('arm')}:{_delivery_status(row)}"
    )


def _finite_keypoints(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == (17, 2) and np.isfinite(array).all())


def _finite_scores(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.shape == (17,) and np.isfinite(array).all()
        and np.all(array >= 0.0)
    )


def _person_index(prediction_id: Any) -> int:
    try:
        return int(str(prediction_id).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return -1


def _source_metadata(source: Path, source_manifest: dict) -> tuple[dict, dict]:
    dataset = source_manifest.get("dataset") or {}
    root = dataset.get("root")
    if not root:
        return {}, {}
    dataset_root = resolve_path(root)
    cuts_path = dataset_root / "cuts.jsonl"
    people_path = dataset_root / "persons.jsonl"
    cuts = {
        row["cut_id"]: row for row in read_jsonl(cuts_path)
    } if cuts_path.exists() else {}
    people = {
        row["person_id"]: row for row in read_jsonl(people_path)
    } if people_path.exists() else {}
    return cuts, people


def _source_matches(source: Path) -> set[tuple[str, str, str]]:
    path = source / "matches.jsonl"
    if not path.exists():
        return set()
    return {
        (str(row.get("cut_id")), str(row.get("person_id")), str(row.get("prediction_id")))
        for row in read_jsonl(path)
        if row.get("match_status") == "matched"
        and row.get("cut_id") and row.get("person_id") and row.get("prediction_id")
    }


def _capability(target: str, timeout: float) -> dict:
    status, _, payload = _request_json(
        _url(target, "/healthz"), timeout=min(timeout, 5.0)
    )
    if status != 200 or not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"unhealthy refine server {target}: HTTP {status}")
    capability = payload.get("refine")
    return {
        "target": target,
        "healthz": payload,
        "refine": capability if isinstance(capability, dict) else None,
    }


def _validate_capability(capability: dict, expected_arm: str, strict: bool) -> list[str]:
    refine = capability.get("refine")
    warnings: list[str] = []
    if not isinstance(refine, dict):
        message = f"{expected_arm} /healthz has no refine capability/config identity"
        if strict:
            raise ValueError(message)
        return [message]
    expected_v2 = expected_arm == B2
    config = refine.get("config") if isinstance(refine.get("config"), dict) else {}
    checks = {
        "enabled": refine.get("enabled") is True,
        "v2_enabled": refine.get("v2_enabled") is expected_v2,
        "config_sha256": isinstance(refine.get("config_sha256"), str)
        and len(refine["config_sha256"]) == 64,
        "code_version": isinstance(refine.get("code_version"), str),
        "config": isinstance(refine.get("config"), dict),
        "feature_version": refine.get("feature_version") is not None,
        "pose_library_version": bool(refine.get("pose_library_version")),
        "deployment_version": bool(refine.get("deployment_version")),
        "source_revision": bool(refine.get("source_revision")),
        "non_placeholder_deployment": str(refine.get("deployment_version", "")).lower()
        not in {"", "development", "unknown", "none"},
        "non_placeholder_source_revision": str(refine.get("source_revision", "")).lower()
        not in {"", "development", "unknown", "none"},
        "search_policy_identity": all(
            key in config
            for key in (
                "distance_metric", "fallback_distance", "min_skeleton_score",
                "fallback_pos_full", "fallback_pos_reduced",
                "fallback_angle_full", "fallback_angle_reduced",
                "fallback_hybrid_full", "fallback_hybrid_reduced",
            )
        ),
    }
    if expected_v2:
        modes = refine.get("supported_modes") or []
        checks["aggressive_mode"] = "aggressive" in modes
        checks["torso_default_off"] = refine.get("torso_enabled") is False
        checks["v2_4_code"] = str(refine.get("code_version", "")).startswith("v2.4")
        checks["v2_lower_body_config"] = "refine_v2_lower_body" in config
    else:
        checks["v1_code"] = str(refine.get("code_version", "")).startswith("v1")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        message = f"{expected_arm} capability mismatch: {', '.join(failed)}"
        if strict:
            raise ValueError(message)
        warnings.append(message)
    return warnings


def _base_url(candidate: dict) -> str:
    return str(candidate.get("bvh_url") or (
        f"/pose/{urllib.parse.quote(str(candidate['pose_id']), safe='')}/bvh"
    ))


def _write_artifact(run_dir: Path, unit_id: str, arm: str, data: bytes) -> Path:
    directory = run_dir / "artifacts" / unit_id.removeprefix("unit:")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm}.bvh"
    path.write_bytes(data)
    return path


def _seal_result_files(
    run_dir: Path, relative_paths: list[str], *, run_identity: dict,
) -> dict:
    """Seal every pre-label result, artifact, render, and assignment snapshot."""
    files: dict[str, dict] = {}
    for relative in sorted(set(relative_paths)):
        target = run_dir / relative
        if not target.is_file():
            raise FileNotFoundError(f"cannot seal missing result file: {target}")
        files[relative] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    result = {
        "schema_version": 1,
        "sealed_at": utc_now(),
        "run_identity": run_identity,
        "files": files,
    }
    write_json(run_dir / "result_manifest.json", result)
    return {
        "path": "result_manifest.json",
        "sha256": sha256_file(run_dir / "result_manifest.json"),
        "file_count": len(files),
    }


def _write_proxy_mesh_template(run_dir: Path, rows: list[dict]) -> None:
    """Create a non-promotable worksheet for the external CSP/avatar mesh pass."""
    template = [
        build_mesh_evidence_template_row(
            unit_id=str(row["unit_id"]),
            arm=str(row["arm"]),
            artifact_id=str(_blind_artifact_id(row) or ""),
            geometry_sha256=str(row.get("geometry_sha256") or ""),
        )
        for row in rows
        if row.get("artifact_path")
    ]
    # Use the same public validator as the post-run evidence sealer.  Template
    # mode permits placeholders/incomplete checks but not malformed identity or
    # false checks without stable hard-violation IDs.
    require_valid_mesh_evidence_bundle(
        template, require_complete=False, allow_placeholders=True,
    )
    write_jsonl(run_dir / "mesh_safety_evidence.template.jsonl", template)


def _request_payload(unit: dict, mode: str) -> dict:
    arm = B2 if mode == "aggressive" else B1
    policy = (unit.get("arm_policies") or {}).get(arm) or {
        "eligible": unit.get("common_eligible"),
        "refinable_limbs": unit.get("refinable_limbs", []),
    }
    return {
        "pose_id": unit["pose_id"],
        "view": unit["view"],
        "keypoints": unit["target_keypoints"],
        "scores": unit["target_scores"],
        "search_distance": unit.get("search_distance"),
        "refine_allowed": bool(policy.get("eligible")),
        "refinable_limbs": policy.get("refinable_limbs", []),
        "skeleton_state": unit.get("skeleton_state"),
        "coverage_class": unit.get("coverage_class"),
        "slot_origin": unit.get("slot_origin"),
        "skeleton_source": unit.get("skeleton_source"),
        "search_stability": unit.get("search_stability"),
        "distance_metric": unit.get("distance_metric"),
        "confidence_threshold": unit.get("confidence_threshold"),
        "gap_type": unit.get("gap_type", "unknown"),
        "refine_mode": mode,
    }


def _base_arm(unit: dict) -> dict:
    return {
        "unit_id": unit["unit_id"],
        "arm": B0,
        "artifact_id": _artifact_id(unit["selected_base_sha256"]),
        "artifact_path": unit["base_artifact_path"],
        "artifact_delivered": True,
        "content_sha256": unit["selected_base_sha256"],
        "bvh_sha256": unit["selected_base_sha256"],
        "eligible": bool(unit["common_eligible"]),
        "endpoint_called": False,
        "attempted": False,
        "refined": False,
        "geometry_changed": False,
        "fallback_required": False,
        "exact_base": True,
        "timeout": False,
        "error": False,
        "contract_error": False,
        "latency_ms": 0.0,
        "latency_kind": "already_ready_base",
        "reason": "no_refine",
        "refine_outcome": "unchanged",
        "refine_version": None,
        "mode_requested": "base",
        "mode_applied": "base",
        "aggressive_attempted": False,
        "aggressive_reason": None,
        "backend": "none",
        "limbs": [],
        "adopted_blocks": [],
        "partial_rollback": False,
        "limb_decisions": {},
        "diagnostics": {},
        "automatic_metrics": {},
        "hard_safety_violations": [],
        "ownership_validated": bool(unit.get("ownership_validation", {}).get("valid")),
    }


def _local_fallback_arm(unit: dict, arm: str, reason: str) -> dict:
    row = _base_arm(unit)
    row.update({
        "arm": arm,
        "eligible": False,
        "fallback_required": True,
        "reason": reason,
        "refine_outcome": "not_attempted",
        "mode_requested": "aggressive" if arm == B2 else "conservative",
        "latency_ms": None,
        "latency_kind": "not_called",
    })
    return row


def _run_endpoint(
    *,
    unit: dict,
    arm: str,
    target: str,
    timeout_seconds: float,
    run_dir: Path,
    expected_cache_hit: bool | None,
    expected_capability: dict,
) -> tuple[dict, dict | None]:
    requested_mode = "aggressive" if arm == B2 else "conservative"
    row = _local_fallback_arm(unit, arm, "request_pending")
    policy = (unit.get("arm_policies") or {}).get(arm) or {}
    row.update({
        "eligible": bool(policy.get("eligible", unit["common_eligible"])),
        "endpoint_called": True,
        "eligibility_policy": policy.get("policy", "common_structural"),
    })
    payload = _request_payload(unit, requested_mode)
    started = time.perf_counter()
    timed_out = False
    failed = False
    response: dict | None = None
    try:
        status, _, response = _request_json(
            _url(target, "/refine"), method="POST",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json", timeout=timeout_seconds,
        )
        if status != 200 or not isinstance(response, dict):
            failed = True
            row.update({
                "error": True,
                "contract_error": status == 200,
                "reason": f"http_{status}",
                "exact_base": False,
                "artifact_id": None,
                "artifact_path": None,
                "artifact_delivered": False,
                "content_sha256": None,
                "bvh_sha256": None,
            })
            return row, response

        diagnostics = response.get("diagnostics") or {}
        context = diagnostics.get("context") or {}
        result_url = _url(target, response.get("bvh_url"))
        artifact_bytes = None
        if result_url:
            result_status, fetched = _fetch_binary(result_url, timeout_seconds)
            if result_status == 200:
                artifact_bytes = fetched
        if artifact_bytes is None:
            failed = True
            row.update({
                "error": True,
                "reason": "artifact_fetch_failed",
                "exact_base": False,
                "artifact_id": None,
                "artifact_path": None,
                "artifact_delivered": False,
                "content_sha256": None,
                "bvh_sha256": None,
            })
            return row, response

        result_hash = sha256_bytes(artifact_bytes)
        artifact_path = _write_artifact(run_dir, unit["unit_id"], arm, artifact_bytes)
        outcome = str(response.get("refine_outcome") or (
            "improved" if response.get("refined") else "not_attempted"
        ))
        backend = str(response.get("backend") or "none")
        attempted = bool(
            backend != "none" or outcome in {"improved", "reverted"}
        )
        mode_applied = diagnostics.get("mode_applied") or (
            requested_mode if response.get("refined") else "base"
        )
        exact_base = result_hash == unit["selected_base_sha256"]
        # The denominator is the runtime event that required restoration; it
        # must not depend on restoration having succeeded exactly.
        fallback_required = bool(
            not response.get("refined") or mode_applied == "base"
        )
        reason = str(response.get("reason") or "unknown")
        row.update({
            "artifact_id": _artifact_id(result_hash),
            "artifact_path": str(artifact_path.resolve()),
            "artifact_delivered": True,
            "content_sha256": result_hash,
            "bvh_sha256": result_hash,
            "attempted": attempted,
            "refined": bool(response.get("refined")),
            "geometry_changed": result_hash != unit["selected_base_sha256"],
            "fallback_required": fallback_required,
            "exact_base": exact_base,
            "conservative_fallback": bool(
                arm == B2 and mode_applied == "conservative"
            ),
            "timeout": reason == "timeout",
            "reason": reason,
            "refine_outcome": outcome,
            "refine_version": response.get("refine_version"),
            "mode_requested": diagnostics.get("mode_requested", requested_mode),
            "mode_applied": mode_applied,
            "aggressive_attempted": bool(diagnostics.get("aggressive_attempted")),
            "aggressive_reason": diagnostics.get("aggressive_reason"),
            "backend": backend,
            "limbs": response.get("limbs") or [],
            "adopted_blocks": response.get("limbs") or [],
            "partial_rollback": bool(
                response.get("refined")
                and response.get("limb_decisions")
                and any(
                    isinstance(value, dict)
                    and str(value.get("decision") or value.get("reason") or "").lower()
                    in {"reverted", "rollback", "rejected", "non_regression"}
                    for value in (response.get("limb_decisions") or {}).values()
                )
            ),
            "limb_decisions": response.get("limb_decisions") or {},
            "diagnostics": diagnostics,
            "cache_hit": diagnostics.get("cache_hit"),
        })
        contract_failures = []
        expected_refine = expected_capability.get("refine") or {}
        lineage_checks = {
            "base_bvh_sha256": unit["selected_base_sha256"],
            "refine_config_sha256": expected_refine.get("config_sha256"),
            "feature_version": expected_refine.get("feature_version"),
            "pose_library_version": expected_refine.get("pose_library_version"),
            "deployment_version": expected_refine.get("deployment_version"),
            "source_revision": expected_refine.get("source_revision"),
        }
        for name, expected_value in lineage_checks.items():
            if not expected_value or context.get(name) != expected_value:
                contract_failures.append(f"response_lineage_{name}_mismatch")
        version = str(row.get("refine_version") or "")
        if arm == B1 and not version.startswith("v1"):
            contract_failures.append("expected_v1_response")
        if arm == B2:
            if not version.startswith("v2.4"):
                contract_failures.append("expected_v2_4_response")
            if row["mode_requested"] != "aggressive":
                contract_failures.append("aggressive_request_not_acknowledged")
        if contract_failures:
            row["error"] = row["contract_error"] = True
            row["contract_failures"] = contract_failures
        if (
            expected_cache_hit is not None
            and row.get("cache_hit") is not expected_cache_hit
        ):
            row["error"] = row["contract_error"] = True
            row.setdefault("contract_failures", []).append(
                f"cache_hit_expected_{str(expected_cache_hit).lower()}"
            )
    except Exception as exc:
        timeout = "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
        timed_out = timeout
        failed = not timeout
        row.update({
            "error": not timeout,
            "timeout": timeout,
            "reason": "request_timeout" if timeout else "runner_error",
            "runner_error": f"{type(exc).__name__}: {exc}",
            "exact_base": False,
            "artifact_id": None,
            "artifact_path": None,
            "artifact_delivered": False,
            "content_sha256": None,
            "bvh_sha256": None,
        })
    finally:
        row["latency_ms"] = (time.perf_counter() - started) * 1000.0
        row["latency_kind"] = "cache_hit" if row.get("cache_hit") else "time_to_ready"
        # A return from inside ``try`` evaluates the row before ``finally``.
        # Re-assert these transport flags so the returned object is complete.
        row["timeout"] = bool(row.get("timeout") or timed_out)
        row["error"] = bool(row.get("error") or failed)
    return row, response


def _evaluate_arm(unit: dict, row: dict) -> None:
    path = row.get("artifact_path")
    if not path:
        row["automatic_metrics"] = {"available": False, "reason": "artifact_unavailable"}
        row["hard_safety_violations"] = ["artifact_unavailable"]
        return
    try:
        from .refine_evaluator import evaluate_refine_artifacts

        policy = (unit.get("arm_policies") or {}).get(row.get("arm")) or {}
        allowed_limbs = policy.get("refinable_limbs", unit.get("refinable_limbs", []))
        allowed_joints = [
            joint
            for limb in allowed_limbs
            for joint in _ALLOWED_JOINTS.get(str(limb), ())
        ]
        evaluation = evaluate_refine_artifacts(
            unit["base_artifact_path"], path,
            unit["target_keypoints"], unit["target_scores"], unit["view"],
            synthetic_gt_3d=unit.get("synthetic_gt_3d"),
            allowed_joint_suffixes=allowed_joints,
        )
        row["external_evaluation"] = evaluation
        row["evaluator_version"] = evaluation.get("evaluator_version")
        row["evaluator_config_sha256"] = (
            evaluation.get("evaluator_config", {}).get("config_sha256")
        )
        metrics = evaluation.get("result_metrics") or {}
        row["automatic_metrics"] = {
            name: value.get("value") if isinstance(value, dict) else None
            for name, value in metrics.items()
            if name != "projection"
        }
        identity = evaluation.get("identity") or {}
        if identity.get("geometry_changed") is not None:
            row["geometry_changed"] = bool(identity["geometry_changed"])
        row["geometry_sha256"] = identity.get("result_geometry_sha256")
        row["geometry_equal_base"] = identity.get("geometry_equal")
        safety = evaluation.get("safety") or {}
        new_violations = safety.get("violations") or []
        absolute_violations = safety.get("absolute_violations") or new_violations
        row["new_hard_safety_violations"] = list(new_violations)
        row["hard_safety_violations"] = list(absolute_violations)
        row["safety_evaluator_kind"] = "skeleton_capsule_proxy"
        row["safety_checks_complete"] = not any(
            "unavailable" in str(item.get("type", ""))
            for item in absolute_violations if isinstance(item, dict)
        )
    except Exception as exc:
        row["automatic_metrics"] = {
            "available": False,
            "reason": "evaluator_failure",
            "message": f"{type(exc).__name__}: {exc}",
        }
        row["hard_safety_violations"] = ["evaluator_failure"]


def _freeze_units(
    source: Path,
    source_manifest: dict,
    selected_rank: int,
) -> list[dict]:
    candidates = [
        row for row in read_jsonl(source / "candidates.jsonl")
        if int(row.get("rank", -1)) == selected_rank
        and row.get("person_id")
        and row.get("surfaced") is True
    ]
    candidates.sort(key=lambda row: (str(row.get("cut_id")), str(row.get("person_id"))))
    duplicates = [
        key for key, count in Counter(
            (row.get("cut_id"), row.get("person_id")) for row in candidates
        ).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"selection rule produced duplicate person units: {duplicates[:3]}")
    cuts, gt_people = _source_metadata(source, source_manifest)
    source_matches = _source_matches(source)
    units: list[dict] = []
    dataset = source_manifest.get("dataset") or {}
    for candidate in candidates:
        cut_id = str(candidate["cut_id"])
        person_id = str(candidate["person_id"])
        response_path = source / "responses" / f"{cut_id}.json"
        response = read_json(response_path) if response_path.exists() else {}
        people = response.get("people") or []
        index = _person_index(candidate.get("prediction_id"))
        person = people[index] if 0 <= index < len(people) else {}
        skeleton = person.get("skeleton") or {}
        keypoints = person.get("keypoints") or skeleton.get("keypoints")
        policy_scores = person.get("scores") or skeleton.get("scores")
        raw_scores = person.get("raw_scores")
        scores = raw_scores if _finite_scores(raw_scores) else policy_scores
        raw_scores_available = _finite_scores(raw_scores)
        valid_keypoints = _finite_keypoints(keypoints)
        valid_scores = _finite_scores(scores)
        target_keypoints = keypoints if valid_keypoints else []
        target_scores = list(scores) if valid_scores else []
        quality_trace = person.get("quality_trace") or {}
        refine_mask = quality_trace.get("refine_valid_joint_mask")
        if (
            valid_scores and isinstance(refine_mask, list)
            and len(refine_mask) == 17
        ):
            target_scores = [
                float(score) if bool(keep) else 0.0
                for score, keep in zip(target_scores, refine_mask)
            ]
        if valid_keypoints and valid_scores:
            from .refine_evaluator import query_evidence

            evidence = query_evidence(target_keypoints, target_scores)
        else:
            evidence = {"valid": False, "error": "missing_or_invalid_skeleton"}
        refinable_limbs = [
            limb for limb, indices in _LIMB_KEYPOINTS.items()
            if valid_scores and all(float(target_scores[index]) >= 0.3 for index in indices)
        ]
        ownership_key = (cut_id, person_id, str(candidate.get("prediction_id")))
        ownership_valid = ownership_key in source_matches
        common_eligible = bool(
            evidence.get("valid")
            and structural_refine_allowed(
                skeleton_state=person.get("skeleton_state"),
                coverage_class=person.get("coverage_class"),
                refinable_limbs=refinable_limbs,
                slot_origin=person.get("slot_origin"),
                skeleton_source=person.get("skeleton_source"),
                ownership_valid=ownership_valid,
            )
        )
        cut = cuts.get(cut_id, {})
        gt_person = gt_people.get(person_id, {})
        query_hash = hash_json({
            "keypoints": target_keypoints,
            "scores": target_scores,
            "skeleton_state": person.get("skeleton_state"),
            "coverage_class": person.get("coverage_class"),
            "slot_origin": person.get("slot_origin"),
            "skeleton_source": person.get("skeleton_source"),
        })
        identity = {
            "dataset_id": dataset.get("dataset_id"),
            "person_id": person_id,
            "pose_id": candidate.get("pose_id"),
            "view": candidate.get("view"),
            "selected_rank": selected_rank,
            "query_preprocess_sha256": query_hash,
        }
        unit_id = "unit:" + hash_json(identity)[:20]
        mask = [
            bool(value >= 0.3) for value in target_scores
        ] if valid_scores else [False] * 17
        try:
            search_distance = float(candidate["distance"])
        except (KeyError, TypeError, ValueError):
            search_distance = None
        if search_distance is None:
            search_distance_band = "unknown"
        elif search_distance < 0.15:
            search_distance_band = "lt_0_15"
        elif search_distance < 0.30:
            search_distance_band = "0_15_to_0_30"
        elif search_distance < 0.45:
            search_distance_band = "0_30_to_0_45"
        else:
            search_distance_band = "gte_0_45"
        units.append({
            "unit_id": unit_id,
            "dataset_id": dataset.get("dataset_id"),
            "cut_id": cut_id,
            "person_id": person_id,
            "artist_id": cut.get("artist_id", gt_person.get("artist_id", "unknown")),
            "project_id": cut.get("project_id", gt_person.get("project_id", "unknown")),
            "scene_group_id": cut.get("scene_group_id", "unknown"),
            "source_image_path": cut.get("image_path"),
            "query_image_sha256": cut.get("image_sha256"),
            "query_preprocess_sha256": query_hash,
            "selected_rank": selected_rank,
            "selection_rule": f"deterministic_rank_{selected_rank}",
            "pose_id": str(candidate["pose_id"]),
            "view": str(candidate["view"]),
            "base_bvh_url": _base_url(candidate),
            "source_candidate_artifact_id": candidate.get("candidate_artifact_id"),
            "source_candidate_bvh_sha256": candidate.get("bvh_sha256"),
            "search_distance": search_distance,
            "search_distance_band": search_distance_band,
            "target_keypoints": target_keypoints,
            "target_scores": target_scores,
            "raw_scores": list(raw_scores) if raw_scores_available else None,
            "raw_scores_available": raw_scores_available,
            "target_valid_mask": mask,
            "query_evidence": evidence,
            "query_evidence_sha256": evidence.get("evidence_sha256"),
            "source_refine_allowed": person.get("refine_allowed"),
            "source_refinable_limbs": person.get("refinable_limbs") or [],
            "score_source": "raw_scores" if _finite_scores(raw_scores) else "policy_scores",
            "common_eligible": common_eligible,
            "refinable_limbs": refinable_limbs,
            "foreshortened_limbs": quality_trace.get("foreshortened_limbs") or [],
            "confidence": person.get("confidence", "low"),
            "skeleton_state": person.get("skeleton_state"),
            "coverage_class": person.get("coverage_class"),
            "slot_origin": person.get("slot_origin"),
            "skeleton_source": person.get("skeleton_source"),
            "search_stability": person.get("search_stability"),
            "distance_metric": person.get("distance_metric"),
            "confidence_threshold": person.get("confidence_threshold"),
            "gap_type": person.get("gap_type", "unknown"),
            "pose_type": gt_person.get("pose_type", cut.get("pose_type")),
            "foreshortening_ambiguity": bool(
                quality_trace.get("foreshortened_limbs")
            ),
            "synthetic_gt_3d": (
                gt_person.get("synthetic_gt_3d")
                or candidate.get("synthetic_gt_3d")
            ),
            "ownership_validation": {
                "valid": ownership_valid,
                "source": "frozen_search_person_assignment",
                "match_status": "matched" if ownership_valid else "unverified",
                "prediction_id": candidate.get("prediction_id"),
                "slot_origin": person.get("slot_origin"),
                "skeleton_source": person.get("skeleton_source"),
            },
            "selected_base_sha256": None,
            "base_artifact_path": None,
        })
    return units


def _freeze_arm_policies(units: list[dict], capabilities: dict[str, dict]) -> None:
    """Reapply each release's eligibility to the same frozen raw evidence."""
    v1_config = (capabilities[B1].get("refine") or {}).get("config") or {}
    v2_config = (capabilities[B2].get("refine") or {}).get("config") or {}
    v1_enabled = {"left_arm", "right_arm"}
    if str(v1_config.get("refine_limbs", "arms")).lower() != "arms":
        v1_enabled.update({"left_leg", "right_leg"})
    v2_enabled = {"left_arm", "right_arm"}
    if bool(v2_config.get("refine_v2_lower_body", True)):
        v2_enabled.update({"left_leg", "right_leg"})

    for unit in units:
        common_limbs = set(unit.get("refinable_limbs") or [])
        foreshortened = set(unit.get("foreshortened_limbs") or [])
        v1_limbs = sorted((common_limbs - foreshortened) & v1_enabled)
        v2_limbs = sorted(common_limbs & v2_enabled)
        structural = bool(unit.get("common_eligible"))
        search_distance = unit.get("search_distance")
        try:
            search_distance_value = float(search_distance)
        except (TypeError, ValueError):
            search_distance_value = None
        metric = str(v1_config.get("distance_metric") or "pos").lower()
        coverage = str(unit.get("coverage_class") or "insufficient").lower()
        v1_threshold = v1_config.get(f"fallback_{metric}_{coverage}")
        try:
            v1_threshold_value = float(v1_threshold)
        except (TypeError, ValueError):
            v1_threshold_value = None
        v1_solver_score_ok = False
        if _finite_scores(unit.get("target_scores")):
            body = np.asarray(unit["target_scores"], dtype=np.float64)[5:17]
            v1_solver_score_ok = bool(
                float(np.mean(body)) >= float(v1_config["min_skeleton_score"])
            )
        v1_search_high = bool(
            search_distance_value is not None
            and v1_threshold_value is not None
            and search_distance_value <= v1_threshold_value
            and coverage in {"full", "reduced"}
            and (
                unit.get("skeleton_state") == "valid"
                or unit.get("search_stability") == "stable"
            )
        )
        v1_eligible = bool(
            structural and v1_limbs
            and v1_search_high
        )
        unit["arm_policies"] = {
            B1: {
                "policy": "v1_production_search_and_structural",
                "eligible": v1_eligible,
                "refinable_limbs": v1_limbs,
                "search_distance": search_distance_value,
                "fallback_distance": v1_threshold_value,
                "solver_score_gate_passed": v1_solver_score_ok,
            },
            B2: {
                "policy": "v2_structural",
                "eligible": bool(structural and v2_limbs),
                "refinable_limbs": v2_limbs,
            },
        }


def _freeze_base_artifacts(
    units: list[dict],
    *,
    v1_target: str,
    v2_target: str,
    timeout_seconds: float,
    run_dir: Path,
) -> None:
    for unit in units:
        payloads = []
        for target in (v1_target, v2_target):
            status, data = _fetch_binary(_url(target, unit["base_bvh_url"]), timeout_seconds)
            if status != 200:
                raise RuntimeError(
                    f"failed to freeze base for {unit['unit_id']} from {target}: HTTP {status}"
                )
            payloads.append(data)
        hashes = [sha256_bytes(data) for data in payloads]
        if hashes[0] != hashes[1]:
            raise ValueError(
                f"v1/v2 base mismatch for {unit['unit_id']}: {hashes[0]} != {hashes[1]}"
            )
        expected = unit.get("source_candidate_bvh_sha256")
        if expected and expected != hashes[0]:
            raise ValueError(
                f"source/server base mismatch for {unit['unit_id']}: {expected} != {hashes[0]}"
            )
        path = _write_artifact(run_dir, unit["unit_id"], B0, payloads[0])
        unit["selected_base_sha256"] = hashes[0]
        unit["base_artifact_path"] = str(path.resolve())
        from .refine_evaluator import evaluate_refine_artifacts

        allowed_joints = [
            joint
            for limb in unit.get("refinable_limbs", [])
            for joint in _ALLOWED_JOINTS.get(str(limb), ())
        ]
        frozen_evaluation = evaluate_refine_artifacts(
            path, path, unit["target_keypoints"], unit["target_scores"],
            unit["view"], synthetic_gt_3d=unit.get("synthetic_gt_3d"),
            allowed_joint_suffixes=allowed_joints,
        )
        base_artifact = frozen_evaluation.get("base_artifact") or {}
        if not base_artifact.get("parse_ok"):
            raise ValueError(
                f"common evaluator cannot freeze {unit['unit_id']}: "
                f"base {base_artifact.get('error_stage')} failure: "
                f"{base_artifact.get('error')}"
            )
        base_metrics = frozen_evaluation.get("base_metrics") or {}
        joint_metric = base_metrics.get("joint_nme") or {}
        if (
            frozen_evaluation.get("query_evidence", {}).get("valid") is True
            and not joint_metric.get("available")
        ):
            raise ValueError(
                f"common evaluator cannot freeze {unit['unit_id']}: "
                "base common 2D projection metric is unavailable"
            )
        unit["query_evidence"] = frozen_evaluation["query_evidence"]
        unit["query_evidence_sha256"] = frozen_evaluation["query_evidence"].get(
            "evidence_sha256"
        )
        unit["evaluator_version"] = frozen_evaluation.get("evaluator_version")
        unit["evaluator_config_sha256"] = frozen_evaluation.get(
            "evaluator_config", {}
        ).get("config_sha256")


def _render_and_blind(
    run_dir: Path,
    units: list[dict],
    rows: list[dict],
    *,
    seed: int,
    renderer_version: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    by_unit: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_unit[row["unit_id"]][row["arm"]] = row
    renders = run_dir / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    independent_items: list[dict] = []
    independent_provenance: list[dict] = []
    pair_items: list[dict] = []
    pair_provenance: list[dict] = []
    pair_labels: list[dict] = []
    rng = random.Random(seed)

    for unit in units:
        arm_rows = by_unit[unit["unit_id"]]
        available = [row for row in arm_rows.values() if row.get("artifact_path")]
        try:
            normalized_target = normalize_pose(
                unit["target_keypoints"], unit["target_scores"],
                valid_mask=unit.get("target_valid_mask"),
            )
            target_poses = [normalized_target]
        except (TypeError, ValueError):
            target_poses = []
        safety_poses = []
        safety_view = "side" if unit["view"] == "three_quarter" else "three_quarter"
        for row in available:
            try:
                target_poses.append(project_bvh(
                    row["artifact_path"], unit["view"],
                    valid_mask=unit.get("target_valid_mask"),
                ))
                safety_poses.append(project_bvh(row["artifact_path"], safety_view))
            except Exception:
                continue
        target_bounds = shared_bounds(target_poses)
        safety_bounds = shared_bounds(safety_poses)
        render_by_artifact: dict[str, str] = {}
        arms_by_artifact: dict[str, list[str]] = defaultdict(list)
        for arm in ARMS:
            row = arm_rows[arm]
            artifact_id = _blind_artifact_id(row)
            if not artifact_id or not row.get("artifact_path"):
                continue
            row["blind_artifact_id"] = artifact_id
            arms_by_artifact[artifact_id].append(arm)
            if artifact_id in render_by_artifact:
                row["render_path"] = render_by_artifact[artifact_id]
                row["render_sha256"] = sha256_file(render_by_artifact[artifact_id])
                row["renderer_version"] = renderer_version
                continue
            render_id = hash_json({
                "unit_id": unit["unit_id"], "artifact_id": artifact_id,
                "target_view": unit["view"], "safety_view": safety_view,
                "renderer_version": renderer_version,
            })[:24]
            output = renders / f"{render_id}.svg"
            try:
                render_blind_artifact(
                    artifact_path=row["artifact_path"],
                    target_keypoints=unit["target_keypoints"],
                    target_scores=unit["target_scores"],
                    target_view=unit["view"], safety_view=safety_view,
                    target_bounds=target_bounds, safety_bounds=safety_bounds,
                    output_path=output, renderer_version=renderer_version,
                    target_valid_mask=unit.get("target_valid_mask"),
                    allow_missing_target=not bool(
                        unit.get("query_evidence", {}).get("valid")
                    ),
                )
                render_by_artifact[artifact_id] = str(output.resolve())
                row["render_path"] = str(output.resolve())
                row["render_sha256"] = sha256_file(output)
                row["renderer_version"] = renderer_version
            except Exception as exc:
                row["render_error"] = f"{type(exc).__name__}: {exc}"

        for artifact_id, artifact_arms in sorted(arms_by_artifact.items()):
            item_id = "item:" + hash_json({
                "unit_id": unit["unit_id"], "artifact_id": artifact_id,
            })[:20]
            independent_items.append({
                "item_id": item_id,
                "unit_id": unit["unit_id"],
                "artifact_id": artifact_id,
                "render_path": render_by_artifact.get(artifact_id),
                "render_sha256": next(
                    (
                        arm_rows[arm].get("render_sha256")
                        for arm in artifact_arms
                        if arm_rows[arm].get("render_sha256")
                    ),
                    None,
                ),
                "renderer_version": renderer_version,
                "source_image_path": unit.get("source_image_path"),
            })
            independent_provenance.append({
                "item_id": item_id,
                "unit_id": unit["unit_id"],
                "artifact_id": artifact_id,
                "arms": sorted(artifact_arms),
                "render_path": render_by_artifact.get(artifact_id),
                "render_sha256": next(
                    (
                        arm_rows[arm].get("render_sha256")
                        for arm in artifact_arms
                        if arm_rows[arm].get("render_sha256")
                    ),
                    None,
                ),
                "renderer_version": renderer_version,
            })

        contrasts = list(CONTRASTS)
        rng.shuffle(contrasts)
        for contrast, arm_a, arm_b in contrasts:
            first, second = arm_rows[arm_a], arm_rows[arm_b]
            first_blind = _pair_artifact_id(first)
            second_blind = _pair_artifact_id(second)
            pair_id = "pair:" + hash_json({
                "unit_id": unit["unit_id"], "contrast": contrast,
                "a": first_blind, "b": second_blind,
            })[:20]
            order = [arm_a, arm_b]
            rng.shuffle(order)
            left, right = arm_rows[order[0]], arm_rows[order[1]]
            left_status, right_status = _delivery_status(left), _delivery_status(right)
            operational_failure = left_status != "ready" or right_status != "ready"
            rateable = bool(
                not operational_failure
                and left.get("render_path") and right.get("render_path")
                and left.get("render_sha256") and right.get("render_sha256")
            )
            pair_items.append({
                "pair_id": pair_id,
                "unit_id": unit["unit_id"],
                "left_artifact_id": _pair_artifact_id(left),
                "right_artifact_id": _pair_artifact_id(right),
                "left_render_path": left.get("render_path"),
                "right_render_path": right.get("render_path"),
                "left_render_sha256": left.get("render_sha256"),
                "right_render_sha256": right.get("render_sha256"),
                "renderer_version": renderer_version,
                "rateable": rateable,
                "operational_failure": operational_failure,
                "source_image_path": unit.get("source_image_path"),
            })
            pair_provenance.append({
                "pair_id": pair_id,
                "unit_id": unit["unit_id"],
                "contrast": contrast,
                "left_variant": order[0],
                "right_variant": order[1],
                "left_artifact_id": _pair_artifact_id(left),
                "right_artifact_id": _pair_artifact_id(right),
                "left_render_path": left.get("render_path"),
                "right_render_path": right.get("render_path"),
                "left_render_sha256": left.get("render_sha256"),
                "right_render_sha256": right.get("render_sha256"),
                "renderer_version": renderer_version,
                "rateable": rateable,
                "operational_failure": operational_failure,
                "left_delivery_status": left_status,
                "right_delivery_status": right_status,
            })
            exact = not operational_failure and first_blind == second_blind
            if operational_failure:
                if left_status != "ready" and right_status != "ready":
                    winner = "both_bad"
                elif left_status != "ready":
                    winner = "right"
                else:
                    winner = "left"
                label_source = "automatic_operational_failure"
            else:
                winner = "tie" if exact else "unknown"
                label_source = (
                    "automatic_exact_geometry" if exact else "human_required"
                )
            pair_labels.append({
                "pair_id": pair_id,
                "unit_id": unit["unit_id"],
                "winner": winner,
                "severity": "major" if operational_failure else (
                    "minor" if exact else "unknown"
                ),
                "body_part": "overall" if (exact or operational_failure) else "unknown",
                "safety_violation": "other" if operational_failure else (
                    "none" if exact else "unknown"
                ),
                "labeler_id": label_source if label_source.startswith("automatic_") else "",
                "label_source": label_source,
            })
    rng.shuffle(independent_items)
    rng.shuffle(pair_items)
    rng.shuffle(pair_labels)
    return (
        independent_items, independent_provenance,
        pair_items, pair_provenance, pair_labels,
    )


def _build_label_assignments(
    independent_items: list[dict],
    pair_items: list[dict],
    pair_labels: list[dict],
    *,
    seed: int,
    duplicate_fraction: float = 0.20,
    hidden_repeat_fraction: float = 0.05,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Pre-assign independent, duplicate, and hidden-repeat review rows.

    Repeat lineage is private.  Public templates expose only a random review id,
    so a UI can present the duplicate later without revealing its purpose.
    """
    rng = random.Random(seed)
    pair_template_by_id = {
        str(row["pair_id"]): row for row in pair_labels
    }
    tasks: list[dict] = []
    for item in independent_items:
        tasks.append({
            "task_type": "independent",
            "source_id": str(item["item_id"]),
            "artifact_identity": str(item.get("artifact_id")),
            "public": item,
        })
    for item in pair_items:
        template = pair_template_by_id[str(item["pair_id"])]
        if template.get("label_source") != "human_required":
            continue
        tasks.append({
            "task_type": "pair",
            "source_id": str(item["pair_id"]),
            "artifact_identity": [
                str(item.get("left_artifact_id")),
                str(item.get("right_artifact_id")),
            ],
            "public": item,
        })
    tasks.sort(key=lambda row: (row["task_type"], row["source_id"]))

    primary_by_key: dict[tuple[str, str], dict] = {}
    assignments: list[dict] = []

    def append_assignment(task: dict, kind: str, slot: str, repeat_of=None) -> None:
        assignment_id = "assignment:" + hash_json({
            "seed": seed,
            "task_type": task["task_type"],
            "source_id": task["source_id"],
            "kind": kind,
            "ordinal": len(assignments),
        })[:24]
        row = {
            "assignment_id": assignment_id,
            "task_type": task["task_type"],
            "source_id": task["source_id"],
            "assignment_kind": kind,
            "rater_slot": slot,
            "repeat_of_assignment_id": repeat_of,
            "artifact_identity": task["artifact_identity"],
        }
        assignments.append(row)
        if kind == "primary":
            primary_by_key[(task["task_type"], task["source_id"])] = row

    for task in tasks:
        append_assignment(task, "primary", "primary")

    def selected(fraction: float) -> list[dict]:
        if not tasks or fraction <= 0.0:
            return []
        count = max(1, int(round(len(tasks) * fraction)))
        return rng.sample(tasks, min(count, len(tasks)))

    for task in selected(duplicate_fraction):
        append_assignment(task, "duplicate", "secondary")
    for task in selected(hidden_repeat_fraction):
        primary = primary_by_key[(task["task_type"], task["source_id"])]
        append_assignment(
            task, "hidden_repeat", "primary",
            repeat_of=primary["assignment_id"],
        )
    rng.shuffle(assignments)

    independent_templates: list[dict] = []
    human_pair_templates: list[dict] = []
    public_queue: list[dict] = []
    independent_by_id = {
        str(row["item_id"]): row for row in independent_items
    }
    pair_by_id = {str(row["pair_id"]): row for row in pair_items}
    for assignment in assignments:
        assignment_id = assignment["assignment_id"]
        source_id = assignment["source_id"]
        if assignment["task_type"] == "independent":
            item = independent_by_id[source_id]
            independent_templates.append({
                "assignment_id": assignment_id,
                "item_id": source_id,
                "unit_id": item["unit_id"],
                "artifact_id": item["artifact_id"],
                "overall_usability": "unknown",
                "reject_reason": None,
                "safety_violation": "unknown",
                "labeler_id": "",
                "label_source": "human_required",
            })
            public_queue.append({
                "assignment_id": assignment_id,
                "task_type": "independent",
                **item,
            })
        else:
            item = pair_by_id[source_id]
            human_pair_templates.append({
                "assignment_id": assignment_id,
                "pair_id": source_id,
                "unit_id": item["unit_id"],
                "winner": "unknown",
                "severity": "unknown",
                "body_part": "unknown",
                "safety_violation": "unknown",
                "labeler_id": "",
                "label_source": "human_required",
            })
            public_queue.append({
                "assignment_id": assignment_id,
                "task_type": "pair",
                **item,
            })
    automatic_pairs = [
        dict(row) for row in pair_labels
        if row.get("label_source") != "human_required"
    ]
    rng.shuffle(independent_templates)
    rng.shuffle(human_pair_templates)
    rng.shuffle(public_queue)
    return (
        assignments,
        public_queue,
        independent_templates,
        automatic_pairs + human_pair_templates,
    )


def run_refine_evaluation(
    *,
    v1_target: str,
    v2_target: str,
    from_run: str | Path,
    output_root: str | Path = "out/eval/runs",
    run_id: str | None = None,
    timeout_seconds: float = 30.0,
    seed: int = 20260805,
    selected_rank: int = 1,
    strict_capabilities: bool = True,
    renderer_version: str = RENDERER_VERSION,
    expected_cache_hit: bool | None = False,
    promotion_criteria: str | Path | dict | None = None,
) -> Path:
    """Execute frozen B0/B1/B2 final-artifact evaluation on one selected pose/person."""
    if selected_rank < 1:
        raise ValueError("selected_rank must be >= 1")
    if not v1_target or not v2_target:
        raise ValueError("both v1_target and v2_target are required")
    source = resolve_run(from_run)
    source_manifest = read_json(source / "manifest.json")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{now}-{slug('refine-three-arm')}"
    run_dir = resolve_path(Path(output_root) / run_id)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    criteria_value: dict | None = None
    criteria_source_sha256: str | None = None
    frozen_criteria_path = run_dir / "promotion_criteria.frozen.json"
    if promotion_criteria is not None:
        if isinstance(promotion_criteria, dict):
            criteria_value = promotion_criteria
        else:
            criteria_source = resolve_path(promotion_criteria)
            criteria_value = read_json(criteria_source)
            criteria_source_sha256 = sha256_file(criteria_source)
        if not isinstance(criteria_value, dict) or not criteria_value:
            raise ValueError("promotion criteria must be a non-empty JSON object")
        _validate_preregistered_criteria(criteria_value)
        # Freeze a canonical local copy before contacting either evaluated server.
        write_json(frozen_criteria_path, criteria_value)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "refine_eval_schema_version": 1,
        "run_id": run_id,
        "mode": "refine_three_arm",
        "created_at": utc_now(),
        "command": " ".join(sys.argv),
        "source_run_id": source_manifest.get("run_id"),
        "dataset": source_manifest.get("dataset"),
        "source_artifacts": source_manifest.get("artifacts"),
        "arms": list(ARMS),
        "confirmatory_primary": "B2_vs_B1",
        "selection_rule": f"deterministic_rank_{selected_rank}",
        "seed": seed,
        "blind_randomization": {
            "method": "private_seeded_rng",
            "seed_public": False,
            "note": "label-order seed is stored only in private provenance metadata",
        },
        "renderer": {
            "version": renderer_version,
            "body": "COCO17 fixed stick figure",
            "camera": "orthographic target view + fixed safety view",
        },
        "mesh_safety_contract": mesh_contract_metadata(),
        "cache_policy": {
            "expected_cache_hit": expected_cache_hit,
            "latency_basis": "cache_off_post_click" if expected_cache_hit is False
            else "mixed_or_unspecified",
        },
        "promotion_criteria": {
            "preregistered": criteria_value is not None,
            "frozen_path": (
                str(frozen_criteria_path.resolve()) if criteria_value is not None else None
            ),
            "sha256": (
                sha256_file(frozen_criteria_path) if criteria_value is not None else None
            ),
            "source_sha256": criteria_source_sha256,
            "frozen_before_server_contact": criteria_value is not None,
        },
        "strict_capabilities": bool(strict_capabilities),
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)
    errors: list[dict] = []
    try:
        capabilities = {
            B1: _capability(v1_target, timeout_seconds),
            B2: _capability(v2_target, timeout_seconds),
        }
        warnings = []
        warnings.extend(_validate_capability(capabilities[B1], B1, strict_capabilities))
        warnings.extend(_validate_capability(capabilities[B2], B2, strict_capabilities))
        v1_refine = capabilities[B1].get("refine") or {}
        v2_refine = capabilities[B2].get("refine") or {}
        shared_identity = (
            ("pose_library_version", "pose-library"),
            ("feature_version", "feature"),
            ("source_revision", "source-revision"),
        )
        for field, label in shared_identity:
            if (
                v1_refine.get(field) is not None
                and v2_refine.get(field) is not None
                and v1_refine[field] != v2_refine[field]
            ):
                raise ValueError(f"v1/v2 servers expose different {label} identities")
        manifest["servers"] = capabilities
        manifest["capability_warnings"] = warnings
        write_json(run_dir / "manifest.json", manifest)

        units = _freeze_units(source, source_manifest, selected_rank)
        if not units:
            raise ValueError(f"source run has no person candidate at rank {selected_rank}")
        _freeze_arm_policies(units, capabilities)
        _freeze_base_artifacts(
            units, v1_target=v1_target, v2_target=v2_target,
            timeout_seconds=timeout_seconds, run_dir=run_dir,
        )
        write_jsonl(run_dir / "frozen_units.jsonl", units)
        write_json(run_dir / "frozen_manifest.json", {
            "schema_version": 1,
            "source_run_id": source_manifest.get("run_id"),
            "selection_rule": f"deterministic_rank_{selected_rank}",
            "n_eval": len(units),
            "units_sha256": sha256_file(run_dir / "frozen_units.jsonl"),
            "evaluator_versions": sorted({
                unit.get("evaluator_version") for unit in units
                if unit.get("evaluator_version")
            }),
            "evaluator_config_sha256": sorted({
                unit.get("evaluator_config_sha256") for unit in units
                if unit.get("evaluator_config_sha256")
            }),
            "promotion_criteria_sha256": (
                sha256_file(frozen_criteria_path) if criteria_value is not None else None
            ),
            "promotion_criteria_preregistered": criteria_value is not None,
            "sealed_at": utc_now(),
        })

        rows: list[dict] = []
        unit_by_id = {unit["unit_id"]: unit for unit in units}
        for unit in units:
            row = _base_arm(unit)
            _evaluate_arm(unit, row)
            rows.append(row)

        tasks = [(unit["unit_id"], arm) for unit in units for arm in (B1, B2)]
        random.Random(seed).shuffle(tasks)
        for unit_id, arm in tasks:
            unit = unit_by_id[unit_id]
            if not (_finite_keypoints(unit["target_keypoints"])
                    and _finite_scores(unit["target_scores"])):
                row = _local_fallback_arm(unit, arm, "invalid_or_missing_frozen_skeleton")
                response = None
            else:
                row, response = _run_endpoint(
                    unit=unit, arm=arm,
                    target=v1_target if arm == B1 else v2_target,
                    timeout_seconds=timeout_seconds, run_dir=run_dir,
                    expected_cache_hit=expected_cache_hit,
                    expected_capability=capabilities[arm],
                )
            _evaluate_arm(unit, row)
            rows.append(row)
            if row.get("error") or row.get("timeout"):
                errors.append({
                    "unit_id": unit_id,
                    "arm": arm,
                    "kind": "timeout" if row.get("timeout") else "arm_error",
                    "message": row.get("runner_error") or row.get("reason"),
                })
        rows.sort(key=lambda row: (row["unit_id"], ARMS.index(row["arm"])))

        blind_seed = secrets.randbits(64)
        manifest["blind_randomization"]["seed_commitment"] = hash_json({
            "run_id": run_id, "blind_seed": blind_seed,
        })
        (
            independent_items, independent_provenance,
            pair_items, pair_provenance, pair_labels,
        ) = _render_and_blind(
            run_dir, units, rows, seed=blind_seed,
            renderer_version=renderer_version,
        )
        for row in independent_provenance:
            row["blind_seed"] = blind_seed
        for row in pair_provenance:
            row["blind_seed"] = blind_seed
        assignment_seed = secrets.randbits(64)
        manifest["label_assignment"] = {
            "duplicate_fraction": 0.20,
            "hidden_repeat_fraction": 0.05,
            "seed_public": False,
            "seed_commitment": hash_json({
                "run_id": run_id, "assignment_seed": assignment_seed,
            }),
        }
        (
            label_assignments, public_label_queue,
            independent_labels, pair_labels,
        ) = _build_label_assignments(
            independent_items, pair_items, pair_labels, seed=assignment_seed,
        )

        write_jsonl(run_dir / "refine_arms.jsonl", rows)
        # Compatibility alias for early harness consumers.
        write_jsonl(run_dir / "refine_pairs.jsonl", rows)
        write_jsonl(run_dir / "refine_independent_items.jsonl", independent_items)
        write_jsonl(
            run_dir / "refine_independent_provenance.private.jsonl",
            independent_provenance,
        )
        write_jsonl(
            run_dir / "refine_independent_labels_template.jsonl",
            independent_labels,
        )
        write_jsonl(run_dir / "refine_pair_items.jsonl", pair_items)
        write_jsonl(
            run_dir / "refine_pair_provenance.private.jsonl", pair_provenance,
        )
        write_jsonl(run_dir / "refine_pair_labels_template.jsonl", pair_labels)
        write_jsonl(run_dir / "refine_label_queue.jsonl", public_label_queue)
        write_jsonl(
            run_dir / "refine_label_assignments.private.jsonl", label_assignments,
        )
        write_jsonl(run_dir / "errors.jsonl", errors)
        _write_proxy_mesh_template(run_dir, rows)
        write_json(run_dir / "blind_randomization.private.json", {
            "run_id": run_id,
            "blind_seed": blind_seed,
            "seed_commitment": manifest["blind_randomization"]["seed_commitment"],
        })
        write_json(run_dir / "label_assignment.private.json", {
            "run_id": run_id,
            "assignment_seed": assignment_seed,
            "seed_commitment": manifest["label_assignment"]["seed_commitment"],
        })

        sealed_paths = [
            "frozen_units.jsonl", "frozen_manifest.json", "refine_arms.jsonl",
            "refine_pairs.jsonl",
            "refine_independent_items.jsonl",
            "refine_independent_provenance.private.jsonl",
            "refine_independent_labels_template.jsonl",
            "refine_pair_items.jsonl", "refine_pair_provenance.private.jsonl",
            "refine_pair_labels_template.jsonl", "errors.jsonl",
            "refine_label_queue.jsonl", "refine_label_assignments.private.jsonl",
            "blind_randomization.private.json", "label_assignment.private.json",
            "mesh_safety_evidence.template.jsonl",
        ]
        if frozen_criteria_path.exists():
            sealed_paths.append("promotion_criteria.frozen.json")
        sealed_paths.extend(
            str(path.relative_to(run_dir))
            for path in sorted((run_dir / "artifacts").rglob("*.bvh"))
        )
        sealed_paths.extend(
            str(path.relative_to(run_dir))
            for path in sorted((run_dir / "renders").glob("*.svg"))
        )
        manifest["result_seal"] = _seal_result_files(
            run_dir, sealed_paths,
            run_identity={
                "run_id": run_id,
                "dataset": manifest.get("dataset"),
                "servers": manifest.get("servers"),
                "renderer": manifest.get("renderer"),
                "cache_policy": manifest.get("cache_policy"),
                "promotion_criteria": manifest.get("promotion_criteria"),
                "blind_randomization": manifest.get("blind_randomization"),
                "label_assignment": manifest.get("label_assignment"),
                "strict_capabilities": manifest.get("strict_capabilities"),
                "capability_warnings": manifest.get("capability_warnings"),
            },
        )

        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["counts"] = {
            "n_eval": len(units),
            "arms": len(rows),
            "common_eligible": sum(bool(unit["common_eligible"]) for unit in units),
            "attempted": sum(bool(row["attempted"]) for row in rows if row["arm"] != B0),
            "geometry_changed": sum(bool(row["geometry_changed"]) for row in rows),
            "timeouts": sum(bool(row["timeout"]) for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "independent_items": len(independent_items),
            "blind_pairs": len(pair_items),
            "automatic_ties": sum(
                row.get("label_source") == "automatic_exact_geometry"
                for row in pair_labels
            ),
        }
        manifest["raw_query_evidence_complete"] = all(
            bool(unit.get("raw_scores_available")) for unit in units
        )
        write_json(run_dir / "manifest.json", manifest)
        try:
            from .refine_report import write_refine_report

            write_refine_report(
                run_dir,
                independent_labels_path=(
                    run_dir / "refine_independent_labels_template.jsonl"
                ),
                pair_labels_path=run_dir / "refine_pair_labels_template.jsonl",
                promotion_criteria=(
                    frozen_criteria_path if criteria_value is not None else None
                ),
            )
        except Exception as exc:
            errors.append({
                "kind": "report_generation",
                "message": f"{type(exc).__name__}: {exc}",
            })
            write_jsonl(run_dir / "errors.jsonl", errors)
        return run_dir
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "manifest.json", manifest)
        raise
