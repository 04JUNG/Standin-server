"""Versioned experiment extension over existing sealed HTTP/replay runs.

No provider call is performed by default. HTTP is opt-in via an explicit adapter
and target; images are never uploaded by this refine-only adapter. Product
flags, assets and endpoints are not changed by the harness.
"""
from __future__ import annotations

import copy
import json
import math
import secrets
import time
import urllib.parse
from pathlib import Path

from .arms import Arm, ArmRegistry, DEFAULT_REGISTRY, EXTENSION_SCHEMA, STATES
from .dataset import load_dataset, validate_dataset
from .http_runner import _fetch_binary, _request_json
from .labels import resolve_run
from .refine_evaluator import evaluate_refine_artifacts
from .refine_three_arm import _build_label_assignments, _freeze_units
from .search_evidence import rows_if_present, source_person, source_candidate, summarize_search_run
from .util import (hash_json, read_json, read_jsonl, resolve_path, sha256_bytes,
                   sha256_file, utc_now, write_json, write_jsonl)


class FixtureAdapter:
    """Offline recorded outcomes; even all-pass fixtures cannot certify quality."""
    execution_kind = "fixture"

    def __init__(self, outcomes: dict):
        self.outcomes = outcomes

    def execute(self, unit: dict, arm: Arm, timeout_seconds: float) -> dict:
        outcome = self.outcomes.get((unit["unit_id"], arm.arm_id),
                                   self.outcomes.get(arm.arm_id))
        if outcome is None:
            return {"state": "missing_dependency", "reason": "fixture_not_recorded"}
        if isinstance(outcome, Exception):
            raise outcome
        result = copy.deepcopy(outcome)
        if result.get("bvh_path") is not None:
            result["bvh_bytes"] = resolve_path(result.pop("bvh_path")).read_bytes()
        return result


class HTTPRefineAdapter:
    execution_kind = "http"

    def __init__(self, target: str):
        if urllib.parse.urlparse(target).scheme not in {"http", "https"}:
            raise ValueError("HTTP adapter needs an explicit http(s) target")
        self.target = target

    def _url(self, path: str) -> str:
        return urllib.parse.urljoin(self.target.rstrip("/") + "/", path.lstrip("/"))

    def base_bytes(self, unit: dict, timeout_seconds: float) -> bytes:
        status, data = _fetch_binary(self._url(unit["base_bvh_url"]), timeout_seconds)
        if status != 200:
            raise RuntimeError(f"base_fetch_http_{status}")
        return data

    def execute(self, unit: dict, arm: Arm, timeout_seconds: float) -> dict:
        started = time.monotonic()
        status, _, health = _request_json(self._url("/healthz"), timeout=timeout_seconds)
        if status != 200 or not health or health.get("ok") is not True:
            return {"state": "missing_dependency", "reason": "unhealthy_endpoint"}
        capability = health.get("refine") or {}
        if arm.adapter == "http_v3":
            v3 = health.get("refine_v3")
            return {"state": "unsupported" if v3 else "missing_dependency",
                    "reason": "v3_has_no_applied_output_contract" if v3 else "v3_absent",
                    "diagnostics": {"v3_capability": v3}, "actual_backend": "none"}
        version = str(capability.get("code_version", ""))
        if arm.expected_version_prefix and not version.startswith(arm.expected_version_prefix):
            return {"state": "unsupported", "reason": "algorithm_version_mismatch",
                    "diagnostics": {"actual_version": version}}
        if capability.get("enabled") is not True:
            return {"state": "missing_dependency", "reason": "refine_disabled"}
        if arm.mode and arm.mode not in (capability.get("supported_modes") or []):
            return {"state": "unsupported", "reason": "refine_mode_unsupported"}
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("health_check_exhausted_deadline")
        payload = {key: unit.get(key) for key in (
            "pose_id", "view", "search_distance", "skeleton_state", "coverage_class",
            "slot_origin", "skeleton_source", "search_stability", "distance_metric",
            "confidence_threshold", "gap_type", "lower_body_observed")}
        payload.update(keypoints=unit["target_keypoints"], scores=unit.get("policy_scores", unit["target_scores"]),
                       refine_allowed=unit.get("source_refine_allowed"),
                       refinable_limbs=unit.get("source_refinable_limbs", []),
                       refine_mode=arm.mode, render_thumbnail=False)
        if unit.get("camera") is not None:
            payload["camera"] = unit["camera"]
        if unit.get("refine_v3") is not None:
            payload["refine_v3"] = unit["refine_v3"]
        status, _, response = _request_json(self._url("/refine"), method="POST",
            data=json.dumps(payload).encode(), content_type="application/json", timeout=remaining)
        if status != 200 or not response:
            return {"state": "error", "reason": f"refine_http_{status}"}
        if response.get("refine_version") != version:
            return {"state": "fail", "reason": "response_algorithm_version_mismatch"}
        context = (response.get("diagnostics") or {}).get("context") or {}
        if context.get("base_bvh_sha256") != unit["selected_base_sha256"]:
            return {"state": "fail", "reason": "selected_base_mismatch", "diagnostics": response.get("diagnostics")}
        expected_context = {key: capability.get(key) for key in (
            "pose_library_version", "feature_version", "source_revision", "deployment_version")}
        expected_context["refine_config_sha256"] = capability.get("config_sha256")
        if any(value is None or context.get(key) != value for key, value in expected_context.items()):
            return {"state": "fail", "reason": "server_binding_changed_or_missing"}
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("refine_exhausted_deadline")
        url = response.get("bvh_url")
        if not url:
            return {"state": "error", "reason": "result_artifact_unavailable"}
        status, data = _fetch_binary(self._url(url), remaining)
        if status != 200:
            return {"state": "error", "reason": f"result_fetch_http_{status}"}
        return {"state": "pass" if response.get("refined") else "abstained",
                "reason": response.get("reason"), "refined": response.get("refined") is True,
                "bvh_bytes": data, "actual_backend": response.get("backend"),
                "bindings": {"source_revision": context.get("source_revision"),
                             "db": context.get("pose_library_version"),
                             "refine_config_sha256": context.get("refine_config_sha256")},
                "diagnostics": response.get("diagnostics"), "refine_version": response.get("refine_version")}


def _policy(value: dict) -> dict:
    policy = copy.deepcopy(value)
    for field in ("timeout_seconds", "refine_budget_ms", "search_budget_ms"):
        number = policy.get(field)
        if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float))
                                   or not math.isfinite(number) or number <= 0):
            raise ValueError(f"{field} must be finite and positive")
    if not policy.get("timeout_seconds"):
        raise ValueError("preregistered timeout_seconds is required; no inherited SLO")
    return policy


def _freeze(source: Path, dataset, manifest: dict, selected_rank: int, selections: list[dict]) -> list[dict]:
    selected = {(r["cut_id"], r["gt_person_id"]): r for r in selections}
    if len(selected) != len(selections):
        raise ValueError("duplicate selections for person")
    rich = {(r["cut_id"], r["person_id"]): r for r in _freeze_units(source, manifest, selected_rank)}
    candidates = rows_if_present(source, "candidates.jsonl")
    matches = rows_if_present(source, "matches.jsonl")
    units = []
    for person in sorted(dataset.target_persons, key=lambda r: (r["cut_id"], r["person_id"])):
        key = (person["cut_id"], person["person_id"])
        unit = copy.deepcopy(rich.get(key) or {})
        unit.pop("source_image_path", None)
        candidate = next((c for c in candidates if (c.get("cut_id"), c.get("person_id")) == key
                          and c.get("rank") == selected_rank and c.get("surfaced") is True), {})
        observed = source_person(source, candidate, matches) if candidate else {}
        candidate = source_candidate(candidate, observed)
        selection = selected.get(key, {})
        if selection.get("mirror") not in (None, candidate.get("mirror")):
            raise ValueError("selection mirror differs from frozen candidate")
        # A selection cannot silently switch the algorithm's selected candidate.
        if selection.get("candidate_artifact_id") not in (None, candidate.get("candidate_artifact_id")):
            raise ValueError("selection candidate differs from frozen rank rule")
        unit.update(unit_id="unit:" + hash_json({"dataset": dataset.dataset_id,
                    "cut_id": key[0], "gt_person_id": key[1], "rank": selected_rank})[:24],
                    cut_id=key[0], gt_person_id=key[1], person_id=key[1],
                    selection_revision=selection.get("selection_revision"),
                    body_revision=selection.get("body_revision"), hand_revision=selection.get("hand_revision"),
                    output_scope=selection.get("output_scope"),
                    profile_id=selection.get("profile_id"), mirror=candidate.get("mirror"),
                    camera=selection.get("camera", candidate.get("camera")),
                    camera_basis="explicit_camera" if selection.get("camera", candidate.get("camera")) else "legacy_fixed_view",
                    candidate_artifact_id=candidate.get("candidate_artifact_id"),
                    joint_provenance=observed.get("joint_provenance", []),
                    refine_evidence=observed.get("refine_evidence"),
                    refine_v3=observed.get("refine_v3"),
                    lower_body_observed=observed.get("lower_body_observed"),
                    policy_scores=observed.get("scores") or (observed.get("skeleton") or {}).get("scores"),
                    source_selected=bool(candidate), bindings=selection.get("bindings", {}))
        unit["selected_base_sha256"] = selection.get("selected_base_sha256", candidate.get("bvh_sha256"))
        unit["base_artifact_path"] = selection.get("base_artifact_path")
        unit["comparison_context_sha256"] = hash_json({key: unit.get(key) for key in (
            "selected_base_sha256", "camera", "mirror", "view", "query_preprocess_sha256", "joint_provenance", "policy_scores",
            "selection_revision", "body_revision", "hand_revision", "output_scope")})
        units.append(unit)
    return units


def _source_snapshot(source: Path, labels_path=None, dataset_root: Path | None = None) -> dict:
    """Content identities only; changes/additions/deletions invalidate the run."""
    paths = {source / "manifest.json", *source.glob("*.jsonl"),
             *(source / "responses").rglob("*.json")}

    def digest(path):
        try:
            return sha256_file(path)
        except OSError:
            return "unavailable"

    value = {"source_files": {path.relative_to(source).as_posix(): digest(path)
                               for path in sorted(paths)},
             "human_labels_sha256": digest(Path(labels_path)) if labels_path else None}
    if dataset_root is not None:
        value["dataset_files"] = {name: digest(dataset_root / name)
                                  for name in ("dataset.json", "cuts.jsonl", "persons.jsonl")}
    return value


def run_experiment(*, from_run, policy, output_root="out/eval/runs", run_id=None,
                   registry: ArmRegistry | dict | None = None, adapters: dict | None = None,
                   dataset=None, selections=None, labels_path=None) -> Path:
    """Run all preregistered persons x explicit arms, preserving failed/missing rows.

    ``selections`` is JSONL or rows with cut_id, gt_person_id, revision IDs,
    body/hand/scope, selected_base_sha256 and local base_artifact_path. These
    paths stay private. Adapters may be injected for offline contract tests.
    """
    source = resolve_run(from_run)
    source_snapshot = _source_snapshot(source, labels_path)
    source_manifest = read_json(source / "manifest.json")
    dataset = load_dataset(dataset or source_manifest["dataset"]["root"])
    source_snapshot["dataset_files"] = _source_snapshot(source, labels_path, dataset.root)["dataset_files"]
    problems = [i.message for i in validate_dataset(dataset) if i.level == "error"]
    if problems:
        raise ValueError("dataset is not sealed/valid: " + "; ".join(problems))
    for key, actual in (("cut_manifest_sha256", dataset.actual_cut_hash), ("gt_sha256", dataset.actual_person_hash)):
        if source_manifest.get("dataset", {}).get(key) != actual:
            raise ValueError("source run and dataset snapshot mismatch: " + key)
    policy = _policy(read_json(policy) if not isinstance(policy, dict) else policy)
    registry = registry or DEFAULT_REGISTRY
    if isinstance(registry, dict):
        registry = ArmRegistry.from_dict(registry)
    adapters = adapters or {}
    selected = read_jsonl(selections) if selections and not isinstance(selections, list) else (selections or [])
    units = _freeze(source, dataset, source_manifest, int(policy.get("selected_rank", 1)), selected)
    # Consume search/camera/label evidence before any adapter can contact or
    # mutate the source. Verify the complete snapshot again after those reads.
    search = summarize_search_run(source, dataset, labels_path=labels_path, budget_ms=policy.get("search_budget_ms"))
    snapshot_after_reads = _source_snapshot(source, labels_path, dataset.root)
    if int(policy.get("selected_rank", 1)) < 1:
        raise ValueError("selected_rank must be positive")
    requested_source = source_manifest.get("requested_backend") or {}
    actual_source = source_manifest.get("actual_backend") or {}
    source_backend_verified = bool(requested_source and actual_source)
    source_backend_mismatch = False
    for key, requested in requested_source.items():
        actual = actual_source.get(key)
        values = actual if isinstance(actual, list) else [actual]
        source_backend_verified = source_backend_verified and bool(actual)
        source_backend_mismatch = source_backend_mismatch or bool(actual and any(v != requested for v in values))
    groups = {(r.get("artist_id"), r.get("project_id"), r.get("scene_group_id")) for r in dataset.cuts
              if all(isinstance(r.get(k), str) and r[k] and not r[k].startswith(("unknown", "unresolved:"))
                     for k in ("artist_id", "project_id", "scene_group_id"))}
    run_id = run_id or "experiment-" + secrets.token_hex(8)
    output = resolve_path(Path(output_root) / run_id)
    output.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": EXTENSION_SCHEMA, "registry": registry.to_dict(),
              "policy": policy, "source_manifest_sha256": source_snapshot["source_files"]["manifest.json"],
              "source_snapshot": source_snapshot, "source_snapshot_sha256": hash_json(source_snapshot),
              "search_evidence_sha256": hash_json(search),
              "units_sha256": hash_json(units), "expected_rows": len(units) * len(registry.arms),
              "expected_unit_ids": [unit["unit_id"] for unit in units],
              "source_execution_kind": source_manifest.get("mode"),
              "source_backend_verified": source_backend_verified,
              "source_backend_mismatch": source_backend_mismatch,
              "source_requested_backend": requested_source, "source_actual_backend": actual_source,
              "source_scene_groups": len(groups),
              "independent_groups": len({(g[0], g[1]) for g in groups}),
              "dataset": {"dataset_id": dataset.dataset_id, "cuts_sha256": dataset.actual_cut_hash,
                          "persons_sha256": dataset.actual_person_hash}}
    # Freeze before adapter contact, including base fetching.
    write_json(output / "experiment.frozen.json", frozen)
    write_jsonl(output / "frozen_units.private.jsonl", units)
    write_json(output / "search_evidence.json", search)
    records = []
    for unit in units:
        base_bytes = None
        base_error = None
        try:
            if unit.get("base_artifact_path"):
                base_bytes = resolve_path(unit["base_artifact_path"]).read_bytes()
            elif unit.get("source_selected"):
                base_adapter = next((a for a in adapters.values() if hasattr(a, "base_bytes")), None)
                if base_adapter:
                    base_bytes = base_adapter.base_bytes(copy.deepcopy(unit), policy["timeout_seconds"])
            if base_bytes is not None and not unit.get("selected_base_sha256"):
                base_error = "selected_base_hash_missing"
                base_bytes = None
            elif base_bytes is not None and sha256_bytes(base_bytes) != unit.get("selected_base_sha256"):
                base_error = "selected_base_hash_mismatch"
                base_bytes = None
        except Exception:
            base_error = "selected_base_unavailable"
        artifact_dir = output / "artifacts" / unit["unit_id"].split(":", 1)[1]
        if base_bytes is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "base.bvh").write_bytes(base_bytes)
        for arm in registry.arms:
            row = {"schema_version": EXTENSION_SCHEMA, "unit_id": unit["unit_id"],
                   "cut_id": unit["cut_id"], "gt_person_id": unit["gt_person_id"], "arm": arm.arm_id,
                   "selection_revision": unit.get("selection_revision"), "artifact_id": None,
                   "body_revision": unit.get("body_revision"), "hand_revision": unit.get("hand_revision"),
                   "output_scope": unit.get("output_scope"),
                   "camera_sha256": hash_json({"camera": unit.get("camera"), "view": unit.get("view")}),
                   "selection_key": {"unit_id": unit["unit_id"], "arm": arm.arm_id},
                   "comparison_context_sha256": unit["comparison_context_sha256"],
                   "selected_base_sha256": unit.get("selected_base_sha256"),
                   "requested_backend": arm.requested_backend, "actual_backend": None,
                   "bindings": dict(unit.get("bindings", {})),
                   "expected_bindings": arm.bindings,
                   "query_measurement_basis": "model_coordinates_and_scores; visibility_not_certified_by_confidence",
                   "execution_kind": "not_executed", "state": "unknown", "reason": None,
                   "elapsed_ms": None, "exact_recovery": "unknown", "full_mesh_safety": "unknown",
                   "quality": "unknown", "proxy_safety": "unknown"}
            if not unit["source_selected"]:
                row.update(state="unknown", reason="selected_candidate_unavailable")
            elif not unit.get("ownership_validation", {}).get("valid"):
                row.update(state="abstained", reason="person_ownership_unverified")
            elif arm.options.get("supported_profiles") is not None and unit.get("profile_id") not in arm.options["supported_profiles"]:
                row.update(state="unsupported", reason="profile_unsupported")
            elif base_error:
                row.update(state="fail" if base_error == "selected_base_hash_mismatch" else
                           "unknown" if base_error == "selected_base_hash_missing" else "error", reason=base_error)
            elif base_bytes is None:
                row.update(state="missing_dependency", reason="selected_base_bytes_unavailable")
            else:
                adapter = adapters.get(arm.arm_id)
                if arm.adapter == "identity":
                    outcome = {"state": "pass", "refined": False, "bvh_bytes": base_bytes,
                               "actual_backend": "identity"}
                    row.update(execution_kind="identity", elapsed_ms=0.0)
                elif adapter is None:
                    outcome = {"state": "missing_dependency", "reason": "adapter_not_configured"}
                else:
                    started = time.monotonic()
                    row["execution_kind"] = adapter.execution_kind
                    try:
                        outcome = adapter.execute(copy.deepcopy(unit), arm, policy["timeout_seconds"])
                    except TimeoutError:
                        outcome = {"state": "timeout", "reason": "adapter_timeout"}
                    except Exception:
                        outcome = {"state": "error", "reason": "adapter_error"}
                    row["elapsed_ms"] = (time.monotonic() - started) * 1000
                    if row["elapsed_ms"] > policy["timeout_seconds"] * 1000:
                        outcome = {"state": "timeout", "reason": "adapter_deadline_exceeded"}
                data = outcome.pop("bvh_bytes", None)
                if outcome.get("state") not in STATES:
                    outcome = {"state": "error", "reason": "invalid_adapter_state"}
                row.update({k: v for k, v in outcome.items() if k in (
                    "state", "reason", "actual_backend", "refined", "diagnostics", "refine_version")})
                row["bindings"].update(outcome.get("bindings", {}))
                if data is not None:
                    digest = sha256_bytes(data)
                    row["artifact_id"] = "sha256:" + digest
                    row["result_bvh_sha256"] = digest
                    path = artifact_dir / (hash_json(arm.arm_id)[:16] + ".bvh")
                    path.write_bytes(data)
                    if outcome.get("refined") is False:
                        row["exact_recovery"] = "pass" if digest == unit["selected_base_sha256"] else "fail"
                        if row["exact_recovery"] == "fail":
                            row.update(state="fail", reason="exact_recovery_mismatch")
                    try:
                        evaluation = evaluate_refine_artifacts(artifact_dir / "base.bvh", path,
                            unit.get("target_keypoints", []), unit.get("target_scores", []), unit.get("view"),
                            synthetic_gt_3d=unit.get("synthetic_gt_3d"),
                            allowed_joint_suffixes=policy.get("allowed_joint_suffixes"),
                            camera=unit.get("camera"))
                        # Evaluator artifact summaries contain private filesystem paths.
                        evaluation.get("base_artifact", {}).pop("path", None)
                        evaluation.get("result_artifact", {}).pop("path", None)
                        row["external_evaluation"] = evaluation
                        row["proxy_safety"] = ("fail" if evaluation.get("safety", {}).get("hard_safety_violation")
                                               else "pass" if evaluation.get("ok") else "unknown")
                    except Exception:
                        row["proxy_safety"] = "unknown"
                elif row["state"] in {"pass", "abstained"}:
                    row.update(state="error", reason="adapter_omitted_artifact")
                if row.get("refined") is True and arm.requested_backend != row.get("actual_backend"):
                    row.update(state="fail", reason="backend_mismatch")
                if row.get("actual_backend") in {"mock", "fixture"} and arm.requested_backend not in {"mock", "fixture"}:
                    row.update(state="fail", reason="backend_mismatch")
            row["binding_status"] = "pass" if policy.get("required_bindings") and all(
                row["bindings"].get(key) not in (None, "", "unknown") for key in policy["required_bindings"]
            ) else "unknown"
            expected = arm.bindings
            if any(k in row["bindings"] and row["bindings"][k] != v for k, v in expected.items()):
                row.update(state="fail", reason="binding_mismatch", binding_status="fail")
            elif any(k not in row["bindings"] for k in expected):
                row["binding_status"] = "unknown"
            records.append(row)
    write_jsonl(output / "experiment_arms.jsonl", records)
    snapshot_after_execution = _source_snapshot(source, labels_path, dataset.root)
    source_changed = source_snapshot != snapshot_after_reads or source_snapshot != snapshot_after_execution
    source_integrity = {"state": "fail" if source_changed else "pass",
                        "source_snapshot_changed": source_changed,
                        "expected_snapshot_sha256": hash_json(source_snapshot),
                        "after_reads_snapshot_sha256": hash_json(snapshot_after_reads),
                        "after_execution_snapshot_sha256": hash_json(snapshot_after_execution)}
    write_json(output / "source_integrity.json", source_integrity)
    # Existing human review assignment engine: duplicate and hidden repeats remain
    # unfilled; delivered artifacts do not automatically become quality labels.
    items = [{"item_id": "item:" + secrets.token_hex(12), "unit_id": r["unit_id"],
              "artifact_id": r["artifact_id"]} for r in records if r["artifact_id"]]
    pair_items, pair_templates, pair_provenance = [], [], []
    by_unit = {}
    for row in records:
        by_unit.setdefault(row["unit_id"], {})[row["arm"]] = row
    for unit_id, arm_rows in by_unit.items():
        baseline_row = arm_rows.get(registry.baseline, {})
        if not baseline_row.get("artifact_id"):
            continue
        for arm_id, row in arm_rows.items():
            if arm_id == registry.baseline or not row.get("artifact_id"):
                continue
            pair_id = "pair:" + secrets.token_hex(12)
            left, right = (baseline_row, row) if secrets.randbits(1) else (row, baseline_row)
            pair_items.append({"pair_id": pair_id, "unit_id": unit_id,
                               "left_artifact_id": left["artifact_id"], "right_artifact_id": right["artifact_id"]})
            pair_templates.append({"pair_id": pair_id, "unit_id": unit_id, "winner": "unknown",
                                   "safety_violation": "unknown", "label_source": "human_required"})
            pair_provenance.append({"pair_id": pair_id, "unit_id": unit_id,
                                    "left_arm": left["arm"], "right_arm": right["arm"]})
    assignments, queue, templates, pair_labels = _build_label_assignments(
        items, pair_items, pair_templates, seed=secrets.randbits(64))
    write_jsonl(output / "refine_pair_items.jsonl", pair_items)
    write_jsonl(output / "refine_pair_provenance.private.jsonl", pair_provenance)
    write_jsonl(output / "refine_pair_labels_template.jsonl", pair_labels)
    write_jsonl(output / "refine_label_assignments.private.jsonl", assignments)
    write_jsonl(output / "refine_label_queue.jsonl", queue)
    write_jsonl(output / "refine_independent_labels_template.jsonl", templates)
    from .experiment_report import build_experiment_report
    report = build_experiment_report(records, frozen=frozen, search=search, source_integrity=source_integrity)
    write_json(output / "experiment_report.json", report)
    result_files = ["experiment_arms.jsonl", "search_evidence.json", "source_integrity.json", "refine_label_queue.jsonl",
                    "refine_independent_labels_template.jsonl", "experiment_report.json",
                    "frozen_units.private.jsonl", "refine_label_assignments.private.jsonl",
                    "refine_pair_items.jsonl", "refine_pair_provenance.private.jsonl", "refine_pair_labels_template.jsonl"]
    result_files.extend(path.relative_to(output).as_posix() for path in sorted((output / "artifacts").rglob("*.bvh")))
    write_json(output / "manifest.json", {"schema_version": EXTENSION_SCHEMA, "mode": "experiment",
        "run_id": run_id, "status": "complete", "completed_at": utc_now(),
        "frozen_sha256": sha256_file(output / "experiment.frozen.json"),
        "result_seal": {name: sha256_file(output / name) for name in result_files},
        "expected_rows": frozen["expected_rows"], "actual_rows": len(records)})
    return output
