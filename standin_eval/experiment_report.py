"""Fixed-denominator reports; delivery, quality, proxy safety and time differ."""
from __future__ import annotations

from collections import Counter

from .arms import EXTENSION_SCHEMA, STATES
from .util import hash_json, percentile, read_json, read_jsonl, sha256_file, write_json
from .labels import resolve_run


def state_counts(values: list[str], denominator: int) -> dict:
    counter = Counter(values)
    return {"denominator": denominator, "pass": counter["pass"], "fail": counter["fail"],
            "unknown": sum(counter[s] for s in STATES - {"pass", "fail"}),
            "states": {state: counter[state] for state in sorted(STATES)},
            "pass_rate": counter["pass"] / denominator if denominator else None}


def build_experiment_report(records: list[dict], *, frozen: dict,
                            search: dict | None = None, human_labels: list[dict] | None = None) -> dict:
    """No row dropping and no conversion of missing labels or measurements to pass.

    Human labels follow existing independent-refine label fields and are bound
    to exact unit/artifact IDs. Pairwise improvement and actual mesh/tool evidence
    remain separate promotion prerequisites, not inferred from a proxy score.
    """
    registry = frozen["registry"]
    policy = frozen["policy"]
    expected = frozen["expected_rows"]
    arms = [r["arm_id"] for r in registry["arms"]]
    expected_units = frozen.get("expected_unit_ids", sorted({r["unit_id"] for r in records}))
    expected_keys = {(unit, arm) for unit in expected_units for arm in arms}
    keys = [(r["unit_id"], r["arm"]) for r in records]
    integrity_errors = []
    if len(keys) != len(set(keys)) or len(records) != expected or set(keys) != expected_keys:
        integrity_errors.append("preregistered_row_set_changed")
    contexts = {}
    for row in records:
        contexts.setdefault(row["unit_id"], set()).add(row.get("comparison_context_sha256"))
    if any(len(values) != 1 or None in values for values in contexts.values()):
        integrity_errors.append("different_selected_base_camera_or_revision")
    labels = {}
    for row in human_labels or []:
        if row.get("unit_id") and row.get("artifact_id") and row.get("labeler_id"):
            labels.setdefault((row["unit_id"], row["artifact_id"]), []).append(row)
    by_arm = {}
    hard_failures = list(integrity_errors)
    minimum_labelers = max(2, int(policy.get("minimum_distinct_labelers", 2)))
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        quality_states = []
        for row in rows:
            bound = labels.get((row["unit_id"], row.get("artifact_id")), [])
            usable = [r for r in bound if r.get("overall_usability") in {"direct", "reference", "unusable"}
                      and r.get("label_source") == "human_required"
                      and r.get("safety_violation") in {"none", "anatomy", "collision", "contact", "ground", "other"}]
            raters = {r["labeler_id"] for r in usable}
            values = {r["overall_usability"] for r in usable}
            if len(raters) < minimum_labelers or len(values) != 1:
                quality_states.append("unknown")
            else:
                quality_states.append("fail" if "unusable" in values or any(
                    r.get("safety_violation") != "none" for r in usable) else "pass")
        timing = []
        budget = policy.get("refine_budget_ms")
        for row in rows:
            elapsed = row.get("elapsed_ms")
            if row["state"] == "timeout":
                timing.append("fail")
            elif budget is None or elapsed is None or row["execution_kind"] in {"fixture", "not_executed"}:
                timing.append("unknown")
            else:
                timing.append("pass" if elapsed <= budget else "fail")
        dimensions = {
            "execution": state_counts([r["state"] for r in rows], len(expected_units)),
            "human_usability": state_counts(quality_states, len(expected_units)),
            "skeleton_capsule_proxy": state_counts([r.get("proxy_safety", "unknown") for r in rows], len(expected_units)),
            "exact_recovery": state_counts([r.get("exact_recovery", "unknown") for r in rows], len(expected_units)),
            "immutable_bindings": state_counts([r.get("binding_status", "unknown") for r in rows], len(expected_units)),
            "refine_time": state_counts(timing, len(expected_units)),
            "full_deformed_mesh_safety": state_counts(["unknown"] * len(rows), len(expected_units)),
        }
        # Every missing preregistered record remains unknown in every dimension.
        for dimension in dimensions.values():
            missing = len(expected_units) - sum(dimension["states"].values())
            if missing > 0:
                dimension["states"]["unknown"] += missing
                dimension["unknown"] += missing
        failures = [r["reason"] for r in rows if r["state"] == "fail"]
        failures += ["proxy_safety_violation"] if any(r.get("proxy_safety") == "fail" for r in rows) else []
        failures += ["latency_budget_violation"] if "fail" in timing else []
        failures += ["human_usability_rejected"] if "fail" in quality_states else []
        hard_failures.extend(failures)
        measured = [r["elapsed_ms"] for r in rows if r.get("elapsed_ms") is not None
                    and r["execution_kind"] not in {"fixture", "identity", "not_executed"}]
        by_arm[arm] = {"dimensions": dimensions,
            "reasons": dict(Counter(r.get("reason") or r["state"] for r in rows)),
            "latency": {"count": len(measured), "p50_ms": percentile(measured, .5),
                        "p95_ms": percentile(measured, .95), "max_ms": max(measured) if measured else None,
                        "basis": "adapter_health_refine_and_artifact_fetch; excludes_source_search_and_base_fetch"},
            "execution_kinds": dict(Counter(r["execution_kind"] for r in rows)),
            "backend_pairs": [{"requested": requested, "actual": actual, "count": count}
                              for (requested, actual), count in Counter(
                                  (r.get("requested_backend"), r.get("actual_backend")) for r in rows).items()]}
    incomplete = ["actual_profile_mesh_evidence_and_native_tool_review_required",
                  "paired_quality_improvement_analysis_required"]
    for field in ("refine_budget_ms", "search_budget_ms", "required_bindings",
                  "minimum_units", "minimum_independent_groups", "quality_improvement_rule"):
        if not policy.get(field):
            incomplete.append("policy_missing:" + field)
    if not records:
        incomplete.append("empty_target_denominator")
    if any(r["execution_kind"] in {"fixture", "not_executed"} for r in records):
        incomplete.append("fixture_or_unexecuted_rows_cannot_certify_real_quality")
    if frozen.get("source_execution_kind") == "replay":
        incomplete.append("replay_does_not_measure_provider_or_product_latency")
    if not frozen.get("source_backend_verified", False):
        incomplete.append("source_requested_actual_backend_unverified")
    if frozen.get("source_backend_mismatch"):
        hard_failures.append("source_backend_mismatch")
    if len(expected_units) < policy.get("minimum_units", 0):
        incomplete.append("insufficient_preregistered_units")
    if frozen.get("independent_groups", 0) < policy.get("minimum_independent_groups", 0):
        incomplete.append("insufficient_verified_independent_groups")
    for arm, value in by_arm.items():
        for name, dimension in value["dimensions"].items():
            if dimension["unknown"]:
                incomplete.append(f"{arm}:{name}:unknown")
    return {"schema_version": EXTENSION_SCHEMA, "baseline": registry["baseline"],
            "decision": "FAIL" if hard_failures else "INCONCLUSIVE" if incomplete else "PASS",
            "expected_rows": expected, "observed_rows": len(records),
            "denominator": {"persons": len(expected_units), "arms": len(arms)},
            "report_scope": "engineering_contract_and_evidence_completeness; use legacy promotion analysis separately",
            "arms": by_arm, "search": (search or {}).get("dimensions", {}),
            "hard_failures": sorted(set(hard_failures)), "incomplete_reasons": sorted(set(incomplete)),
            "limitations": ["static_thumbnail_usability_is_not_fbx_safety",
                            "capsule_and_2d_checks_are_not_deformed_mesh_certification",
                            "no_uninspected_profile_pose_or_scope_is_certified"]}


def write_experiment_report(run, *, human_labels_path=None) -> dict:
    """Validate sealed evidence/assignments and write an unsealed reviewed report.

    The original engineering report remains immutable. Unknown, unassigned,
    mismatched or unfinished labels cannot become a measured quality pass.
    """
    root = resolve_run(run)
    manifest = read_json(root / "manifest.json")
    if manifest.get("mode") != "experiment":
        raise ValueError("not an experiment run")
    if sha256_file(root / "experiment.frozen.json") != manifest.get("frozen_sha256"):
        raise ValueError("experiment preregistration seal mismatch")
    required = {"experiment_arms.jsonl", "search_evidence.json", "refine_label_assignments.private.jsonl",
                "refine_label_queue.jsonl", "frozen_units.private.jsonl"}
    if not required <= set(manifest.get("result_seal", {})):
        raise ValueError("experiment result seal incomplete")
    for name, expected in manifest["result_seal"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or sha256_file(path) != expected:
            raise ValueError("experiment result seal mismatch")
    frozen = read_json(root / "experiment.frozen.json")
    if hash_json(read_jsonl(root / "frozen_units.private.jsonl")) != frozen["units_sha256"]:
        raise ValueError("experiment units changed")
    queue = {r["assignment_id"]: r for r in read_jsonl(root / "refine_label_queue.jsonl")}
    assignments = {r["assignment_id"]: r for r in read_jsonl(root / "refine_label_assignments.private.jsonl")}
    valid_labels, label_errors = [], []
    for row in read_jsonl(human_labels_path) if human_labels_path else []:
        assignment = assignments.get(row.get("assignment_id"))
        item = queue.get(row.get("assignment_id"))
        if not item or not assignment or assignment.get("task_type") != "independent":
            label_errors.append("unknown_independent_assignment")
        elif any(row.get(key) != item.get(key) for key in ("unit_id", "item_id", "artifact_id")):
            label_errors.append("assignment_unit_or_artifact_mismatch")
        else:
            valid_labels.append(row)
    report = build_experiment_report(read_jsonl(root / "experiment_arms.jsonl"), frozen=frozen,
        search=read_json(root / "search_evidence.json"), human_labels=valid_labels)
    if label_errors:
        report["decision"] = "FAIL"
        report["hard_failures"] += sorted(set(label_errors))
    report["review_labels_sha256"] = sha256_file(human_labels_path) if human_labels_path else None
    report["human_review_scope"] = "validated_assignment_artifact_and_distinct_raters; paired_promotion_analysis_separate"
    write_json(root / "experiment_report.reviewed.json", report)
    return report
