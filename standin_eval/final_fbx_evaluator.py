"""Fixed-denominator final-file evaluation with independent quality dimensions."""
from __future__ import annotations

from collections import Counter
import math

from .artifact_lineage import BINDINGS, SCHEMA, is_sha, resolved_identity, validate_lineage
from .util import canonical_json, hash_json

DIMENSIONS = ("integrity", "structure", "mesh", "scene_interaction", "human_usability", "native_save", "time")
FINAL_REQUIRED = ("integrity", "structure", "mesh", "human_usability", "native_save", "time")
STATES = ("pass", "fail", "unknown", "unsupported", "missing_dependency", "timeout", "error", "abstained")


def seal_export_policy(tasks, policy):
    """No inherited SLO/sample defaults. Explicit requirements are frozen first."""
    ids = [task.get("task_id") for task in tasks]
    if not ids or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty preregistered task IDs required")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise ValueError("explicit export policy ID required")
    required = policy.get("required_dimensions")
    if (not isinstance(required, list) or not required or len(set(required)) != len(required)
            or any(k not in DIMENSIONS for k in required)):
        raise ValueError("explicit required dimensions required")
    if "time" in required:
        value = policy.get("task_budget_seconds")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("explicit positive task budget required")
    if type(policy.get("minimum_tasks")) is not int or policy["minimum_tasks"] < 1:
        raise ValueError("explicit minimum task count required")
    payload = {"schema": SCHEMA, "tasks": tasks, "policy": policy}
    # JSON roundtrip prevents mutation of the caller's preregistered objects.
    import json
    return {**json.loads(canonical_json(payload)),
            "seal_sha256": hash_json(payload)}


def evaluate_export_tasks(sealed, observations):
    payload = {k: sealed[k] for k in ("schema", "tasks", "policy")}
    if hash_json(payload) != sealed.get("seal_sha256"):
        raise ValueError("export preregistration seal mismatch")
    tasks, policy = sealed["tasks"], sealed["policy"]
    by_id = {}
    for row in observations:
        ident = row.get("task_id")
        if ident in by_id:
            raise ValueError("duplicate export observation")
        if ident not in {t["task_id"] for t in tasks}:
            raise ValueError("unregistered export observation")
        by_id[ident] = row
    results = []
    human_labelers, independent_groups = set(), set()
    for task in tasks:
        obs = by_id.get(task["task_id"], {})
        dims = {name: "unknown" for name in DIMENSIONS}
        reasons = []
        if obs.get("selections") is not None:
            lineage = validate_lineage(obs["selections"], obs.get("artifacts", []))
            dims["integrity"] = lineage["state"]
            reasons += lineage["failures"] + lineage["unknown_reasons"]
        structural = obs.get("structure", {})
        if structural.get("evidence_kind") == "structural_roundtrip":
            dims["structure"] = structural.get("state", "unknown")
        mesh = obs.get("mesh", {})
        required_checks = task.get("required_mesh_checks", [])
        if mesh.get("evidence_kind") == "actual_mesh":
            if mesh.get("state") == "fail":
                dims["mesh"] = "fail"
            elif mesh.get("state") == "pass" and required_checks and all(mesh.get("checks", {}).get(k) == "pass" for k in required_checks):
                dims["mesh"] = "pass"
            elif mesh.get("state") in STATES:
                dims["mesh"] = mesh["state"] if mesh["state"] != "pass" else "unknown"
        scene = obs.get("scene_interaction", {})
        if task.get("person_count", 1) == 1 and not task.get("props_present", False):
            # Explicit policy may exclude this dimension, never manufacture a pass.
            reasons.append("scene_interaction_not_applicable")
        elif scene.get("independent_checker") and scene.get("checks_complete") is True:
            dims["scene_interaction"] = scene.get("state", "unknown")
        else:
            reasons.append("scene_interaction_unverified")
        humans = list(obs.get("human_reviews", []))
        if obs.get("human_review"):
            humans.append(obs["human_review"])
        human_states = []
        for human in humans:
            if (human.get("assignment_id") and human.get("labeler_id") and human.get("completed") is True
                    and human.get("tool_version") and human.get("os") and human.get("opened") is True):
                if human.get("usability") in ("immediate", "after_edit"):
                    human_states.append("pass")
                elif human.get("usability") == "unusable":
                    human_states.append("fail")
                identity_bound = all(human.get(k) == task.get(k) and task.get(k) for k in
                                     ("artifact_sha256", "profile_sha256", "scope", "selection_revision"))
                if identity_bound and human.get("usability") in ("immediate", "after_edit", "unusable"):
                    human_labelers.add(human["labeler_id"])
        if "fail" in human_states:
            dims["human_usability"] = "fail"
        elif human_states and all(state == "pass" for state in human_states):
            dims["human_usability"] = "pass"
        native = obs.get("native_save", {})
        if native.get("state") in ("cancelled", "write_failed"):
            dims["native_save"] = "abstained" if native["state"] == "cancelled" else "fail"
        elif (native.get("state") == "saved" and native.get("readback_verified") is True
              and is_sha(native.get("artifact_sha256")) and native.get("artifact_sha256") == native.get("saved_sha256")
              and type(native.get("bytes")) is int and native["bytes"] > 0):
            dims["native_save"] = "pass"
        timing = obs.get("elapsed_seconds")
        if (type(timing) in (int, float) and math.isfinite(timing) and timing >= 0
                and policy.get("task_budget_seconds") is not None):
            dims["time"] = "pass" if timing <= policy["task_budget_seconds"] else "fail"
        # Apply immutable identity checks AFTER all dimension calculations;
        # later good labels/readbacks may never erase a stale-file failure.
        identity_fields = ("artifact_sha256", "profile_sha256", "scope", "selection_revision")
        binding_missing = any(not task.get(k) for k in identity_fields)
        binding_missing |= any(not is_sha(task.get(k)) for k in ("artifact_sha256", "profile_sha256"))
        if not task.get("selection_sha256s"):
            binding_missing = True
        if not task.get("required_mesh_checks"):
            reasons.append("mesh_check_policy_missing")
            if dims["mesh"] != "fail":
                dims["mesh"] = "unknown"
        for key in BINDINGS:
            expected_binding = task.get("bindings", {}).get(key)
            if not resolved_identity(expected_binding):
                binding_missing = True
            elif obs.get("bindings", {}).get(key) != expected_binding:
                if dims["integrity"] != "fail":
                    dims["integrity"] = "fail" if obs.get("bindings", {}).get(key) else "unknown"
                reasons.append("runtime_binding_mismatch_or_missing:" + key)
        if obs.get("selections") is not None and task.get("selection_sha256s"):
            if sorted(hash_json(s) for s in obs["selections"]) != sorted(task["selection_sha256s"]):
                dims["integrity"] = "fail"
                reasons.append("frozen_selection_mismatch")
        delivery = [n for n in obs.get("artifacts", []) if n.get("stage") == "native_save"]
        if delivery and any(n.get("sha256") != task.get("artifact_sha256") for n in delivery):
            dims["integrity"] = "fail"
            reasons.append("final_artifact_task_mismatch")
        if binding_missing:
            reasons.append("frozen_task_bindings_missing")
            if dims["integrity"] != "fail":
                dims["integrity"] = "unknown"
        for evidence_name, target in (("structure", "structure"), ("mesh", "mesh"),
                                      ("human_review", "human_usability"), ("native_save", "native_save"),
                                      ("scene_interaction", "scene_interaction")):
            evidence = obs.get(evidence_name, {})
            for field in identity_fields:
                if evidence and task.get(field) and evidence.get(field) != task[field]:
                    if dims[target] != "fail":
                        dims[target] = "fail" if evidence.get(field) else "unknown"
                    reasons.append(evidence_name + "_binding_mismatch_or_missing:" + field)
        for human in humans:
            for field in identity_fields:
                if task.get(field) and human.get(field) != task[field]:
                    if dims["human_usability"] != "fail":
                        dims["human_usability"] = "fail" if human.get(field) else "unknown"
                    reasons.append("human_review_binding_mismatch_or_missing:" + field)
        if resolved_identity(task.get("independent_group_id")) and task.get("grouping_confirmed") is True:
            independent_groups.add(task["independent_group_id"])
        for name in dims:
            if dims[name] not in STATES:
                dims[name] = "unknown"
        required_names = set(policy["required_dimensions"]) | set(FINAL_REQUIRED)
        if task.get("person_count", 1) > 1 or task.get("props_present", False):
            required_names.add("scene_interaction")
        required = [dims[k] for k in required_names]
        fixture = obs.get("execution_kind") != "product_observation" or any(
            s.get("backend", {}).get("actual") in ("mock", "fixture") for s in obs.get("selections", []))
        state = "fail" if "fail" in dims.values() else "unknown" if any(s != "pass" for s in required) or fixture or binding_missing else "pass"
        if fixture:
            reasons.append("offline_or_unmeasured_cannot_certify_product")
        results.append({"task_id": task["task_id"], "unit_id": task.get("unit_id"), "state": state,
                        "dimensions": dims, "reasons": sorted(set(reasons)),
                        "execution_kind": obs.get("execution_kind", "not_executed")})
    totals = {"denominator": len(tasks), **{s: sum(r["state"] == s for r in results) for s in ("pass", "fail", "unknown")}}
    dimensions = {name: {"denominator": len(tasks), **dict(Counter(r["dimensions"][name] for r in results))} for name in DIMENSIONS}
    policy_gaps = []
    reviewers = policy.get("minimum_reviewers")
    groups = policy.get("minimum_independent_groups")
    if type(reviewers) is not int or reviewers < 2:
        policy_gaps.append("minimum_two_distinct_reviewers_not_preregistered")
    elif len(human_labelers) < reviewers:
        policy_gaps.append("insufficient_distinct_reviewers")
    if type(groups) is not int or groups < 1:
        policy_gaps.append("independent_group_requirement_not_preregistered")
    elif len(independent_groups) < groups or any(t.get("grouping_confirmed") is not True for t in tasks):
        policy_gaps.append("independent_groups_unconfirmed_or_insufficient")
    if not resolved_identity(policy.get("sample_size_rationale")):
        policy_gaps.append("sample_size_rationale_missing")
    decision = ("FAIL" if totals["fail"] else "PASS"
                if totals["pass"] == len(tasks) and len(tasks) >= policy["minimum_tasks"] and not policy_gaps
                else "INCONCLUSIVE")
    return {"schema": SCHEMA, "evaluation_kind": "final_fbx", "decision": decision,
            "seal_sha256": sealed["seal_sha256"], "mandatory_dimensions": list(FINAL_REQUIRED), "totals": totals, "dimensions": dimensions,
            "tasks": results, "coverage_gaps": policy_gaps, "distinct_reviewers": len(human_labelers),
            "confirmed_independent_groups": len(independent_groups),
            "certification_scope": "preregistered_finite_tasks_only", "release_certification": False,
            "operational_deployment": False}
