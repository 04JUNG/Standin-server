"""Adapters for existing Converter evidence; no renderer/proxy safety promotion."""
from __future__ import annotations

from dataclasses import asdict
import math

from .artifact_lineage import is_sha
from .util import hash_json

MESH_DIMENSIONS = ("collision", "folding", "degeneration", "joint_limits", "contact", "ground")


def adapt_actual_snapshot(report, *, expected, execution_kind="imported_blender"):
    """Bind finite-actual-fbx-safety-qa-v1 to ONE exact source/profile/full FBX.

    Reuse Converter's receipt and target profile validators. A historical report
    or fixture is useful evidence of its own finite snapshot, never a general
    profile certification. Independently missing joint/contact/ground checks
    remain unknown even when the finite skin audit passed.
    """
    result = {"schema": "evaluation-extension-v1", "evidence_kind": "actual_mesh",
              "execution_kind": execution_kind, "state": "unknown", "failures": [],
              "unknown_reasons": [], "checks": {k: "unknown" for k in MESH_DIMENSIONS},
              "scene_interaction": "unknown", "coverage": "finite_snapshot_only",
              "adapter_version": "converter-snapshot-adapter-v1", "report_sha256": hash_json(report), **{k: expected.get(k) for k in
                  ("unit_id", "arm", "artifact_id", "artifact_sha256", "profile_sha256", "scope", "source_sha256", "selection_revision")}}
    bad, unknown = result["failures"], result["unknown_reasons"]
    required = ("artifact_sha256", "profile_sha256", "source_sha256")
    if not all(is_sha(expected.get(k)) for k in required) or not expected.get("scope"):
        unknown.append("expected_identity_missing")
    if report.get("schema") == "finite-actual-fbx-safety-failure-v1":
        result["state"] = "fail" if report.get("status") == "validation_blocked" else "unknown"
        result["failures" if result["state"] == "fail" else "unknown_reasons"].append("converter_snapshot:" + str(report.get("reason", "unknown")))
        return result
    if report.get("schema") != "finite-actual-fbx-safety-qa-v1":
        unknown.append("actual_snapshot_schema_required")
    if expected.get("scope") != "full":
        unknown.append("unsupported_scope_derivation")
    for key in required:
        if expected.get(key) and report.get(key) != expected[key]:
            bad.append("snapshot_identity_mismatch:" + key)
    if report.get("safe_final_snapshot_count") != 1:
        unknown.append("finite_snapshot_count_unbound")
    try:
        from src.target_profile import TargetProfile
        profile = TargetProfile(**report.get("profile", {}))
        if profile.digest != report.get("profile_sha256"):
            bad.append("profile_digest_mismatch")
        bindings = expected.get("bindings", {})
        for key in ("controller_sha256", "blender_build_hash", "converter_manifest_sha256", "hand_sha256"):
            if not bindings.get(key):
                unknown.append("checker_binding_missing:" + key)
            elif bindings[key] != asdict(profile).get(key):
                bad.append("checker_binding_mismatch:" + key)
        from converter.artifact_receipt import ArtifactReceipt, VERSION, _digest as converter_digest
        from converter.final_artifact import valid_receipt
        from converter.mesh_validation import VERSION as MESH_VERSION
        receipt = ArtifactReceipt(**report.get("receipt", {}))
        audit = report.get("mesh_audit", {})
        integrity = report.get("artifact_integrity", {})
        if receipt.version != VERSION or receipt.digest != report.get("receipt_sha256"):
            bad.append("receipt_digest_mismatch")
        for key in required:
            if expected.get(key) and getattr(receipt, key) != expected[key]:
                bad.append("receipt_identity_mismatch:" + key)
        if (receipt.scope != expected.get("scope") or receipt.artifact_size != report.get("artifact_size")
                or receipt.artifact_size <= 0 or receipt.revision != expected.get("selection_revision")
                or receipt.context_sha256 != expected.get("context_sha256")):
            bad.append("receipt_revision_scope_size_or_context_mismatch")
        if receipt.mesh_audit_sha256 != converter_digest(audit) or receipt.integrity_audit_sha256 != converter_digest(integrity):
            bad.append("receipt_audit_hash_mismatch")
        if not valid_receipt(integrity, expected.get("artifact_sha256"), expected.get("scope")):
            unknown.append("final_skin_reimport_unverified")
        if (audit.get("version") != MESH_VERSION or audit.get("profile_sha256") != expected.get("profile_sha256")
                or audit.get("full_body_artifact_sha256") != expected.get("artifact_sha256")
                or receipt.full_body_artifact_sha256 != expected.get("artifact_sha256")
                or receipt.mesh_policy_sha256 != audit.get("mesh_policy_sha256")
                or receipt.mesh_policy_version != audit.get("mesh_policy_version")):
            bad.append("mesh_audit_binding_mismatch")
        if audit.get("hard_violations"):
            bad.extend("actual_mesh:" + str(v) for v in audit["hard_violations"])
            for key in ("collision", "folding", "degeneration"):
                result["checks"][key] = "fail"
        complete = (audit.get("status") == "mesh_validated" and audit.get("full_body") is True
                    and audit.get("complete_collision_certificate") is True
                    and audit.get("hard_violations") == [] and audit.get("unverified_reasons") == []
                    and is_sha(audit.get("mesh_policy_sha256")) and bool(audit.get("mesh_policy_version")))
        if complete and not bad:
            for key in ("collision", "folding", "degeneration"):
                result["checks"][key] = "pass"
        else:
            unknown.extend(audit.get("unverified_reasons") or ["mesh_checks_incomplete"])
        stages = report.get("stages", [])
        if [stage.get("stage") for stage in stages] != ["neutral-repair", "body", "hands"]:
            unknown.append("deformation_stage_coverage_incomplete")
        previous = ""
        for stage in stages:
            witness, replay = stage.get("witness", {}), stage.get("replay", {})
            witness_sha = converter_digest(witness)
            if (stage.get("witness_sha256") != witness_sha or replay.get("witness_sha256") != witness_sha
                    or stage.get("mesh_audit", {}).get("deformation_witness_sha256") != witness_sha
                    or witness.get("profile_sha256") != expected.get("profile_sha256")
                    or witness.get("source_sha256") != expected.get("source_sha256")
                    or witness.get("previous_witness_sha256") != previous):
                bad.append("deformation_witness_binding_mismatch")
            error, tolerance = replay.get("max_vertex_error_m"), replay.get("vertex_tolerance_m")
            if (replay.get("discrete_shell_orientation_verified") is not True
                    or type(error) not in (int, float) or type(tolerance) not in (int, float)
                    or not math.isfinite(error) or not math.isfinite(tolerance)
                    or not 0 <= error <= tolerance or not 0 < tolerance <= 1e-5):
                unknown.append("deformation_replay_incomplete")
            if stage.get("mesh_audit", {}).get("status") != "mesh_validated":
                unknown.append("deformation_stage_mesh_incomplete")
            previous = witness_sha
        if stages and (stages[-1].get("artifact_sha256") != expected.get("artifact_sha256")
                       or stages[-1].get("mesh_audit") != audit):
            bad.append("final_stage_artifact_mismatch")
        # Pose/contact/ground have different contracts. Do not invent passes
        # from skin intersection or from library preset metadata.
        pose_safety = report.get("pose_safety")
        if isinstance(pose_safety, dict) and pose_safety.get("final_body", {}).get("status") == "pose_validated":
            result["checks"]["joint_limits"] = "pass"
    except ImportError:
        unknown.append("missing_dependency:converter_mesh")
    except (TypeError, ValueError, AttributeError, KeyError) as error:
        unknown.append("malformed_converter_evidence:" + type(error).__name__)
    if execution_kind != "imported_blender" and execution_kind != "local_blender":
        unknown.append("fixture_cannot_certify_real_mesh")
    if execution_kind == "imported_blender" and expected.get("artifact_readback_verified") is not True:
        unknown.append("artifact_bytes_not_read_back")
    if bad or unknown:
        for key in MESH_DIMENSIONS:
            if result["checks"][key] == "pass":
                result["checks"][key] = "unknown"
    result["state"] = "fail" if bad else "unsupported" if "unsupported_scope_derivation" in unknown else "unknown" if unknown else "pass"
    return result


def adapt_roundtrip(report, *, artifact_sha256, scope, execution_kind="imported_blender", required_checks=()):
    """Single Converter reimport checks structural integrity only."""
    names = ("converter_ok", "finite", "reopened_same_bytes") + tuple(required_checks)
    optional = ("hand_body_bones_preserved", "shape_controls_match", "shape_digest_matches")
    failure = any(report.get(k) is False for k in names + optional)
    complete = (all(report.get(k) is True for k in names)
                and report.get("scope") == scope
                and all(type(report.get(k)) is int and report[k] > 0 for k in ("meshes", "vertices", "bones", "bytes")))
    return {"state": "fail" if failure else "pass" if complete else "unknown",
            "artifact_sha256": artifact_sha256, "scope": scope,
            "evidence_kind": "structural_roundtrip", "execution_kind": execution_kind,
            "mesh_safety": "unknown", "report_sha256": hash_json(report)}


SCENE_CASES = {"single": 1, "two_people": 2, "three_repeated": 3}


def adapt_scene_roundtrip(report, *, case=None, artifact_sha256=None, expected=None):
    """The existing tool certifies placement for 1/2/3 cases, not eight people.

    An aggregate report has no single artifact identity. Only an explicitly
    selected case with local output readback can feed a final-file task.
    """
    cases = report.get("cases", [])
    failure = report.get("ok") is False or any(c.get("ok") is False for c in cases)
    result = {"evidence_kind": "scene_placement_roundtrip", "report_sha256": hash_json(report),
              "state": "fail" if failure else "pass" if report.get("ok") is True and cases else "unknown",
              "covered_cases": [c.get("case") for c in cases],
              "scene_interaction": "unknown", "unknown_reasons": ["scene_interaction_unverified"]}
    if case is None:
        return result
    if case not in SCENE_CASES:
        return {**result, "state": "unsupported", "reason": "scene_case_not_implemented"}
    selected = [row for row in cases if row.get("case") == case]
    if len(selected) != 1 or not is_sha(artifact_sha256):
        return {**result, "state": "unknown", "reason": "selected_scene_case_or_bytes_missing"}
    row = selected[0]
    errors = [row.get("independent_world_error_m"), row.get("max_world_error_m")]
    tolerance = row.get("tolerance_m")
    finite = (type(tolerance) in (int, float) and math.isfinite(tolerance) and tolerance > 0
              and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= tolerance for value in errors))
    complete = (report.get("ok") is True and row.get("ok") is True and row.get("roundtrip_verified") is True
                and row.get("person_count") == SCENE_CASES[case] and finite)
    bad = failure or row.get("roundtrip_verified") is False or row.get("person_count") != SCENE_CASES[case]
    if finite is False and all(type(value) in (int, float) for value in errors) and type(tolerance) in (int, float):
        bad = True
    return {**result, "evidence_kind": "structural_roundtrip", "state": "fail" if bad else "pass" if complete else "unknown",
            "artifact_sha256": artifact_sha256, "scene_case": case, "person_count": SCENE_CASES[case],
            "mesh_safety": "unknown", **{k: (expected or {}).get(k) for k in ("profile_sha256", "scope", "selection_revision")}}
