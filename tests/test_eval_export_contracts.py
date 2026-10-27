"""Offline final-export contracts; synthetic bytes never assert product quality."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from standin_eval.artifact_lineage import BINDINGS, artifact_node, validate_lineage
from standin_eval.export_runner import blender_plan, execute_blender
from standin_eval.final_fbx_evaluator import evaluate_export_tasks, seal_export_policy
from standin_eval.mesh_evidence_adapter import adapt_actual_snapshot, adapt_roundtrip, adapt_scene_roundtrip
from standin_eval.save_scenarios import run_save_scenarios, verify_scene_download, SceneContractError
from standin_eval.util import hash_json, sha256_bytes

SHA = "a" * 64


def selection():
    return dict(unit_id="unit-1", cut_id="cut-1", gt_person_id="permanent-person-7", person_index=2,
                candidate_id="candidate-4", arm="F25", selection_revision="r1", body_revision="body1",
                hand_revision="hands1", scope="full", source_sha256=sha256_bytes(b"fixture BVH"),
                camera_sha256=SHA, profile_sha256=SHA,
                bindings={k: "fixture-build" if k == "blender_build" else SHA for k in BINDINGS},
                backend={"requested": "provider-build-1", "actual": "provider-build-1"})


def lineage(directory):
    selected = selection()
    nodes, parent = [], []
    for index, stage in enumerate(("selected_bvh", "body_fbx", "hands_fbx", "individual_fbx", "download", "native_save")):
        path = Path(directory) / stage
        data = b"fixture BVH" if index == 0 else b"fixture final FBX" if index >= 3 else stage.encode()
        path.write_bytes(data)
        node = artifact_node(path, artifact_id=stage, stage=stage, parents=parent, selections=[selected])
        nodes.append(node)
        parent = [stage]
    return selected, nodes


def report_case(directory):
    selected, nodes = lineage(directory)
    identity = dict(artifact_sha256=nodes[-1]["sha256"], profile_sha256=SHA, scope="full", selection_revision="r1")
    task = dict(task_id="task-1", unit_id="unit-1", person_count=1, **identity,
                independent_group_id="synthetic-group", grouping_confirmed=True,
                required_mesh_checks=["collision", "folding", "degeneration"],
                selection_sha256s=[hash_json(selected)], bindings=selected["bindings"])
    policy = dict(policy_id="offline-test-policy", required_dimensions=["integrity", "structure", "mesh", "human_usability", "native_save", "time"],
                  minimum_tasks=1, minimum_reviewers=2, minimum_independent_groups=1,
                  sample_size_rationale="Finite synthetic contract assertion only", task_budget_seconds=17)
    obs = dict(task_id="task-1", execution_kind="product_observation", selections=[selected], artifacts=nodes,
               bindings=selected["bindings"], elapsed_seconds=12,
               structure=dict(**identity, evidence_kind="structural_roundtrip", state="pass"),
               mesh=dict(**identity, evidence_kind="actual_mesh", state="pass", checks={k: "pass" for k in task["required_mesh_checks"]}),
               human_review=dict(**identity, assignment_id="assignment", labeler_id="reviewer", completed=True,
                                 tool_version="CSP-fixture-version", os="fixture-os", opened=True, usability="immediate"),
               native_save=dict(**identity, state="saved", readback_verified=True, saved_sha256=nodes[-1]["sha256"], bytes=nodes[-1]["bytes"]))
    obs["human_reviews"] = [{**obs["human_review"], "labeler_id": "reviewer-2", "assignment_id": "assignment-2"}]
    return task, policy, obs


def snapshot_fixture():
    """Real pure-math mesh/receipt contracts, fabricated skin-import evidence."""
    from src.target_profile import TargetProfile
    from converter.mesh_validation import MeshPolicy, audit_mesh, topology_sha256
    from converter.artifact_receipt import issue_receipt
    from converter.deformation_witness import build_lbs_witness, compose_lbs_witness, verify_witness
    profile = TargetProfile("fixture-profile", *([SHA] * 9), "revision-1", "hand-revision-1", "fixture-adapter", "fixture-converter", "fixture-blender")
    # A known closed tetrahedron: face normals come from its independent rest.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    policy = MeshPolicy(profile.digest, topology_sha256(faces, len(vertices)), "fixture-policy", ("skin",) * 4)
    data = b"synthetic fixture bytes, not FBX"
    artifact_sha = sha256_bytes(data)
    weights = np.ones((4, 1))
    transforms = np.eye(4)[None]
    witness = build_lbs_witness(neutral_vertices=vertices, corrected_rest_vertices=vertices, faces=faces,
                                vertex_weights=weights, bone_world_transforms=transforms, mesh_world_transform=np.eye(4),
                                stage="neutral-repair", stage_input_sha256=SHA, source_asset_sha256=SHA,
                                profile_sha256=profile.digest, source_sha256=SHA)
    stages = []
    for stage in ("neutral-repair", "body", "hands"):
        if stage != "neutral-repair":
            witness = compose_lbs_witness(witness, faces=faces, vertex_weights=weights,
                                         bone_world_transforms=transforms, stage=stage, stage_input_sha256=artifact_sha)
        replay = verify_witness(witness, vertices=vertices, faces=faces, stage_input_sha256=witness.stage_input_sha256,
                                 source_asset_sha256=SHA, profile_sha256=profile.digest, source_sha256=SHA)
        audit = audit_mesh(vertices, faces, policy=policy, profile_sha256=profile.digest,
                           expected_normals=normals, reference_double_areas=np.linalg.norm(normals, axis=1))
        assert audit["status"] == "mesh_validated", audit
        audit.update(full_body_artifact_sha256=artifact_sha, deformation_witness_sha256=witness.digest)
        stages.append(dict(stage=stage, artifact_sha256=artifact_sha, witness=witness.manifest(),
                           witness_sha256=witness.digest, replay=replay, mesh_audit=audit))
    integrity = dict(version="final-fbx-v1", artifact_sha256=artifact_sha, scope="full", skin_verified=True,
                     bones=1, meshes=1, vertices=4, polygons=4)
    receipt = issue_receipt(data, source_sha256=SHA, context_sha256=SHA, profile_sha256=profile.digest,
                            revision="r1", scope="full", mesh_audit=audit, integrity_audit=integrity)
    report = dict(schema="finite-actual-fbx-safety-qa-v1", profile=asdict(profile), profile_sha256=profile.digest,
                  source_sha256=SHA, artifact_sha256=artifact_sha, safe_final_snapshot_count=1, stages=stages,
                  artifact_integrity=integrity, mesh_audit=audit, receipt=asdict(receipt), receipt_sha256=receipt.digest,
                  artifact_size=len(data), pose_safety=None)
    expected = dict(unit_id="unit", arm="v3", artifact_id="artifact", artifact_sha256=artifact_sha,
                    profile_sha256=profile.digest, source_sha256=SHA, scope="full", selection_revision="r1",
                    context_sha256=SHA, artifact_readback_verified=True,
                    bindings={k: asdict(profile)[k] for k in ("controller_sha256", "blender_build_hash", "converter_manifest_sha256", "hand_sha256")})
    return report, expected


class ExportLineageTests(unittest.TestCase):
    def test_complete_byte_lineage_and_missing_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            selected, nodes = lineage(directory)
            self.assertEqual(validate_lineage([selected], nodes)["state"], "pass")
            self.assertEqual(validate_lineage([selected], nodes[:-1])["state"], "unknown")
            self.assertEqual(validate_lineage([selected], nodes[:-1], require_native=False)["state"], "pass")

    def test_native_tamper_and_wrong_selection_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            selected, nodes = lineage(directory)
            nodes[-1]["sha256"] = "b" * 64
            self.assertIn("delivery_bytes_mismatch:native_save", validate_lineage([selected], nodes)["failures"])
            nodes[-1]["selection_sha256s"] = [SHA]
            self.assertEqual(validate_lineage([selected], nodes)["state"], "fail")

    def test_explicit_permanent_person_not_request_index(self):
        with tempfile.TemporaryDirectory() as directory:
            selected, nodes = lineage(directory)
            selected.pop("gt_person_id")
            result = validate_lineage([selected], nodes)
            self.assertIn("missing_selection:gt_person_id", result["unknown_reasons"])
            self.assertEqual(result["state"], "fail")  # existing nodes pin the original selection

    def test_parent_cycle_and_missing_body_hand_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            selected, nodes = lineage(directory)
            nodes[1]["parents"] = ["native_save"]
            self.assertEqual(validate_lineage([selected], nodes)["state"], "fail")
            self.assertNotEqual(validate_lineage([selected], [nodes[0], *nodes[3:]])["state"], "pass")

    def test_duplicate_person_and_mock_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            selected, nodes = lineage(directory)
            self.assertEqual(validate_lineage([selected, selected], nodes)["state"], "fail")
            selected["backend"]["actual"] = "mock"
            self.assertIn("backend_fallback", validate_lineage([selected], [])["unknown_reasons"])


class SnapshotAdapterTests(unittest.TestCase):
    def test_exact_finite_snapshot_contract_not_general_profile(self):
        report, expected = snapshot_fixture()
        result = adapt_actual_snapshot(report, expected=expected)
        self.assertEqual(result["state"], "pass", result)
        self.assertEqual(result["coverage"], "finite_snapshot_only")
        self.assertEqual(result["checks"]["contact"], "unknown")
        self.assertEqual(result["checks"]["ground"], "unknown")
        self.assertEqual(result["scene_interaction"], "unknown")

    def test_fixture_cannot_certify_real_and_no_readback_unknown(self):
        report, expected = snapshot_fixture()
        self.assertEqual(adapt_actual_snapshot(report, expected=expected, execution_kind="fixture")["state"], "unknown")
        expected.pop("artifact_readback_verified")
        self.assertEqual(adapt_actual_snapshot(report, expected=expected)["state"], "unknown")

    def test_wrong_profile_scope_or_receipt_never_pass(self):
        report, expected = snapshot_fixture()
        for field, value in (("profile_sha256", SHA), ("selection_revision", "r2"), ("artifact_sha256", SHA)):
            wrong = {**expected, field: value}
            self.assertEqual(adapt_actual_snapshot(report, expected=wrong)["state"], "fail", field)
        wrong = {**expected, "scope": "head"}
        self.assertNotEqual(adapt_actual_snapshot(report, expected=wrong)["state"], "pass")

    def test_missing_replay_or_stage_is_unknown(self):
        report, expected = snapshot_fixture()
        report["stages"][0]["replay"]["discrete_shell_orientation_verified"] = False
        self.assertEqual(adapt_actual_snapshot(report, expected=expected)["state"], "unknown")
        report["stages"] = []
        self.assertEqual(adapt_actual_snapshot(report, expected=expected)["state"], "unknown")

    def test_hard_snapshot_failure_and_incomplete_timeout_preserved(self):
        expected = {"unit_id": "unit"}
        self.assertEqual(adapt_actual_snapshot({"schema": "finite-actual-fbx-safety-failure-v1", "status": "validation_blocked", "reason": "skin_folding"}, expected=expected)["state"], "fail")
        self.assertEqual(adapt_actual_snapshot({"schema": "finite-actual-fbx-safety-failure-v1", "status": "mesh_unverified", "reason": "validation_deadline"}, expected=expected)["state"], "unknown")

    def test_open_file_structural_check_is_not_mesh_safety(self):
        report = dict(converter_ok=True, finite=True, reopened_same_bytes=True, scope="full", meshes=1, vertices=4, bones=1, bytes=40)
        result = adapt_roundtrip(report, artifact_sha256=SHA, scope="full")
        self.assertEqual(result["state"], "pass")
        self.assertEqual(result["mesh_safety"], "unknown")
        self.assertEqual(adapt_roundtrip({**report, "shape_controls_match": False}, artifact_sha256=SHA, scope="full")["state"], "fail")
        self.assertEqual(adapt_roundtrip(report, artifact_sha256=SHA, scope="full", required_checks=["shape_controls_match"])["state"], "unknown")
        scene = adapt_scene_roundtrip({"ok": True, "cases": [{"case": "two_people"}]})
        self.assertEqual(scene["scene_interaction"], "unknown")
        self.assertEqual(scene["covered_cases"], ["two_people"])

    def test_scene_exact_case_binding_and_unsupported_eight_people(self):
        report = {"ok": True, "cases": [dict(case="two_people", ok=True, person_count=2,
            roundtrip_verified=True, independent_world_error_m=0.00001, max_world_error_m=0.00001, tolerance_m=0.0001)]}
        bound = adapt_scene_roundtrip(report, case="two_people", artifact_sha256=SHA,
                                      expected=dict(profile_sha256=SHA, scope="full", selection_revision="r1"))
        self.assertEqual(bound["state"], "pass")
        self.assertEqual(bound["artifact_sha256"], SHA)
        self.assertEqual(bound["evidence_kind"], "structural_roundtrip")
        self.assertEqual(bound["mesh_safety"], "unknown")
        self.assertEqual(adapt_scene_roundtrip(report, case="eight_people", artifact_sha256=SHA)["state"], "unsupported")
        report["cases"][0]["independent_world_error_m"] = 0.1
        self.assertEqual(adapt_scene_roundtrip(report, case="two_people", artifact_sha256=SHA)["state"], "fail")
        self.assertEqual(adapt_scene_roundtrip({"ok": False})["state"], "fail")

    def test_legacy_mesh_contract_unmodified(self):
        from standin_eval.refine_mesh import build_mesh_evidence_template_row, validate_mesh_evidence_row
        row = build_mesh_evidence_template_row(unit_id="unit", arm="B0_no_refine", artifact_id="a", geometry_sha256=SHA)
        self.assertEqual(validate_mesh_evidence_row(row, require_complete=False, allow_placeholders=True), [])


class ExportReportTests(unittest.TestCase):
    def test_fixed_denominator_missing_labels_and_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            # Positive schema contract only; generated synthetic files are never
            # reported outside this test as measured product execution.
            seal = seal_export_policy([task], policy)
            self.assertEqual(evaluate_export_tasks(seal, [obs])["decision"], "PASS")
            obs.pop("human_review")
            obs.pop("human_reviews")
            result = evaluate_export_tasks(seal, [obs])
            self.assertEqual(result["decision"], "INCONCLUSIVE")
            self.assertEqual(result["totals"], {"denominator": 1, "pass": 0, "fail": 0, "unknown": 1})
            missing = evaluate_export_tasks(seal, [])
            self.assertEqual(missing["totals"]["denominator"], 1)

    def test_stale_human_and_native_binding_fail_survives_good_content(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            seal = seal_export_policy([task], policy)
            for key in ("human_review", "native_save"):
                wrong = copy.deepcopy(obs)
                wrong[key]["artifact_sha256"] = "b" * 64
                wrong[key].pop("selection_revision")
                result = evaluate_export_tasks(seal, [wrong])
                self.assertEqual(result["decision"], "FAIL", key)
                self.assertEqual(result["tasks"][0]["dimensions"]["human_usability" if key == "human_review" else key], "fail")

    def test_mock_never_product_pass_and_unknown_scene_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            obs["execution_kind"] = "fixture"
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")
            obs["execution_kind"] = "product_observation"
            policy["required_dimensions"].append("scene_interaction")
            task["person_count"] = 2
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")

    def test_policy_and_denominator_cannot_change_after_execution(self):
        sealed = seal_export_policy([{"task_id": "task"}], dict(policy_id="p", required_dimensions=["mesh"], minimum_tasks=1))
        sealed["tasks"].append({"task_id": "later"})
        with self.assertRaises(ValueError):
            evaluate_export_tasks(sealed, [])
        with self.assertRaises(ValueError):
            seal_export_policy([{"task_id": "task"}], dict(policy_id="p", required_dimensions=["time"], minimum_tasks=1))

    def test_unbound_all_pass_observation_and_missing_hash_native_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            task.pop("bindings")
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")
            task, policy, obs = report_case(directory)
            obs["native_save"].pop("saved_sha256")
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")

    def test_policy_cannot_remove_human_or_multi_person_safety_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            obs.pop("human_review")
            obs.pop("human_reviews")
            policy["required_dimensions"] = ["integrity"]
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")
            task, policy, obs = report_case(directory)
            task["person_count"] = 2
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")

    def test_runtime_model_or_hand_asset_changed_hard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            obs["bindings"] = {**obs["bindings"], "model_sha256": "b" * 64}
            obs["bindings"].pop("db_sha256")
            result = evaluate_export_tasks(seal_export_policy([task], policy), [obs])
            self.assertEqual(result["decision"], "FAIL")

    def test_global_worker_group_and_sample_requirements_stay_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            obs.pop("human_reviews")
            result = evaluate_export_tasks(seal_export_policy([task], policy), [obs])
            self.assertEqual(result["tasks"][0]["dimensions"]["human_usability"], "pass")
            self.assertEqual(result["decision"], "INCONCLUSIVE")
            self.assertEqual(result["distinct_reviewers"], 1)
            self.assertIn("insufficient_distinct_reviewers", result["coverage_gaps"])
            task, policy, obs = report_case(directory)
            task["grouping_confirmed"] = False
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")
            task, policy, obs = report_case(directory)
            policy.pop("sample_size_rationale")
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")

    def test_explicit_mock_and_placeholder_binding_cannot_product_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            task, policy, obs = report_case(directory)
            obs["selections"][0]["backend"] = {"requested": "mock", "actual": "mock"}
            digest = hash_json(obs["selections"][0])
            task["selection_sha256s"] = [digest]
            for node in obs["artifacts"]:
                node["selection_sha256s"] = [digest]
            self.assertEqual(evaluate_export_tasks(seal_export_policy([task], policy), [obs])["decision"], "INCONCLUSIVE")
            for placeholder in ("unknown", "REQUIRED", "unresolved:blender"):
                selected = selection()
                selected["bindings"]["blender_build"] = placeholder
                self.assertIn("missing_binding:blender_build", validate_lineage([selected], [])["unknown_reasons"])

    def test_duplicate_or_unregistered_observation_rejected(self):
        sealed = seal_export_policy([{"task_id": "task"}], dict(policy_id="p", required_dimensions=["mesh"], minimum_tasks=1))
        with self.assertRaises(ValueError):
            evaluate_export_tasks(sealed, [{"task_id": "task"}, {"task_id": "task"}])
        with self.assertRaises(ValueError):
            evaluate_export_tasks(sealed, [{"task_id": "other"}])


class SaveScenarioTests(unittest.TestCase):
    def test_all_offline_scenarios_and_no_quality_certification(self):
        result = run_save_scenarios()
        self.assertEqual(result["contract_tests"], {"denominator": 50, "pass": 50, "fail": 0})
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        for row in result["scenarios"]:
            self.assertEqual(row["product_quality"], "unknown")
            if row["action"] == "http_cancel":
                self.assertIs(row["server_operation_stopped"], False)

    def test_bad_download_bytes_metadata_and_magic_rejected(self):
        from converter.scene_contract import FBX_MAGIC
        data = FBX_MAGIC + b"synthetic bytes"
        headers = {"X-Standin-Scene-Contract": "scene-save-v1", "X-Standin-Scene-Save-Mode": "original-selection",
                   "X-Standin-Person-Count": "3", "X-Standin-Scene-Version": "rough-layout-v1", "X-Standin-Artifact-SHA256": sha256_bytes(data)}
        self.assertTrue(verify_scene_download(data, headers, person_count=3, mode="original-selection")["verified"])
        with self.assertRaises(SceneContractError):
            verify_scene_download(data + b"tamper", headers, person_count=3, mode="original-selection")
        with self.assertRaises(SceneContractError):
            verify_scene_download(data, headers, person_count=8, mode="original-selection")


class BlenderPlanTests(unittest.TestCase):
    def plan(self, directory, kind="single"):
        source = Path(directory) / "source"
        source.write_bytes(b"fixture")
        item = {"path": str(source), "sha256": sha256_bytes(b"fixture")}
        inputs = {"bvh": item, "character": item} if kind == "single" else {"source": [item, item]}
        return blender_plan(kind, blender="missing-blender", output_dir=Path(directory) / "fresh", inputs=inputs)

    def test_explicit_execution_disabled_and_plan_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory)
            with patch("standin_eval.export_runner.subprocess.run") as run:
                self.assertEqual(execute_blender(plan, expected={}, timeout_seconds=3)["state"], "unsupported")
                self.assertFalse(run.called)
                plan["command"] = ["untrusted-program"]
                with self.assertRaises(ValueError):
                    execute_blender(plan, expected={}, timeout_seconds=3, allow_blender=True)
                self.assertFalse(run.called)

    def test_input_rechecked_and_timeout_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory)
            with self.assertRaises(ValueError):
                execute_blender(plan, expected={}, timeout_seconds=float("inf"), allow_blender=True)
            (Path(directory) / "source").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                execute_blender(plan, expected={}, timeout_seconds=3, allow_blender=True)

    def test_missing_blender_and_timeout_rows_without_real_execution(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory)
            with patch("standin_eval.export_runner.subprocess.run", side_effect=FileNotFoundError):
                self.assertEqual(execute_blender(plan, expected={}, timeout_seconds=3, allow_blender=True)["state"], "missing_dependency")
            with patch("standin_eval.export_runner.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 3)):
                self.assertEqual(execute_blender(plan, expected={}, timeout_seconds=3, allow_blender=True)["state"], "timeout")

    def test_scene_source_minimum_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(directory, "scene")
            self.assertEqual(plan["kind"], "scene")
            item = plan["inputs"]["source"][0]
            with self.assertRaises(ValueError):
                blender_plan("scene", blender="blender", output_dir=Path(directory) / "out", inputs={"source": [item]})
            with self.assertRaises(ValueError):
                blender_plan("single", blender="blender", output_dir=directory, inputs={"bvh": item, "character": item})


if __name__ == "__main__":
    unittest.main()
