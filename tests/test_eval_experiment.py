"""Offline experiment contracts. No models, providers, private library or Blender."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from standin_eval.arms import Arm, ArmRegistry, DEFAULT_REGISTRY, LEGACY_ARMS
from standin_eval.dataset import EvalDataset, seal_dataset
from standin_eval.experiment_report import build_experiment_report, write_experiment_report
from standin_eval.experiment_runner import FixtureAdapter, HTTPRefineAdapter, run_experiment
from standin_eval.search_evidence import candidate_evidence, source_person
from standin_eval.util import hash_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from tests.test_smoke import _synthetic_bvh, _target_kp, _bvh_with_rotation


def source_fixture(root: Path, n=2):
    image = root / "synthetic.png"
    Image.new("RGB", (100, 100), "white").save(image)
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    cut = {"cut_id": "cut", "image_path": str(image), "image_sha256": sha256_file(image),
           "artist_id": "synthetic-artist", "project_id": "synthetic-project",
           "scene_group_id": "synthetic-scene", "split": "engineering", "expected_route": "core",
           "num_people_gt": n, "image_width": 100, "image_height": 100}
    persons = [{"person_id": f"person-{i}", "cut_id": "cut", "bbox_source": "synthetic_gt",
                "bbox_xyxy": [i * 30, 0, i * 30 + 25, 90], "eligible": True,
                "out_of_scope": False, "scale_class": "near"} for i in range(n)]
    dataset = EvalDataset(dataset_root, {"schema_version": 1, "dataset_id": "synthetic-test"}, [cut], persons)
    write_jsonl(dataset_root / "cuts.jsonl", [cut])
    write_jsonl(dataset_root / "persons.jsonl", persons)
    seal_dataset(dataset)
    base = Path(_synthetic_bvh(str(root), "base.bvh"))
    points, scores = _target_kp(str(base))
    source = root / "source"
    (source / "responses").mkdir(parents=True)
    manifest = {"run_id": "synthetic-source", "mode": "replay", "dataset": {
        "root": str(dataset_root), "dataset_id": dataset.dataset_id,
        "cut_manifest_sha256": dataset.actual_cut_hash, "gt_sha256": dataset.actual_person_hash},
        "requested_backend": {"vlm": "replay", "pose": "replay"},
        "actual_backend": {"vlm": "replay", "pose": "replay"}}
    write_json(source / "manifest.json", manifest)
    candidate = {"cut_id": "cut", "person_id": "person-0", "prediction_id": "cut:pred:0",
                 "rank": 1, "pose_id": "synthetic-pose", "view": "front", "distance": .1,
                 "bvh_sha256": sha256_file(base), "surfaced": True,
                 "candidate_artifact_id": "candidate-synthetic"}
    write_jsonl(source / "candidates.jsonl", [candidate])
    write_jsonl(source / "matches.jsonl", [{"cut_id": "cut", "person_id": "person-0",
                "prediction_id": "cut:pred:0", "match_status": "matched"}])
    write_jsonl(source / "cut_results.jsonl", [{"cut_id": "cut", "status": "ok", "route": "core",
                                               "latency_ms": 10, "within_time_budget": True}])
    write_jsonl(source / "predictions.jsonl", [{"cut_id": "cut", "prediction_id": "cut:pred:0",
        "box_xyxy": [0, 0, 25, 90], "person_index": 0, "skeleton_state": "valid", "coverage_class": "full"}])
    provenance = [{"joint_index": i, "person_id": "slot-123", "source": "pose_model",
                   "visibility": "occluded" if i == 9 else "unknown"} for i in range(17)]
    person = {"keypoints": points.tolist(), "scores": scores.tolist(), "raw_scores": scores.tolist(),
              "refine_allowed": True, "refinable_limbs": ["left_arm", "right_arm"],
              "skeleton_state": "valid", "coverage_class": "full", "slot_origin": "vlm",
              "skeleton_source": "full_image", "joint_provenance": provenance,
              "lower_body_observed": False, "quality_trace": {"side_labels_verified": False}}
    write_json(source / "responses" / "cut.json", {"people": [person]})
    selection = {"cut_id": "cut", "gt_person_id": "person-0", "selection_revision": "selection-1",
                 "body_revision": "body-1", "hand_revision": "hands-1", "output_scope": "full",
                 "selected_base_sha256": sha256_file(base), "base_artifact_path": str(base),
                 "bindings": {"model": "model-sha", "db": "db-sha"}}
    return source, selection, base, person


class ExperimentTests(unittest.TestCase):
    def test_registry_preserves_historical_arms_and_explicit_baseline(self):
        self.assertEqual(DEFAULT_REGISTRY.baseline, "F25")
        self.assertEqual(LEGACY_ARMS["B2_v24_aggressive"].expected_version_prefix, "v2.4")
        self.assertNotEqual(LEGACY_ARMS["B2_v24_aggressive"].algorithm, DEFAULT_REGISTRY.arms[1].algorithm)
        with self.assertRaises(ValueError):
            ArmRegistry(DEFAULT_REGISTRY.arms, "not-registered")

    def test_fixed_denominator_unknown_labels_missing_selected_person_and_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, person = source_fixture(root)
            fixture = FixtureAdapter({"F25": {"state": "abstained", "refined": False,
                                             "actual_backend": "none", "bvh_bytes": base.read_bytes()}})
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                adapters={"F25": fixture}, output_root=root / "runs", run_id="fixed")
            rows = read_jsonl(run / "experiment_arms.jsonl")
            report = read_json(run / "experiment_report.json")
            self.assertEqual(len(rows), 6)
            self.assertEqual(report["denominator"]["persons"], 2)
            self.assertEqual(report["decision"], "INCONCLUSIVE")
            f25 = next(r for r in rows if r["arm"] == "F25" and r["gt_person_id"] == "person-0")
            self.assertEqual(f25["exact_recovery"], "pass")
            self.assertEqual(f25["state"], "abstained")
            self.assertEqual(len({r["comparison_context_sha256"] for r in rows if r["gt_person_id"] == "person-0"}), 1)
            self.assertEqual(report["arms"]["F25"]["dimensions"]["human_usability"]["unknown"], 2)
            self.assertEqual(report["search"]["candidate_coverage_at_5"], {"pass": 0, "fail": 1, "unknown": 1, "denominator": 2})
            frozen = read_jsonl(run / "frozen_units.private.jsonl")
            self.assertEqual(frozen[0]["joint_provenance"], person["joint_provenance"])
            self.assertFalse(frozen[0]["lower_body_observed"])
            self.assertEqual(frozen[0]["gt_person_id"], "person-0")
            self.assertNotEqual(frozen[0]["gt_person_id"], "slot-123")
            self.assertNotIn(str(root), json.dumps(report))

    def test_recovery_mismatch_is_failure_and_not_a_visual_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, _ = source_fixture(root, n=1)
            changed = Path(_bvh_with_rotation(str(root), "changed.bvh", "LeftArm", 12))
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                adapters={"F25": FixtureAdapter({"F25": {"state": "abstained", "refined": False,
                    "actual_backend": "none", "bvh_bytes": changed.read_bytes()}})},
                output_root=root / "runs", run_id="wrong-recovery")
            report = read_json(run / "experiment_report.json")
            self.assertEqual(report["decision"], "FAIL")
            self.assertIn("exact_recovery_mismatch", report["hard_failures"])

    def test_timeout_and_model_fallback_preserve_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, _ = source_fixture(root, n=1)
            run = run_experiment(from_run=source, policy={"timeout_seconds": .2}, selections=[selection],
                adapters={"F25": FixtureAdapter({"F25": TimeoutError()}),
                          "v3": FixtureAdapter({"v3": {"state": "pass", "refined": True,
                           "actual_backend": "mock", "bvh_bytes": base.read_bytes()}})},
                output_root=root / "runs", run_id="timeout-fallback")
            rows = read_jsonl(run / "experiment_arms.jsonl")
            self.assertEqual(next(r for r in rows if r["arm"] == "F25")["state"], "timeout")
            self.assertEqual(next(r for r in rows if r["arm"] == "v3")["reason"], "backend_mismatch")
            self.assertEqual(len(rows), 3)

    def test_policy_units_sealed_before_any_adapter_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, _ = source_fixture(root, n=1)
            class Probe(FixtureAdapter):
                def execute(self, unit, arm, timeout_seconds):
                    frozen = read_json(root / "runs" / "sealed" / "experiment.frozen.json")
                    self_test.assertEqual(frozen["expected_rows"], 3)
                    unit["target_scores"][0] = 999
                    return super().execute(unit, arm, timeout_seconds)
            self_test = self
            fixture = Probe({"F25": {"state": "abstained", "refined": False, "bvh_bytes": base.read_bytes()}})
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                adapters={"F25": fixture}, output_root=root / "runs", run_id="sealed")
            frozen_units = read_jsonl(run / "frozen_units.private.jsonl")
            self.assertNotEqual(frozen_units[0]["target_scores"][0], 999)

    def test_http_v3_shadow_is_unsupported_and_does_not_refine(self):
        adapter = HTTPRefineAdapter("http://local.test")
        with patch("standin_eval.experiment_runner._request_json", return_value=(200, b"{}", {
            "ok": True, "refine_v3": {"mode": "shadow", "regional_apply": False, "certified_profiles": 0}})) as request:
            result = adapter.execute({}, DEFAULT_REGISTRY.arms[2], 1)
        self.assertEqual(result["state"], "unsupported")
        self.assertEqual(result["reason"], "v3_has_no_applied_output_contract")
        self.assertEqual(request.call_count, 1)

    def test_http_f25_contract_forwards_same_camera_and_policy_scores(self):
        unit = {"pose_id": "pose", "view": "front", "target_keypoints": [[1, 2]] * 17,
                "target_scores": [.9] * 17, "policy_scores": [.2] * 17,
                "source_refine_allowed": True, "source_refinable_limbs": ["left_arm"],
                "lower_body_observed": False, "selected_base_sha256": "a" * 64,
                "camera": {"source_bvh_sha256": "a" * 64, "version": "candidate-camera-v1"}}
        context = {"base_bvh_sha256": "a" * 64, "pose_library_version": "db-version",
                   "feature_version": 7, "source_revision": "code-sha", "deployment_version": "build-sha",
                   "refine_config_sha256": "b" * 64}
        health = {"ok": True, "refine": {"enabled": True, "code_version": "v2.5.3",
                  "supported_modes": ["aggressive"], "config_sha256": "b" * 64,
                  **{key: context[key] for key in ("pose_library_version", "feature_version",
                                                 "source_revision", "deployment_version")}}}
        response = {"refined": False, "backend": "none", "reason": "no_gain", "refine_version": "v2.5.3",
                    "bvh_url": "/pose/pose/bvh", "diagnostics": {"context": context}}
        with patch("standin_eval.experiment_runner._request_json", side_effect=[(200, b"", health), (200, b"", response)]) as request, patch(
                "standin_eval.experiment_runner._fetch_binary", return_value=(200, b"same-base")):
            outcome = HTTPRefineAdapter("http://local.test").execute(unit, DEFAULT_REGISTRY.arms[1], 1)
        payload = json.loads(request.call_args_list[1].kwargs["data"])
        self.assertEqual(payload["camera"], unit["camera"])
        self.assertEqual(payload["scores"], unit["policy_scores"])
        self.assertFalse(payload["lower_body_observed"])
        self.assertFalse(payload["render_thumbnail"])
        self.assertEqual(outcome["state"], "abstained")
        self.assertEqual(outcome["bindings"]["db"], "db-version")

    def test_validated_human_labels_remain_separate_from_mesh_and_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, _, _ = source_fixture(root, n=1)
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                                 output_root=root / "runs", run_id="valid-human")
            templates = read_jsonl(run / "refine_independent_labels_template.jsonl")
            labels = [{**row, "overall_usability": "direct", "safety_violation": "none",
                       "labeler_id": f"rater-{i % 2}"} for i, row in enumerate(templates)]
            path = root / "reviewed-labels.jsonl"
            write_jsonl(path, labels)
            report = write_experiment_report(run, human_labels_path=path)
            self.assertEqual(report["arms"]["B0_no_refine"]["dimensions"]["human_usability"]["pass"], 1)
            self.assertEqual(report["arms"]["B0_no_refine"]["dimensions"]["full_deformed_mesh_safety"]["unknown"], 1)
            self.assertEqual(report["decision"], "INCONCLUSIVE")

    def test_joint_ownership_and_unknown_source_are_not_inferred(self):
        candidate = {"pose_id": "generated-looking-name"}
        evidence = candidate_evidence(candidate, {})
        self.assertEqual(evidence["candidate_source"], "unknown")
        self.assertEqual(evidence["generation_state"], "unknown")
        self.assertEqual(evidence["poster_state"], "unknown")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _, _ = source_fixture(root, n=1)
            candidate = read_jsonl(source / "candidates.jsonl")[0]
            self.assertEqual(source_person(source, candidate, []), {})

    def test_missing_policy_and_modified_dataset_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, _, _ = source_fixture(root, n=1)
            with self.assertRaises(ValueError):
                run_experiment(from_run=source, policy={}, output_root=root / "runs")
            manifest = read_json(source / "manifest.json")
            manifest["dataset"]["gt_sha256"] = "changed"
            write_json(source / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
                run_experiment(from_run=source, policy={"timeout_seconds": 1}, output_root=root / "runs")

    def test_fixed_camera_projects_same_base_and_rejects_incompatible_binding(self):
        import numpy as np
        from src.bvh import load_coco17
        from standin_eval.refine_evaluator import evaluate_refine_artifacts, EVALUATOR_VERSION, CAMERA_EVALUATOR_VERSION
        with tempfile.TemporaryDirectory() as directory:
            base = Path(_synthetic_bvh(directory))
            points, scores = load_coco17(str(base))
            angle = np.deg2rad(35)
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                 [-np.sin(angle), 0, np.cos(angle)]])
            hip = points[[11, 12]].mean(axis=0)
            torso = np.linalg.norm(points[[5, 6]].mean(axis=0) - hip)
            camera_points = (points - hip) @ rotation.T
            target = camera_points[:, :2].copy()
            target[:, 1] *= -1
            target /= (camera_points[:, 2] + 4 * torso)[:, None]
            camera = {"version": "candidate-camera-v2", "rotation": rotation.tolist(),
                      "source_bvh_sha256": sha256_file(base), "projection": "perspective",
                      "distance_torso": 4., "labels_swapped": False, "observed_joint_mask": [True] * 17}
            result = evaluate_refine_artifacts(base, base, target.tolist(), scores.tolist(), "front", camera=camera)
            self.assertTrue(result["ok"])
            self.assertEqual(result["evaluator_version"], CAMERA_EVALUATOR_VERSION)
            self.assertLess(result["result_metrics"]["joint_nme"]["value"], 1e-6)
            legacy = evaluate_refine_artifacts(base, base, target.tolist(), scores.tolist(), "front")
            self.assertEqual(legacy["evaluator_version"], EVALUATOR_VERSION)
            self.assertGreater(legacy["result_metrics"]["joint_nme"]["value"], 1e-3)
            invalid = evaluate_refine_artifacts(base, base, target.tolist(), scores.tolist(), "front",
                                               camera={**camera, "source_bvh_sha256": "0" * 64})
            self.assertFalse(invalid["ok"])
            self.assertIn("invalid_frozen_camera", invalid["query_evidence"]["error"])
            camera["labels_swapped"] = True
            swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            swapped = evaluate_refine_artifacts(base, base, target[swap].tolist(), scores[swap].tolist(), "front", camera=camera)
            self.assertLess(swapped["result_metrics"]["joint_nme"]["value"], 1e-6)

    def test_review_assignments_are_sealed_and_unassigned_labels_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, _ = source_fixture(root, n=1)
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                                 output_root=root / "runs", run_id="review")
            records = read_jsonl(run / "experiment_arms.jsonl")
            identity = next(r for r in records if r["arm"] == "B0_no_refine")
            labels = root / "labels.jsonl"
            write_jsonl(labels, [{"assignment_id": "not-assigned", "item_id": "fake", "unit_id": identity["unit_id"],
                "artifact_id": identity["artifact_id"], "labeler_id": "rater-1", "label_source": "human_required",
                "overall_usability": "direct", "safety_violation": "none"}])
            report = write_experiment_report(run, human_labels_path=labels)
            self.assertIn("unknown_independent_assignment", report["hard_failures"])
            self.assertEqual(report["arms"]["B0_no_refine"]["dimensions"]["human_usability"]["pass"], 0)
            self.assertEqual(read_json(run / "experiment_report.json")["decision"], "INCONCLUSIVE")
            (run / "artifacts" / identity["unit_id"].split(":")[1] / "base.bvh").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                write_experiment_report(run)

    def test_raw_camera_and_generation_review_evidence_are_preserved_without_inline_assets(self):
        from standin_eval.search_evidence import summarize_search_run, compare_search_evidence
        from standin_eval.dataset import load_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, person = source_fixture(root, n=1)
            camera = {"version": "candidate-camera-v1", "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      "source_bvh_sha256": sha256_file(base)}
            person["candidates"] = [{"pose_id": "synthetic-pose", "view": "front", "camera": camera}]
            person["pose_presentation"] = {"status": "generation_failed", "generation_status": "timeout",
                "generated_candidates": [{"source": "generation", "kind": "generated_review", "rank": 1,
                    "refine_allowed": False, "export_allowed": False,
                    "asset": {"transport": "inline_bvh", "bvh_sha256": "a" * 64, "bvh": "PRIVATE_RAW_BVH"}}]}
            write_json(source / "responses" / "cut.json", {"people": [person]})
            dataset = load_dataset(root / "dataset")
            search = summarize_search_run(source, dataset)
            self.assertEqual(search["candidates"][0]["camera"], camera)
            self.assertEqual(search["candidates"][0]["pose_presentation"]["generation_status"], "timeout")
            self.assertNotIn("PRIVATE_RAW_BVH", json.dumps(search))
            self.assertEqual(search["dimensions"]["assist_success_at_5"]["unknown"], 1)
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                                 output_root=root / "runs", run_id="camera-source")
            unit = read_jsonl(run / "frozen_units.private.jsonl")[0]
            self.assertEqual(unit["camera"], camera)
            compared = compare_search_evidence(runs={"search-old": source, "search-new": source},
                                               baseline="search-old", dataset=dataset)
            self.assertEqual(compared["comparisons"]["search-new"]["paired"]["candidate_coverage_at_5"]["unresolved"], 1)

    def test_fixture_timing_is_excluded_and_expected_rows_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, _ = source_fixture(root, n=1)
            fixture = FixtureAdapter({"F25": {"state": "pass", "refined": False, "bvh_path": str(base)}})
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                adapters={"F25": fixture}, output_root=root / "runs", run_id="fixture-time")
            report = read_json(run / "experiment_report.json")
            frozen = read_json(run / "experiment.frozen.json")
            self.assertEqual(report["arms"]["F25"]["latency"]["count"], 0)
            self.assertEqual(report["arms"]["B0_no_refine"]["latency"]["count"], 0)
            self.assertEqual(len(frozen["expected_unit_ids"]), 1)

    def test_mutated_source_cannot_mix_evidence_or_clear_failure_on_rereport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection, base, _ = source_fixture(root, n=1)
            original = (source / "candidates.jsonl").read_bytes()

            class MutatingAdapter(FixtureAdapter):
                def execute(self, unit, arm, timeout_seconds):
                    # Valid JSON mutation must not change the already computed
                    # search cohort or quietly become a new source snapshot.
                    write_jsonl(source / "candidates.jsonl", [])
                    return super().execute(unit, arm, timeout_seconds)

            adapter = MutatingAdapter({"F25": {"state": "abstained", "refined": False,
                                               "actual_backend": "none", "bvh_bytes": base.read_bytes()}})
            run = run_experiment(from_run=source, policy={"timeout_seconds": 1}, selections=[selection],
                adapters={"F25": adapter}, output_root=root / "runs", run_id="source-mutated")
            report = read_json(run / "experiment_report.json")
            self.assertEqual(len(read_jsonl(run / "experiment_arms.jsonl")), 3)
            self.assertEqual(report["decision"], "FAIL")
            self.assertIn("source_snapshot_changed", report["hard_failures"])
            # Still uses pre-adapter candidate evidence: label unknown, not the
            # mutated empty-source interpretation of zero successful candidates.
            self.assertEqual(report["search"]["candidate_coverage_at_5"]["unknown"], 1)
            self.assertTrue(read_json(run / "source_integrity.json")["source_snapshot_changed"])
            self.assertIn("source_integrity.json", read_json(run / "manifest.json")["result_seal"])
            frozen = read_json(run / "experiment.frozen.json")
            self.assertIn("responses/cut.json", frozen["source_snapshot"]["source_files"])
            self.assertNotEqual(frozen["source_snapshot"]["source_files"]["candidates.jsonl"],
                                sha256_file(source / "candidates.jsonl"))
            (source / "candidates.jsonl").write_bytes(original)
            reviewed = write_experiment_report(run)
            self.assertEqual(reviewed["decision"], "FAIL")
            self.assertIn("source_snapshot_changed", reviewed["hard_failures"])

    def test_report_detects_dropped_rows_and_changed_comparison_context(self):
        frozen = {"registry": DEFAULT_REGISTRY.to_dict(), "policy": {}, "expected_rows": 3,
                  "expected_unit_ids": ["unit-1"]}
        row = {"unit_id": "unit-1", "arm": "F25", "state": "unknown", "execution_kind": "fixture"}
        report = build_experiment_report([row], frozen=frozen)
        self.assertEqual(report["decision"], "FAIL")
        self.assertIn("preregistered_row_set_changed", report["hard_failures"])


if __name__ == "__main__":
    unittest.main()
