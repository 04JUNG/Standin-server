import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.hybrid_pose import HybridPosePipeline
from src.hybrid_pose import contracts, schema
from src.hybrid_pose.gemini import GeminiReviewer, GeminiStageError, provider_schema
from src.hybrid_pose.pipeline import normalize_rtm
from src.hybrid_pose.render import overlay
from src.schema import Skeleton


def evaluation():
    return {"decision": "pass", "issues": [], "uncertainties": []}


def repair_evaluation(kind="obvious_pose_error"):
    return {
        "decision": "repair",
        "issues": [
            {
                "issue_id": "a",
                "type": kind,
                "person_index": 0,
                "joint_names": ["left_elbow"],
                "region_bbox": None,
                "evidence": "Wrong bend.",
            }
        ],
        "uncertainties": [],
    }


def patch_data():
    return {
        "edits": [
            {
                "issue_id": "a",
                "person_index": 0,
                "name": "left_elbow",
                "x": 400,
                "y": 500,
                "state": "visible",
                "evidence": "Arm bend.",
            }
        ],
        "added_people": [],
        "removed_people": [],
        "unresolved": [],
    }


def skeleton():
    return Skeleton(
        np.asarray([[10 + i, 15 + i] for i in range(17)], dtype=np.float32),
        np.full(17, 0.8, dtype=np.float32),
    )


def proposal():
    return normalize_rtm([skeleton()], 100, 150, 0.3)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.image = Path(tmp.name) / "러프.png"
        Image.new("RGB", (100, 150), (255, 0, 0)).save(self.image)
        self.pose = Mock()
        self.pose.estimate.return_value = [skeleton()]
        self.reviewer = Mock(metadata={"model": "offline-test"})
        self.artifacts = {}
        self.pipeline = HybridPosePipeline(self.pose, self.reviewer, mode="two-stage")

    def run_pipeline(self, responses):
        self.reviewer.request.side_effect = responses
        return self.pipeline.extract(self.image, self.artifacts.__setitem__)

    def test_pass_once_no_repair_preserves_every_joint(self):
        result = self.run_pipeline([(evaluation(), {})])
        self.assertEqual(result["final"], result["rtm"])
        self.assertEqual(result["delivery_status"], "rtm_kept")
        self.assertEqual(result["quality_review"], "unreviewed")
        self.assertEqual(result["api_calls_started"], 1)
        self.assertFalse(result["visual_reassessment"])
        self.pose.estimate.assert_called_once()
        self.assertEqual(list(self.pose.estimate.call_args.args[0][0, 0]), [0, 0, 255])
        self.assertGreaterEqual(
            result["timings_ms"]["total"],
            sum(ms for name, ms in result["timings_ms"].items() if name != "total"),
        )

    def test_single_repair_no_reassessment_and_same_inputs(self):
        result = self.run_pipeline([(repair_evaluation(), {}), (patch_data(), {})])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["delivery_status"], "repair_unverified")
        calls = self.reviewer.request.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["evaluation", "repair"])
        self.assertEqual(calls[0].args[1:4], calls[1].args[1:4])
        old = result["rtm"]["people"][0]["keypoints"]
        new = result["final"]["people"][0]["keypoints"]
        for a, b in zip(old, new):
            if a["name"] == "left_elbow":
                self.assertNotIn("score", b)
                self.assertEqual(b["x"], 400)
            else:
                self.assertEqual(a, b)
        self.assertEqual(result["audit"]["unchanged_joint_count"], 16)
        self.assertEqual(result["keypoint_bounds"][0]["bbox"]["x2"], 400)

    def test_uncertainty_without_issues_never_repairs(self):
        e = evaluation()
        e.update(
            decision="review_needed",
            uncertainties=[
                {
                    "person_index": 0,
                    "joint_names": ["left_wrist"],
                    "region_description": "Behind table",
                    "reason": "Ambiguous",
                }
            ],
        )
        result = self.run_pipeline([(e, {})])
        self.assertEqual(result["delivery_status"], "review_needed")
        self.assertEqual(result["api_calls_started"], 1)
        self.assertEqual(result["final"], result["rtm"])

    def test_unresolved_is_not_successful_visual_verification(self):
        p = patch_data()
        p["edits"] = []
        p["unresolved"] = [{"issue_id": "a", "reason": "Insufficient evidence."}]
        result = self.run_pipeline([(repair_evaluation(), {}), (p, {})])
        self.assertEqual(result["delivery_status"], "repair_unverified_review_needed")
        self.assertEqual(result["final"], result["rtm"])

    def test_errors_not_retried_or_returned_as_pass(self):
        for error in [
            TimeoutError("credential-must-not-leak"),
            GeminiStageError("Gemini HTTP 429; stage not retried"),
        ]:
            with self.subTest(error=type(error).__name__):
                self.reviewer.request.reset_mock()
                result = self.run_pipeline([error])
                self.assertEqual(self.reviewer.request.call_count, 1)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["final"])
                self.assertIsNotNone(result["rtm"])
                self.assertNotIn("credential-must-not-leak", json.dumps(result))

    def test_repair_failure_preserves_base_no_fallback_success(self):
        result = self.run_pipeline([(repair_evaluation(), {}), TimeoutError()])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["api_calls_started"], 2)
        self.assertEqual(result["error"]["stage"], "repair")
        self.assertIsNone(result["final"])
        self.assertIsNotNone(result["rtm"])
        self.assertNotIn("final.json", self.artifacts)

    def test_invalid_evaluation_never_calls_repair(self):
        e = repair_evaluation()
        e["decision"] = "pass"
        result = self.run_pipeline([(e, {})])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["api_calls_started"], 1)

    def test_invalid_patch_is_atomic(self):
        p = patch_data()
        p["edits"].append({**p["edits"][0], "name": "nose"})
        result = self.run_pipeline([(repair_evaluation(), {}), (p, {})])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["rtm"], proposal())
        self.assertIsNone(result["final"])

    def test_empty_rtm_still_evaluated_and_can_add_person(self):
        self.pose.estimate.return_value = []
        e = repair_evaluation()
        e["issues"][0].update(
            type="missing_person",
            person_index=None,
            joint_names=[],
            region_bbox={"x1": 1, "y1": 1, "x2": 900, "y2": 900},
        )
        p = patch_data()
        p["edits"] = []
        person = proposal()["people"][0]
        p["added_people"] = [
            {
                "issue_id": "a",
                "new_person_id": "new0",
                "bbox": person["bbox"],
                "keypoints": [
                    {k: v for k, v in x.items() if k != "score"}
                    for x in person["keypoints"]
                ],
                "evidence": "Separate visible person.",
            }
        ]
        result = self.run_pipeline([(e, {}), (p, {})])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["final"]["people"]), 1)

    def test_invalid_image_calls_no_model(self):
        result = self.pipeline.extract(self.image.with_name("missing.png"))
        self.assertEqual(result["status"], "error")
        self.pose.estimate.assert_not_called()
        self.reviewer.request.assert_not_called()


class SingleCallTest(unittest.TestCase):
    def setUp(self):
        PipelineTest.setUp(self)
        self.pipeline = HybridPosePipeline(self.pose, self.reviewer)

    def run_single(self, e=None, p=None):
        response = {
            "evaluation": e if e is not None else evaluation(),
            "patch": (
                p
                if p is not None
                else {
                    "edits": [],
                    "added_people": [],
                    "removed_people": [],
                    "unresolved": [],
                }
            ),
        }
        self.reviewer.request.return_value = (response, {"saved_raw": True})
        return self.pipeline.extract(self.image, self.artifacts.__setitem__)

    def test_default_single_pass_preserves_all_coordinates(self):
        r = self.run_single()
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["final"], r["rtm"])
        self.assertEqual(r["api_calls_started"], 1)
        self.assertEqual(
            self.reviewer.request.call_args.args[0], "single_review_repair"
        )
        self.assertIn("single_review_repair.response.json", self.artifacts)
        self.assertNotIn("evaluation.response.json", self.artifacts)
        self.assertNotIn("repair.response.json", self.artifacts)

    def test_single_repair_preserves_unedited_no_reevaluation(self):
        r = self.run_single(repair_evaluation(), patch_data())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["api_calls_started"], 1)
        self.reviewer.request.assert_called_once()
        self.assertEqual(r["audit"]["unchanged_joint_count"], 16)
        self.assertFalse(r["visual_reassessment"])
        self.assertEqual(r["delivery_status"], "repair_unverified")
        self.assertEqual(r["rtm"], proposal())
        for a, b in zip(
            r["rtm"]["people"][0]["keypoints"], r["final"]["people"][0]["keypoints"]
        ):
            if a["name"] != "left_elbow":
                self.assertEqual(a, b)
        self.assertGreaterEqual(
            r["timings_ms"]["total"],
            sum(v for k, v in r["timings_ms"].items() if k != "total"),
        )

    def test_invalid_single_patch_atomic_and_raw_preserved(self):
        for kind in ["wrong_type", "absent_coords", "duplicate", "pass_with_patch"]:
            with self.subTest(kind=kind):
                self.artifacts.clear()
                self.reviewer.request.reset_mock()
                e, p = repair_evaluation(), patch_data()
                if kind == "wrong_type":
                    e["issues"][0].update(
                        type="false_or_duplicate_person", joint_names=[]
                    )
                elif kind == "absent_coords":
                    p["edits"][0]["state"] = "not_present"
                elif kind == "duplicate":
                    p["edits"].append(copy.deepcopy(p["edits"][0]))
                else:
                    e = evaluation()
                r = self.run_single(e, p)
                self.assertEqual(r["status"], "error")
                self.assertIsNone(r["final"])
                self.assertEqual(r["rtm"], proposal())
                self.assertNotIn("final.json", self.artifacts)
                self.assertIn("single_review_repair.response.json", self.artifacts)
                self.reviewer.request.assert_called_once()

    def test_single_timeout_does_not_retry(self):
        self.reviewer.request.side_effect = TimeoutError("private")
        r = self.pipeline.extract(self.image)
        self.assertIsNone(r["final"])
        self.assertEqual(r["api_calls_started"], 1)
        self.reviewer.request.assert_called_once()
        self.assertNotIn("private", json.dumps(r))

    def test_single_uncertainty_keeps_base(self):
        e = evaluation()
        e.update(
            decision="review_needed",
            uncertainties=[
                {
                    "person_index": 0,
                    "joint_names": ["left_wrist"],
                    "region_description": "Behind table",
                    "reason": "Ambiguous",
                }
            ],
        )
        r = self.run_single(e)
        self.assertEqual(r["delivery_status"], "review_needed")
        self.assertEqual(r["final"], r["rtm"])

    def test_low_vs_default_payload_only_thinking_config_differs(self):
        payloads = []
        for level in ["default", "low"]:
            client = GeminiReviewer("secret", "test-model", thinking_level=level)
            with patch(
                "urllib.request.urlopen", return_value=GeminiTest().response({})
            ) as call:
                client.request("single_review_repair", b"a", b"b", {"people": []})
            payloads.append(json.loads(call.call_args.args[0].data))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(client.metadata["thinking_level"], level)
        self.assertEqual(
            payloads[1]["generationConfig"].pop("thinkingConfig"),
            {"thinkingLevel": "low"},
        )
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(
            set(payloads[0]["generationConfig"]["responseJsonSchema"]["properties"]),
            {"evaluation", "patch"},
        )

    def test_invalid_modes_fail_before_api(self):
        with self.assertRaises(ValueError):
            HybridPosePipeline(self.pose, self.reviewer, mode="typo")
        with self.assertRaises(ValueError):
            GeminiReviewer("secret", "test-model", thinking_level="typo")


class ValidationTest(unittest.TestCase):
    def test_bad_prediction_rejected(self):
        for mutation in [
            "duplicate_id",
            "duplicate_joint",
            "bool_coord",
            "absent_coord",
            "infinite_score",
        ]:
            data = proposal()
            person = data["people"][0]
            if mutation == "duplicate_id":
                data["people"].append(copy.deepcopy(person))
            elif mutation == "duplicate_joint":
                person["keypoints"][1] = person["keypoints"][0]
            elif mutation == "bool_coord":
                person["keypoints"][0]["x"] = True
            elif mutation == "absent_coord":
                person["keypoints"][0]["state"] = "not_present"
            else:
                person["keypoints"][0]["score"] = float("inf")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                schema.validate_prediction(data)

    def test_nonfinite_rtm_is_null_and_json_serializable(self):
        s = skeleton()
        s.scores[0] = float("nan")
        s.keypoints[1, 0] = float("inf")
        out = normalize_rtm([s], 100, 150, 0.3)
        json.dumps(out, allow_nan=False)
        self.assertEqual(out["people"][0]["keypoints"][0]["state"], "not_present")
        self.assertEqual(out["people"][0]["keypoints"][1]["state"], "not_present")

    def test_degenerate_rtm_box_and_recovered_bounds(self):
        s = skeleton()
        s.scores[:] = 0
        base = normalize_rtm([s], 100, 150, 0.3)
        self.assertIsNone(schema.keypoint_bounds(base)[0]["bbox"])
        out, _ = contracts.merge_repair(
            base, repair_evaluation("missing_joint"), patch_data()
        )
        self.assertEqual(schema.keypoint_bounds(out)[0]["bbox"]["x1"], 400)
        self.assertEqual(out["people"][0]["bbox"], base["people"][0]["bbox"])

    def test_removal_keeps_surviving_id(self):
        base = proposal()
        base["people"].append(copy.deepcopy(base["people"][0]))
        base["people"][1]["person_index"] = 7
        e = repair_evaluation()
        e["issues"][0].update(type="false_or_duplicate_person", joint_names=[])
        p = patch_data()
        p["edits"] = []
        p["removed_people"] = [
            {"issue_id": "a", "person_index": 0, "evidence": "Duplicate."}
        ]
        out, _ = contracts.merge_repair(base, e, p)
        self.assertEqual([x["person_index"] for x in out["people"]], [7])

    def test_font_fallback_no_macos_dependency(self):
        from PIL import ImageFont

        font = ImageFont.load_default()
        with patch(
            "src.hybrid_pose.render.ImageFont.truetype", side_effect=OSError
        ), patch("src.hybrid_pose.render.ImageFont.load_default", return_value=font):
            self.assertEqual(
                overlay(Image.new("RGB", (100, 150)), proposal()).size, (100, 150)
            )


class GeminiTest(unittest.TestCase):
    def setUp(self):
        self.client = GeminiReviewer("secret-test-key", "test-model", timeout=1)

    def response(self, value, finish="STOP"):
        return io.BytesIO(
            json.dumps(
                {
                    "candidates": [
                        {
                            "finishReason": finish,
                            "content": {"parts": [{"text": json.dumps(value)}]},
                        }
                    ]
                }
            ).encode()
        )

    def test_actual_prompt_and_two_images_sent(self):
        with patch(
            "urllib.request.urlopen", return_value=self.response(evaluation())
        ) as call:
            value, _ = self.client.request(
                "evaluation", b"image1", b"image2", {"people": []}
            )
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        parts = payload["contents"][0]["parts"]
        self.assertIn("인물 누락", parts[0]["text"])
        self.assertEqual(sum("inlineData" in p for p in parts), 2)
        self.assertNotIn("secret-test-key", request.full_url)
        self.assertEqual(value, evaluation())
        self.assertEqual(call.call_count, 1)

    def test_http_errors_never_retried(self):
        for status in [400, 401, 403, 429, 500, 503]:
            error = urllib.error.HTTPError(
                "https://example.test", status, "secret-test-key", {}, None
            )
            with self.subTest(status=status), patch(
                "urllib.request.urlopen", side_effect=error
            ) as call:
                with self.assertRaises(GeminiStageError) as raised:
                    self.client.request("evaluation", b"a", b"b", {})
                self.assertNotIn("secret-test-key", str(raised.exception))
                self.assertEqual(call.call_count, 1)

    def test_incomplete_valid_json_still_rejected(self):
        with patch(
            "urllib.request.urlopen",
            return_value=self.response(evaluation(), "MAX_TOKENS"),
        ):
            with self.assertRaises(GeminiStageError):
                self.client.request("evaluation", b"a", b"b", {})

    def test_schema_adaptation_does_not_mutate_local_validation(self):
        before = copy.deepcopy(contracts.REPAIR_SCHEMA)
        adapted = provider_schema(contracts.REPAIR_SCHEMA)
        self.assertNotIn("additionalProperties", adapted)
        self.assertEqual(contracts.REPAIR_SCHEMA, before)

    def test_explicit_model_required_and_path_safe(self):
        for model in [None, "", "../other", "model?key=secret"]:
            with self.assertRaises(ValueError):
                GeminiReviewer("key", model)
