from io import BytesIO
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from api.app import STATE, app
from src.fullbody_routing import FullBodySearch
from src.pipeline import CFG, Pipeline
from src.schema import Action, BBox, Relationship, Shot, Skeleton, View, VLMAnalysis
from src.search_routing import DescribeOutcome, PoseDescription


class ArmContactCorrectionTest(TestCase):
    def test_analyze_uses_the_same_corrected_joints_for_search_and_response(self):
        points = np.array([
            [476, 184], [504, 144], [440, 147], [547, 149], [395, 154],
            [615, 331], [343, 324], [639, 551], [343, 615], [648, 707],
            [192, 508], [559, 674], [393, 676], [594, 951], [421, 955],
            [592, 991], [433, 974],
        ], dtype=np.float32)
        expected = points.copy()
        expected[[8, 10]] = points[[10, 8]]
        image = BytesIO()
        Image.new("RGB", (840, 1100)).save(image, format="PNG")

        cases = (
            ("compact tag", points, "한 손 허리 · 한 팔 내리기", {"left": "lowered"}, True),
            ("structured waist", points, "한 손을 허리에 대고 반대 팔을 내림",
             {"left": "lowered", "right": "waist"}, True),
            ("no waist evidence", points, "한쪽 팔을 굽히고 반대 팔은 아래로 내림",
             {"left": "lowered"}, False),
            ("already correct", expected, "한 손 허리 · 한 팔 내리기", {"left": "lowered"}, False),
            ("different arm tagged", points, "한 손 허리 · 한 팔 내리기", {"left": "waist"}, False),
        )
        for label, coordinates, body, arm_states, applied in cases:
            with self.subTest(label=label):
                original = coordinates.copy()
                pose = SimpleNamespace(
                    self_detecting=True,
                    estimate=Mock(return_value=[Skeleton(coordinates, np.full(17, .85, dtype=np.float32))]),
                    estimate_crop_candidates=Mock(return_value=[]),
                )
                vlm = SimpleNamespace(
                    analyze=Mock(return_value=VLMAnalysis(
                        1, Shot.FULL_HALF, Action.STANDING, View.FRONT, Relationship.SOLO,
                        approx_boxes=[BBox(150, 120, 700, 1010)], lower_body_visible=[True],
                    )),
                    describe_people=Mock(return_value=DescribeOutcome("ok", {0: PoseDescription(
                        actions=["pose_for_display"], body_details=body,
                        search_text="한 팔을 굽히고 반대쪽 팔을 내리고 서 있다.",
                        certainty="medium", arm_states=arm_states, visible_scope="full",
                    )})),
                )
                engine = FullBodySearch([], None, None)
                engine.search_evidence = SimpleNamespace(
                    observe_depth=Mock(return_value=None),
                    semantic_contacts=lambda branch, *_: branch,
                    rerank=lambda ranked, *_: (ranked, None),
                )
                engine.geometry_search = Mock(return_value={"status": "no_result", "hits": []})
                engine.semantic_search = Mock(return_value={"status": "no_result", "hits": []})
                pipeline = Pipeline([], vlm_client=vlm, pose_model=pose, fullbody_search=engine,
                                    skeleton_repairer=False)
                with patch.dict(STATE, {"pipeline": pipeline}), patch.dict(
                    "os.environ", {"PARALLEL_POSE_DESCRIPTION": "0"},
                ), patch.multiple(CFG, vlm_hand_shapes=False, vlm_person_tags=False,
                                  body_matching_mode="off"):
                    response = TestClient(app).post(
                        "/analyze", files={"file": ("rough.png", image.getvalue(), "image/png")})
                self.assertEqual(response.status_code, 200, response.text)
                person = response.json()["people"][0]
                engine.geometry_search.assert_called_once()
                engine.semantic_search.assert_called_once()
                vlm.describe_people.assert_called_once()
                geometry = engine.geometry_search.call_args.args[0]
                np.testing.assert_array_equal(geometry["keypoints"], expected if applied else original)
                np.testing.assert_array_equal(person["keypoints"], geometry["keypoints"])
                np.testing.assert_array_equal(coordinates, original)
                correction = person["quality_trace"].get("arm_contact_correction")
                self.assertEqual(correction["side"] if correction else None, "right" if applied else None)
