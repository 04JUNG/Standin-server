import copy
import unittest

from src.hybrid_pose import contracts as c, schema as v


def base():
    return {
        "people": [
            {
                "person_index": 0,
                "bbox": {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000},
                "keypoints": [
                    {
                        "name": n,
                        "x": 100 + j * 20,
                        "y": 150 + j * 20,
                        "state": "visible",
                        "score": 0.8,
                    }
                    for j, n in enumerate(v.COCO17)
                ],
            }
        ]
    }


def evaluation(decision="pass"):
    return {
        "decision": decision,
        "issues": (
            []
            if decision != "repair"
            else [
                {
                    "issue_id": "a",
                    "type": "obvious_pose_error",
                    "person_index": 0,
                    "joint_names": ["left_elbow"],
                    "region_bbox": None,
                    "evidence": "Elbow is on a sleeve corner.",
                }
            ]
        ),
        "uncertainties": [],
    }


def repair():
    return {
        "edits": [
            {
                "issue_id": "a",
                "person_index": 0,
                "name": "left_elbow",
                "x": 300,
                "y": 400,
                "state": "visible",
                "evidence": "Actual arm bend.",
            }
        ],
        "added_people": [],
        "removed_people": [],
        "unresolved": [],
    }


class ContractTest(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(
            c.validate_evaluation(evaluation(), base())["decision"], "pass"
        )

    def test_no_false_pass(self):
        e = evaluation("repair")
        e["decision"] = "pass"
        with self.assertRaises(ValueError):
            c.validate_evaluation(e, base())

    def test_unknown_person(self):
        e = evaluation("repair")
        e["issues"][0]["person_index"] = 8
        with self.assertRaises(ValueError):
            c.validate_evaluation(e, base())

    def test_patch_preserves_unedited(self):
        b = base()
        old = copy.deepcopy(b)
        out, a = c.merge_repair(b, evaluation("repair"), repair())
        self.assertEqual(old, b)
        self.assertEqual(a["unchanged_joint_count"], 16)
        self.assertFalse(a["visual_reassessment"])
        for x, y in zip(b["people"][0]["keypoints"], out["people"][0]["keypoints"]):
            if x["name"] != "left_elbow":
                self.assertEqual(x, y)

    def test_unrelated_edit_rejected(self):
        p = repair()
        p["edits"][0]["name"] = "nose"
        with self.assertRaises(ValueError):
            c.merge_repair(base(), evaluation("repair"), p)

    def test_unknown_issue_rejected(self):
        p = repair()
        p["edits"][0]["issue_id"] = "unknown"
        with self.assertRaises(ValueError):
            c.merge_repair(base(), evaluation("repair"), p)

    def test_silent_omission_rejected(self):
        p = repair()
        p["edits"] = []
        with self.assertRaises(ValueError):
            c.merge_repair(base(), evaluation("repair"), p)
        p["unresolved"] = [{"issue_id": "a", "reason": "Cannot locate"}]
        out, _ = c.merge_repair(base(), evaluation("repair"), p)
        self.assertEqual(out, base())

    def test_add_and_stable_ids(self):
        b = base()
        b["people"][0]["person_index"] = 5
        e = evaluation("repair")
        e["issues"][0].update(
            type="missing_person",
            person_index=None,
            joint_names=[],
            region_bbox={"x1": 1, "y1": 1, "x2": 800, "y2": 900},
        )
        p = repair()
        p["edits"] = []
        new = base()["people"][0]
        p["added_people"] = [
            {
                "issue_id": "a",
                "new_person_id": "new0",
                "bbox": new["bbox"],
                "keypoints": [
                    {k: x[k] for k in ["name", "x", "y", "state"]}
                    for x in new["keypoints"]
                ],
                "evidence": "Separate person",
            }
        ]
        out, _ = c.merge_repair(b, e, p)
        self.assertEqual([x["person_index"] for x in out["people"]], [5, 6])

    def test_bad_coord(self):
        for x in [True, 1001, float("nan")]:
            p = repair()
            p["edits"][0]["x"] = x
            with self.assertRaises(ValueError):
                c.merge_repair(base(), evaluation("repair"), p)
