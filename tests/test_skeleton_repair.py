"""04J-286: Gemini 스켈레톤 보완 연결 계약(가짜 reviewer, 네트워크 없음)."""
import numpy as np

from src.config import CFG
from src.schema import BBox, Skeleton
from src.skeleton_extraction import PersonSlot, finalize_slot
from src.skeleton_repair import (REPAIR_SOURCE, RepairConfig, SkeletonRepairer,
                                 build_repairer, slots_to_prediction)
from src.refine_policy import structural_refine_allowed
from src.hybrid_pose.schema import COCO17

W, H = 400, 600


def _standing(cx=200.0, top=100.0, scale=1.0):
    p = np.zeros((17, 2), np.float32)
    s = scale
    p[0] = (cx, top)
    p[1], p[2] = (cx - 5 * s, top - 5 * s), (cx + 5 * s, top - 5 * s)
    p[3], p[4] = (cx - 12 * s, top), (cx + 12 * s, top)
    p[5], p[6] = (cx - 40 * s, top + 60 * s), (cx + 40 * s, top + 60 * s)
    p[7], p[8] = (cx - 50 * s, top + 140 * s), (cx + 50 * s, top + 140 * s)
    p[9], p[10] = (cx - 55 * s, top + 210 * s), (cx + 55 * s, top + 210 * s)
    p[11], p[12] = (cx - 30 * s, top + 220 * s), (cx + 30 * s, top + 220 * s)
    p[13], p[14] = (cx - 32 * s, top + 330 * s), (cx + 32 * s, top + 330 * s)
    p[15], p[16] = (cx - 34 * s, top + 440 * s), (cx + 34 * s, top + 440 * s)
    return p


def _slot(slot_id, points=None, scores=None, box=(120, 80, 280, 560)):
    slot = PersonSlot(slot_id=slot_id, slot_origin="vlm",
                      vlm_box=BBox(*box, source="vlm"))
    if points is not None:
        slot.skeleton = Skeleton(keypoints=points, scores=scores)
        slot.skeleton_source = "full_image"
    return finalize_slot(slot, CFG)


class FakeReviewer:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, 0
        self.metadata = {}

    def request(self, stage, original, marked, ctx, evaluation=None):
        self.calls += 1
        assert stage == "single_review_repair"
        assert original[:4] == b"\x89PNG" and marked[:4] == b"\x89PNG"
        self.ctx = ctx
        if self.error:
            raise self.error
        return self.response, {}


def _repairer(response=None, error=None, mode="auto"):
    return SkeletonRepairer(FakeReviewer(response, error), RepairConfig(mode=mode, model="m"))


def _image():
    from PIL import Image
    return Image.new("RGB", (W, H), "white")


def _empty_patch():
    return {"edits": [], "added_people": [], "removed_people": [], "unresolved": []}


def test_all_valid_skips_gemini_call():
    pts = _standing()
    slot = _slot(0, pts, np.full(17, 0.9, np.float32))
    assert slot.state == "valid"
    rep = _repairer({"evaluation": {}, "patch": {}})
    trace = rep.repair(_image(), [slot], W, H, CFG)
    assert rep.reviewer.calls == 0 and trace["called"] is False and trace["status"] == "skipped"


def test_legs_filled_promote_partial_and_block_refine():
    pts = _standing()
    scores = np.full(17, 0.9, np.float32)
    scores[11:17] = 0.05                       # 다리 못 잡음
    slot = _slot(0, pts, scores)
    assert slot.state != "valid"
    legs = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    edits = []
    for name in legs:
        j = COCO17.index(name)
        edits.append({"issue_id": "i1", "person_index": 0, "name": name,
                      "x": float(pts[j, 0] / W * 1000), "y": float(pts[j, 1] / H * 1000),
                      "state": "visible", "evidence": "legs drawn"})
    response = {"evaluation": {"decision": "repair", "uncertainties": [], "issues": [
        {"issue_id": "i1", "type": "missing_joint", "person_index": 0, "joint_names": legs,
         "region_bbox": None, "evidence": "legs visible"}]},
        "patch": dict(_empty_patch(), edits=edits)}
    rep = _repairer(response)
    trace = rep.repair(_image(), [slot], W, H, CFG)
    assert trace["status"] == "ok" and rep.reviewer.calls == 1
    assert slot.skeleton_source == REPAIR_SOURCE
    assert slot.state == "valid", slot.reasons
    assert float(slot.skeleton.scores[13]) > CFG.skeleton_kpt_threshold
    assert float(slot.skeleton.scores[0]) == np.float32(0.9)   # 손대지 않은 관절 점수 보존
    assert trace["repaired_slots"][0]["state_after"] == "valid"
    assert not structural_refine_allowed(
        skeleton_state=slot.state, coverage_class=slot.evidence.coverage_class,
        refinable_limbs=slot.evidence.refinable_limbs, slot_origin=slot.slot_origin,
        skeleton_source=slot.skeleton_source)


def test_added_person_goes_to_empty_slot_by_overlap():
    good = _slot(0, _standing(100), np.full(17, 0.9, np.float32), box=(40, 80, 160, 560))
    empty = _slot(1, box=(240, 80, 360, 560))
    assert empty.state == "missing"
    pts = _standing(300)
    kps = [{"name": n, "x": float(pts[j, 0] / W * 1000), "y": float(pts[j, 1] / H * 1000),
            "state": "visible"} for j, n in enumerate(COCO17)]
    response = {"evaluation": {"decision": "repair", "uncertainties": [], "issues": [
        {"issue_id": "p", "type": "missing_person", "person_index": None, "joint_names": [],
         "region_bbox": {"x1": 600, "y1": 133, "x2": 900, "y2": 933}, "evidence": "person"}]},
        "patch": dict(_empty_patch(), added_people=[{"issue_id": "p", "new_person_id": "n1",
                      "bbox": {"x1": 600, "y1": 133, "x2": 900, "y2": 933},
                      "keypoints": kps, "evidence": "drawn"}])}
    rep = _repairer(response)
    trace = rep.repair(_image(), [good, empty], W, H, CFG)
    assert trace["added_people_matched"] == 1
    assert empty.skeleton is not None and empty.state == "valid"
    assert good.skeleton_source == "full_image"


def test_reviewer_error_keeps_rtm_joints():
    pts = _standing()
    scores = np.full(17, 0.9, np.float32)
    scores[11:17] = 0.05
    slot = _slot(0, pts, scores)
    before = (slot.state, slot.skeleton_source, slot.skeleton.scores.copy())
    trace = _repairer(error=RuntimeError("boom")).repair(_image(), [slot], W, H, CFG)
    assert trace["status"] == "error" and trace["message"] is None
    assert (slot.state, slot.skeleton_source) == before[:2]
    assert np.array_equal(slot.skeleton.scores, before[2])


def test_invalid_response_is_rejected_atomically():
    pts = _standing()
    scores = np.full(17, 0.9, np.float32)
    scores[11:17] = 0.05
    slot = _slot(0, pts, scores)
    # 지적하지 않은 팔 관절 수정 → 계약 위반 → 아무것도 안 바뀜
    response = {"evaluation": {"decision": "repair", "uncertainties": [], "issues": [
        {"issue_id": "i1", "type": "missing_joint", "person_index": 0,
         "joint_names": ["left_knee"], "region_bbox": None, "evidence": "x"}]},
        "patch": dict(_empty_patch(), edits=[{"issue_id": "i1", "person_index": 0,
                      "name": "left_wrist", "x": 1.0, "y": 1.0, "state": "visible",
                      "evidence": "x"}])}
    trace = _repairer(response).repair(_image(), [slot], W, H, CFG)
    assert trace["status"] == "error" and slot.skeleton_source == "full_image"


def test_prediction_uses_vlm_box_for_empty_slot():
    pred = slots_to_prediction([_slot(0, box=(40, 60, 200, 300))], W, H, 0.3)
    person = pred["people"][0]
    assert person["bbox"] == {"x1": 100.0, "y1": 100.0, "x2": 500.0, "y2": 500.0}
    assert all(k["state"] == "not_present" for k in person["keypoints"])


def test_build_repairer_off_for_mock_or_env(monkeypatch):
    assert build_repairer(None) is None or CFG.vlm_provider.lower() == "gemini"
    monkeypatch.setenv("SKELETON_REPAIR", "off")
    assert build_repairer(None) is None


def test_pipeline_runs_repair_before_routing_and_records_trace():
    from tests.test_search_routing import _CountingVLM, _RoughPose, _StubSearcher, _Img, _signature
    from src.library import build_synthetic_index
    from src.pipeline import Pipeline

    class Spy:
        def __init__(self):
            self.calls = []

        def repair(self, image, slots, w, h, cfg):
            self.calls.append([s.state for s in slots])
            return {"called": True, "status": "ok", "targets": [1], "elapsed_ms": 3.0,
                    "repaired_slots": []}

    old = CFG.search_routing_enabled
    CFG.search_routing_enabled = True
    try:
        base = Pipeline(build_synthetic_index(), vlm_client=_CountingVLM(), pose_model=_RoughPose(),
                        semantic_searcher=_StubSearcher(), skeleton_repairer=False)
        assert base.skeleton_repairer is None
        before = base.process_cut(_Img("full_half standing front 2p rough"))
        spy = Spy()
        pipe = Pipeline(build_synthetic_index(), vlm_client=_CountingVLM(), pose_model=_RoughPose(),
                        semantic_searcher=_StubSearcher(), skeleton_repairer=spy)
        after = pipe.process_cut(_Img("full_half standing front 2p rough"))
        assert len(spy.calls) == 1 and len(spy.calls[0]) == 2
        assert after.routing_summary["skeleton_repair"]["status"] == "ok"
        assert any("Gemini 스켈레톤 보완" in n for n in after.notes)
        # 보완이 아무것도 안 바꾸면 Top-5는 그대로
        assert _signature(before)["candidates"] == _signature(after)["candidates"]
    finally:
        CFG.search_routing_enabled = old


def test_repair_that_makes_state_worse_is_reverted():
    pts = _standing()
    scores = np.full(17, 0.9, np.float32)
    scores[[13, 15]] = 0.05                    # 왼다리 아래만 못 잡음 → partial
    slot = _slot(0, pts, scores)
    before = slot.state
    torso = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    edits = [{"issue_id": "i1", "person_index": 0, "name": n, "x": None, "y": None,
              "state": "not_present", "evidence": "x"} for n in torso]
    response = {"evaluation": {"decision": "repair", "uncertainties": [], "issues": [
        {"issue_id": "i1", "type": "obvious_pose_error", "person_index": 0, "joint_names": torso,
         "region_bbox": None, "evidence": "x"}]}, "patch": dict(_empty_patch(), edits=edits)}
    trace = _repairer(response).repair(_image(), [slot], W, H, CFG)
    assert trace["status"] == "ok" and trace["reverted_slots"][0]["state_before"] == before
    assert slot.state == before and slot.skeleton_source == "full_image"
    assert "vlm_repair:edit" not in slot.reasons


def test_salvage_keeps_valid_issue_when_other_issue_breaks_contract():
    pts = _standing()
    scores = np.full(17, 0.9, np.float32)
    scores[11:17] = 0.05
    slot = _slot(0, pts, scores)
    legs = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    edits = [{"issue_id": "good", "person_index": 0, "name": n,
              "x": float(pts[COCO17.index(n), 0] / W * 1000), "y": float(pts[COCO17.index(n), 1] / H * 1000),
              "state": "visible", "evidence": "legs"} for n in legs]
    edits.append({"issue_id": "bad", "person_index": 0, "name": "nose", "x": 1.0, "y": 1.0,
                  "state": "visible", "evidence": "x"})
    response = {"evaluation": {"decision": "repair", "uncertainties": [], "issues": [
        {"issue_id": "good", "type": "missing_joint", "person_index": 0, "joint_names": legs,
         "region_bbox": None, "evidence": "legs"},
        {"issue_id": "bad", "type": "missing_joint", "person_index": 0, "joint_names": ["left_wrist"],
         "region_bbox": None, "evidence": "x"}]}, "patch": dict(_empty_patch(), edits=edits)}
    trace = _repairer(response).repair(_image(), [slot], W, H, CFG)
    assert trace["status"] == "ok" and trace["decision"] == "repair_partial"
    assert trace["issues_kept"] == ["good"] and trace["issues_dropped"][0]["issue_id"] == "bad"
    assert slot.state == "valid" and float(slot.skeleton.scores[0]) == np.float32(0.9)


def test_added_people_fill_boxless_empty_slots_left_to_right():
    a = PersonSlot(slot_id=0, slot_origin="vlm"); finalize_slot(a, CFG)
    b = PersonSlot(slot_id=1, slot_origin="vlm"); finalize_slot(b, CFG)
    rep = _repairer(None)
    assert rep.needs([a, b]) == [0, 1]
    adds, issues = [], []
    for n, cx in (("right", 300), ("left", 100)):
        pts = _standing(cx)
        kps = [{"name": nm, "x": float(pts[j, 0] / W * 1000), "y": float(pts[j, 1] / H * 1000),
                "state": "visible"} for j, nm in enumerate(COCO17)]
        box = {"x1": (cx - 60) / W * 1000, "y1": 100.0, "x2": (cx + 60) / W * 1000, "y2": 950.0}
        issues.append({"issue_id": n, "type": "missing_person", "person_index": None, "joint_names": [],
                       "region_bbox": box, "evidence": "p"})
        adds.append({"issue_id": n, "new_person_id": n, "bbox": box, "keypoints": kps, "evidence": "p"})
    rep.reviewer.response = {"evaluation": {"decision": "repair", "uncertainties": [], "issues": issues},
                             "patch": dict(_empty_patch(), added_people=adds)}
    trace = rep.repair(_image(), [a, b], W, H, CFG)
    assert trace["added_people_matched"] == 2
    assert float(a.skeleton.keypoints[0, 0]) < float(b.skeleton.keypoints[0, 0])
