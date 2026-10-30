"""Explicit evidence, anatomical order and complete skin measurements are gates."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.refine_v4.leg_adapter import plan_legs
from src.refine_v4.pipeline import Stage, skin_nonregression


def rejected(fn):
    try: fn()
    except ValueError: return
    raise AssertionError("expected rejection")


def test_anatomical_side_and_cross_family():
    for first, second, expected in [("오른", "왼", "right"), ("왼", "오른", "left")]:
        plan = plan_legs(f"앉아서 {first}쪽 허벅지를 {second}쪽 허벅지 위로 포개고 무릎을 굽힌다",
                         selection="both", source="user_confirmed", reference="user-photo")
        assert plan["intent"]["top_side"] == expected
        assert plan["intent"]["cross_style"] == "seated_knee_over_knee"
        assert not plan["automatic_image_extraction"]
    rejected(lambda: plan_legs("앉아서 다리를 꼰다", selection="both", source="user_confirmed", reference="photo"))
    rejected(lambda: plan_legs("오른발목을 왼발목 위에 교차", selection="both", source="user_confirmed", reference="photo"))


def test_no_unattributed_or_screen_side_application():
    rejected(lambda: plan_legs("무릎을 굽힘", selection="right", source="guess", reference=""))
    rejected(lambda: Stage("legs", ("screen_right",), None).roots())
    rejected(lambda: Stage("legs", ("left", "left"), None).roots())


def test_skin_missing_uncertain_nan_and_new_collision_rejected():
    clean = {"regions":14, "collisions":[], "region_closure_failures":{}}
    assert skin_nonregression(clean, clean)["accepted"]
    assert not skin_nonregression(clean, {})["accepted"]
    uncertain = copy.deepcopy(clean); uncertain["region_closure_failures"] = {"Torso":1}
    assert not skin_nonregression(uncertain, uncertain)["accepted"]
    for depth, expected in [(float("nan"),False), (2.001,False), (1.99,True)]:
        changed = copy.deepcopy(clean)
        changed["collisions"] = [{"regions":["RightHand","LeftThigh"], "max_depth_mm":depth}]
        assert skin_nonregression(clean, changed)["accepted"] is expected


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("PASS", name)
