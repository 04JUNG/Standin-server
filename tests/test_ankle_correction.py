"""Contact-preserving stepwise ankle correction on a synthetic T-pose skeleton."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_curation import ankle_correction as ac
from pose_curation import plausibility as pl
from test_pose_plausibility import _arms_down, _frame, _skeleton


def _ankle(joints, frame, side="right"):
    sk = pl.Skeleton(joints)
    _, rot, _ = pl.fk_world(joints, frame)
    return ac.ankle_angles(sk, rot, side, pl.SIDES[side][1])


def _sole(joints, frame, side="right"):
    sk = pl.Skeleton(joints)
    pos, rot, _ = pl.fk_world(joints, frame)
    return ac.sole_points(pos, rot, ac.feet(sk)[side])


def test_rest_feet_are_planted_flat():
    j = _skeleton()
    contact, floor = ac.planted(j, _frame(j, **_arms_down()))
    assert set(contact) == {"left", "right"}
    assert set(contact["right"]) == {"heel", "ball", "tip"}
    for p in _sole(j, _frame(j, **_arms_down())).values():
        assert abs(p[1] - floor) < 1e-6


def test_toes_up_heel_planted_is_lowered_stepwise():
    j = _skeleton()
    # Straight knee, toes pulled up 40 deg: only the heel stays on the floor.
    frame = _frame(j, **_arms_down(RightFoot=(0, -40, 0)))
    contact, floor = ac.planted(j, frame)
    assert set(contact["right"]) == {"heel"}
    assert _ankle(j, frame)[0] < -35
    result = ac.correct_ankles(j, frame)
    assert result["status"] == "corrected", result["steps"]
    assert [s["alpha"] for s in result["steps"]] == list(ac.STEPS)
    gains = [s["ankle_gain"] for s in result["steps"]]
    assert gains == sorted(gains), gains  # each step corrects a bit more
    fixed = result["frame"]
    assert _ankle(j, fixed)[0] > -22  # inside the straight-knee normal range
    heel0, heel1 = contact["right"]["heel"], _sole(j, fixed)["heel"]
    leg = ac.leg_length(pl.Skeleton(j))
    assert np.linalg.norm(heel1 - heel0) < ac.GATE_CONTACT * leg
    pos, _, _ = pl.fk_world(j, fixed)
    assert pos[:, 1].min() > floor - ac.GATE_FLOOR * leg


def test_reversed_planted_foot_is_rebuilt_on_the_floor():
    j = _skeleton()
    frame = _frame(j, **_arms_down(RightFoot=(0, 0, 175)))
    result = ac.correct_ankles(j, frame)
    assert result["status"] == "gross" and result["gross_sides"] == ["right"]
    sk = pl.Skeleton(j)
    pos, _, _ = pl.fk_world(j, result["frame"])
    toe = pos[sk.role["right_toe"]] - pos[sk.role["right_foot"]]
    assert toe[2] > 0  # toes forward again
    _, floor = ac.planted(j, frame)
    for p in _sole(j, result["frame"]).values():
        assert abs(p[1] - floor) < ac.GATE_CONTACT * ac.leg_length(sk)


def test_airborne_foot_is_left_to_plain_clamping():
    j = _skeleton()
    # Right knee bent 90 with the foot behind: not on the floor.
    frame = _frame(j, **_arms_down(RightLeg=(0, 90, 0), RightFoot=(0, -60, 0)))
    contact, _ = ac.planted(j, frame)
    assert "right" not in contact
    assert ac.correct_ankles(j, frame)["status"] == "nothing"


def test_pipeline_keeps_planted_feet_on_the_floor():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from audit_pose_plausibility import corrected_frame
    j = _skeleton()
    frame = _frame(j, **_arms_down(RightFoot=(0, -40, 0), LeftForeArm=(0, 0, 40)))
    fixed, changed, ankle = corrected_frame(j, frame)
    assert "left_elbow" in changed and "planted_corrected" in changed
    contact, _ = ac.planted(j, frame)
    leg = ac.leg_length(pl.Skeleton(j))
    assert np.linalg.norm(_sole(j, fixed)["heel"] - contact["right"]["heel"]) < ac.GATE_CONTACT * leg


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
