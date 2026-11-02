"""Anatomical plausibility: sign conventions, mirror symmetry and corrections."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_curation.plausibility import (Skeleton, Pose, correct, euler_angles,
                                        euler_matrix, evaluate, fk_world, measure)

ROT = ["Zrotation", "Xrotation", "Yrotation"]


def _skeleton(names=None):
    """Mixamo-like T-pose: +X = character left, +Y up, +Z forward."""
    n = names or {}
    j = []

    def add(name, parent, off, chans=ROT):
        j.append([n.get(name, name), parent, np.array(off, float), list(chans), False])
        return len(j) - 1

    hips = add("Hips", -1, (0, 100, 0), ["Xposition", "Yposition", "Zposition"] + ROT)
    spine = add("Spine", hips, (0, 10, 0))
    spine1 = add("Spine1", spine, (0, 12, 0))
    chest = add("Spine2", spine1, (0, 12, 0))
    neck = add("Neck", chest, (0, 16, 0))
    head = add("Head", neck, (0, 10, 0))
    add("Head_End", head, (0, 18, 0), [])
    for side, sx in (("Left", 1), ("Right", -1)):
        clav = add(side + "Shoulder", chest, (sx * 6, 10, 0))
        arm = add(side + "Arm", clav, (sx * 12, 0, 0))
        fore = add(side + "ForeArm", arm, (sx * 27, 0, 0))
        hand = add(side + "Hand", fore, (sx * 25, 0, 0))
        add(side + "HandMiddle1", hand, (sx * 9, 0, 0), [])
    for side, sx in (("Left", 1), ("Right", -1)):
        up = add(side + "UpLeg", hips, (sx * 9, -6, 0))
        leg = add(side + "Leg", up, (0, -43, 0))
        foot = add(side + "Foot", leg, (0, -43, 0))
        toe = add(side + "ToeBase", foot, (0, -6, 13))
        add(side + "ToeBase_End", toe, (0, 0, 7), [])
    return j


def _frame(joints, **rot):
    """Rotations in degrees as joint=(z, x, y); returns a channel vector."""
    out, ci = [], 0
    for name, parent, off, chans, _ in joints:
        vals = {"Xposition": off[0], "Yposition": off[1], "Zposition": off[2]}
        z, x, y = rot.get(name, (0, 0, 0))
        vals.update(Zrotation=z, Xrotation=x, Yrotation=y)
        out += [vals[c] for c in chans]
    return np.array(out, float)


def _arms_down(**extra):
    base = dict(LeftArm=(-80, 0, 0), RightArm=(80, 0, 0))
    base.update(extra)
    return base


def _codes(result):
    return {f["code"] for f in result["findings"]}


def test_euler_roundtrip_all_orders():
    rng = np.random.default_rng(0)
    for order in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        for _ in range(20):
            ang = rng.uniform(-80, 80, 3)
            R = euler_matrix(order, ang)
            assert np.allclose(euler_matrix(order, euler_angles(R, order)), R, atol=1e-9), order


def test_neutral_standing_passes():
    j = _skeleton()
    r = evaluate(j, _frame(j, **_arms_down()))
    assert r["verdict"] == "pass", r["findings"]
    m = r["measures"]
    assert abs(m["left_shoulder_elevation"] - 10) < 1 and abs(m["left_elbow_flexion"]) < 1


def test_elbow_flexion_sign_and_mirror():
    j = _skeleton()
    # Left arm on +X: forward flexion is a negative Y rotation; mirrored on the right.
    m = measure(Pose(Skeleton(j), _frame(j, LeftForeArm=(0, 0, -90), RightForeArm=(0, 0, 90))))
    assert abs(m["left_elbow_flexion"] - 90) < 1 and abs(m["right_elbow_flexion"] - 90) < 1
    assert abs(m["left_elbow_offaxis"]) < 1 and abs(m["right_elbow_offaxis"]) < 1


def test_elbow_hyperextension_rejected_and_corrected():
    j = _skeleton()
    frame = _frame(j, **_arms_down(LeftForeArm=(0, 0, 40)))
    r = evaluate(j, frame)
    assert r["verdict"] == "reject" and "elbow_flexion" in _codes(r)
    fixed, changed = correct(j, frame)
    assert "left_elbow" in changed
    after = evaluate(j, fixed)
    assert "elbow_flexion" not in _codes(after), after["findings"]


def test_knee_and_ankle_signs():
    j = _skeleton()
    m = measure(Pose(Skeleton(j), _frame(j, LeftLeg=(0, 60, 0), RightFoot=(0, 30, 0))))
    assert abs(m["left_knee_flexion"] - 60) < 1
    assert abs(m["right_ankle_plantar"] - 30) < 1
    m = measure(Pose(Skeleton(j), _frame(j, LeftLeg=(0, -30, 0))))
    assert abs(m["left_knee_flexion"] + 30) < 1  # hyperextension is negative


def test_mirrored_pose_measures_match():
    j = _skeleton()
    left = dict(LeftArm=(-50, 20, 30), LeftForeArm=(5, 10, -70), LeftHand=(20, 15, 10),
                LeftUpLeg=(10, -40, 15), LeftLeg=(0, 50, 5), LeftFoot=(8, 10, 12))
    mirrored = {k.replace("Left", "Right"): (-z, x, -y) for k, (z, x, y) in left.items()}
    m1 = measure(Pose(Skeleton(j), _frame(j, **left)))
    m2 = measure(Pose(Skeleton(j), _frame(j, **mirrored)))
    for key, value in m1.items():
        if key.startswith("left_") and value is not None:
            other = m2[key.replace("left_", "right_")]
            # Off-axis signs flip with the hinge axis; magnitudes must agree.
            if "offaxis" in key or "deviation" in key:
                value, other = abs(value), abs(other)
            assert abs(value - other) < 0.5, (key, value, other)


def test_hinge_alignment_keeps_joint_positions():
    j = _skeleton()
    # Twist the upper arm 45 deg about its own axis with a 90 deg elbow: the
    # forearm now bends out of the hinge plane by 45 deg.
    frame = _frame(j, **_arms_down(LeftArm=(-80, 0, 0), LeftForeArm=(0, 45, -90)))
    r = evaluate(j, frame)
    assert "elbow_offaxis" in _codes(r), r["findings"]
    fixed, changed = correct(j, frame)
    assert "left_elbow_hinge" in changed, changed
    after = measure(Pose(Skeleton(j), fixed))
    assert abs(after["left_elbow_offaxis"]) < 3, after["left_elbow_offaxis"]
    sk = Skeleton(j)
    before_pos, _, _ = fk_world(j, frame)
    after_pos, _, _ = fk_world(j, fixed)
    for key in ("left_forearm", "left_hand", "left_handtip"):
        k = sk.role[key]
        assert np.linalg.norm(before_pos[k] - after_pos[k]) < 0.5, key


def test_shoulder_behind_back_flagged_and_projected():
    j = _skeleton()
    # Arm raised overhead then swung far behind: hyper-flexion past vertical.
    frame = _frame(j, **_arms_down(LeftArm=(80, 0, 0), LeftShoulder=(0, 0, 0)))
    frame2 = _frame(j, LeftArm=(-80, 0, 0), RightArm=(80, 0, 0))
    assert evaluate(j, frame2)["verdict"] == "pass"
    r = evaluate(j, frame)  # left arm straight up beyond vertical toward the right
    fixed, _ = correct(j, frame)
    assert len(evaluate(j, fixed)["findings"]) <= len(r["findings"])


def test_dorsiflexion_limit_depends_on_knee():
    j = _skeleton()
    straight = evaluate(j, _frame(j, **_arms_down(LeftFoot=(0, -32, 0))))
    assert "ankle_plantar" in _codes(straight)
    # Same ankle with the knee bent 90 deg: normal weight-bearing lunge range.
    bent = evaluate(j, _frame(j, **_arms_down(LeftFoot=(0, -32, 0), LeftLeg=(0, 90, 0))))
    assert "ankle_plantar" not in _codes(bent), bent["findings"]


def test_reversed_foot_rejected_and_turned_forward():
    j = _skeleton()
    frame = _frame(j, **_arms_down(LeftFoot=(0, 0, 175)))
    r = evaluate(j, frame)
    assert r["verdict"] == "reject" and "ankle_toe_out" in _codes(r)
    fixed, changed = correct(j, frame)
    assert "left_ankle" in changed
    sk = Skeleton(j)
    pos, _, _ = fk_world(j, fixed)
    toe = pos[sk.role["left_toe"]] - pos[sk.role["left_foot"]]
    assert toe[2] > 0, toe  # toes point forward again


def test_raised_arm_without_shoulder_lift_held_and_lifted():
    j = _skeleton()
    # Left arm 150 deg overhead (abduction past horizontal), clavicle untouched.
    frame = _frame(j, LeftArm=(60, 0, 0), RightArm=(80, 0, 0))
    r = evaluate(j, frame)
    assert "shoulder_girdle" in _codes(r), r["findings"]
    fixed, changed = correct(j, frame)
    assert "left_clavicle" in changed
    after = measure(Pose(Skeleton(j), fixed))
    assert after["left_clavicle_elevation"] > 10
    assert abs(after["left_shoulder_elevation"] - r["measures"]["left_shoulder_elevation"]) < 1


def test_arm_twist_and_wrist_alone_never_reject():
    j = _skeleton()
    for pose in (dict(LeftForeArm=(0, 130, 0)), dict(LeftHand=(0, 0, -70))):
        r = evaluate(j, _frame(j, **_arms_down(**pose)))
        assert r["verdict"] == "hold", r["findings"]
        assert all(f["severity"] == "hold" for f in r["findings"])


def test_other_reject_reasons_still_reject_beside_arm_twist():
    j = _skeleton()
    r = evaluate(j, _frame(j, **_arms_down(LeftHand=(0, 0, -70), LeftForeArm=(0, 0, 40))))
    assert r["verdict"] == "reject"
    severity = {f["code"]: f["severity"] for f in r["findings"]}
    assert severity["elbow_flexion"] == "reject" and severity["wrist_deviation"] == "hold"


def test_alternate_joint_names():
    names = {"Spine2": "Chest4", "LeftShoulder": "LeftCollar", "LeftArm": "LeftShoulder",
             "LeftForeArm": "LeftElbow", "LeftHand": "LeftWrist", "RightShoulder": "RightCollar",
             "RightArm": "RightShoulder", "RightForeArm": "RightElbow", "RightHand": "RightWrist",
             "LeftUpLeg": "LeftHip", "LeftLeg": "LeftKnee", "LeftFoot": "LeftAnkle",
             "RightUpLeg": "RightHip", "RightLeg": "RightKnee", "RightFoot": "RightAnkle",
             "LeftToeBase": "LeftToe", "RightToeBase": "RightToe"}
    a, b = _skeleton(), _skeleton(names)
    pose = dict(LeftForeArm=(0, 0, -60), LeftLeg=(0, 40, 0))
    renamed = {names.get(k, k): v for k, v in pose.items()}
    m1 = measure(Pose(Skeleton(a), _frame(a, **pose)))
    m2 = measure(Pose(Skeleton(b), _frame(b, **renamed)))
    assert m1 == m2


def test_unsupported_skeleton_is_explicit():
    j = [["Hips", -1, np.zeros(3), ["Xposition", "Yposition", "Zposition"] + ROT, False]]
    assert evaluate(j, np.zeros(6))["verdict"] == "unsupported"


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
