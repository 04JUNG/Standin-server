"""상·하체 분리 합성(split_compose)과 보이는 범위 렌더 테스트. 실제 라이브러리 없이 돈다."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from src.bvh import channel_starts, coco17_from_fk, fk
from src.experimental.split_compose import compose_upper_lower
from src.experimental.volume_mannequin import render, visible_extent
from tests.test_part_composition import rig


def _posed(arm_left=0.0, arm_right=0.0, root_yaw=0.0, knee=0.0):
    joints, frame = rig()
    start = channel_starts(joints)
    names = {j[0]: i for i, j in enumerate(joints)}
    frame[start[names["LeftArm"]]] = arm_left
    frame[start[names["RightArm"]]] = arm_right
    for side in ("Left", "Right"):
        # 테스트 리그의 허벅지는 몸 뒤(+z)로 뻗어 있어 사람이 할 수 없는 자세다. 아래로 내린다(X축 90°).
        frame[start[names[side + "UpLeg"]] + 1] = 90
        if side + "Leg" in names:
            frame[start[names[side + "Leg"]]] = knee
    frame[3:6] = [0, root_yaw, 0]
    return joints, frame


def test_upper_comes_from_donor_and_lower_stays_from_base():
    base_j, base_f = _posed(knee=80)
    donor_j, donor_f = _posed(arm_left=70, arm_right=-70)
    result = compose_upper_lower(base_j, base_f, donor_j, donor_f)
    d = result.diagnostics
    assert d["pass"], d["reasons"]
    assert d["lower_position_delta_torso"] < 1e-6 and d["bone_length_delta_torso"] < 1e-6
    assert max(d["arm_direction_error_deg"].values()) < 1e-3
    before = fk(base_j, base_f)
    after = fk(base_j, result.frame)
    names = {j[0]: i for i, j in enumerate(base_j)}
    assert np.allclose(after[names["LeftFoot"]] if "LeftFoot" in names else after[0],
                       before[names["LeftFoot"]] if "LeftFoot" in names else before[0])
    # 팔은 donor 쪽으로 바뀌었다
    donor_pts, _ = coco17_from_fk(donor_j, fk(donor_j, donor_f))
    comp_pts, _ = coco17_from_fk(base_j, after)
    base_pts, _ = coco17_from_fk(base_j, before)
    donor_arm = donor_pts[9] - donor_pts[5]
    assert np.dot(comp_pts[9] - comp_pts[5], donor_arm) > np.dot(base_pts[9] - base_pts[5], donor_arm) - 1e-9


def test_pelvis_heading_transport_follows_base_facing():
    base_j, base_f = _posed(root_yaw=90)
    donor_j, donor_f = _posed(arm_left=70, arm_right=-70)
    d = compose_upper_lower(base_j, base_f, donor_j, donor_f).diagnostics
    assert d["transport_frame"] == "pelvis_full"
    assert d["waist_twist_deg"] < 5          # 상체가 base 골반 방향을 따른다


def test_matched_camera_transport_can_flag_large_twist():
    base_j, base_f = _posed()
    donor_j, donor_f = _posed(arm_left=70, arm_right=-70)
    d = compose_upper_lower(base_j, base_f, donor_j, donor_f, view_yaws=(np.pi / 2, 0.0)).diagnostics
    assert d["transport_frame"] == "matched_camera"
    assert "waist_twist_too_large" in d["reasons"]


def test_visible_extent_and_cropped_render():
    mask = [True] * 17
    assert visible_extent(mask) == "full"
    knee = mask[:15] + [False, False]
    assert visible_extent(knee) == "knee"
    hip = mask[:13] + [False] * 4
    assert visible_extent(hip) == "hip"
    assert visible_extent(mask[:11] + [False] * 6) == "chest"
    assert visible_extent(None) == "full"
    joints, frame = rig()
    full = np.asarray(render(joints, frame, size=(100, 130), extent="full"))[..., 3] > 0
    chest = np.asarray(render(joints, frame, size=(100, 130), extent="chest"))[..., 3] > 0
    assert full.any() and chest.any() and not np.array_equal(full, chest)
    with pytest.raises(ValueError):
        render(joints, frame, extent="feet")


def test_anatomy_flags_impossible_hip_angle():
    from src.experimental.split_compose import anatomy, anatomy_problems
    pelvis = np.eye(3)
    p = np.zeros((17, 3))
    p[11], p[12] = [-.1, 0, 0], [.1, 0, 0]
    p[5], p[6] = [-.15, .5, 0], [.15, .5, 0]
    p[13], p[14] = [-.1, -.4, 0], [.1, -.4, 0]
    ok = anatomy(p, pelvis)
    assert abs(ok["torso_flexion_deg"]) < 1e-6 and not anatomy_problems(ok, ok)
    bad = p.copy()
    bad[13] = [-.1, -.1, -.4]          # 허벅지가 몸 뒤로 크게 젖혀짐
    assert any(r.startswith("hip_angle_impossible") for r in anatomy_problems(anatomy(bad, pelvis), ok))
