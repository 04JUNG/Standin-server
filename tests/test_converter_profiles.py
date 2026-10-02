"""운영 FBX 변환기의 리그 프로파일 판별(converter.bone_map).

100STYLE 프로파일을 더해도 기존 리그의 판별 결과는 그대로여야 한다. 프로파일 표는
solver 동결(converter/SHA256SUMS.v325) 안에 있어서, 판별이 바뀌면 이미 배포된 포즈의
내보내기 결과가 조용히 달라진다.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converter.bone_map import (
    CANONICAL_BONES,
    CMU_BVH,
    MIXAMO,
    MIXAMO_NOPREFIX,
    PROFILES,
    REQUIRED_BONES,
    STYLE100,
    resolve_profile,
)
from tests.test_smoke import _synthetic_bvh

FINGERS = [f"{side}Hand{finger}{n}" for side in ("Left", "Right")
           for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for n in (1, 2, 3)]
# 100STYLE 원본 계층: Chest3는 표에 없지만 계층에는 있다. 정리 단계가 손가락 30개를 더한다.
STYLE100_RIG = list(STYLE100.values()) + ["Chest3"] + FINGERS
# CMU cgspeed BVH 계층(목·손가락 보조 관절 포함)
CMU_RIG = list(CMU_BVH.values()) + ["Neck1", "LThumb", "RThumb", "LeftFingerBase",
                                    "RightFingerBase", "LeftHandIndex1", "RightHandIndex1"]


def test_100style_profile_covers_every_canonical_bone():
    assert set(STYLE100) == set(CANONICAL_BONES)
    assert REQUIRED_BONES <= set(STYLE100)
    assert PROFILES["100style"] is STYLE100
    # 100STYLE의 LeftShoulder는 상완이다(CMU·Mixamo에서는 쇄골).
    assert STYLE100["upperarm.L"] == "LeftShoulder" and STYLE100["shoulder.L"] == "LeftCollar"


def test_100style_rig_resolves_to_100style():
    assert resolve_profile(STYLE100_RIG) == "100style"


@pytest.mark.parametrize("names, expected", [
    (list(MIXAMO.values()), "mixamo"),
    (list(MIXAMO_NOPREFIX.values()) + FINGERS, "mixamo_noprefix"),
    (CMU_RIG, "cmu_bvh"),
    (list(CANONICAL_BONES), "canonical"),
])
def test_existing_rigs_resolve_as_before(names, expected):
    assert resolve_profile(names) == expected


def test_every_profile_table_resolves_to_itself():
    for name, table in PROFILES.items():
        if table:
            assert resolve_profile(list(table.values())) == name


def test_minimal_rig_used_by_refine_tests_is_unchanged(tmp_path):
    path = Path(_synthetic_bvh(str(tmp_path), "t.bvh"))
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        head, _, rest = line.strip().partition(" ")
        if head in ("ROOT", "JOINT"):
            names.append(rest.strip())
    assert resolve_profile(names) == "mixamo_noprefix"


def test_unknown_rig_is_still_rejected():
    with pytest.raises(ValueError):
        resolve_profile([f"Bone{i:03d}" for i in range(30)])
