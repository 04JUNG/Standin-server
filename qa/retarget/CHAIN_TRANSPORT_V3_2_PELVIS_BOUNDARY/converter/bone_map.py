"""
내부 표준 스켈레톤(Canonical Skeleton) 정의 및 프로파일별 매핑 테이블.

설계 원칙
---------
1. 우리 시스템의 "진짜 표준"은 내부 canonical 이름 하나뿐이다.
   외부 규격(Mixamo / CMU BVH / CSP)은 전부 이 canonical 로 들어오고 나가는
   매핑 테이블로만 존재한다.
2. 새 규격이 추가되면 이 파일에 dict 하나만 추가하면 된다.
   변환 로직(retarget.py)은 규격 이름을 전혀 모른다.
3. 레스트 포즈 전제: 모든 프로파일은 **T-pose** 를 rest 로 갖는다고 가정한다.
   (A-pose 리그를 쓰려면 REST_HINT 에 보정 회전을 추가해야 한다 — 미구현/명시적 실패)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 내부 표준 22본 (Unity Humanoid 코어 호환 · RTMPose 17kp 를 전부 커버)
# ---------------------------------------------------------------------------
CANONICAL_BONES: list[str] = [
    "hips",
    "spine",
    "spine1",
    "spine2",
    "neck",
    "head",
    "shoulder.L", "upperarm.L", "forearm.L", "hand.L",
    "shoulder.R", "upperarm.R", "forearm.R", "hand.R",
    "upleg.L", "leg.L", "foot.L", "toe.L",
    "upleg.R", "leg.R", "foot.R", "toe.R",
]

# 부모 관계 (검증·씬 조립용)
CANONICAL_PARENT: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "spine1": "spine",
    "spine2": "spine1",
    "neck": "spine2",
    "head": "neck",
    "shoulder.L": "spine2", "upperarm.L": "shoulder.L",
    "forearm.L": "upperarm.L", "hand.L": "forearm.L",
    "shoulder.R": "spine2", "upperarm.R": "shoulder.R",
    "forearm.R": "upperarm.R", "hand.R": "forearm.R",
    "upleg.L": "hips", "leg.L": "upleg.L", "foot.L": "leg.L", "toe.L": "foot.L",
    "upleg.R": "hips", "leg.R": "upleg.R", "foot.R": "leg.R", "toe.R": "foot.R",
}

# 좌우 미러 페어 (CSP 좌우 반전 대응 — v3 14장 잔여 이슈)
MIRROR_PAIRS: list[tuple[str, str]] = [
    ("shoulder.L", "shoulder.R"),
    ("upperarm.L", "upperarm.R"),
    ("forearm.L", "forearm.R"),
    ("hand.L", "hand.R"),
    ("upleg.L", "upleg.R"),
    ("leg.L", "leg.R"),
    ("foot.L", "foot.R"),
    ("toe.L", "toe.R"),
]

# 포즈 유사도·검증에 쓰이는 핵심 본 (이게 없으면 변환 실패로 간주)
REQUIRED_BONES: set[str] = {
    "hips", "spine", "neck", "head",
    "upperarm.L", "forearm.L", "upperarm.R", "forearm.R",
    "upleg.L", "leg.L", "upleg.R", "leg.R",
}


# ---------------------------------------------------------------------------
# 프로파일: canonical -> 해당 규격의 본 이름
# ---------------------------------------------------------------------------

MIXAMO: dict[str, str] = {
    "hips": "mixamorig:Hips",
    "spine": "mixamorig:Spine",
    "spine1": "mixamorig:Spine1",
    "spine2": "mixamorig:Spine2",
    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",
    "shoulder.L": "mixamorig:LeftShoulder",
    "upperarm.L": "mixamorig:LeftArm",
    "forearm.L": "mixamorig:LeftForeArm",
    "hand.L": "mixamorig:LeftHand",
    "shoulder.R": "mixamorig:RightShoulder",
    "upperarm.R": "mixamorig:RightArm",
    "forearm.R": "mixamorig:RightForeArm",
    "hand.R": "mixamorig:RightHand",
    "upleg.L": "mixamorig:LeftUpLeg",
    "leg.L": "mixamorig:LeftLeg",
    "foot.L": "mixamorig:LeftFoot",
    "toe.L": "mixamorig:LeftToeBase",
    "upleg.R": "mixamorig:RightUpLeg",
    "leg.R": "mixamorig:RightLeg",
    "foot.R": "mixamorig:RightFoot",
    "toe.R": "mixamorig:RightToeBase",
}

# Mixamo 를 BVH 로 내보내면 콜론이 빠지는 경우가 있어 별칭 프로파일을 둔다
MIXAMO_NOPREFIX: dict[str, str] = {
    k: v.split(":", 1)[1] for k, v in MIXAMO.items()
}

# CMU MoCap (cgspeed BVH 변환본 기준)
CMU_BVH: dict[str, str] = {
    "hips": "Hips",
    "spine": "LowerBack",
    "spine1": "Spine",
    "spine2": "Spine1",
    "neck": "Neck",
    "head": "Head",
    "shoulder.L": "LeftShoulder",
    "upperarm.L": "LeftArm",
    "forearm.L": "LeftForeArm",
    "hand.L": "LeftHand",
    "shoulder.R": "RightShoulder",
    "upperarm.R": "RightArm",
    "forearm.R": "RightForeArm",
    "hand.R": "RightHand",
    "upleg.L": "LeftUpLeg",
    "leg.L": "LeftLeg",
    "foot.L": "LeftFoot",
    "toe.L": "LeftToeBase",
    "upleg.R": "RightUpLeg",
    "leg.R": "RightLeg",
    "foot.R": "RightFoot",
    "toe.R": "RightToeBase",
}

# CSP 표준 본 — ⚠️ 실험(EXP-FBX-04) 전까지 확정 불가. 자리표시자.
# 정적 메시 베이크 경로에서는 이 매핑이 필요 없다(본이 산출물에 없음).
CSP: dict[str, str] = {}

PROFILES: dict[str, dict[str, str]] = {
    "canonical": {b: b for b in CANONICAL_BONES},
    "mixamo": MIXAMO,
    "mixamo_noprefix": MIXAMO_NOPREFIX,
    "cmu_bvh": CMU_BVH,
    "csp": CSP,
}


def to_canonical(profile: str) -> dict[str, str]:
    """해당 규격 본 이름 -> canonical 이름 (역방향 테이블)."""
    fwd = PROFILES[profile]
    return {v: k for k, v in fwd.items()}


def resolve_profile(bone_names: list[str]) -> str:
    """
    실제 아마추어의 본 이름 목록을 보고 어떤 프로파일인지 자동 판별한다.
    라이브러리 씨앗이 CMU/Mixamo 혼재라 자동 판별이 운영 비용을 크게 줄인다.
    """
    names = set(bone_names)
    # 점수 = (일치 본 수, -필수본 누락 수). 동점일 때 **필수 본을 덜 빠뜨리는 쪽**을 고른다.
    #
    # 실제로 필요한 규칙이다: Spine2 가 없는 Mixamo 계열 BVH 는
    #   mixamo_noprefix (spine2="Spine2" 누락 — 선택 본)
    #   cmu_bvh         (spine="LowerBack" 누락 — **필수 본**)
    # 둘 다 21본 일치로 동점이 된다. 누락의 등급으로 갈라야 올바른 쪽이 뽑힌다.
    best, best_score = "canonical", (-1, 0)
    for name, table in PROFILES.items():
        if not table:
            continue
        hit = sum(1 for v in table.values() if v in names)
        req_missing = sum(1 for c in REQUIRED_BONES if table.get(c) not in names)
        score = (hit, -req_missing)
        if score > best_score:
            best, best_score = name, score
    best_hit = best_score[0]
    if best_hit < len(REQUIRED_BONES) * 0.7:
        raise ValueError(
            f"알려진 리그 프로파일과 매칭 실패 (최대 일치 {best_hit}본). "
            f"입력 본 예시: {sorted(names)[:8]}"
        )
    return best


def mirror_name(canonical: str) -> str:
    """canonical 이름의 좌우를 뒤집는다."""
    if canonical.endswith(".L"):
        return canonical[:-2] + ".R"
    if canonical.endswith(".R"):
        return canonical[:-2] + ".L"
    return canonical
