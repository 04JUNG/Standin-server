"""포즈 라이브러리 번들에 들어가도 되는 것의 규칙.

번들 빌더(`build_pose_bundle.py`)와 배포 검증기(`deploy_pose_library.py`)가 같은 규칙을
쓴다. 자산 버킷은 버전 관리라 한 번 올라간 값은 이전 버전에 영구히 남는다. 그래서
"올린 뒤 지우기"가 아니라 "올리기 전에 막기"만 의미가 있다.

1. 사용자 식별자: pose_id와 meta 어디에도 설치·작업 ID, 사용자 입력 해시 접두사
   (`user_<sha12>` — `jobs.input_sha256`으로 작업까지 다시 연결된다), BetaData 버킷 경로가
   들어가면 안 된다.
2. meta 허용 목록: 정리 도구가 붙이는 meta에는 로컬 경로(`batch_directory`), 검수 참고
   러프(`reference_roughs`), 보정에 쓴 입력 이미지 해시(`calibration_*`) 같은 내부 값이
   섞여 있다. 번들에는 허용 목록의 키만 남긴다.
3. 경로 형식: `bvh_path`는 컨테이너 WORKDIR 기준 `data/bvh/<파일>.bvh`만 허용한다.
4. 리그 호환: 운영 FBX 변환기(`converter.bone_map.resolve_profile`)가 모르는 리그는
   내보내기에서 실패한다. 검색에는 나오는데 저장이 안 되는 포즈를 만들지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

# ── 1. 사용자 식별자 ────────────────────────────────────────────────────
PRIVACY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("user_input_hash", re.compile(r"user_[0-9a-f]{12}")),
    # BFF가 `inst_${randomUUID()}`·`job_${randomUUID()}`로 발급한다.
    ("installation_id", re.compile(r"inst_[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)),
    ("job_id", re.compile(r"job_[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)),
    ("beta_data_key", re.compile(r"installations/")),
    ("beta_data_bucket", re.compile(r"betadata", re.IGNORECASE)),
    ("calibration_image", re.compile(r"calibration_image")),
    ("local_path", re.compile(r"(?:^|[\s\"'])(?:[A-Za-z]:[\\/]|/Users/|/home/)")),
)

# ── 2. meta 허용 목록 ───────────────────────────────────────────────────
# 값이 dict인 키는 허용할 하위 키 집합을 둔다. None이면 스칼라만 허용한다.
META_ALLOWLIST: dict[str, frozenset[str] | None] = {
    "source": None,
    "license": None,
    "license_url": None,
    "author": None,
    "source_url": None,
    "source_frame_0based": None,
    "pose_family_id": None,
    "batch_id": None,
    "curation_group": None,
    "review_status": None,
    "bvh_sha256": None,
    "category": None,
    "category_label": None,
    "style": None,
    "hand_augmentation": frozenset({"captured_from_source", "left", "right", "joint_count"}),
    "composition_variant": frozenset({"base_pose_id", "base_sha256", "root_rotation_matrix",
                                      "pitch_yaw_roll_degrees", "limb_rotations_preserved"}),
    "gap": frozenset({"schema", "method", "cluster_key"}),
}

# ── 3. 경로 형식 ────────────────────────────────────────────────────────
BVH_PATH_PATTERN = re.compile(r"^data/bvh/[^/\\]+\.bvh$")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def privacy_findings(pose_id: str, meta: dict | None = None,
                     bvh_path: str | None = None) -> list[str]:
    """찾은 문제를 `<label>:<위치>` 형태로 돌려준다. 값 자체는 출력하지 않는다."""
    findings: list[str] = []
    places = [("pose_id", pose_id)]
    if bvh_path is not None:
        places.append(("bvh_path", bvh_path))
    places.extend(("meta", text) for text in _strings(meta or {}))
    for place, text in places:
        for label, pattern in PRIVACY_PATTERNS:
            if label == "local_path" and place == "pose_id":
                continue
            if pattern.search(text):
                findings.append(f"{label}:{place}")
    return sorted(set(findings))


def disallowed_meta_keys(meta: dict | None) -> list[str]:
    bad: list[str] = []
    for key, value in (meta or {}).items():
        if key not in META_ALLOWLIST:
            bad.append(key)
            continue
        allowed = META_ALLOWLIST[key]
        if allowed is None:
            if isinstance(value, (dict, list)):
                bad.append(key)
        elif isinstance(value, dict):
            bad.extend(f"{key}.{sub}" for sub in value if sub not in allowed)
        elif value is not None:
            bad.append(key)
    return sorted(bad)


def sanitize_meta(meta: dict | None) -> dict:
    """허용 목록의 키만 남긴다. 허용되지 않는 하위 키도 떨어낸다."""
    out: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if key not in META_ALLOWLIST:
            continue
        allowed = META_ALLOWLIST[key]
        if allowed is None:
            if not isinstance(value, (dict, list)):
                out[key] = value
        elif isinstance(value, dict):
            kept = {sub: item for sub, item in value.items() if sub in allowed}
            if kept:
                out[key] = kept
    return out


def bvh_path_ok(bvh_path: str | None) -> bool:
    return bool(bvh_path) and bool(BVH_PATH_PATTERN.fullmatch(bvh_path.replace("\\", "/")))


# ── 4. 리그 호환 ────────────────────────────────────────────────────────
def bvh_joint_names(path: Path) -> list[str]:
    """HIERARCHY의 ROOT/JOINT 이름만 읽는다(End Site 제외). MOTION은 읽지 않는다."""
    names: list[str] = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            stripped = line.strip()
            if stripped.startswith("MOTION"):
                break
            head, _, rest = stripped.partition(" ")
            if head in ("ROOT", "JOINT") and rest.strip():
                names.append(rest.strip())
    return names


def converter_profile(path: Path) -> str:
    """운영 변환기가 이 BVH를 어떤 리그로 보는지. 모르는 리그면 ValueError."""
    from converter.bone_map import resolve_profile

    names = bvh_joint_names(path)
    if not names:
        raise ValueError("BVH에 ROOT/JOINT가 없습니다")
    return resolve_profile(names)
