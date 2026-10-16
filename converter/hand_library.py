"""손 포즈 라이브러리(04J-290): 정의 로드·검증과 리그 독립 손가락 회전 계산.

이 모듈은 표준 라이브러리만 쓴다(converter API 프로세스에는 numpy가 없다). 분석 서버는
프리셋 이름·후보 목록을, converter API는 요청 검증을 위해 읽는다. 회전 계산은
``converter/hand_geometry.py``(numpy), 실제 리그 적용은 ``converter/hands.py``(Blender)다.

설계 원칙
---------
1. **몸은 절대 바꾸지 않는다.** 손 굽기는 V3.2.5 몸 결과 이후의 별도 단계이며 손가락
   30본의 로컬 회전만 정한다. 동결 solver 파일(SHA256SUMS.v325)은 건드리지 않는다.
2. **각도는 해부학 단위로 정의한다.** 프리셋은 "몇 도 접는다/벌린다"만 말한다. 접는 축은
   리그의 rest 기하에서 매번 계산한다(``hand_geometry.py``).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

LIBRARY_PATH = Path(__file__).with_name("hand_library.v1.json")
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS_PER_FINGER = 3
# 해부학적으로 무리 없는 범위. 이보다 큰 값은 정의 오류로 본다.
CURL_LIMITS = (-10.0, 110.0)
SPREAD_LIMITS = (-30.0, 30.0)

# Mixamo 본 이름. 우리 남·여 캐릭터와 라이브러리 BVH의 손가락 이름이 이 규약이다.
_SIDE_PREFIX = {"left": "Left", "right": "Right"}
_FINGER_NAME = {"thumb": "Thumb", "index": "Index", "middle": "Middle",
                "ring": "Ring", "pinky": "Pinky"}


class HandLibraryError(ValueError):
    """정의 파일 또는 요청이 계약을 어겼다."""


@dataclass(frozen=True)
class FingerPose:
    curl: tuple[float, float, float]
    spread: float


@dataclass(frozen=True)
class HandPreset:
    preset_id: str
    label_ko: str
    vlm_hint: str
    fingers: Mapping[str, FingerPose]
    category: str = "basic"


@dataclass(frozen=True)
class HandLibrary:
    library_version: str
    sha256: str
    presets: Mapping[str, HandPreset]
    fallback_order: tuple[str, ...]
    # 카테고리 id → 표시 이름. 정의 순서가 화면·프롬프트의 묶음 순서다.
    categories: Mapping[str, str] = None  # type: ignore[assignment]

    def in_category(self, category: str) -> tuple[str, ...]:
        return tuple(p.preset_id for p in self.presets.values() if p.category == category)

    def preset(self, preset_id: str) -> HandPreset:
        try:
            return self.presets[preset_id]
        except KeyError:
            raise HandLibraryError(f"unknown hand preset: {preset_id!r}") from None

    @property
    def preset_ids(self) -> tuple[str, ...]:
        return tuple(self.presets)


def _number(value: Any, field: str, limits: tuple[float, float]) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise HandLibraryError(f"{field} must be a finite number")
    if not limits[0] <= value <= limits[1]:
        raise HandLibraryError(f"{field}={value} outside {limits}")
    return float(value)


def parse_library(raw: bytes) -> HandLibrary:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandLibraryError("hand library JSON is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise HandLibraryError("hand library schema_version must be 1")
    version = payload.get("library_version")
    if not isinstance(version, str) or not version:
        raise HandLibraryError("library_version must be a non-empty string")
    categories = payload.get("categories")
    if (not isinstance(categories, dict) or not categories
            or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in categories.items())):
        raise HandLibraryError("categories must map ids to display names")
    raw_presets = payload.get("presets")
    if not isinstance(raw_presets, dict) or not raw_presets:
        raise HandLibraryError("presets must be a non-empty object")
    presets: dict[str, HandPreset] = {}
    for preset_id, body in raw_presets.items():
        if not isinstance(preset_id, str) or not preset_id.isidentifier():
            raise HandLibraryError(f"invalid preset id {preset_id!r}")
        if not isinstance(body, dict):
            raise HandLibraryError(f"{preset_id} must be an object")
        fingers_raw = body.get("fingers")
        if not isinstance(fingers_raw, dict) or set(fingers_raw) != set(FINGERS):
            raise HandLibraryError(f"{preset_id}.fingers must define exactly {FINGERS}")
        fingers = {}
        for finger in FINGERS:
            spec = fingers_raw[finger]
            curl = spec.get("curl") if isinstance(spec, dict) else None
            if not isinstance(curl, list) or len(curl) != JOINTS_PER_FINGER:
                raise HandLibraryError(f"{preset_id}.{finger}.curl must have 3 numbers")
            fingers[finger] = FingerPose(
                tuple(_number(v, f"{preset_id}.{finger}.curl", CURL_LIMITS) for v in curl),
                _number(spec.get("spread", 0), f"{preset_id}.{finger}.spread", SPREAD_LIMITS),
            )
        label = body.get("label_ko")
        hint = body.get("vlm_hint")
        if not isinstance(label, str) or not label or not isinstance(hint, str) or not hint:
            raise HandLibraryError(f"{preset_id} needs label_ko and vlm_hint")
        category = body.get("category")
        if category not in categories:
            raise HandLibraryError(f"{preset_id}.category must be one of {sorted(categories)}")
        presets[preset_id] = HandPreset(preset_id, label, hint, fingers, category)
    order = payload.get("fallback_order")
    # 후보를 채우는 범용 기본값 몇 개. 모든 프리셋을 나열할 필요는 없다.
    if (not isinstance(order, list) or not order or len(set(order)) != len(order)
            or any(preset_id not in presets for preset_id in order)):
        raise HandLibraryError("fallback_order must be a non-empty list of unique known presets")
    empty = [c for c in categories if not any(p.category == c for p in presets.values())]
    if empty:
        raise HandLibraryError(f"categories without presets: {empty}")
    return HandLibrary(version, hashlib.sha256(raw).hexdigest(), presets, tuple(order),
                       dict(categories))


@lru_cache(maxsize=1)
def load_library(path: Optional[str] = None) -> HandLibrary:
    source = Path(path) if path else LIBRARY_PATH
    return parse_library(source.read_bytes())


def normalize_request(raw: Any, library: Optional[HandLibrary] = None) -> Optional[dict]:
    """``{"left": id|None, "right": id|None}`` 요청을 검증한다. 둘 다 없으면 None.

    None을 돌려주면 변환기는 손 단계를 실행하지 않는다 — 기존 FBX와 바이트 단위로 같다.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - set(SIDES):
        raise HandLibraryError("hands must be an object with optional left/right")
    library = library or load_library()
    result = {}
    for side in SIDES:
        value = raw.get(side)
        if value is None:
            result[side] = None
            continue
        if not isinstance(value, str):
            raise HandLibraryError(f"hands.{side} must be a preset id or null")
        library.preset(value)
        result[side] = value
    if result["left"] is None and result["right"] is None:
        return None
    return result


def bone_name(side: str, finger: str, joint: int, prefix: str = "mixamorig:") -> str:
    return f"{prefix}{_SIDE_PREFIX[side]}Hand{_FINGER_NAME[finger]}{joint + 1}"


def hand_bone_name(side: str, prefix: str = "mixamorig:") -> str:
    return f"{prefix}{_SIDE_PREFIX[side]}Hand"


def finger_bone_names(prefix: str = "mixamorig:") -> list[str]:
    return [bone_name(side, finger, joint, prefix)
            for side in SIDES for finger in FINGERS for joint in range(JOINTS_PER_FINGER)]


def candidate_order(primary: Iterable[Optional[str]], library: Optional[HandLibrary] = None,
                    limit: int = 4) -> list[str]:
    """후보 목록: 주어진 순서(VLM 1순위·대안) → 1순위와 같은 카테고리 → fallback_order.

    예: VLM이 ``gun_trigger``를 골랐다면 같은 무기 묶음(검 쥐기·총 쥐기…)이 먼저 이어진다.
    자동 선택이 '비슷하지만 다른 손'으로 틀리는 경우가 많아서다.
    """
    library = library or load_library()
    primary = [p for p in primary if p in library.presets]
    siblings = library.in_category(library.presets[primary[0]].category) if primary else ()
    out: list[str] = []
    for preset_id in (*primary, *siblings, *library.fallback_order):
        if preset_id not in out:
            out.append(preset_id)
        if len(out) >= limit:
            break
    return out
