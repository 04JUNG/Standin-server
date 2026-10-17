"""러프에서 읽은 인물별 손 모양과 손 후보(04J-290).

VLM 별도 호출로 인물마다 왼손·오른손의 손 프리셋을 묻는다. 결과는 두 가지로 나간다.

* ``preset`` — FBX에 바로 구울 값. VLM이 손을 보고 프리셋을 골랐으면 그 값(source=vlm),
  손이 안 보이거나 호출이 실패·누락됐으면 라이브러리 기본값 ``relaxed``(source=default).
* ``candidates`` — 사용자가 고를 손 후보. VLM 1순위·대안을 앞에 두고 라이브러리의
  ``fallback_order``로 채운다. 자동 선택이 틀려도 화면에서 한 번에 바꿀 수 있게 한다.

손 모양은 검색·라우팅·refine의 입력이 아니다(CLAUDE.md 불변식 1과 같은 이유). 분석 프롬프트도
바꾸지 않는다 — route·인원수·박스가 이 기능 때문에 흔들릴 수 없게 별도 호출로 받는다.
전신 API는 순위 계산 이후 실제 FBX 피부 검사에 이 프리셋을 전달한다. 관통 후보의
출력 자격만 제한하며 검색 점수·설명·관절·refine solver 입력을 바꾸지 않는다.
배열 길이가 인원수와 다르면 통째로 버린다. 한 칸이 밀리면 다른 사람의 손이 붙기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from converter.hand_library import HandLibrary, candidate_order, load_library

if TYPE_CHECKING:
    from .schema import VLMAnalysis

DEFAULT_PRESET = "relaxed"
CANDIDATE_LIMIT = 4
HAND_SHAPES_KEY = "hand_shapes"


@dataclass(frozen=True)
class HandChoice:
    preset: str = DEFAULT_PRESET
    source: Literal["vlm", "default"] = "default"
    # VLM이 말한 확신도. default면 None.
    confidence: Optional[Literal["high", "low"]] = None
    # VLM이 그 손이 러프에 그려져 있다고 했는지. None=모름(호출 안 함·실패).
    visible: Optional[bool] = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PersonHands:
    left: HandChoice = field(default_factory=HandChoice)
    right: HandChoice = field(default_factory=HandChoice)
    library_version: str = ""
    # vlm=이 인물에 대해 VLM이 답함 · not_requested=기능 꺼짐 · unavailable=호출 실패/형식 불일치
    status: Literal["vlm", "not_requested", "unavailable"] = "not_requested"


def _default_choice(library: HandLibrary, visible: Optional[bool] = None,
                    hinted: tuple[Optional[str], ...] = ()) -> HandChoice:
    return HandChoice(DEFAULT_PRESET, "default", None, visible,
                      tuple(candidate_order(hinted, library, CANDIDATE_LIMIT)))


def default_hands(status: str = "not_requested", library: Optional[HandLibrary] = None) -> PersonHands:
    library = library or load_library()
    return PersonHands(_default_choice(library), _default_choice(library),
                       library.library_version, status)  # type: ignore[arg-type]


def _parse_side(raw: object, library: HandLibrary) -> HandChoice:
    if not isinstance(raw, dict):
        return _default_choice(library)
    shape = raw.get("shape")
    alternative = raw.get("alternative")
    shape = shape if isinstance(shape, str) and shape in library.presets else None
    alternative = (alternative if isinstance(alternative, str) and alternative in library.presets
                   else None)
    visible = raw.get("visible") if isinstance(raw.get("visible"), bool) else None
    confidence = raw.get("confidence") if raw.get("confidence") in ("high", "low") else None
    if shape is None or visible is False:
        # 안 보이는 손에 VLM이 모양을 붙여도 굽지 않는다. 대신 후보 앞쪽에는 둔다.
        return _default_choice(library, visible, (shape, alternative))
    return HandChoice(shape, "vlm", confidence or "low", visible,
                      tuple(candidate_order((shape, alternative), library, CANDIDATE_LIMIT)))


def parse_hand_shapes(payload: object, count: int,
                      library: Optional[HandLibrary] = None) -> list[PersonHands]:
    """VLM 응답 → 인물별 PersonHands. 형식이 어긋나면 전원 unavailable(기본값)."""
    library = library or load_library()
    people = payload.get("people") if isinstance(payload, dict) else None
    if not isinstance(people, list) or len(people) != count:
        return [default_hands("unavailable", library) for _ in range(count)]
    out = []
    for entry in people:
        if not isinstance(entry, dict):
            out.append(default_hands("unavailable", library))
            continue
        out.append(PersonHands(_parse_side(entry.get("left"), library),
                               _parse_side(entry.get("right"), library),
                               library.library_version, "vlm"))
    return out


def apply_hand_shapes(vlm: "VLMAnalysis", payload: Optional[dict]) -> None:
    """별도 호출의 응답을 VLMAnalysis에 싣는다. 실패(None)도 '물어봤다'로 기록한다."""
    vlm.raw[HAND_SHAPES_KEY] = payload
    vlm.person_hands = parse_hand_shapes(payload, vlm.num_people)


def detect_person_hands(vlm: "VLMAnalysis", person_index: Optional[int]) -> PersonHands:
    """descriptor 한 명의 손. VLM 슬롯이 아닌 사람(검출기만 찾은 사람)은 기본값이다."""
    hands = getattr(vlm, "person_hands", None) or []
    if person_index is not None and 0 <= person_index < len(hands):
        return hands[person_index]
    asked = HAND_SHAPES_KEY in vlm.raw
    return default_hands("unavailable" if asked else "not_requested")
