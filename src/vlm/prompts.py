"""
VLM 프롬프트. 원칙(설계문서 v2 §3-4): 개수·종류·의미까지만, 좌표 생성 금지.
approx_boxes는 0~1 정규화 좌표의 '대략' 박스만 허용(정밀 박스는 검출기 몫).

프롬프트는 버전으로 고른다(env `VLM_PROMPT_VERSION`, 기본 `p1-scope`). 새 버전은 같은
이미지에서 route·인원수·박스가 흔들리지 않는지 평가로 확인한 뒤에 켠다. 응답의
`inference_metadata.vlm_prompt_version`이 어느 프롬프트가 답했는지 남긴다.
"""

SYSTEM = (
    "너는 웹툰 러프 콘티(선화) 컷을 분석하는 도우미다. "
    "정밀한 관절 좌표나 픽셀 단위 위치는 절대 추정하지 마라. "
    "사람 수, 샷 종류, 행동, 시점, 관계 같은 '의미'만 판단한다."
)

# 반드시 JSON만 출력하도록 강제. 열거값은 schema의 Controlled Vocabulary와 일치해야 함.
USER_TEMPLATE = """이 컷을 분석해서 아래 JSON 스키마로만 답하라. 설명 문장 금지.

{{
  "num_people": <0~20 정수>,
  "shot": "full_half" | "bust" | "face",
  "action": "standing" | "sitting" | "walking" | "running" | "reaching" | "lying" | "other",
  "view": "front" | "side" | "back" | "three_quarter",
  "relationship": "solo" | "talking" | "hugging" | "holding_hands" | "fighting",
  "approx_boxes": [ {{"x1":0~1,"y1":0~1,"x2":0~1,"y2":0~1}}, ... ],
  "lower_body_visible": [<인물별 true | false>, ...],
  "body_scopes": [<인물별 "full" | "half" | "bust" | "head" | null>, ...],
  "dialogue": "<말풍선 텍스트 있으면, 없으면 null>"
}}

판단 기준:
- shot: 전신/반신이면 full_half, 가슴 위만 보이면 bust, 얼굴 클로즈업이면 face.
- approx_boxes: 각 인물의 '대략' 위치만. 정밀할 필요 없다(좌/중/우 수준).
- lower_body_visible: approx_boxes와 같은 순서. 해당 인물의 양쪽 골반·무릎·발목이 러프에 실제로
  그려져 식별 가능할 때만 true. 반신 컷, 가구/다른 인물에 가림, 화면 밖 잘림, 추측해야 하면 false.
- num_people는 0~20 범위이며 approx_boxes 개수와 반드시 같아야 한다.
- body_scopes: approx_boxes와 같은 순서·개수로 인물별 화면 구도를 판단한다.
  full=머리부터 발까지 전신 구도, half=허리/골반/허벅지 부근에서 잘린 반신 구도,
  bust=가슴 위와 어깨가 보이는 흉상, head=머리·얼굴 중심(목 일부 포함)의 두상.
  다른 인물/물체에 가려진 관절과 화면 밖으로 잘린 구도는 다르다. 다리가 가려졌거나
  포즈 추정이 어려운 것만으로 half로 바꾸지 마라. 무릎 근처의 애매한 잘림,
  구도 판단이 어려운 러프는 null. 컷 안의 모든 사람에게 같은 값을 복사하지 마라.
  body_scopes는 출력 범위 메타데이터이며 기존 shot 및 lower_body_visible 기준을 바꾸지 않는다.
- 확신이 없으면 가장 그럴듯한 값을 고르되, 좌표를 지어내지 마라(대략이면 충분).
"""

# p2-person-tags: p1-scope에 인물별 action·view만 더한다. 나머지 문장은 그대로 둔다.
# 기존 항목의 판단이 함께 달라지면 평가에서 원인을 가를 수 없기 때문이다.
_PERSON_TAG_FIELDS = (
    '  "person_actions": [<인물별 "standing" | "sitting" | "walking" | "running" | '
    '"reaching" | "lying" | "other" | null>, ...],\n'
    '  "person_views": [<인물별 "front" | "side" | "back" | "three_quarter" | null>, ...],\n'
)
_PERSON_TAG_RULES = (
    "- person_actions·person_views: approx_boxes와 같은 순서·개수로 인물마다 따로 판단한다.\n"
    "  컷 대표값(action·view)을 모든 인물에 복사하지 마라. 그 인물의 행동이나 몸이 향한 방향을\n"
    "  러프에서 판단하기 어려우면 null. 기록용 값이며 다른 항목의 판단 기준을 바꾸지 않는다.\n"
)


def _insert_before(text: str, anchor: str, addition: str) -> str:
    if text.count(anchor) != 1:
        raise RuntimeError(f"prompt anchor is not unique: {anchor!r}")
    return text.replace(anchor, addition + anchor)


USER_TEMPLATE_P2 = _insert_before(
    _insert_before(USER_TEMPLATE, '  "dialogue":', _PERSON_TAG_FIELDS),
    "- 확신이 없으면 가장 그럴듯한 값을 고르되", _PERSON_TAG_RULES,
)

DEFAULT_PROMPT_VERSION = "p1-scope"
USER_TEMPLATES = {
    "p1-scope": USER_TEMPLATE,
    "p2-person-tags": USER_TEMPLATE_P2,
}
# 인물별 action·view를 묻는 버전. mock VLM이 같은 모양의 응답을 흉내 낼 때 쓴다.
PERSON_TAG_VERSIONS = frozenset({"p2-person-tags"})


def user_template(version: str) -> str:
    try:
        return USER_TEMPLATES[version]
    except KeyError:
        raise ValueError(
            f"unknown VLM_PROMPT_VERSION {version!r}; expected one of {sorted(USER_TEMPLATES)}"
        ) from None
