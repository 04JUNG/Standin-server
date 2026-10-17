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

# ── 인물별 태그 전용 호출 ────────────────────────────────────────────
#
# 분석 프롬프트에 인물별 항목을 더하면 route가 함께 흔들린다. 2026-10-06~08에 세 번
# 측정했고(48컷 2회·96컷 2회, 총 2,880호출) 문구를 두 번 고쳐도 초과분이 2.5~2.7%p에
# 머물렀다. 고치면 걸리는 컷만 바뀌었다 — 96컷 중 세 컷이 초과분 전부를 만들었고 그
# 세 컷은 수정할 때마다 다른 컷으로 옮겨 갔다. 기록은 docs/POSE_GAP_LOOP.md에 있다.
#
# 그래서 태그는 **별도 호출**로 받는다. 분석 프롬프트(p1-scope)는 한 글자도 바뀌지 않으므로
# route·인원수·박스가 구조적으로 흔들릴 수 없다. 대가는 호출 1회와 그 비용이다.
PERSON_TAGS_SYSTEM = (
    "너는 이미 분석이 끝난 웹툰 러프 컷에서 인물별 의미 태그만 채우는 도우미다. "
    "사람 수와 위치는 이미 정해져 있다. 그 수를 바꾸거나 사람을 새로 찾지 마라. "
    "관절 좌표나 픽셀 위치는 추정하지 마라."
)

PERSON_TAGS_TEMPLATE = """이 컷에는 사람이 {count}명 있고, 위치는 아래와 같다(0~1 정규화, 좌→우 순서).

{boxes}

각 인물의 행동과 몸이 향한 방향만 판단해 아래 JSON으로만 답하라. 설명 문장 금지.

{{
  "person_actions": [<인물별 "standing" | "sitting" | "walking" | "running" | "reaching" | "lying" | "other" | null>, ...],
  "person_views": [<인물별 "front" | "side" | "back" | "three_quarter" | null>, ...]
}}

규칙:
- 두 배열의 길이는 반드시 {count}이고, 순서는 위 목록과 같다.
- 한 인물의 값을 다른 인물에 복사하지 마라. 러프에서 판단하기 어려우면 그 자리만 null.
- 사람 수를 다시 세지 마라. 위에 적힌 {count}명이 전부다.
"""


def person_tags_prompt(boxes) -> str:
    """태그 전용 프롬프트. boxes는 0~1 정규화된 (x1, y1, x2, y2) 목록이다."""
    lines = []
    for index, box in enumerate(boxes):
        if box is None:
            lines.append(f"- {index}번: 위치 불명")
            continue
        x1, y1, x2, y2 = (round(float(value), 3) for value in box)
        lines.append(f"- {index}번: x {x1}~{x2}, y {y1}~{y2}")
    return PERSON_TAGS_TEMPLATE.format(count=len(boxes), boxes="\n".join(lines))


USER_TEMPLATE_SCOPE_OUTPUT = USER_TEMPLATE.replace(
    '"full" | "half" | "bust" | "head" | null',
    '"full" | "half" | "bust" | "shoulder_head" | "head" | null',
).replace(
    'bust=가슴 위와 어깨가 보이는 흉상, head=머리·얼굴 중심(목 일부 포함)의 두상.',
    'bust=가슴 위와 어깨가 보이는 흉상, '
    'shoulder_head=머리·목·어깨선만 보이고 가슴·배·팔이 화면 밖인 어깨 위 클로즈업, '
    'head=머리·얼굴 중심(목 일부 포함)의 두상. shoulder_head와 가슴이 보이는 bust를 구분하라.',
)
DEFAULT_PROMPT_VERSION = "p3-output-scope"

# ── 인물별 손 모양 전용 호출(04J-290) ─────────────────────────────────
#
# person tags와 같은 이유로 분석 프롬프트와 분리한다. 손 모양은 FBX 손가락 굽기와
# 손 후보에만 쓰이고 검색·라우팅의 입력이 아니다(src/hand_shapes.py).
HAND_SHAPES_SYSTEM = (
    "너는 이미 분석이 끝난 웹툰 러프 컷에서 인물별 손 모양만 고르는 도우미다. "
    "사람 수와 위치는 이미 정해져 있다. 그 수를 바꾸거나 사람을 새로 찾지 마라. "
    "관절 좌표나 픽셀 위치는 추정하지 마라. 정해진 손 모양 이름 중에서만 고른다."
)

HAND_SHAPES_TEMPLATE = """이 컷에는 사람이 {count}명 있고, 위치는 아래와 같다(0~1 정규화, 좌→우 순서).

{boxes}

각 인물의 왼손과 오른손이 어떤 모양인지 아래 목록에서 골라라.

{shapes}

아래 JSON으로만 답하라. 설명 문장 금지.

{{
  "people": [
    {{
      "left":  {{"visible": true | false, "shape": <모양 이름 | null>, "alternative": <모양 이름 | null>, "confidence": "high" | "low"}},
      "right": {{"visible": true | false, "shape": <모양 이름 | null>, "alternative": <모양 이름 | null>, "confidence": "high" | "low"}}
    }}
  ]
}}

규칙:
- people 배열의 길이는 반드시 {count}이고, 순서는 위 목록과 같다.
- left/right는 **그 인물 자신의** 왼손/오른손이다. 화면의 왼쪽/오른쪽이 아니다.
  정면을 보는 인물은 화면 오른쪽에 있는 손이 그 인물의 왼손이다. 뒷모습이면 화면과 같다.
- visible: 그 손이 러프에 실제로 그려져 있으면 true. 화면 밖·몸이나 물건에 가림·흉상 구도면 false.
- shape: 보이는 손이 위 목록 중 가장 가까운 모양. 안 보이면 null.
- alternative: shape 다음으로 그럴듯한 모양. 없으면 null. shape와 같은 값을 쓰지 마라.
- confidence: 손가락 모양이 그림에서 분명하면 high, 선이 러프해 추측했으면 low.
- 물건을 들고 있으면 그 물건 전용 손을 먼저 고른다(칼=sword_grip, 총=gun_trigger, 활=bow_draw,
  큰 공=ball_palm, 야구공=ball_grip, 컵=hold_cup, 휴대폰=hold_phone, 펜=pen). 전용 손이 없으면
  grip·cup·pinch 같은 일반 쥐기 손을 고른다. 제스처(브이·엄지척 등)는 손가락 모양이 보일 때만 고른다.
- 기침·헛기침으로 입 앞에 주먹을 대면 cough_fist, 손바닥으로 입을 덮으면 cover_mouth를 고른다.
- 한 인물의 값을 다른 인물에 복사하지 마라. 사람 수를 다시 세지 마라.
"""


def hand_shapes_prompt(boxes, presets) -> str:
    """손 모양 전용 프롬프트. presets는 (id, label_ko, vlm_hint[, 묶음 이름]) 목록이다."""
    lines = []
    for index, box in enumerate(boxes):
        if box is None:
            lines.append(f"- {index}번: 위치 불명")
            continue
        x1, y1, x2, y2 = (round(float(value), 3) for value in box)
        lines.append(f"- {index}번: x {x1}~{x2}, y {y1}~{y2}")
    groups: dict[str, list[str]] = {}
    for entry in presets:
        preset_id, label, hint = entry[:3]
        group = entry[3] if len(entry) > 3 else ""
        groups.setdefault(group, []).append(f"- {preset_id} ({label}): {hint}")
    shapes = "\n".join(
        (f"[{group}]\n" if group else "") + "\n".join(lines) for group, lines in groups.items())
    return HAND_SHAPES_TEMPLATE.format(count=len(boxes), boxes="\n".join(lines), shapes=shapes)


USER_TEMPLATES = {
    "p1-scope": USER_TEMPLATE,
    "p2-person-tags": USER_TEMPLATE_P2,
    "p3-output-scope": USER_TEMPLATE_SCOPE_OUTPUT,
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


# ── 인물별 자세 설명 호출(04J-286 시맨틱 라우팅) ──────────────────────────
#
# 관절 근거가 부족한 인물에 한해서만 부른다(정상 러프는 호출 0회). 분석 프롬프트와 분리돼
# 있어 route·인원수·박스가 이 호출 때문에 달라질 수 없다. 태그 어휘는 라이브러리 태깅과
# 같은 search-taxonomy-v2 키만 허용한다(src/semantic_vocab.py). 판단 원칙은
# pose-library-viewer/tagging/ROUGH_TO_LANGUAGE.md를 따른다 — 관찰과 해석을 나누고,
# 형태에서 행동으로 넘어가되 근거 없는 감정·종목을 붙이지 않는다.
POSE_DESCRIBE_VERSION = "d4-visible-action-relations"

POSE_DESCRIBE_SYSTEM = (
    "너는 웹툰 러프 콘티에서 지정된 인물의 자세를 검색용 태그와 짧은 문장으로 설명하는 도우미다. "
    "관절 좌표나 픽셀 위치는 추정하지 마라. 사람 수를 다시 세거나 새 인물을 찾지 마라. "
    "먼저 실제 보이는 팔·다리·몸통 배치를 관찰하고 행동 해석과 구분한다. "
    "무릎 굽힘만으로 점프·낙하·종목을 확정하지 않는다. "
    "읽을 수 없으면 솔직히 visible=false 또는 certainty=low로 답한다."
)

POSE_DESCRIBE_TEMPLATE = """이 컷에서 아래 {count}명의 자세를 설명하라(0~1 정규화 박스, 순서 고정).

{boxes}

허용 태그(영문 키만 사용, 목록 밖 값 금지):
- postures: {postures}
- actions: {actions}
- domains: {domains}
- activities: {activities}
- objects: {objects}
- intents: {intents}
- emotions: {emotions}

아래 JSON으로만 답하라. 설명 문장 금지.

{{
  "people": [
    {{
      "visible": <이 인물의 몸 자세를 러프에서 읽을 수 있으면 true>,
      "postures": [...], "actions": [...], "domains": [...], "activities": [...],
      "objects": [...], "intents": [...], "emotions": [...],
      "held_object_hands": {{"<실제로 쥔 objects 태그>": "left|right|both|unknown"}},
      "held_object_hand_counts": {{"<실제로 쥔 objects 태그>": "one|two|unknown"}},
      "arm_states": {{"left":"lowered|waist|forward|raised|spread|unknown", "right":"lowered|waist|forward|raised|spread|unknown"}},
      "visible_scope": "full|half|shoulder_head|head",
      "head_view": "front|profile_left|profile_right|back|three_quarter_left|three_quarter_right|unknown",
      "pose_relations": [{{"kind":"touch_head|touch_face|check_wrist|hand_chest|arm_raised", "side":"left|right|both|unknown", "chain_visible":true, "side_confidence":"high|unknown", "evidence":"손목에서 어깨로 연결되는 근거. 안 보이면 명시"}}],
      "body_details": "<실제로 보이는 팔·다리·몸통 배치 관찰 한 문장(한국어)>",
      "body_observations": {{
        "knees": "bent|straight|mixed|unknown",
        "legs": "apart|together|staggered|unknown",
        "arms": "lowered|raised|spread|mixed|unknown",
        "torso": "forward|backward|upright|unknown"
      }},
      "context_evidence": "<상황·종목·대상 태그를 뒷받침하는 관찰/맥락 근거. 없으면 빈 문자열>",
      "search_text": "<핵심 행동 또는 자세 + 구별되는 팔·다리 배치의 자연스러운 검색 문장>",
      "certainty": "high" | "medium" | "low"
    }}
  ]
}}

규칙:
- 팔·다리가 없어도 머리·목·어깨 방향을 읽을 수 있으면 visible=true. 보이는 부위의 자세에 대해서만 certainty를 매겨라. 없는 부위를 추정하지 마라.
- visible_scope: 팔이 보이면 shoulder_head 금지, half로. head_view의 좌우는 화면에서 코가 향하는 방향이다(손의 인물 기준 좌우와 다름).
- pose_relations는 물건 유무와 무관하게 핵심 동작을 기록한다. 손목시계/기기 확인은 check_wrist로, 머리를 짚는 것은 touch_head로. 옆에 컵을 들었다고 핵심 동작을 hold로 대체하지 마라. 해당 동작이 없으면 빈 배열.
- side는 인물 본인 기준이며 화면 위치만으로 고르지 마라. 손-팔꿈치-어깨 연결이 가려지거나 불확실하면 chain_visible=false, side=unknown. search_text에도 동작과 확인된 좌우를 보존하라.
- people 배열 길이는 반드시 {count}이고 순서는 위 목록과 같다.
- body_details에는 관찰만 쓴다. 뛰어오르거나 낙하한다는 추측을 관찰 문장에 넣지 마라.
- held_object_hands는 물건별로 실제로 쥔 손을 적는다. 화면 좌우가 아니라 인물 본인 기준이다. 정면 인물의 왼손은 화면 오른쪽이다. 뒤돌아선 인물은 반대다. 물건에서 손목·팔꿈치·어깨 연결을 따라 확인하고 가림·옆모습으로 모호하면 unknown. 물건을 다른 손으로 만지는 것과 쥐는 것을 구분한다. 물건이 없으면 빈 객체다.
- held_object_hand_counts는 좌우와 독립이다. 좌우가 모호해도 양손이 같은 물건을 잡는 것이 보이면 two다. arm_states는 각 팔을 별도로 관찰한다. 허리 손은 waist, 앞으로 뻗음은 forward, 내린 팔은 lowered. 팔 연결이 모호하거나 안 보이면 해당 팔만 unknown. 옆에 놓인 물건과 실제로 쥔 물건을 구분하라.
- body_observations는 파이프(|)로 구분된 허용값 중 하나씩 고른다. 좌표·각도는 출력하지 마라.
- knees=mixed는 한쪽은 굽히고 다른 쪽은 편 경우, legs=staggered는 앞뒤로 엇갈린 경우다.
- 두 팔이 내려가면 lowered, 들어가면 raised, 옆으로 뻗으면 spread, 서로 다르면 mixed다.
- 다리가 굽어 보인다는 이유로 crouching을 확정하지 말고 지지가 불명확하면 자세·행동은 비울 수 있다.
- 보이는 단서가 부족한 속성은 unknown. 팔·다리는 관찰하되 전후 기울기가 모호하면 torso=unknown.
- 행동은 근거가 가장 강한 하나만 actions에 넣는다. jump와 fall을 동시에 주된 행동으로 넣지 마라.
- 행동이 모호해도 구체적인 몸 배치는 유지한다. 두 무릎을 굽혀 다리를 벌린 형태만으로 검색할 수 있다.
- 종목·상황·도구를 채울 근거가 없으면 domains·activities·objects·intents·emotions는 빈 배열로 둔다.
- 스포츠·판타지·전투처럼 활용 가능한 대안은 주된 검색에 확정 사실로 섞지 마라.
- 대상이 안 보여도 행동 맥락으로 연결할 수 있으나 context_evidence에 관찰과 추론을 구분해 근거를 쓴다.
- 서 있기만 하고 다른 단서가 없으면 postures=["standing"]만 두고 certainty를 low로 둔다.
- 선이 너무 거칠어 팔다리 배치를 읽을 수 없으면 visible=false.
"""


def pose_describe_prompt(boxes) -> str:
    """boxes는 0~1 정규화된 (x1, y1, x2, y2) 목록이다(None이면 위치 불명)."""
    from ..semantic_vocab import VOCABULARY

    lines = []
    for index, box in enumerate(boxes):
        if box is None:
            lines.append(f"- {index}번: 위치 불명")
            continue
        x1, y1, x2, y2 = (round(float(value), 3) for value in box)
        lines.append(f"- {index}번: x {x1}~{x2}, y {y1}~{y2}")

    def keys(group):
        return ", ".join(VOCABULARY[group])

    return POSE_DESCRIBE_TEMPLATE.format(
        count=len(boxes), boxes="\n".join(lines),
        postures=keys("posture"), actions=keys("action"), domains=keys("domain"),
        activities=keys("activity"), objects=keys("object"), intents=keys("intent"),
        emotions=keys("emotion"),
    )
