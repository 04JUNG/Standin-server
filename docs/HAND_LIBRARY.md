# 손 라이브러리 (04J-290)

> 상태: 구현 · 단위 테스트 완료 / 실제 남·여 캐릭터 Blender 검증 대기
> (`tests/converter/verify_hand_library.py`)

러프에서 손 모양을 읽어 FBX 손가락에 굽고, 자동 선택이 틀릴 때 사용자가 고를 수 있게 손 후보를
함께 준다. 몸 포즈(V3.2.5)는 바꾸지 않는다.

화면(refine)·BFF·API 설계: [`HAND_SELECTION_DESIGN.md`](HAND_SELECTION_DESIGN.md)

```
/analyze ──(VLM 별도 호출, VLM_HAND_SHAPES=1)──▶ people[].hands.{left,right}
                                                   preset      ← 자동으로 구울 값
                                                   candidates  ← 사용자가 고를 후보
        사용자가 후보를 바꾸면 그 값으로 ↓
converter POST /convert · /convert-bundle · /convert-framed
        hand_left=<preset> hand_right=<preset>
        ──▶ V3.2.5 몸 변환 → 원본 손가락 전달 → 선택한 손 프리셋 → framing·미리보기
```

## 손 프리셋 (hand-library-v1, 79종 · 9묶음)

원본 포즈의 손가락은 `converter/source_fingers.py`의 `source-fingers-v1` 단계에서
먼저 보존한다. 기본 몸 변환의 22본 매핑에 손가락이 없어서 모든 포즈가 편 손으로
보이던 문제를 고친 것이다. 손목 움직임을 제거한 원본 손가락 회전을 해부학적
손바닥 좌표로 전달하며 몸·손목·본 길이·위치는 바꾸지 않는다. 좌우 반전도 지원한다.
프리셋을 지정한 쪽은 원본 전달을 건너뛰므로 선택한 손 모양을 중복해서 접지 않는다.
원본 체인이 불완전한 손은 `missing_finger_chain` 진단과 함께 대상 기본 손을 유지한다.
이 진단은 손이 정확히 복원됐거나 관통이 없다는 인증이 아니다.

기존 저장 썸네일은 코드 수정만으로 바뀌지 않는다.
`scripts/rebuild_library_source_hands.py`가 별도 출력 폴더에 4범위×8방향 이미지를
생성하고 같은 최종 FBX로 `pose-previews/`의 GLB 캐시도 다시 만든다.
새 렌더 방식은 `frozen-native-source-fingers-v1`이며 기존 이미지·검수
결과를 새 버전으로 재명명하지 않는다. 변경된 렌더의 설명·태그는 출력된
`semantic-rereview.jsonl`을 기준으로 재검수해야 한다. 기존 입력 DB·BVH는 수정하지 않는다.

| 묶음 | 프리셋 |
|---|---|
| 기본 | `open` 편 손, `relaxed` 힘 뺀 손, `tense` 힘준 손, `blade` 손날, `fist` 주먹, `claw` 갈퀴손, `soft_open` 살짝 편 손, `hyper_open` 쫙 젖힌 손, `limp` 늘어진 손 |
| 물건 쥐기 | `grip` 쥐는 손, `cup` 오므린 손, `pinch` 집는 손, `support` 받치는 손, `hold_phone` 휴대폰 쥐기, `hold_cup` 컵 쥐기, `pen` 펜 쥐기, `hold_bag_handle` 가방 손잡이 들기, `hold_chopsticks` 젓가락 쥐기, `hold_spoon` 숟가락 쥐기, `hold_book` 책 받쳐 들기, `hold_handrail` 손잡이·난간 잡기 |
| 무기 | `sword_grip` 검 쥐기, `gun_trigger` 권총 쥐기, `gun_ready` 총 쥐기(검지 펴기), `bow_draw` 활시위 당기기, `staff_grip` 창·봉 쥐기, `dagger_reverse` 단검 역수 쥐기, `rifle_support` 총 받침손, `throw_knife` 던지기용 단검 쥐기, `shield_grip` 방패 손잡이 쥐기, `wand_grip` 지팡이·완드 쥐기 |
| 스포츠·공 | `ball_palm` 공 받치기, `ball_grip` 공 쥐기(야구공), `racket_grip` 라켓·배트 쥐기, `basketball_shoot` 슛 손, `volleyball_set` 토스 손, `bowling` 볼링공 쥐기, `climbing_crimp` 클라이밍 크림프, `swim_stroke` 수영 손 |
| 도구·악기 | `piano` 피아노 치기, `guitar_chord` 기타 코드 잡기, `hammer_grip` 망치 쥐기, `mic_grip` 마이크 쥐기 |
| 일상 동작 | `typing` 타이핑, `press_button` 버튼 누르기, `phone_swipe` 휴대폰 넘기기, `scratch_head` 머리 긁기, `rest_chin` 턱 괴기, `tap_table` 탁자 두드리기, `floor_press` 바닥 짚기, `cough_fist` 기침 주먹, `cover_mouth` 입 가리기 |
| 제스처 | `point` 가리키기, `v_sign` 브이, `thumbs_up` 엄지척, `ok_sign` 오케이, `finger_heart` 손가락 하트, `call_me` 전화해(샤카), `rock` 락(뿔), `beckon` 이리 와, `count_three` 숫자 셋, `count_four` 숫자 넷, `cross_fingers` 행운 빌기, `pinky_promise` 새끼손가락 약속, `gun_finger` 손가락 총, `snap` 손가락 튕기기, `wave` 손 흔들기 |
| 감정 표현 | `tremble_fist` 부들부들 주먹, `clutch_chest` 가슴 움켜쥐기, `cover_face` 얼굴 가리기, `pray` 기도하는 손, `reach_out` 손 뻗기 |
| 액션·감정 | `loose_fist` 느슨한 주먹, `grab` 움켜쥐기, `spell_cast` 마법 시전, `catch_ready` 받을 준비, `pull_hook` 끌어당기기, `push` 밀치기, `fingertip_floor` 손끝 짚기 |

자동 선택 기본값은 `relaxed`. 후보는 VLM 1순위·대안 → 1순위와 같은 묶음 → `fallback_order` 순이다.

정의는 `converter/hand_library.v1.json` 한 파일이다. 손가락마다 세 마디의 굽힘(`curl`)과
첫 마디 벌림(`spread`)을 **도 단위**로 적는다. 새 프리셋은 이 파일에 항목을 더하고
`category`를 지정하면 끝이다(VLM 프롬프트의 선택지도 이 파일에서 만든다).
값을 바꾸면 `library_sha256`이 바뀌어 변환 report lineage로 구분된다.

## 미리보기·각도 실험

`python scripts/hand_library_preview.py out/hand-library.html` → 프리셋 전부를 표준 손 골격으로 3D 표시하고,
마디 각도를 슬라이더로 바꿔 보고 JSON으로 복사할 수 있는 단일 HTML. 페이지가 변환기 계산(Python)과
자기 JS 재계산이 일치하는지 스스로 검사한다. 실제 캐릭터 메시 렌더는 아니다.

## 관통 검사 (`converter/hand_collision.py`)

손가락 마디를 캡슐, 손바닥을 두께 2.5cm 판으로 보고 서로 파고드는 깊이를 잰다(손 길이 18cm 기준
두께, 손 길이에 비례). 0.5mm 넘게 파고들면 관통이다.

- 라이브러리 전 프리셋 × 양손이 표준 손에서 관통 0이어야 한다(`test_no_preset_has_finger_or_palm_penetration_on_either_hand`).
- 집기·오케이·손가락 하트·튕기기는 끝이 **닿아** 있어야 한다(관통도, 떨어짐도 아님).
- 변환기 손 굽기는 실제 리그 손 길이로 같은 검사를 돌려 report `hands.sides.*.self_collisions`에 남긴다.
  Blender 검증 스크립트는 하나라도 있으면 실패한다.
- 미리보기 페이지의 마네킹 손은 같은 두께로 그려지고, 슬라이더로 바꾸면 관통한 마디를 빨갛게 표시한다.
- 각도를 고쳐 관통을 푸는 도구: `scripts/resolve_hand_collisions.py`(다른 손가락 모양을 지키며 최소 변경),
  `scripts/fit_wrapped_thumb.py`(주먹 계열에서 엄지가 손가락 바깥을 감싸는 자리 전수 탐색).

## 리그에 적용하는 방법 (`converter/hand_geometry.py`)

- 접는 축은 **그 리그의 rest 기하**(손목·손가락 첫 마디 위치)에서 매번 계산한다.
  본 roll·좌우 이름·리그마다 다른 로컬 축과 무관하게 손바닥 쪽으로 접힌다.
- 엄지는 손바닥 정면이 아니라 손바닥을 가로질러 새끼 쪽으로 접힌다(`THUMB_ACROSS_WEIGHT`).
- 손바닥 방향은 `cross(검지 뿌리, 새끼 뿌리)`의 손잡이성으로 정한다. rigged_rest 산출물은 포즈가
  rest에 구워져 손 방향이 포즈마다 다르므로 세계 축(“T-pose 손바닥=아래”)을 쓰지 않는다.
- 엄지가 검지 쪽에 있는지 다시 확인한다. 좌우 이름이 뒤바뀐 리그는 반대로 접지 않고 실패한다.
- 대상: Mixamo 손가락 30본(`mixamorig:*Hand{Thumb,Index,Middle,Ring,Pinky}{1,2,3}`)을 가진 리그.
  남성 `standin-master-v2`와 여성 `standin-female-v3.3.1`은 30본을 가진다
  (`CHAIN_TRANSPORT_V3_2_1_PALM_ROLL_QA.md` §2.1). 운영 여성 리그는 검증 스크립트로 확인한다.
  손가락 본이 없으면 변환이 실패한다
  (조용히 무시하지 않는다).

## 변환기 단계 (`converter/hands.py`)

framing과 같은 후처리 위치다. 산출 FBX를 다시 불러 손가락 30본의 로컬 회전만 정하고,
`convert.py`와 같은 rigged_rest 규약(메시에 굽고 포즈를 rest로 승격)으로 다시 내보낸다.

불변식
1. **손을 요청하지 않으면 이 단계는 실행되지 않는다.** 기존 FBX와 바이트 단위로 같다.
2. 동결 solver 파일(SHA256SUMS.v325)은 수정하지 않는다. lineage 검사는 그대로 통과한다.
3. 손가락이 아닌 모든 pose bone 행렬은 손 적용 전후 1e-5 이내로 같아야 한다(아니면 실패).
4. `force_exact_v324` 비상 모드에서는 손 요청을 거부한다.
5. report의 `hands`(`left`·`right`·`library_sha256`·`body_bones_preserved`)가 요청과 다르면
   runner가 `WORKER_INTEGRITY_ERROR`로 막는다.
6. 손 굽기는 framing·카메라 회전·미리보기보다 먼저 돈다. 미리보기는 손이 적용된 실제 산출물이다.

API
- `POST /convert`, `/convert-bundle`, `/convert-framed`: 선택 form 필드 `hand_left`, `hand_right`
  (프리셋 id, 비우면 그 손은 리그 기본 손). 모르는 id는 `400 INVALID_HAND_PRESET`.
- 응답 헤더 `X-Standin-Hands: left=<id|rest>;right=<id|rest>` 또는 `none`.
- `/convert-framed` JSON에 `hands: {left, right, library_version, version} | null`.
- `left`/`right`는 **캐릭터 자신의** 왼손/오른손이다. `mirror=true` 변환에서 손을 화면 기준으로
  맞추려면 호출자가 좌우를 바꿔 보내야 한다.

## 러프 손 인식 (`src/hand_shapes.py`)

- `VLM_HAND_SHAPES=1`일 때 분석과 **별도 호출**로 인물별 좌/우 손 모양을 묻는다
  (기본 꺼짐 — 켜면 컷당 Gemini 호출 1회 추가). 분석 프롬프트는 한 글자도 바뀌지 않는다.
- 응답 `people[].hands`:

```json
{
  "left":  {"preset": "grip", "source": "vlm", "confidence": "high", "visible": true,
            "candidates": ["grip", "fist", "relaxed", "open"]},
  "right": {"preset": "relaxed", "source": "default", "confidence": null, "visible": false,
            "candidates": ["fist", "relaxed", "open", "grip"]},
  "library_version": "hand-library-v1",
  "status": "vlm"
}
```

- 자동 선택 규칙: VLM이 보이는 손에 프리셋을 골랐으면 그 값(`source=vlm`, 확신도 그대로).
  안 보이거나·어휘 밖이거나·호출이 실패하면 `relaxed`(`source=default`).
- 후보: VLM 1순위 → VLM 대안 → `fallback_order` 순으로 중복 없이 최대 4개.
  안 보이는 손에 VLM이 모양을 말했으면 굽지는 않고 후보 맨 앞에만 둔다.
- `confidence=low`이면 UI가 후보를 먼저 펼쳐 보이는 것을 권장한다.
- 배열 길이가 인원수와 다르면 전원 `status=unavailable`(기본값). 한 칸이 밀려 다른 사람 손이
  붙는 일을 막는다. 검출기만 찾은 사람(VLM 슬롯이 아님)도 기본값이다.
- 손은 검색·라우팅·refine의 입력이 아니다(`tests/test_hand_shapes.py`가 검색 결과 불변과
  사용처를 검사한다).

## 남은 일 / 제약

- 실제 남·여 캐릭터에서 7종 × 좌우 렌더 검수(`verify_hand_library.py`) — 각도 값은 렌더를 보고
  한 번 조정할 수 있다. 조정은 JSON만 바꾸면 된다.
- `pinch`·`ok_sign`·`finger_heart`는 표준 손에서 엄지·검지 끝이 0.7cm 안으로 모이도록 맞췄다. 손 크기가 다른 캐릭터에서는 조금 벌어질 수 있다.
- `hyper_open`·`push`·`floor_press`·`climbing_crimp`는 의도적으로 음수 굽힘(젖힘)을 쓴다.
- 소품 잡기(칼 각도 등)·손목 방향은 범위 밖이다. 손목은 V3.2.1 palm roll이 담당한다.
- BFF/클라이언트 연결: `/analyze`의 `hands.*.preset`을 변환 요청 `hand_left/right`로 넘기고,
  후보 선택 UI를 붙이는 작업은 클라이언트 담당이다.
- 프리셋 렌더 썸네일(후보 UI용 작은 손 그림)은 아직 없다.
