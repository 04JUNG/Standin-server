# 남녀 우선 대분류·대표 프리셋 추천 — 2026-10-17

사용자 요청에 따라 기본 경로를 **남녀 표현 계열 → 해당 계열 BASE → 명확한 특징이 있을 때만 프리셋 변경**으로 단순화한 `coarse` 전략을 구현했다.
윤곽 분할·단면 폭·포즈 비교는 이 전략의 입력이나 게이트가 아니다. 이전 윤곽 실험은 별도 기록으로 보존한다.
모든 평가 인물에 `recommended_body_id`를 기록하며, 자동 적용 여부를 추천 평가의 결과로 대체하지 않는다.

## 판독·추천 규칙

VLM은 그림 전체의 얼굴/상체/체격/의상/비율 인상을 읽는다. 옷을 입었거나 선화라는 이유만으로 모든 항목을 지우지 않는다.
남성형·여성형은 캐릭터 디자인 계열이며 생물학적 성별·나이 판정이 아니다.

1. `presentation`: masculine / feminine / androgynous / unknown을 가장 먼저 판독한다.
2. 해당 계열 `BASE`를 기본 추천으로 둔다. 남녀가 명확한 경우 대안도 같은 계열에서만 고른다. 남녀 판독 자체가 tentative이면 주 추천과 같은 계열 BASE를 보존하면서 반대 계열 BASE도 대안으로 제공한다(최대 3개).
3. 나머지 축은 `muscle`(일반/근육질/매우 근육질), `fullness`(마름/보통/부드러움/통통함), `proportion`(아담/표준/길쭉함)이다.
4. 특징이 **clear**일 때만 기본형을 변경한다. tentative/unknown은 기록하지만 기본형을 바꾸지 않는다.
5. 특징이 겹치면 근육 → 부드럽고 큰 체격 → 비율 → 슬림 순으로 대표값을 선택하고, 다른 특징 후보와 BASE를 대안으로 남긴다. 이 우선순위는 제품 규칙이며 학습된 점수가 아니다.
6. 남녀도 불명확하면 설정 기본 계열의 BASE를 명시적으로 추천하고, 반대 계열 BASE도 대안에 기록한다. `observed_family=null`, `recommendation_basis=catalog_default`로 판독과 기본값을 구분한다.

| 명확한 특징 | 남성 계열 | 여성 계열 |
|---|---|---|
| 특징 없음·약한 인상 | BASE | BASE |
| 근육질 | MUSCULAR | MUSCLE_MAX |
| 매우 근육질 | MUSCLE_MAX | NEW_MUSCULAR |
| 매우 근육질 + 부드럽거나 큰 볼륨 | BULK | NEW_MUSCULAR |
| 통통한 체격 | FAT_MAX | FAT_MAX |
| 부드러운 체격 | SOFT | SOFT |
| 길쭉한 비율 | IDEAL83 | IDEAL83 |
| 아담한 비율 | TOON64 | TOON64 |
| 슬림 | BASE (전용 남성 SLIM 없음) | SLIM |

여성 자산의 MUSCULAR 표시명은 탄탄형, MUSCLE_MAX는 근육형, NEW_MUSCULAR는 고근육형이다.
최신 확정 20종의 실제 이름을 사용하며 존재하지 않는 체형 ID를 만들지 않는다.
TOON64는 아동 판정이 아니고, IDEAL83도 러프에서 정확히 8.3등신을 측정했다는 뜻이 아니다.

## 실제 러프 28명 원본 대조 리뷰 — 최신 결과

최신 검수는 [COARSE_REVIEW.md](COARSE_REVIEW.md), 수치는 [COARSE_REVIEW_METRICS.json](COARSE_REVIEW_METRICS.json),
28명 각각의 전후 추천과 판독 근거는 [COARSE_REVIEW_CASES.json](COARSE_REVIEW_CASES.json)을 따른다.

원본을 전수 확인한 결과 `131112:p4`가 옆 남학생을 읽고 남성 BASE를 추천하는 오류를 확인했다.
머리까지 확장한 인물 crop + 동일 crop에 대상 박스를 표시한 입력으로 수정하여 **여성 BASE**를 추천한다.
`124637:p0`는 사용자가 **남자**라고 확인하여 기존 남성 BASE 추천이 정답이다. 이전 AI 리뷰의 여성형 판독을 정정했다.
`124702:p0`는 사용자가 **여자**라고 확인했으므로 중성형 판독 → 설정 남성 BASE 주 추천은 **오답**이다. 여성 BASE는 대안에만 있으며 이 오류는 미해결이다.

| 추천 프리셋 | 인원 |
|---|---:|
| male BASE | 14 |
| female BASE | 7 |
| male MUSCULAR | 3 |
| male MUSCLE_MAX | 1 |
| male IDEAL83 | 3 |
| 합계 | **28** |

판독 계열: 남성형 20명 / 여성형 7명 / 중성적 1명(설정 남성 BASE 추천).
Gemini 7배치 성공, 오류 0건. 실제 모델 `gemini-3.5-flash-lite`를 유지했다.
다른 모델에서 새 불일치가 생기거나 기존 불일치가 남아 Flash 3.8/Pro 3.1로 교체하지 않았다.

사용자 확인 정답 2건 기준은 **1/2(50%)**다. 나머지 AI 리뷰 라벨 24건의 일치는 23/24 → 24/24이며,
이를 합친 혼합 기준은 24/26 → 25/26이다. 혼합 일치율 96.2%를 작가 정답 정확도로 표현하지 않는다.
최초 AI 리뷰는 변경하지 않고 사용자 정답을 별도 파일에 기록하여 평가에 우선 적용했다.
추가 단순 소체 지침은 4개 crop에서 오류를 해결하지 못해 채택하지 않았다.

로컬 보고서: `artifacts/body-matching/2026-10-17-coarse-review/REPORT.html`.
원본 전체(로컬 검수 전용), 이전 crop, 새 대상 crop, 전후 추천, 모든 판독 근거를 28명 각각 표시한다.
이전 결과는 `2026-10-17-coarse/family-first/results.json`에 보존한다.

## API와 활성화

`BODY_CLASSIFICATION_STRATEGY=coarse`를 선택하면 기존 `/analyze` sidecar에 연결된다.
설정 기본값은 `legacy`이며 실제 `.env`, 스테이징, 런타임 카탈로그는 변경하지 않았다.
이 변경은 배포하지 않았다. 기존 수동 선택·자동 선택·기본값 처리도 legacy 모드에서 유지한다.

```dotenv
BODY_MATCHING_MODE=shadow
BODY_CLASSIFICATION_STRATEGY=coarse
BODY_COARSE_CATALOG_PATH=config/body_morphology.v1.json
BODY_VLM_PROVIDER=gemini
BODY_VLM_MODEL=gemini-flash-lite-latest
```

주요 응답:

```text
body_matching.people[i].coarse_classification.axes.presentation
body_matching.people[i].coarse_recommendation.recommended_body_id
body_matching.people[i].coarse_recommendation.recommended_preset
body_matching.people[i].coarse_recommendation.recommendation_basis
body_matching.people[i].coarse_recommendation.candidates
```

`recommendation_basis`: recognized_feature / presentation_default / catalog_default.
VLM 오류·budget 초과·crop 누락도 인물별 추천은 BASE로 남기고 오류 사유를 별도로 기록한다.
분류용 마네킹 프리셋과 런타임 QA를 통과한 FBX 권한은 구분하므로 `auto_apply=false`, `applied_body_id=null`이다.
기존 `selection_state=unavailable`은 자동 적용 자산이 없음을 뜻하며 **추천 미생성으로 해석하지 않는다**.
coarse의 기존 `candidates`는 빈 배열로 유지하여 미승인 추천 ID를 기존 BFF의 실행 후보에 섞지 않는다. `suggested_asset`도 null이고 BFF/Export 주문을 만들지 않는다. UI는 `coarse_recommendation`을 읽어야 한다.

## 재현·회귀

```bash
python scripts/eval_body_coarse.py --protocol /path/to/protocol.json \
  --catalog config/body_morphology.v1.json --out /path/to/result \
  --remote --env-file /path/to/.env
# 동일 관찰의 추천 규칙만 다시 평가(추가 API 호출 없음)
python scripts/eval_body_coarse.py --protocol /path/to/protocol.json \
  --catalog config/body_morphology.v1.json --out /path/to/replay \
  --recorded /path/to/result/results.json
```

원본 이미지는 해시 확인, 원격 요청에는 불투명 ID와 인물 crop만 사용한다. 원본 전체를 원격으로 보내지 않는다. `camera_real_rough_20260921_v2/protocol.json` 형식의 원본 이미지+대상 box 프로토콜은 서비스와 동일한 `coarse_input.prepare_input`을 사용한다. 예전 crop-only 프로토콜도 비교 재현용으로 지원한다. API 키는 출력·커밋하지 않는다.
프리셋과 성별을 나타내는 파일 이름을 모델에 주지 않는다.

대분류 회귀 49개: 양 계열 기본형, 약한 특징 기본형 유지, 선명한 특징 경로,
근육질 여성 계열 유지, 잘린 하체와 근육 판독의 독립성, 스틱/얼굴 전용 입력,
정보 부족·provider 오류에도 모든 추천 기록, typed API 및 BFF 계약, legacy 기본값 유지.

전체 CI 범위 회귀: **921 passed, 8 skipped, 28 subtests passed**. 스모크 **54/54**, compileall 통과.

입력 회귀는 얼굴 보존·가장자리 대상과 옆 인물 분리·동일 crop 두 장만 전송·서비스/평가 입력 일치·원본 해시 변조 거부·잠정 계열 대안의 개수와 순위를 검증한다.

사용자 확인 라벨 채점은 `--labels docs/BODY_SELECTION/COARSE_CONFIRMED_LABELS.json`으로 재현한다. 라벨은 요청 프롬프트·이미지·추천 규칙에 들어가지 않고 평가 단계에서만 사용한다. `metrics.confirmed_family`는 해당 라벨 사례만의 정확도이며 전체 28명 정확도가 아니다.
