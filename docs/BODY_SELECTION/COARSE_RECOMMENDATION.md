# 남녀 우선 대분류·대표 프리셋 추천 — 2026-10-17

사용자 요청에 따라 기본 경로를 **남녀 표현 계열 → 해당 계열 BASE → 명확한 특징이 있을 때만 프리셋 변경**으로 단순화한 `coarse` 전략을 구현했다.
윤곽 분할·단면 폭·포즈 비교는 이 전략의 입력이나 게이트가 아니다. 이전 윤곽 실험은 별도 기록으로 보존한다.
모든 평가 인물에 `recommended_body_id`를 기록하며, 자동 적용 여부를 추천 평가의 결과로 대체하지 않는다.

## 판독·추천 규칙

VLM은 그림 전체의 얼굴/상체/체격/의상/비율 인상을 읽는다. 옷을 입었거나 선화라는 이유만으로 모든 항목을 지우지 않는다.
남성형·여성형은 캐릭터 디자인 계열이며 생물학적 성별·나이 판정이 아니다.

1. `presentation`: masculine / feminine / androgynous / unknown을 가장 먼저 판독한다.
2. 해당 계열 `BASE`를 기본 추천으로 둔다. 남녀가 명확한 경우 대안도 같은 계열에서만 고른다.
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

## 실제 러프 28명 결과

Gemini 요청 7배치 성공 / provider 오류 0건. 실제 반환 모델은 `gemini-3.5-flash-lite`.
원본 crop만 사용했고, 기존 포즈 기반 ownership 10명 제외 조건을 판독 전에 적용하지 않았다.
모델이 이 crop의 대상 인물을 직접 구분하고, 모호한 대상은 기본형 추천으로 처리한다.

| 추천 프리셋 | 인원 |
|---|---:|
| male BASE | 16 |
| female BASE | 6 |
| male MUSCULAR | 2 |
| male MUSCLE_MAX | 1 |
| male IDEAL83 | 3 |
| 합계 | **28** |

남녀 대분류: 남성형 22명 / 여성형 6명. 계열 기본형 22명, 명확한 특징을 보고 변경한 추천 6명.
추천 기록률 28/28은 정확도가 아니다. 작가 정답이 없으므로 정확도는 null이다.
이 데이터에 통통형·여성 근육형의 추천이 없으므로 해당 클래스의 실데이터 성능을 검증했다고 주장하지 않는다.
해당 분기는 회귀 테스트에서 확인했으며 다양한 실제 러프를 추가해 평가해야 한다.

특징 추천 사례:

- `131040:p0`, `2.16.04:p1` → 남성 근육형 `male-muscular-v0-7-2`.
- `2.16.04:p0` → 남성 고근육형 `male-muscle-max-v0-7-2`.
- `124629:p0`, `131056:p0`, `청룡성:p1` → 남성 8.3등신형 `male-ideal83-v0-7-2`.
- 여성형으로 판독한 6명은 명확한 특수 체형 근거가 없어 모두 `female-base-v0-3-4`.

로컬 보고서: `artifacts/body-matching/2026-10-17-coarse/REPORT.html`.
모든 인물의 추천 ID/이름/대안/기본값 여부/근거가 표와 카드에 있다.
`recommendations.csv`, `family-first/results.json`에도 28명 모두 기록했다.
요약은 [COARSE_RECOMMENDATION_METRICS.json](COARSE_RECOMMENDATION_METRICS.json).

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

원본 이미지는 해시 확인, 원격 요청에는 불투명 ID만 사용한다. API 키는 출력·커밋하지 않는다.
프리셋과 성별을 나타내는 파일 이름을 모델에 주지 않는다.

신규 31개 회귀: 양 계열 기본형, 약한 특징 기본형 유지, 선명한 특징 경로,
근육질 여성 계열 유지, 잘린 하체와 근육 판독의 독립성, 스틱/얼굴 전용 입력,
정보 부족·provider 오류에도 모든 추천 기록, typed API 및 BFF 계약, legacy 기본값 유지.

전체 CI 범위 회귀: **903 passed, 8 skipped, 28 subtests passed**. 스모크 **54/54**, compileall 통과.
