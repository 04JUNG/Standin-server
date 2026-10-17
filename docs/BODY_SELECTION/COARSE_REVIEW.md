# 실제 러프 28명 원본 대조 리뷰 — 2026-10-17

## 사용자 확인 정답 반영

사용자가 `124637: 남자`, `124702: 여자`로 확인했다. 최초 AI 리뷰는 원본 그대로 보존하고 [COARSE_CONFIRMED_LABELS.json](COARSE_CONFIRMED_LABELS.json)에 정답과 출처를 별도로 기록했다. 정답은 평가에만 사용하며 VLM 요청이나 추천 규칙에 넣지 않는다.

| 사례 | 사용자 정답 | 현재 판독 | 현재 주 추천 | 결과 |
|---|---|---|---|---|
| 124637:p0 | 남성 | 잠정 남성형 | 남성 BASE | 정답. 여성으로 읽었던 AI 리뷰를 정정 |
| 124702:p0 | 여성 | 중성형 | 설정 남성 BASE | 오답. 여성 BASE는 대안에만 있음 |

**사용자 확인 2건의 추천 계열 정확도는 1/2(50%)**다. 여성 BASE가 대안에 있다는 이유로 두 번째 사례를 성공으로 세지 않는다. 일반화된 단순 소체 판독 지침을 4개 crop에 시험했지만 두 번째는 여전히 중성형이었고 첫 번째도 남성에서 중성으로 약해져 미채택했다. 특정 ID별 결과 덮어쓰기는 없다.

## 확인한 입력 오류와 수정

`131112:p4`는 화면 맨 오른쪽 여성 대신 옆 안경 학생을 읽었다. 이전 포즈 기반 crop은 여성의 얼굴을 잘랐고 프롬프트는 crop 중앙을 대상으로 지시했다. 머리 문맥과 대상 위치를 보존하여 남성 BASE → **여성 BASE**로 수정됐다.

`coarse_input.prepare_input`을 서비스와 평가가 공유한다. 기존 descriptor box를 기준으로 상단에 높이 30% 또는 폭 35% 중 큰 여백, 좌우 폭 20%, 하단 높이 5%를 더하고 이미지 경계로 제한한다. 얼굴 검출 보장이 아닌 문맥 보존 휴리스틱이다.

깨끗한 인물 crop과 **동일 범위**의 대상 박스 표시본만 원격으로 보낸다. 원본 전체는 로컬 보고서에서만 검수한다. VLM 좌표 생성·pose 검색 변경·윤곽 측정은 추가하지 않았다. 프롬프트에서 crop 중앙/옆 사람 대신 표시된 대상의 얼굴과 몸을 연결하고, 이름·작품 지식·어깨 폭·머리 길이 하나에 의존하지 않도록 지시한다. 텍스트 마스킹을 구현한 것은 아니다.

모든 잠정 계열 판독은 주 추천 + 같은 계열 BASE + 반대 계열 BASE를 최대 3개로 제공한다. clear 계열은 같은 계열 대안만 제공한다. 근거 부족도 추천 ID를 기록하며 자동 적용은 하지 않는다.

## 평가 수치의 구분

28명 모두 원본 검수·추천 기록을 완료했다. 현재 평가 기준은 사용자 확인 2건 + AI 리뷰 라벨 24건 + AI가 계열을 확정하지 못한 2건이다. 최초 AI 리뷰 파일과 이후 사용자 정답 파일 해시를 모두 보존한다.

| 조건 | 사용자 정답 2건 판독 | 나머지 AI 리뷰 24건 일치 | 혼합 기준 판독 일치 |
|---|---:|---:|---:|
| 기존 crop + Lite | 1/2 | 23/24 | 24/26 |
| 새 crop·대상 표시·지침 + Lite (채택) | 1/2 | 24/24 | 25/26 |
| 같은 새 입력 + Flash 3.8 (비교) | 0/2 | 23/24 | 23/26 |
| 같은 새 입력 + Pro 3.1 (비교) | 1/2 | 23/24 | 24/26 |

Pro의 추천 계열은 혼합 기준 25/26이지만 판독 계열은 24/26이다. 미확정 판독의 설정 남성 BASE가 우연히 리뷰와 일치한 것을 시각 판독 성공으로 세지 않았다.

**혼합 기준 24/26(92.3%) → 25/26(96.2%)는 전체 정확도가 아니다.** 정답 확인 사례만의 추천 정확도는 50%이며 표본이 2건에 불과하다. 이 28명은 개발 세트이고 독립 검증 성능도 아니다. 이전 보고서의 23/26 → 24/26은 사용자 정답 반영 전 AI 리뷰 기준 수치다. 수치 상승의 일부는 정답 수정이며 모델 개선으로 주장하지 않는다.

채택 모델은 실제 응답 `gemini-3.5-flash-lite`로 유지한다. 다른 모델은 새 불일치 또는 기존 불일치가 남아 교체하지 않았다. Gemini 2.5 Flash는 새 키에서 404(`no longer available to new users`)로 실패해 정확도 분모에서 제외했다. 실제 .env는 변경하지 않았다.

근육·체격·비율의 정확도는 이번 남녀 리뷰로 주장하지 않는다. 최종 추천 28명은 남성 BASE 14, 여성 BASE 7, 남성 MUSCULAR 3, MUSCLE_MAX 1, IDEAL83 3이다.

## 재현·검증

```bash
python scripts/eval_body_coarse.py \
  --protocol /path/to/camera_real_rough_20260921_v2/protocol.json \
  --out /path/to/run --remote --env-file /path/to/.env \
  --labels docs/BODY_SELECTION/COARSE_CONFIRMED_LABELS.json
python scripts/eval_body_coarse.py \
  --protocol /path/to/camera_real_rough_20260921_v2/protocol.json \
  --out /path/to/replay --recorded /path/to/run/results.json \
  --labels docs/BODY_SELECTION/COARSE_CONFIRMED_LABELS.json
```

원본/crop/context/target 해시와 입력 버전, 원격 관찰·추천·정답 출처를 저장한다. `metrics.confirmed_family`에 확인된 사례만 채점한다. 중복/없는 ID·미확정 정답 라벨은 거부하며 라벨을 모델 입력으로 사용하지 않는다. 원격 판독 코드와 추천 재생 코드 해시를 구분한다.

- 대분류·평가 테스트 49개. 전체 CI 범위 921 passed / 8 skipped / 28 subtests passed.
- compileall 통과. 직전 런타임 변경 smoke 54/54; 이번에는 판독 런타임 변경 없음.
- HTML 28개 카드·84개 이미지와 파일 연결 정적 검증. file URL 정책으로 브라우저 렌더링 검증은 미수행.
- Docker build는 로컬 daemon 부재로 CI에서 확인한다.
- 기본 legacy 및 기존 BFF/Export 계약 유지. 배포·머지 없음.

로컬 보고서: `artifacts/body-matching/2026-10-17-coarse-review/REPORT.html`.
개별 기록은 `COARSE_REVIEW_CASES.json`, 수치는 `COARSE_REVIEW_METRICS.json`.
이미지·원격 원문·키는 Git에 넣지 않는다.
