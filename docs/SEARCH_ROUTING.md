# 러프함 기반 인물별 검색 경로 분기 (04J-286)

상태: 최신 develop(PR #84·#86 포함)에 구현 반영·오프라인 검증 완료.
실 Gemini 재현은 러프 이미지 외부 전송 승인 대기 · 기본값 **꺼짐**(`SEARCH_ROUTING=0`)

## 1. 무엇을 하나

컷 안의 **인물마다** 검색 경로를 고른다.

| 경로 | 조건 | 하는 일 |
|---|---|---|
| `geometric` | 관절 근거 충분 | 기존 기하 kNN. 후보·순위·confidence·refine 정책 변경 없음 |
| `semantic` | 관절 부족 + VLM 자세 설명이 구체적 | 시맨틱 검색 → `semantic_candidates`(기존 `candidates`는 빈 채로) |
| `undetermined` | 관절 부족 + 설명이 없거나 너무 일반적 | 검색하지 않고 판정 수치만 남김 |
| `error` | 관절 부족 + 자세 설명 VLM 호출 자체 실패 | 오류로 남김(reason 유지). `undetermined`와 섞지 않음 |

얽힘 관계(hug/fight)와 두상 인물은 기존 정책이 검색을 막는다. 라우터는 이 경로를 바꾸지 않고
`undetermined`·`reason=existing_policy:<코드>`·`search.status=not_run`으로 기록만 한다.

상하체 별도 검색 신규 구현·상하체 합성은 범위 밖이다.

## 2. 흐름과 위치

```
Gemini 분석(기존) → RTMPose·슬롯 배정·crop·Human-Art rescue(기존)
  → [_route_slots] 인물별 관절 근거 판정
       └ 부족한 인물이 있으면 그 인물들만 묶어 자세 설명 1회 호출(describe_people)
  → 인물별: geometric이면 기존 _evaluate_slot / 아니면 _route_non_geometric
```

- 위치: `src/pipeline.py::Pipeline._process_slots` — rescue가 끝난 뒤, 인물별 검색 직전.
  crop 재시도·Human-Art rescue는 모두 그 앞에서 끝나므로 라우터는 **최종** 관절 근거만 본다.
- 순수 판정: `src/search_routing.py`(`assess_joint_evidence`, `assess_semantic_evidence`, `decide_route`).
- 시맨틱 검색 계약: `src/semantic_search.py`.
- 자세 설명 호출: `src/vlm/client.py::GeminiVLMClient.describe_people`, 프롬프트 `src/vlm/prompts.py::POSE_DESCRIBE_*`
  (`d1-taxonomy-v2`). 태그 어휘는 라이브러리 태깅과 같은 search-taxonomy-v2(`src/semantic_vocab.py`).

정상 러프(관절 충분)는 자세 설명 호출이 **0회**다. 비용·지연이 그대로다.

## 3. 판정 기준과 초기값 근거

### 관절 근거 (env `ROUTING_JOINT_*`)

| 기준 | 초기값 | 근거 |
|---|---|---|
| 허용 coverage | full, reduced, sparse, upper_only | `insufficient`는 기존 파이프라인도 검색을 포기하는 등급 |
| 유효 뼈 수(전신 계열) | ≥ 6 | 아래 운영 기록 sweep에서 후보를 받던 인물을 하나도 잃지 않는 최대값 |
| 몸통 뼈 수 | ≥ 4 | 실패 인물은 모두 ≤2, 후보를 받은 인물은 모두 4 |
| 상체 전용 유효 뼈 | ≥ 3 | upper_only 검색의 최소 관측(어깨+팔) |
| 박스 높이 | ≥ 64px 그리고 ≥ 이미지 높이의 5% | 운영에서 12~46px 박스는 전부 잘못 잡힌 영역 |

평균 keypoint score는 **문턱으로 쓰지 않는다**(CLAUDE.md 불변식 2: rtmlib score scale은 모델마다 다름).
기록용으로만 `mean_valid_body_score`에 남긴다. 운영 기록에서도 실패 0.31~0.47 / 성공 0.38~0.93으로 겹쳐
판별력이 없었다.

**운영 기록 재적용**(`scripts/eval_routing_thresholds.py`, 클로즈베타 85 Job·85명, 사람 눈 시각 분류 대조):

| 시각 분류 | geometric | not_geometric |
|---|---:|---:|
| 결과 없음(추천 불가) | 0 | 18 |
| 결과 있음·포즈 차이 | 50 | 1 (후보 0개였던 인물) |
| 결과 있음·유사 | 16 | 0 |

유효 뼈 기준을 7로 올리면 시각적으로 '유사'였던 인물 1명(sparse, 뼈 6개)이 geometric에서 빠진다.
→ 6이 회귀 없는 최대값. 박스 기준 32/64/96px은 이 데이터에서 결과 차이가 없다.

### 의미 근거 (env `ROUTING_SEMANTIC_*`)

유효(`valid`) 조건 — 모두 만족:
- `visible=true`, `certainty`가 low가 아님
- 구체 태그 ≥ 1: 자세(서기 제외)·행동(기다리기·바라보기·쉬기·듣기·말하기·발 내딛기·손짓 제외)·활동 중 하나
- 검색 문장 ≥ 8자, 몸 배치 관찰 문장 ≥ 6자

그 외 상태: `too_generic`(예: "서 있는 사람"), `low_certainty`, `not_visible`, `empty`, `vlm_error`.

## 4. 응답·로그 (라우팅 켰을 때만 채워짐)

- `people[i].search_routing`: `route`, `reason`, `joint_evidence{sufficient, reasons, metrics, thresholds}`,
  `semantic_evidence{status, reasons, metrics, thresholds, description, error_reason}`,
  `search{status: ok|no_result|error|not_run, backend, reason, top}`, `elapsed_ms{geometric_search_ms | semantic_search_ms, describe_ms_cut}`
- `people[i].semantic_candidates`: `[{pose_id, score, matched, pose}]` — `pose`가 포즈 JSON(태그·설명·`bvh_url`)
- `search_routing`(컷): 자세 설명 호출 상태·provider·error_reason·retryable·retry_after·소요 시간
- 로그 이벤트 `search_routing`(인물별), `gemini_pose_describe`(호출별)

기존 필드는 모두 그대로다. semantic 후보를 기존 `candidates`에 섞지 않은 이유: `distance`(관절 거리)와 의미 점수는
비교할 수 없고, 앱이 distance를 관절 거리로 해석하는 곳이 있으면 의미가 섞인다. 앱 표시를 붙일 때 `semantic_candidates`를 읽는다.

## 5. 실패 분리와 재시도 정책

- 경로와 검색 결과는 따로 남는다: `semantic + no_result`, `geometric + no_result`(기하 검색이 hard fallback),
  `semantic + error`(색인 없음 등).
- 자세 설명 호출 실패: 04J-294(`src/vlm/failures.py`, PR #86)가 병합돼 있으면 그 분류(rate_limited ·
  daily_quota_exhausted · upstream_unavailable · timeout · upstream_error)를 그대로 쓴다. 아직 없으면 같은
  이름의 최소 분류로 대신한다. 파싱/스키마 오류는 `invalid_response`, 그 밖의 SDK·4xx는 `provider_error`.
- **자동 재시도·경로 전환 없음(기본).** 제안 정책(미구현):
  1. `geometric + no_result` → 같은 인물에 자세 설명을 묻고 semantic을 *보조 후보*로만 추가(기하 후보는 그대로)
  2. `error`(retryable=true) → 재시도하지 않고 앱에 retry_after만 전달. daily_quota는 재시도 금지
  3. `semantic + no_result` → 재시도 없음. 라이브러리 공백 후보로 pose_gaps에 적재

## 6. 끄고 켜기

`SEARCH_ROUTING=0`(기본): 기존과 같은 결과. 새 필드는 기본값(`search_routing: null`, `semantic_candidates: []`),
quality_trace·notes·후보·confidence·refine 정책 불변(`tests/test_search_routing.py::test_routing_off_matches_legacy_and_adds_nothing`).
켜도 geometric 인물의 후보·순위·confidence는 끈 것과 같다(같은 파일 `test_clean_people_stay_geometric_*`, 실모델 재현 39/39 동일).

시맨틱 색인: `SEMANTIC_TAGS_PATH`(기본 `data/search-v2.json`, pose-library-viewer `dist/data/search-v2.json`).
라이브러리에 없는 pose_id는 후보로 내지 않는다. 파일이 없으면 semantic 경로는 `search.status=error`
(`semantic_index_missing`)로 끝난다 — 조용한 빈 결과가 아니다.

## 7. 04J-285와의 경계

04J-285(자연어 → 기존 포즈 JSON 검색)는 아직 들어오지 않았다. 지금 semantic 경로 뒤에는 **임시 태그 검색기**
(`TagSemanticSearcher`, 태그 겹침 + 설명 문장 bigram, 임베딩/LLM 아님)가 붙어 있다. 285가 완성되면
`SemanticSearcher` 계약(`search(query, top_k) -> SemanticSearchResult(status, hits)`)을 구현해
`build_semantic_searcher`만 바꾼다. 라우터·응답 형식은 그대로다.

## 8. 재현

```bash
# 운영 기록으로 기준값 점검(네트워크·모델 불필요)
python scripts/eval_routing_thresholds.py --jobs <beta>/jobs.json \
  --visual <beta>/visual_classification.csv --out out/routing-thresholds

# 대표 사례: 기존 Top-5 vs 라우팅 Top-5 나란히(실 Gemini)
SEARCH_ROUTING=0 VLM_PROVIDER=gemini POSE_BACKEND=rtmlib \
python scripts/eval_search_routing.py --vlm live \
  --manifest <beta>/testset_50_balanced_20261011/manifest.csv \
  --images <beta>/testset_50_balanced_20261011 \
  --db <bundle>/poses.db --tags <viewer>/dist/data/search-v2.json --thumbs <bundle>/thumbs \
  --out out/routing-eval-live
```

출력 `side_by_side.csv`·`index.html`이 04J-291 제품팀 공동 실험(Top-5 활용 가능 여부, 성공→실패 회귀,
처리 시간, 실패 원인)의 입력이다.

`--test-ids T008,T009,T044`로 사례를 선택할 수 있다. 실호출 모드에서 VLM이나 RTMPose가
mock으로 폴백하면 평가를 중단한다. 완료한 사례는 매번 `results.json`에 저장한다.
geometric의 Top-5(포즈 ID·view·거리)가 바뀌거나 사례 처리 오류가 있으면 종료 코드 1을 낸다.
실호출 보고서는 설명 호출 지연 p50/p95를 집계한다. replay에서는 fixture 지연을 Gemini 지연으로 보고하지 않는다.

### 2026-10-13 로컬 재검증

- 기준: develop `c838051`, Python 3.12, 실제 current-X·실제 라이브러리.
- 50장·53명: geometric 39명, semantic 8명, undetermined 6명, 사례 오류 0장.
- geometric 39명 모두 기존 Top-5의 포즈 ID·view·거리 동일.
- semantic 8명 모두 라이브러리 후보 반환. 설명은 인수인계의 Claude fixture를 사용했으므로
  Gemini 품질을 입증하는 결과가 아니다.
- API의 혼합 인물 응답·OFF 기본값, 복수 인물 설명 1회, 시맨틱 검색 예외 격리,
  Gemini 설명 호출 timeout·일일 쿼터 분류·재시도 금지를 추가 검증했다.
- 운영 활성화·클라이언트 표시 연결은 진행하지 않았다.
