# 러프함 기반 인물별 검색 경로 분기 (04J-286)

상태: 최신 develop(PR #84·#86 포함)에 구현 반영·오프라인 검증 완료.
사용자 승인 후 실제 Gemini 3장 파일럿 완료. 전체 50장 실호출 품질 평가는 미진행.
기본값 **꺼짐**(`SEARCH_ROUTING=0`)

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
- 구체 태그 ≥ 1, 또는 구별되는 몸 관찰 ≥ 2
  (`ROUTING_SEMANTIC_MIN_BODY_OBSERVATIONS`, 기본 2).
  구체 태그는 자세(서기 제외)·행동(일반 행동 제외)·활동 중 하나다.
  무릎 굽힘·다리 벌림·팔 올림·몸 기울임은 구별되는 관찰이며,
  무릎 펴짐·다리 모음·팔 내림·상체 곧음만으로 일반 서기를 유효로 승격하지 않는다.
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
(`TagSemanticSearcher`, 구체 태그 + 관찰 몸 배치 + 문장 bigram, 임베딩/LLM 아님)가 붙어 있다. 285가 완성되면
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
- 최종 로컬 전체 테스트: 801 passed, 8 skipped, 22 subtests passed.
  스모크 54/54, compileall·git diff --check 통과.
- 기능 브랜치: `san400b/04j-286-dynamic-routing`. GitHub 업로드·PR 생성은
  자동 승인 검토에서 외부 게시 승인이 필요하다고 차단되어 사용자 승인 대기다. CI는 미실행이다.
- 운영 활성화·클라이언트 표시 연결은 진행하지 않았다.

### 실제 Gemini 3장 파일럿(사용자 승인 후)

사용자 승인 범위인 T008·T009·T044를 실제 Gemini로 실행했다.
기본 gemini-2.5-flash는 현재 키에서 404를 반환해 실험 env에서만 gemini-3.8-flash로 변경했다.

- T008·T009: 기존 후보 0개 → semantic 후보 각 5개. 설명 추가 호출 각각 8.699초·7.870초.
- T044: geometric Top-5의 ID·view·거리 동일, 설명 추가 호출 0회.
- 후보 프런트 썸네일 확인: T008의 3위·T009의 2위는 일부 형태가 유사하나,
  두 사례의 1위는 팔다리 배치가 달라 후보 반환만으로 품질 성공을 선언할 수 없다.
- 두 실패 러프는 같은 구도의 크기 다른 이미지이며 Gemini가 서로 다른 종목·장면 태그를 붙였다.
  크기와 응답 변동의 영향을 구분하려면 추가 대조가 필요하다.
- 전체 50장 실호출·PR·배포는 미진행. 검색 기준·설명 프롬프트·운영 기본 모델은 변경하지 않았다.
- 로컬 증거: `tmp/routing-eval-live-pilot-g38/{REPORT.md,results.json,side_by_side.csv,index.html}`.

## 9. 몸 관찰을 연결한 검색 업그레이드(d2-observed-body)

원인은 분기 이후의 설명·순위에 있었다. d1의 body_details는 쿼리에서 누락됐고,
장면·활동 태그와 일반 standing이 몸 배치보다 순위를 크게 좌우했다.
행동 태그가 없는 구체적인 설명은 구형 검색기에서 검색할 수 없었다.

- 설명: 몸 관찰과 해석을 분리하고 knees·legs·arms·torso를 정성 값으로 받는다.
  좌표·각도는 생성하지 않는다. 주된 행동은 하나이며 불확실하면 비울 수 있다.
- 파서: body_observations를 허용값으로 검증한다. 근거(context_evidence)가 없는 장면·대상·의도는
  주 쿼리에서 제외하고 dropped에 남긴다. 이것은 사실성 검증이 아닌 최소한의 근거 기록 검사다.
- 검색: body_details와 관찰 값을 전달한다. 라이브러리의 설명·검토 메모에서 정성 속성을 읽고,
  일치에는 무릎 3·다리 간격 3·팔 2·몸통 1.5점을 주며 충돌에는 같은 크기로 감점한다.
  모르는 속성은 중립이다. 단순 standing은 가점하지 않고 여러 행동 추측도 중복 가점하지 않는다.
- 장면 태그는 합계 최대 0.5점이다. 이 가중치는 임시 규칙이며 학습·보정된 정답 확률이 아니다.
- 구체적인 몸 관찰 2개 이상이면 행동 태그가 비어도 후보를 찾을 수 있다.
  일반 서기 단서만 있는 설명, 두상·얽힘 정책, low certainty 차단은 유지한다.
- SemanticQuery에 기본값이 있는 body_details·body_observations를 추가했다.
  SemanticSearcher의 search(query, top_k)와 SemanticHit 응답은 유지한다.
- 후보 matched에 body_matches·body_conflicts·score_components를 기록한다.
  backend는 `tag-v2-body-grounded-interim`이다. v1/v2 점수는 직접 비교하지 않는다.

### 전후 검증

실제 Gemini 3.8 Flash로 승인된 동일 3장을 d2 프롬프트로 재실행했다.
T008·T009의 몸 관찰 4개가 일치했고, 종목·판타지·전투 태그는 비어 있었다.
행동 해석은 T008=fall, T009=없음으로 여전히 불확실하므로 관찰 일치와 행동 정확도를 구분한다.

기존 Gemini 설명을 고정한 검색기 대조에서도 두 사례의 1위가 모두
`combat_ual1_Jump_Start_f00000`(무릎을 굽히고 다리를 벌려 팔을 내린 형태)로 바뀌었다.
새 설명에서는 T008의 1위가 같은 포즈, T009의 1위가 `cmu_18_14_00277`
(다리를 벌려 앉고 손을 내린 형태)였다. 사람이 확인한 부분적 형태 개선이며 작가의 정답 판정은 아니다.

- 설명 추가 호출: T008 7.208초, T009 8.606초, T044 0회. 지연 개선은 입증되지 않았다.
- T044의 같은 분석을 공유한 off/on Top-5는 ID·view·거리 모두 동일.
- 전체 테스트 813 passed·8 skipped, 22 subtests passed. compileall·diff check 통과.
- 50장 실제 current-X 재생: geometric 39명 전원 기존 Top-5 동일.
  이 50장의 설명은 Claude fixture이며 실 Gemini 품질 근거가 아니다.
- 원본 실호출: `tmp/routing-eval-live-body-v2/results.json`.
  문장 해석기 보완 후 같은 설명으로 검색만 재생한 최종 결과:
  `tmp/routing-eval-live-body-v2-final/results.json`. 추가 VLM 호출 0회.
- 통제 대조: `scripts/compare_semantic_upgrade.py`,
  `tmp/routing-eval-live-body-v2/ablation.json`(기존/새 설명 × 기존/새 검색기).
- 원본 이미지·라이브러리·키는 커밋하지 않았다. SEARCH_ROUTING 기본 OFF, PR·배포 미진행.

남은 한계: 정성 속성은 거친 구분이다. 라이브러리 문장에 없는 속성은 비교할 수 없으며
한국어 규칙이 모든 문장·관점을 해석하지 못한다. 다른 동작의 홀드아웃 평가, 작가의 동작 의도,
카메라·세밀한 팔 방향과 검색 지연을 별도로 검증해야 한다.

## 10. 실제 BVH 몸 배치 색인(선택적 v3 실험)

라이브러리 설명에 없는 관절 배치를 실제 BVH의 기존 COCO17 매핑에서 추출한다.
러프의 스켈레톤을 복원하거나 기하 검색을 수정하지 않는다. `bvh-body-v1` 색인은
무릎 내각, 골반 축 기준 무릎/발 간격, 몸통 길이 대비 손·팔꿈치 높이, 몸통 기울기를 저장한다.
대표 프레임은 기존 라이브러리와 같은 0이다. 요청 단계에서는 메모리 속성만 비교한다.

- 무릎: 양쪽 140° 이하=bent, 160° 이상=straight, 한쪽씩 다르면 mixed.
  중간값·결측·퇴화한 뼈는 unknown이며 감점하지 않는다.
- 다리: 골반의 수평 좌우 축으로 무릎·발 간격을 측정한다. 골반 폭의 2배 이상=apart,
  1.35배 이하=together. 앞뒤 발 간격이 몸통 길이의 0.7배 이상이면 staggered를 우선한다
  (단 넓은 좌우 벌림이 있으면 apart). 관점별 실루엣과 같다는 보장은 없다.
- 팔: 손·팔꿈치 높이와 좌우 벌림으로 lowered/raised/spread/mixed를 구분한다.
  모호한 팔은 unknown이다. torso는 12° 이하 upright만 판정한다.
  소스별 몸 정면 방향이 불확실하므로 forward/backward는 추정하지 않는다.
- 임계값은 임시 규칙이며 작가 라벨로 보정한 값이 아니다. BVH 매핑 presence 점수는
  RTMPose score와 다르며 러프의 raw score 기준값을 추가하지 않았다.

### 활성화·재현

```bash
python scripts/build_semantic_body_index.py --db <bundle>/poses.db \
  --tags <viewer>/dist/data/search-v2.json --bvh-root <bundle>/bvh \
  --out tmp/semantic-body-v1.json

# 실험 환경에서만 지정. 비우면 v2, SEARCH_ROUTING은 계속 기본 0.
SEMANTIC_BODY_INDEX_PATH=tmp/semantic-body-v1.json

# Gemini 설명을 고정한 검색기 대조. 추가 VLM 호출 0회.
python scripts/compare_semantic_body_index.py --results <saved>/results.json \
  --tags <viewer>/dist/data/search-v2.json --db <bundle>/poses.db \
  --body-index tmp/semantic-body-v1.json --out tmp/body-ablation \
  --images <testset> --thumbs <bundle>/thumbs
```

색인은 알고리즘 버전·임계값·프레임·태그 코퍼스 SHA256·BVH별 SHA256을 보관한다.
로더는 버전·임계값·프레임·코퍼스·관찰 허용값·유한 수치를 검사한다.
BVH의 현재 내용은 요청 중 다시 읽지 않으므로, **BVH/대표 프레임/매핑/태그 변경 시 오프라인
색인을 재생성해야 한다**. 소스 해시는 재현·감사용이며 변경을 실시간 감지하는 기능은 아니다.
빌더는 poses.db를 read-only로 연다. 파일이 실패한 개별 포즈는 오류 목록에 남고,
유효한 부분 색인의 누락 포즈는 기존 문장 속성을 쓴다. 손상·불일치한 색인 전체는 v2 폴백한다.

### 검증에서 발견한 실패와 보완

단서 하나만으로 BVH를 적용하면 ‘한쪽 무릎 굽힘’에 맞는 투구가 착지보다 높아졌다.
또한 문장 어댑터의 mixed는 ‘위로+옆으로’라는 동시 속성일 수도 있어 실제 좌우 비대칭과 같지 않다.
따라서 knees/legs/arms 중 구별되는 intrinsic 관찰이 기본 2개 이상일 때만 v3 몸 근거를 적용한다.
문장에서 추론한 mixed는 이 게이트에서 제외하고 명시적인 mixed만 허용한다.
그 외 쿼리는 v2와 같은 점수/순위를 유지한다(backend 기록만 v3).

실 Gemini T008에는 fall 해석의 3점이 추가 팔 관찰(2점)보다 커서 형태가 덜 맞는 후보가 올라왔다.
행동 태그를 비운 통제 대조에서 더 많은 몸 관찰을 만족한 후보가 1위로 바뀌었다.
이를 보완해 intrinsic 관찰이 충분한 v3 쿼리의 행동 가중치는 1점으로 제한한다.
약한 쿼리/v2의 기존 3점 정책은 유지한다. 이 정책 역시 미보정이며 점수는 확률이 아니다.

### 2026-10-13 최종 결과

- 실제 라이브러리 1,981/1,981개 색인 생성, 파일 처리 실패 0개(약 1.35초).
  이는 모든 관절 속성이 알려졌다는 뜻이 아니다: 무릎 unknown 877개, 팔 unknown 1,141개다.
- d2 실 Gemini T008/T009/T044 설명을 그대로 재사용해 v2/v3 검색을 비교했다.
  T008/T009의 새 1위는 `g1__M_Move5__n03_f1076`이다.
  기존 1위는 각각 `combat_ual1_Jump_Start_f00000`, `cmu_18_14_00277`이었다.
  전면 썸네일에서 넓게 벌린 굽힌 다리·낮은 팔의 부분적 개선을 확인했지만,
  몸통·팔 방향까지 일치하지 않으며 작가의 정답 라벨이 없다. 정확도 상승 수치는 주장하지 않는다.
- T008/T009는 같은 구도의 크기 다른 러프이므로 서로 다른 동작 두 개로 계산하면 안 된다.
  T044 기하 Top-5는 ID·view·distance 동일하고 추가 설명 호출도 없다.
- 실제 current-X·실제 라이브러리의 50장 전체 재생: geometric 39, semantic 8,
  undetermined 6. 기하 39명은 이전 v2 결과와도 Top-5 동일, 사례 오류 0개.
  시맨틱 8명 모두 후보 반환. Claude fixture 대조에서 T008/T009만 v3 적용,
  나머지 6명은 안전 게이트로 기존 검색을 유지한다. 따라서 50장은 새 Gemini 품질 평가가 아니다.
- 새 테스트 29개, 전체 842 passed·8 skipped·22 subtests passed. 스모크 54/54,
  compileall·git diff --check 통과. 합성 자세 테스트는 통제된 구현 검증이지 실제 정확도가 아니다.
- 원본 실호출과 초기 실패 결과는 보존했다. 이번 추가 VLM 호출 0회, PR·push·배포 없음.
  새 옵션과 SEARCH_ROUTING 운영 기본값 모두 비활성이다.

증거: `tmp/semantic-body-v1.json`, `tmp/routing-eval-bvh-body-v3-final/{ablation.json,REPORT.md}`,
`tmp/routing-eval-bvh-fixture-ablation-final/ablation.json`,
`tmp/routing-eval-replay-bvh-v3-release/results.json`.

## 핵심 팔 관계와 미리보기 시점(v4, 2026-10-13)

기본 비활성 `SEMANTIC_SHAPE_INDEX_PATH` 옵션을 추가했다. `scripts/build_semantic_shape_index.py`
는 원본 BVH frame 0에서 몸통 축 기준 손 위치를 계산하고 24개 정사영 시점의 가시성을
미리 기록한다. 새 raw RTMPose 기준·기하 피처·러프 좌표 생성은 없다.

- **팔짱**: reviewed fold_arms 검색 태그 AND 가슴 높이 손 교차가 필요하다.
  무릎을 감싸 손목만 겹친 자세나 머리 위 스트레칭은 팔짱으로 인정하지 않는다.
- **V자 팔**: 양손 모두 몸통 축 위쪽·바깥쪽에 있고 양팔 벌림각이 범위 안에 있어야 한다.
  한 팔만 올리거나 T자·좁은 수직 팔은 제외한다. 임계값은 실험적이며 정확도 점수가 아니다.
- 명확한 관계를 query의 action 또는 관찰 문장에서 읽었으면 hard filter다.
  약한 coarse arms=raised/spread 점수나 장면/행동 점수가 필수 관계를 덮어쓰지 못한다.
- 구체적인 자세도 함께 요구한다(예: sitting+fold_arms). 검증 조합 없음은 no_result와
  `critical_pose_combination_unverified`를 반환한다. 실제 라이브러리 부재 증명은 아니다.
- yaw 0/45/90/180 × pitch 0/30/60/80/-30/-60, roll=0. 관계 가시성 통과 안에서 작은 pitch를
  우선해 투영 몸통이 짧아진 각도의 비율 점수만 극대화하지 않는다. 명시한 부감/앙각 단서는
  body_details·search_text·context_evidence에서만 읽고 낙하라는 행동명으로 시점을 추정하지 않는다.
- `pose.preview_camera`는 미리보기일 뿐 저장된 BVH 회전이 아니다. 동일한 포즈를 카메라만 바꿔
  비교한다. 원근·얼굴 방향·전체 실루엣 픽셀 맞춤은 미구현이다. API의 matched에 선택 근거와
  hint_satisfied를 기록한다. 4 yaw 썸네일에는 pitch가 적용되지 않았음을 eval 보고서에 명시한다.
- 명시한 v4 색인이 손상되면 error로 끝낸다. 핵심 조건을 누락했던 v3로 조용히 폴백하지 않는다.
  v4를 지정하지 않은 기존 v2/v3 검색과 SEARCH_ROUTING=0 경로는 유지한다.

생성 예:

```sh
python scripts/build_semantic_shape_index.py --db <bundle>/poses.db \
  --tags <viewer>/dist/data/search-v2.json --bvh-root <bundle>/bvh \
  --detail-tags <viewer>/dist/data/tagging-v1.json --out tmp/semantic-shapes-v1.json
```

색인은 태그 canonical SHA·설정·frame·개별 BVH SHA를 보존한다. 입력 BVH/태그가 바뀌면
재생성해야 한다. 요청 중 파일별 BVH SHA 재검증이나 BVH 렌더링은 수행하지 않는다.

### 같은 러프 전후 실험

1,981개 BVH 처리 성공(실패 0). 팔짱 검증 2개는 standing 원본/미러이고 몸 기준 V 검증은
30개다(정확한 러프 일치 30개라는 뜻 아님). 기존 Gemini 설명을 고정한 8장/11슬롯에서
T013 슬롯 2는 스트레칭 오답 5개 → 검증 후보 0개, T028 슬롯 1은 다이빙 5개 → V 후보 3개다.
나머지 semantic 6슬롯의 Top5는 동일하며 geometric T044와 부분 인물의 미판정도 유지된다.

새 Gemini 3.8 Flash/current-X 동일 승인 8장 실험은 분석 8회+설명 7회(5성공/2upstream_error).
T013·T028 별도 진단 1회에서는 T013이 다시 upstream_error, T028이 성공했다.
새 T028은 행동 stretch를 출력했지만 V 조건을 보존하여 동일한 새 1위를 반환한다.
context_evidence 시점 전달 및 작은 pitch 우선 수정은 저장된 실호출 설명의 최종 재검색으로
확인했다. 원본 live 기록을 덮어쓰지 않았다. T013의 새 end-to-end 성공은 아직 미검증이다.
정면 부감 비교에서는 V 실루엣이 깨지거나 반대 yaw에서 몸/다리 방향이 달라지는 한계가 확인됐다.
팔 형태는 개선됐지만 전체 구도 일치는 해결되지 않았으며 accuracy 수치로 표현하지 않는다.

증거: `tmp/routing-shapes-v4-frozen-eight/ablation.json`,
`tmp/routing-eval-live-v4-eight-20261013/results.json`,
`tmp/routing-eval-live-v4-critical-diagnostic-20261013/results.json`,
`tmp/routing-shapes-v4-fresh-diagnostic-ablation/ablation.json`,
`tmp/routing-shapes-v4-report/{COMPARISON_REPORT.md,camera-evidence.json}`.
카메라 PNG는 기존 BVH의 COCO17 관절 마네킹이며 실제 FBX 메시 재렌더가 아니다.
Gemini 프롬프트는 d2 그대로, 자동 설명 재시도 없음. PR·push·배포 없음.
검증: 서버 테스트 872 passed/8 skipped/22 subtests, 스모크 54/54.
Blender 전용 tests/converter는 bpy/mathutils 런타임이 없어 제외했다.
전후 화면의 두 러프×두 설명 출처×전체 후보 버튼·이미지와 320/360/736px 표시를 검증했다.

### 04J-285 검색기 통합 경계(아직 통합하지 않음)

라우터는 기존 SemanticSearcher 계약에만 의존한다. 최종 통합에서는 285의
`SemanticSearchService.search`/`semantic_search(index,text,source='vlm',tags,top_k,settings)`를
어댑터로 호출하고 ok/no_result/error를 SemanticSearchResult로 변환한다.
관찰·critical_relations·카메라 진단은 후보 검증 경계에서 보존해야 한다. Top5 반환 뒤에만
필터하면 일치 포즈를 놓칠 수 있으므로 전체 후보 풀 또는 검색기의 필수 조건 인터페이스가 필요하다.
기하 후보와 의미 score는 계속 분리하고 refine_allowed=false를 유지한다.
현재 v4는 임시 검색기 실험이며 285의 worktree/태그/BVH를 수정하거나 별도의 운영 검색기로 배포하지 않는다.
후속: 관점별 몸 배치, 좌우 팔·다리 구분, 작가 판정이 있는 다른 동작 홀드아웃이 필요하다.
