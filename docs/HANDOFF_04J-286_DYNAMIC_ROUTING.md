# 인수인계: 04J-286 러프함 기반 동적 라우팅

작성 2026-10-13 · Linear 04J-286 (기한 10/14 중간 점검) · 다음 작업자: Codex

> 아래 본문은 최초 인수인계 시점의 기록이다. Codex는 같은 날 최신 develop `c838051`
> (PR #84·#86 포함)에 패치를 반영하고 API·오류 분리 테스트와 실제 current-X 50장 재생 검증을
> 완료했다. geometric 39명 모두 Top-5 동일, semantic 8명 모두 후보 반환.
> 자세 설명은 Claude fixture이므로 Gemini 품질·지연은 여전히 미측정이다.
> 실 Gemini 검증은 자동 승인 검토가 러프 이미지 외부 전송을 차단해 사용자 승인 대기 중이다.
> 로컬 전체 테스트는 801 passed·8 skipped이며, GitHub 업로드도 외부 코드 게시 승인이
> 필요하다는 자동 승인 검토에 의해 차단됐다. PR·CI는 아직 미실행이다.
> 최신 구현 설명은 `docs/SEARCH_ROUTING.md`를 따른다.

이 문서 하나만 읽고 이어서 작업할 수 있게 썼다. 설계 상세는 `docs/SEARCH_ROUTING.md`, 응답 계약은
`docs/API_CONTRACT.md` 마지막 절, 저장소 불변식은 `CLAUDE.md`에 있다. **작업 전에 CLAUDE.md의 핵심 설계
불변식 7개를 먼저 읽을 것.**

---

## 1. 한 줄 요약

러프 속 **인물마다** 검색 경로를 고른다. 관절(스켈레톤)이 충분하면 기존 기하 검색, 부족하면 Gemini에게 자세
설명을 1회 받아 구체적이면 시맨틱 검색, 일반적이면 판단 불가, 설명 호출이 실패하면 오류로 남긴다.
기본값은 꺼짐(`SEARCH_ROUTING=0`)이고, 끄면 기존 결과와 같다.

## 2. 현재 상태

| 항목 | 상태 |
|---|---|
| 브랜치 | `san400b/04j-286-dynamic-routing` (base `origin/develop` = `fdc22d3`) |
| 저장소 반영 | **아직 안 됨.** 패치 파일로만 전달됨 → §3 |
| 단위 테스트 | `tests/test_search_routing.py` 19개 통과 |
| 기존 테스트 | 실패 목록이 develop과 동일(이전 환경에 fastapi·google-genai가 없어서 생긴 실패뿐) |
| 실모델 재현 | current-X + 실제 라이브러리 1,981 포즈로 50장 재현 완료. **실제 Gemini 호출은 미실행** |
| PR | 미생성. 대상은 `develop` |
| 운영 배포 | 미배포. 켜지 말 것(§7-4 선행 필요) |

## 3. 코드 받기

패치 위치(로컬): `Standin-server/out/_claude_04j286/04J-286-dynamic-routing.patch`
(`out/`은 gitignore 대상)

```bash
cd Standin-server
git fetch origin
git checkout -b san400b/04j-286-dynamic-routing origin/develop
git am out/_claude_04j286/04J-286-dynamic-routing.patch
python -m pytest -q tests/test_search_routing.py      # 19 passed
python tests/test_smoke.py                             # 기존 스모크
```

`origin/develop`이 `fdc22d3`보다 앞서 있으면 `git am`이 충돌할 수 있다. 충돌 지점은 대개
`src/pipeline.py::_process_slots`의 슬롯 루프, `api/app.py`의 `PersonOut(...)` 생성부,
`src/vlm/client.py`(04J-294 PR #86과 같은 파일)다.

## 4. 동작 방식

```
Gemini 분석(기존) → RTMPose·슬롯 배정·crop·Human-Art rescue(기존)
  → Pipeline._route_slots: 인물별 관절 근거 판정
       └ 부족한 인물이 있으면 그 인물들만 묶어 describe_people 1회
  → 인물별: geometric → 기존 _evaluate_slot (변경 없음)
            그 외      → _route_non_geometric (기하 검색 안 함)
```

| route | 조건 | 결과 |
|---|---|---|
| `geometric` | 관절 충분 | 기존 `candidates`, 기존 confidence·refine 정책 |
| `semantic` | 관절 부족 + 설명 valid | `candidates=[]`, `semantic_candidates`에 Top-5 + 포즈 JSON, refine 금지 |
| `undetermined` | 관절 부족 + 설명 무효 / 기존 정책 차단(얽힘·두상) | 검색 안 함 |
| `error` | 관절 부족 + 설명 호출 실패 | 검색 안 함, `error_reason` 유지 |

검색 결과 상태는 경로와 따로 남긴다: `search.status` = `ok | no_result | error | not_run`.

**"모양으로 찾기" = 기존 스켈레톤 기하 kNN**: RTMPose COCO17 → `features.normalize_skeleton`
→ 라이브러리 3D 포즈를 가상 카메라 4방향으로 투영한 2D와 거리 비교. 이번 작업은 이 부분을 전혀 바꾸지 않았다.

## 5. 파일 지도

| 파일 | 역할 |
|---|---|
| `src/search_routing.py` | 순수 판정. `RoutingConfig`(env `ROUTING_*`), `assess_joint_evidence`, `assess_semantic_evidence`, `decide_route`, `PoseDescription`, `DescribeOutcome` |
| `src/semantic_search.py` | 04J-285 계약(`SemanticQuery`, `SemanticSearchResult`, `SemanticSearcher`) + 임시 `TagSemanticSearcher` + `build_semantic_searcher` |
| `src/semantic_vocab.py` | search-taxonomy-v2 어휘 스냅샷. 원본이 바뀌면 `scripts/sync_semantic_vocab.py`로 재생성 |
| `src/pipeline.py` | `_route_slots`, `_route_non_geometric`, `_routing_record`, `_process_slots` 루프 분기 |
| `src/vlm/client.py` | `BaseVLMClient.describe_people`(기본=error), Mock 구현(hint: `describe_fail`·`vague`·`unreadable`), `GeminiVLMClient.describe_people`, `parse_pose_descriptions`, `describe_failure_reason` |
| `src/vlm/prompts.py` | `POSE_DESCRIBE_*`, 버전 `d1-taxonomy-v2`. 분석 프롬프트(p1-scope)는 한 글자도 안 바꿈 |
| `src/config.py` | `search_routing_enabled`(`SEARCH_ROUTING`), `routing_describe_timeout_ms`, `semantic_top_k` |
| `src/schema.py` | `CutResult.person_routing`, `person_semantic_candidates`, `routing_summary` |
| `api/models.py`, `api/app.py` | `SemanticCandidateOut`, `PersonOut.search_routing`·`semantic_candidates`, `CutResultOut.search_routing`, `_routing_fields` |
| `scripts/eval_routing_thresholds.py` | 운영 기록(jobs.json)에 관절 판정을 재적용, 기준값 sweep |
| `scripts/eval_search_routing.py` | 기존(off)·라우팅(on) Top-5 나란히 저장. `--vlm live` 또는 `--vlm replay` |
| `tests/test_search_routing.py` | 19개. off=기존 동일, geometric 후보 불변, 각 경로, 오류 분리, 파서, 검색기 |

## 6. 결정된 사항 (다시 논의하지 말 것)

사용자가 직접 정했거나 근거를 남긴 결정이다. 바꾸려면 사용자에게 먼저 물을 것.

1. **semantic 후보는 별도 필드** `semantic_candidates`에 싣는다. 기존 `candidates`에 섞지 않는다. 이유: `distance`(관절 거리)와 의미 점수는 비교할 수 없다. (사용자 결정)
2. **자세 설명 호출은 관절 부족 인물이 있는 컷에서만 1회**, 재시도는 없다. 정상 러프는 호출 0회. (사용자 결정)
3. **04J-285 전까지는 임시 태그 검색기**를 쓰고, 계약은 고정한다. (사용자 결정)
4. 관절 판정 문턱에 **raw keypoint score를 쓰지 않는다**. 구조 mask(뼈·몸통·coverage)만 쓴다. CLAUDE.md 불변식 2 때문이다.
5. 얽힘(hug/fight)·두상 인물은 기존 정책을 바꾸지 않는다. `undetermined` + `reason=existing_policy:*`로 기록만 한다.
6. 경로 자동 전환·재시도는 **기본 없음**. 정책은 제안만 해 두었다(`docs/SEARCH_ROUTING.md` §5).
7. 설명 호출에는 스켈레톤 박스가 아니라 **VLM 인물 박스**를 넘긴다. 관절이 깨진 스켈레톤 박스는 실제 인물보다 작다.

## 7. 남은 작업 (우선순위 순)

### 7-1. 저장소 반영 + PR — 먼저
- §3대로 적용 → 테스트 → `develop` 대상으로 PR. 제목 예: `feat(search): 러프함에 따라 인물별 검색 경로를 분기한다 (04J-286)`
- CI(Python 3.12, `requirements.txt`)에서 fastapi가 필요한 테스트까지 통과하는지 확인한다. 특히 `tests/test_smoke.py::test_analyze_route_maps_full_response_contract`. 이전 환경에서는 fastapi가 없어서 못 돌렸다.
- **완료 기준:** CI green, 기존 테스트 회귀 0.

### 7-2. 실제 Gemini로 재현 — 10/14 점검의 핵심 미확인 항목
`.env`에 `GEMINI_API_KEY`가 있는 환경에서 실행한다.

```bash
SEARCH_ROUTING=0 VLM_PROVIDER=gemini POSE_BACKEND=rtmlib POSE_MODEL_VARIANT=cascade \
python scripts/eval_search_routing.py --vlm live \
  --manifest ../../../../standin_beta_2026-09-04/testset_50_balanced_20261011/manifest.csv \
  --images   ../../../../standin_beta_2026-09-04/testset_50_balanced_20261011 \
  --db       ../../../../pose-library-viewer/source_bundle/poses.db \
  --tags     ../../../../pose-library-viewer/dist/data/search-v2.json \
  --thumbs   ../../../../pose-library-viewer/source_bundle/thumbs \
  --out out/routing-eval-live
```

- 경로는 저장소(`Desktop/AI소마 17기/AI소마 17기/3D 모델 배치 자동화서비스/기술팀/Standin-server`)에서 `Desktop/AI소마 17기/`까지 4단계 위로 올라간 상대경로다. 실제 위치에 맞게 고칠 것.
- cascade(Human-Art rescue)는 `POSE_MODEL_MANIFEST`가 필요하다. 없으면 `POSE_MODEL_VARIANT=current-x`로 돌리고 보고서에 적는다.
- 확인할 것:
  - (a) 경로 분포
  - (b) semantic 인물의 `semantic_evidence.status`가 valid로 나오는 비율
  - (c) `search_routing.describe.elapsed_ms`(설명 호출 지연, **아직 미측정**)
  - (d) geometric 인물의 Top-5가 off와 같은지(`side_by_side.csv`의 `same_top5`)
- 결과물: `out/routing-eval-live/{REPORT.md, side_by_side.csv, index.html}` → 04J-291 제품팀 실험 입력.
- **완료 기준:** (d) 전부 True. 설명 지연 p50/p95 기록. Gemini 설명이 너무 일반적이라 undetermined가 많으면 §7-5.

### 7-3. 04J-285 교체
- 285 구현이 들어오면 `SemanticSearcher` 프로토콜을 구현하게 하고 `src/semantic_search.py::build_semantic_searcher`만 바꾼다. 라우터·응답은 건드리지 않는다.
- 계약: `search(query: SemanticQuery, top_k) -> SemanticSearchResult(status ∈ {ok,no_result,error}, hits=[SemanticHit(pose_id, score, matched, pose)], reason, backend, elapsed_ms)`. `pose`는 포즈 JSON이다.
- 라이브러리에 없는 pose_id는 내지 않는다(`library_pose_ids`).
- **완료 기준:** `tests/test_search_routing.py` 통과 + §7-2 재실행 결과 비교.

### 7-4. 운영 켜기 전 선행 작업 (켜지 말 것, 준비만)
- `search-v2.json`을 운영 이미지/번들에 넣는 방식을 정한다. 지금 운영에서 켜면 semantic이 전부 `error(semantic_index_missing)`이 된다.
- 앱/BFF가 `people[i].semantic_candidates`를 읽어야 화면에 나온다. 클라이언트는 다른 팀원(Standin-client) 담당이므로 계약만 전달한다.
- 04J-294(PR #86)가 병합되면 `describe_failure_reason`이 `src/vlm/failures.classify_failure`를 자동으로 쓴다. 병합 후 reason 값이 일치하는지 확인한다(`tests/test_search_routing.py::test_describe_failure_reason_mapping`).

### 7-5. 기준값 조정 (필요 시)
- 관절: `ROUTING_JOINT_MIN_VALID_BONES=6`, `..._TORSO_BONES=4`, `..._UPPER_BONES=3`, `..._BOX_PX=64`, `..._BOX_RATIO=0.05`.
- 근거: 운영 85명 재적용 결과 실패 18명은 전원 부족, 후보를 받은 66명은 전원 충분이었다. 뼈 기준을 7로 올리면 시각적으로 '유사'했던 1명이 빠진다.
- 재확인: `python scripts/eval_routing_thresholds.py --jobs <beta>/jobs.json --visual <beta>/visual_classification.csv --out out/routing-thresholds`
- 의미: `ROUTING_SEMANTIC_*`. 막대 인형 정면 서기(T005)가 undetermined가 됐는데, 일반 태그 규칙이 과하게 엄격할 수 있다. 실 Gemini 결과를 본 뒤에 조정한다.
- 바꾸면 `docs/SEARCH_ROUTING.md` §3 표도 같이 고칠 것.

### 7-6. 다음 단계 제안 (이번 이슈 범위 밖, 결정 필요)
- 관절은 뽑혔지만 자세가 틀린 러프(운영 '포즈 차이' 50명)는 이번 분기로 개선되지 않는다. geometric `no_result`이거나 거리가 임계를 넘을 때 semantic을 **보조 후보**로 추가하는 정책이 후보다.
- 상하체 별도 검색·합성은 범위 밖(이슈 본문에 명시)이다.

## 8. 재현 결과 요약 (기록용)

- 50장(실제 current-X, 실제 라이브러리, VLM은 운영 Job 재생): geometric 39명 전원 Top-5가 off와 동일하다. 컷 처리 시간 중앙값은 1,534ms → 1,530ms(CPU).
- 대표 사례:
  - T044·T040 → geometric
  - T008·T009·T011·T014 → semantic
  - T005 → undetermined(too_generic)
  - T006 → undetermined(얽힘 정책)
  - T012 + 429 주입 → error(`rate_limited`)
- 실패 14장 중 4장(T001~T004)은 운영 VLM이 인물 0명으로 본 컷이라 라우터까지 오지 않는다.
- ⚠ 위 semantic 결과의 자세 설명은 **Gemini 출력이 아니다**. Claude가 러프를 보고 같은 스키마로 쓴 대체값이다(`out/_claude_04j286/describe_fixtures_claude.json`). 품질 판단 근거로 쓰지 말 것.
- 결과 묶음: `out/_claude_04j286/routing-eval.tgz` (`eval_offline/`, `eval_fixture/`, `eval_fault/`, `routing-thresholds/`)

## 9. 주의사항

- **CLAUDE.md 불변식 1**(VLM 태그는 매칭에 안 씀)과의 관계: geometric 경로는 여전히 VLM 의미를 쓰지 않는다. 의미 설명은 관절 근거가 없는 인물의 별도 경로에서만 쓴다. 이 경계를 흐리지 말 것.
- **refine**: semantic·undetermined·error 인물은 `refine_allowed=false`다. 조정할 기하 베이스가 없기 때문이다. 이것을 풀지 말 것(불변식 7).
- `SEARCH_ROUTING=0`일 때 기존 필드 값은 완전히 같다. 새 필드는 기본값(`search_routing: null`, `semantic_candidates: []`)으로 응답에 나타난다.
- 로그 이벤트: `search_routing`(인물별 route·reason·상태·시간), `gemini_pose_describe`(호출별).
- MockVLM hint 규칙(테스트용): 이미지 hint에 `describe_fail` → rate_limited 오류, `vague` → 일반 설명, `unreadable` → visible=false.
- 로컬 `.venv`는 Python 3.9이지만 운영·CI는 3.12다. 새 코드는 `from __future__ import annotations`를 써서 3.9 문법으로도 import된다.
- 원본 데이터(러프·라이브러리·jobs.json)는 저작물·개인정보 문제로 저장소에 커밋하지 않는다(`.gitignore`). 경로 인자로만 넘길 것.
