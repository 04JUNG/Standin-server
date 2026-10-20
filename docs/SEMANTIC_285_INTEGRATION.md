# 시맨틱 검색 세션 → 동적 라우팅 인수인계 — 2026-10-14

## 후속 직접 입력 API (2026-10-17)

현재 `/semantic-search`는 기동 완료한 전신 어댑터를 재사용하여 문장만 검색한다.
아래 snapshot 수입 기록에서 "HTTP endpoint 제외"는 당시 범위를 뜻한다.
이번 후속 연결은 #87 전체 병합이 아니며, 개발용 검수 화면·피드백 저장은 추가하지 않았다.
최신 요청/응답·오류·비생성 계약은 `docs/API_CONTRACT.md`의 직접 문장 검색 항목이 기준이다.

## 가져온 작업

원본 채팅: **라이브러리 태깅(sp2) & 시멘틱 검색(sp2)**
(`01a12de7-4fcf-7ca1-8e66-23c50864ef87`). 원본 worktree는
`../standin-semantic-text-search`, HEAD `b2d0a91`이며 아직 커밋되지 않은 최신 규칙·선호·격리 수정까지 snapshot으로 가져왔다.
기존 286의 `src/semantic_search.py`와 충돌하므로 덮어쓰거나 전체 브랜치를 merge하지 않았다.

| 내용 | 이 작업 폴더의 위치 |
|---|---|
| 285 검색 core·JSON/ID 검증·service | src/semantic_search_285.py |
| 공통 relevance 규칙·선호 loader | src/semantic_relevance.py |
| 선택적 로컬 E5 encoder | src/semantic_embedding.py |
| 어휘 / 선호 / 시맨틱 격리 snapshot | config/semantic_lexicon.v1.json / semantic_review_preferences_285.v1.json / semantic_quarantine_285.v1.json |
| 286 연결 | src/semantic_adapter_285.py |
| 원본 세션 문서 (참고용, 본 브랜치 API와 구분) | docs/SEMANTIC_285_SOURCE_REFERENCE.md |
| 파일별 원본/수입 SHA·수정 내역 | docs/semantic285-import-manifest.json |
| 30개 입력·원래 판독·최신 재검색·검수 로그·5개 probe | tmp/semantic285-handoff/ |

원본 worktree·기존 검수 화면·사용자 기록을 수정하지 않았다. 새 세션을 만들거나 원본 채팅으로 메시지를 보내지 않았다.
DB/BVH/태그/무거운 모델/이미지를 Git에 넣지 않았다. 모델과 자산은 기존 명시한 로컬 경로를 읽기만 한다.
로컬 실험 원문과 검수 기록은 ignored tmp에 별도 보존한다.

## 원본 성과와 한계

- 직접 입력 15개 + 실제 러프 15개 = 30개. 러프는 세션에서 승인받아 실제 Gemini 판독했고,
  가져온 저장 판독은 새 입력의 정답 메모로 대체하지 않았다.
- 저장 검수 이벤트 35회 / 평가가 저장된 입력 28개. 최신 판정 맞음 8 / 다름 3 / 보류 17,
  Q015·Q016 저장 판정 없음. 보류를 오답이나 정답으로 임의 분류하지 않는다.
- 일부 순위 변경은 동일 정규화 문장에 대한 사용자 선호이며 학습·일반화 효과가 아니다.
  동적 라우팅 연결에서는 선호 및 문장별 거절을 기본 비활성으로 유지한다.
- 공통 개선: 엎드림 vs 바로 눕기, 양손 vs 한손 폰, 팔짱+쪼그리기 복합 조건,
  이마/입 근처 손, 턱걸이, 책상 앞 설명, 어깨 스트레칭 등. 모두 태그/설명 기반이며 실제 3D 접촉 검증이 아니다.
- 순위 지정 없는 5개 probe에서 잘못된 후보를 제거하는 효과는 있었으나 적합 후보 확보와 독립 정확도 개선은 미확정이다.
- 기존 draft PR #87은 원본 작업의 배경이다. 이번 snapshot 수입·연결과 미커밋 수정이 PR에 반영됐다는 뜻이 아니다.

## 연결 방식

기본 검색은 유지하고 `SEMANTIC_BACKEND=semantic285`를 명시하면 lazy loader로 사용한다.
`SEARCH_ROUTING=0`이면 기존 경로를 유지한다. Pipeline에 직접 adapter를 주입해 실험할 수도 있다.
설정 파일·schema·DB·Controlled Vocabulary를 통째로 가져오지 않았다. 신규 `/semantic-search` API나 배포는 하지 않았다.

285가 전체 기준 통과 후보군을 반환하고, 286은 팔짱/V 핵심 BVH 관계·자세·카메라와
충분한 관찰이 있을 때 알려진 몸 속성 충돌을 검증한 뒤 family Top-K를 고른다.
Top5/Top20 이후 필터링하지 않도록 내부 pool 크기와 mirror collapse 순서를 분리했다.
known body conflict는 보류, unknown은 중립이다. 두 검색기의 수치 점수를 더하지 않는다.
카메라 24개 비교는 정성 가시성이지 러프 pixel fit이나 3D 정답 증명이 아니다.

상태 ok/no_result/error를 그대로 구분하고, ID/JSON 불일치·DB/태그 mismatch·encoder 오류는 error다.
명시한 캐시가 낡았으면 error이며 구버전으로 조용히 돌아가지 않는다. 기존 기하 후보·refine 정책은 보존한다.
전신 정상 → 기존 검색 / 관절 실패 → 시맨틱 검색 / 후보 없으면 → 부위 보완·생성 연구 순서의 기반이다.
부위 검색·합성·생성은 아직 자동 제품 적용으로 승격하지 않았다.

## 현재 검증

- 가져온 검색·relevance core: 50 passed / 1 skipped.
  skip은 원본 세션의 별도 HTTP endpoint 테스트다. endpoint를 이 브랜치에 수입하지 않아 명시적으로 제외했고,
  286 어댑터 계약을 별도로 검사한다. 실제 라이브러리 개수를 1981로 고정하지 않도록 테스트를 현재 태그 개수와 비교하게 변경했다.
- 어댑터: 12 passed. 전체 pool 선필터·미러 후중복제거·핵심 관계·자세·카메라·body conflict/unknown·상태 오류·기본 선호 off 및 설정 factory 연결 검증.
- 기존 라우팅·관계 검사 포함 집중 테스트: 115 passed / 1 skipped.
- 전체 서버 회귀: 1012 passed / 9 skipped / 22 subtests passed (Blender 전용 tests/converter 제외).
  기존 deprecation 경고 2개는 남아 있다.
- 30개 입력과 저장된 실제 판독을 문장별 선호/거절 없이 재실행: 검색 31건, ok 21 / no_result 10 / 시스템 오류 0.
  다인 슬롯 때문에 입력 수와 검색 요청 수는 다르다. 결과 수는 정확도나 성공률이 아니다.
  모든 후보의 ID=포즈 JSON ID, refine 금지 확인. 추가 이미지 전송 및 VLM 호출 0회.
- 재실행에는 기존 로컬 E5(weight .5, lo .72, hi .88)를 사용했다. 이는 운영 기본값으로 승격하지 않았다.
  실행 환경에 onnxruntime이 없으면 encoder error이며 임베딩 없는 검색으로 몰래 전환하지 않는다.
  이번 실행은 기존 RTM 평가 runtime 경로를 이용했고, 모델을 다운로드하지 않았다.

근거: `tmp/semantic285-routing-replay-30-20261014-rtm-runtime/results.json`.
태그/라이브러리는 현재 1995개로 업데이트돼 과거 1981개 캐시와 맞지 않았다.
원본 자산을 수정하지 않고 새 몸/관계 캐시를 별도로 만들었다.
현재 “앉은 팔짱”은 새 기존 자산 `ext_seated_folded_arms`가 조회된다.
이는 이번 수입 작업이 라이브러리에 추가한 자산이 아니며, 검색되는 경우에는 생성 hook이 실행되지 않는다.

## 여기서 이어갈 일

1. 30개 검수 메모를 회귀 기준으로 삼되 문장별 순위 지정 없이 평가한다.
2. 전신의 조건이 충족되면 기존 자산 유지. 부족한 상체/하체만 분리하고 핵심 행동·도구·접촉 조건 보존.
3. 검색/태그 문제, 부위 조합 문제, 실제 자산 부재를 분리한다. no_result를 자산 부재 증명으로 쓰지 않는다.
4. 실제 제품 BFF/앱 표시·오류 수용 검증과 리뷰/배포 승인은 별도다. 이번에 PR/push/deploy하지 않았다.
5. 생성 초안은 별도로 보존했다. 제한된 문장 조건에서 새 BVH를 만드는 연구 hook이며 실제 mesh·손목·물리 검증 미완료다.

## 로컬 재현

```sh
python -m pytest tests/test_semantic_adapter_285.py tests/test_semantic_search_285.py tests/test_semantic_relevance_285.py -q
python scripts/replay_semantic285_handoff.py --library <viewer-root> --inputs tmp/semantic285-handoff/inputs.json --baseline tmp/semantic285-handoff/baseline-results.json --shape-index <current-shape-cache> --body-index <current-body-cache> --dense-model-dir <existing-local-model> --out <new-result-dir>
```

실제 라우팅 실험에 켜려면 SEARCH_ROUTING=1, SEMANTIC_BACKEND=semantic285,
SEMANTIC_TAGS_PATH, SEMANTIC_TAXONOMY_PATH, SEMANTIC_285_DB_PATH,
현재 코퍼스와 맞는 SEMANTIC_BODY_INDEX_PATH/SEMANTIC_SHAPE_INDEX_PATH를 지정한다.
E5 옵션은 `.env.example`에만 예시로 기록했고 실제 `.env`나 운영 설정을 바꾸지 않았다.
