# 다중 시점 설명 + 동작 활용 의미 색인

2026-09-27 · 상태: 코드 연결·로컬 검증 완료, 전체 외부 VLM 분석 승인 대기

## 구현한 흐름

오프라인: 현재 DB/BVH/격리 목록 확인 → 실제 모델 정면·측면·3/4 렌더 계보·해시 검증 → Gemini Flash Lite에 포즈별 세 이미지를 함께 전송 → 부위별 관찰 설명과 활용 가설 생성 → BVH 구조 대조 → 별도 E5 색인.

온라인: 러프 이미지 → 기존 1회 VLM 분석의 의미 슬롯 → 기존 라우터 → 확장 의미 색인 → 기존 지지 상태·사지 구조 검토 → 실제 BVH 후보와 썸네일 및 의미 설명 반환.

검색 시 라이브러리 이미지를 다시 VLM에 보내지 않는다. E5는 기존 텍스트 임베딩 모델이며 이미지/텍스트 공동학습 모델로 바뀐 것이 아니다. 기하 검색과의 최종 통합, 카메라 맞춤, 상하체 조합, refine은 이번 범위에 포함하지 않는다. 그룹 관계·하체 단독 입력 경로는 추가하지 않는다.

## 색인과 자동 검수

- 포즈 원본은 1,215개를 유지한다. 시점은 포즈에 딸린 설명 자료이며 포즈를 3배 복제하지 않는다.
- 세 시점은 같은 포즈의 관찰 근거다. VLM은 상체/하체/전신의 짧은 관찰 문장(최대 6개)과 활용 가설(최대 2개)을 생성한다.
- 관찰 문장은 최소 1시점, 활용 가설은 최소 2시점 근거가 필요하다.
- 알려진 BVH 측정 조건과 모순인 문장, 알 수 없는 조건 키, 부위 경계를 침범한 문장, 소품·접촉·시간 동작의 미검증 주장을 제외한다.
- 활용 가설은 측정 가능한 조건이 적어도 하나 있어야 하고, 모든 관련 feature cost가 0.35 이하여야 한다. 지지 상태가 있으면 해당 구조와 compatible이어야 한다.
- 텍스트에서 파싱 가능한 조건도 검사한다. VLM이 불리한 조건 키를 누락해 검수를 우회할 수 없게 한다. 다만 자유형 문장 전체를 검증하는 완전한 의미 논리는 아니다.
- 검수 결과를 `auto_geometry_consistent`, `visual_only`, `rejected`로 구분한다. `human_reviewed=false`, `intent_verified=false`, `ground_contact_verified=false`를 유지한다. 구조 일치는 동작 의도나 물리 접촉의 정답 판정이 아니다.
- 활용 의미는 원본 VLM 설명에서 얻으며, 원본 파일명·클립명을 동작 정답으로 사용하지 않는다.

## 점수 결합

기존 구조 설명 0.65 + VLM 관찰 0.25 + 활용 가설 0.10으로 의미 점수를 결합한다. 각 채널은 질의 절마다 가장 가까운 문서 점수를 선택한 다음 절별 평균을 낸다. 같은 문장이 반복돼도 투표 수처럼 가산하지 않는다. 해당 채널에 통과한 설명이 없으면 그 가중치는 구조 설명 점수로 대체한다.

이 비율은 초기 실험값이며 최적값이 검증된 것은 아니다. 기존 지지 구조의 compatible/unknown/contradiction 우선순위와 사지 불일치 비용은 그대로 적용한다. 상체 질의에 하체나 전신 설명을 넣지 않는다. 전신 질의에는 상체·하체 설명도 사용할 수 있다.

## API 연결

기존 `rough_semantic.people[].queries[].results[]`에 `semantic_enrichment`를 추가한다:

- observations: 채택된 관찰 문장, 부위, 근거 시점, 자동 검수 상태, 조건 비용
- usage_hypotheses: 채택된 활용 가설과 동일한 검수 근거
- source_scores: 구조/관찰/활용 채널별 의미 점수
- model_version, views, human_reviewed, intent_verified

기존 기하 후보와 `/semantic-search` 사용자 텍스트 API 계약은 변경하지 않는다. 기본값은 기존처럼 off. 확장은 명시적으로 준비된 색인과 caption 파일이 있을 때 사용한다. 원본 BVH·렌더 이미지·VLM 응답·검수 코드·caption 파일 해시가 맞지 않으면 준비를 거부하고, API는 기존 기하 결과를 보존하며 의미 경로 unavailable을 반환한다.

## 실제 실행 상태와 차단 사유

실제 Gemini Flash Lite 시범 호출 3개(포즈 3개 × 시점 3개 = 이미지 9장)는 성공했다. `pilot-captions/`에 원본 응답과 검수 결과가 있다. 자동 검수 결과 관찰 설명 1개, 활용 가설 1개가 통과했다. 7개 관찰 설명과 4개 활용 가설은 제외됐다. 모델이 약한 무릎 굽힘을 깊은 굽힘으로, 몸통 회전을 기울기로 분류한 사례가 있었다. 이후 프롬프트에 조건 정의와 문장의 원자성을 명시했다. 수정 프롬프트의 전체 호출은 아직 수행하지 않았다.

자동 승인 검토가 전체 1,215개 포즈의 3,645장 이미지를 외부 Gemini 서비스로 보내는 작업을 거절했다. VLM 연동은 승인됐지만 전체 라이브러리 전송 범위와 동시 외부 전송에는 명시적 승인이 필요하다는 사유다. 사용자에게 전송 범위와 API 비용을 알리고 승인을 요청한 상태다. 거절을 우회하는 추가 호출은 하지 않았다.

따라서 **전체 다중 시점 색인은 아직 생성되지 않았고, 새 색인을 사용한 전체 검색·실시간 러프 VLM 통합 평가·정확도 향상은 검증 전**이다. 시범 3개를 전체 라이브러리 결과로 제시하지 않는다.

## 완료한 로컬 검증

- 새 다중 시점 테스트 11/11: 하체 모순, 부위 누출, 소품/지지 미검증, 단일 시점 활용 가설 거부, 중복 문서 점수 불변, provenance 변경 거부, 실제 검색의 지지 조건 우선순위 등.
- 기존 구조 검색 테스트 10/10, 기존 의미 실행 테스트 5/5, smoke 51/51. 합계 77개.
- 기존 경로 재현: 같은 17컷·35슬롯·162요청에서 이전 release-run과 후보 전체 및 점수 정확히 동일. 유효 32명·포즈 요청 142개.
- 기존 `/analyze` HTTP 4케이스: 실제 E5/current DB, frozen Flash Lite, NoPose 어댑터. 썸네일 반환과 off/on 기하 후보 동일성 통과. 새로운 다중 시점 런타임의 실제 HTTP 검증은 전체 색인 생성 이후 진행한다.
- 컴파일 확인 통과.

기존 검색 결과 화면: `artifacts/rough_multiview_20260927/baseline-run/review/index.html`. 이는 새 다중 시점 결과가 아니다.

## 승인 후 실행 순서

```bash
# 실제 렌더 외부 전송: 명시적 승인 후에만 실행
.venv/bin/python scripts/build_pose_multiview_captions.py \
  --out artifacts/rough_multiview_20260927/captions --workers 6

# 실패 응답만 재시도할 때 동일 out에 --retry-errors 사용.
# 누락/실패가 남으면 full-coverage runtime은 해당 색인의 준비를 거부한다.

.venv/bin/python scripts/eval_rough_semantic.py \
  --responses artifacts/rough_vlm_review_20260926/flash-lite \
  --implementation multiview_v3 \
  --captions artifacts/rough_multiview_20260927/captions/captions.json \
  --out artifacts/rough_multiview_20260927/multiview-run \
  --build artifacts/rough_multiview_20260927/multiview-index

.venv/bin/python scripts/eval_rough_multiview.py \
  --root artifacts/rough_multiview_20260927

.venv/bin/python scripts/test_rough_semantic_http.py \
  --captions artifacts/rough_multiview_20260927/captions/captions.json \
  --build artifacts/rough_multiview_20260927/multiview-index \
  --out artifacts/rough_multiview_20260927/multiview-http.json

# 신규 러프 VLM 호출까지 검사하려면 위 HTTP 명령에 --live-vlm 추가.
```

전체 색인 생성·검증 후 적용 설정:

```dotenv
EXPERIMENTAL_ROUGH_ROUTER_MODE=shadow
EXPERIMENTAL_ROUGH_SLOTS_ENABLED=1
ROUGH_SEMANTIC_ENABLED=1
ROUGH_SEMANTIC_BUILD_DIR=artifacts/rough_multiview_20260927/multiview-index
ROUGH_SEMANTIC_FACTS_PATH=artifacts/rough_router_phase2_20260922/semantic-refresh/current-member-facts.jsonl
ROUGH_SEMANTIC_CAPTIONS_PATH=artifacts/rough_multiview_20260927/captions/captions.json
```

설정 변경 후 프로세스 재시작이 필요하다. `.env`나 운영 서버는 변경하지 않았다.

비교 화면은 기존/다중 시점 Top-1, 펼쳐 보는 Top-5, 실제 모델 정면/3·4, 러프, VLM 관찰/활용 가설, 검수 근거 및 평가 JSON 내보내기를 제공하도록 구현했다. 전체 평가 실행 전이라 비교 화면은 아직 생성되지 않았다. 자동 구조 지표의 개선·회귀는 별도로 기록하며, 작가의 바로 사용/수정 사용 채택률과 혼동하지 않는다. 단일 시점 대 세 시점의 효과를 독립적으로 분리한 실험은 아직 없으며, 현재 비교 설계는 구조 설명 색인 대 세 시점 의미 보강 전체의 비교다.

## 롤백

- 의미 검색 전체 해제: `ROUGH_SEMANTIC_ENABLED=0` 후 재시작.
- 다중 시점 보강만 해제: `ROUGH_SEMANTIC_CAPTIONS_PATH`를 비우고 `ROUGH_SEMANTIC_BUILD_DIR=artifacts/rough_multiview_20260927/baseline-index`로 변경 후 재시작.
- 확장 hook 추가로 코드 해시가 바뀌므로 옛 release-index를 새 코드로 강제 로드하지 않는다. 이번 baseline-index는 같은 결과임을 검증한 재빌드다.
- 파일 복구 dry-run:
  `.venv/bin/python scripts/rollback_rough_router.py artifacts/rough_multiview_20260927/checkpoint.json`
- `--apply` 시 이 작업 이후 수정이 없는 파일만 복구한다. 이전 체크포인트보다 이번 것을 먼저 복구한다. 생성한 평가/색인/원본 응답은 감사 기록으로 유지한다.
