# Semantic search API 구현 보고서

> 구현일: 2026-08-18  
> 상태: 내부 추론 API와 development 검증 완료 · holdout/release promotion 전  
> 인터넷 직접 공개 금지

## 결과

내부 semantic runtime을 제품 HTTP 경계에 연결했다.

- `POST /semantic-search`: 사용자 문장, `top_k`, 미리보기 `view_hint` 입력
- `GET /healthz.semantic`: build·encoder·unit 수·cache·concurrency readiness
- 동일 build/request LRU cache: 기본 256개
- ONNX query 실행 bounded concurrency: 기본 2, 대기 250ms 뒤 `503 semantic_busy`
- optional 실패 격리: `SEMANTIC_REQUIRED=0`이면 `/analyze` health 유지
- required fail-closed: `SEMANTIC_REQUIRED=1`이면 semantic 준비 실패 시 startup/readiness 실패
- production fail-closed: 명시적 `SEMANTIC_BUILD_DIR`와 promoted manifest가 아니면 semantic 비활성

## 응답 안전선

- 서버가 `match_source=semantic_user`를 부여한다.
- 응답과 모든 후보가 `refine_allowed=false`다.
- RRF 점수를 확률로 표시하지 않고 `retrieval_score`로 노출한다.
- source 이름 기반 후보는 `contextual`, `exact_pose_claim=false`다.
- `view_hint`는 썸네일 선택에만 사용하고 semantic exact 판정을 바꾸지 않는다.
- 모델·DB 불일치는 silent fallback 없이 `semantic_not_ready`다.

## 설정

```bash
pip install -r requirements-semantic.txt
python scripts/provision_semantic_encoder.py

SEMANTIC_ENABLED=1 \
SEMANTIC_REQUIRED=0 \
SEMANTIC_BUILD_DIR=data/semantic/builds/217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196 \
uvicorn api.app:app --reload
```

`SEMANTIC_ENABLED=0`이 기본이므로 core/mock 서버는 pinned model 없이 계속 실행된다.

## 검증

- 실제 E5/DB startup: 616 unit, 1,232 member, build `sha256:217d…0196` 확인
- 실제 endpoint: 조합형 문장 Top-3 `success/exact`, 미러 member 반환, refine 금지 확인
- API/service 회귀 9/9: readiness, cache, concurrency, OpenAPI, optional/required 실패 격리,
  production의 unpromoted build 차단
- semantic 전체 59/59, 기존 geometry smoke 48/48, refine v2 25/25 통과
- FastAPI lifespan 실제 HTTP: health 200/semantic ready, 검색 200, 두 번째 동일 요청 cache hit 확인
- golden development 30개 결과는 기존 기준을 유지
- holdout 15개는 실행하지 않음

## 남은 승격 조건

1. 앱 서버/클라이언트 검색창과 Top-5 선택·저장 연결
2. semantic/geometry/BVH/model atomic release bundle과 current pointer
3. 실제 다인 `pose_set` 데이터가 들어오면 set assembler 구현
4. 설정·bundle 동결 후 holdout 최종 1회 실행
5. 승인된 manifest를 `production_ready=true`로 승격
