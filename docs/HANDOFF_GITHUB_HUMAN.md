# Standin 기술팀 인수인계 — GitHub `main` 기준 사람용

> 문서 소유: Standin 기술팀 · 확인일: 2026-09-28
> 기준 저장소: `04JUNG/Standin-server` · 기준 브랜치: `main` · 기준 커밋: [`b80dc29`](https://github.com/04JUNG/Standin-server/commit/b80dc29c6b6365d45877ef6418975a74ed0f8d09)
> 이 문서는 **GitHub에 추적된 파일만** 설명한다. 기술팀 로컬 작업 트리의 후속 실험은 [로컬 기준 인수인계](HANDOFF_HUMAN.md)에서 따로 본다. AI 작업용 상세본은 [GitHub 기준 AI 인수인계](HANDOFF_GITHUB_AI.md)다.

## 현재 `main`이 제공하는 것

웹툰 러프 컷 **1장**을 받아 인물별 3D 포즈 Top-K를 반환한다. 작가가 고른 후보는 `/refine`으로 조정할 수 있고, 결과 BVH와 썸네일은 응답에 인라인된다. base BVH와 후보 썸네일도 API로 제공한다.

선택된 BVH를 FBX로 바꾸는 Converter도 같은 저장소에 있지만 추론 서버와 **별도 프로세스·이미지·배포**다. BFF는 인증·Job·기록을 맡고, 최종 base/refined BVH를 골라 Converter에 전달한다.

현재 `main`에는 `/semantic-search`, 손 포즈 라이브러리, 제품용 6~8컷 일괄 요청, 페이지 컷 분할, 새 image-only 라우터, 상·하체 조합 검색이 없다. `scripts/run_batch_pipeline.py`는 여러 파일을 순서대로 평가하는 오프라인 도구이며 제품용 다중 컷 API가 아니다.

## 기능별 위치

| 기능 | 먼저 볼 위치 | GitHub `main`의 역할 |
|---|---|---|
| 추론 HTTP 계약 | `api/app.py`, `api/models.py`, [API 계약](API_CONTRACT.md) | `/analyze`, `/refine`, pose BVH/썸네일, `/export-order`, `/healthz`, `/ops/metrics` |
| 한 컷 파이프라인 | `src/pipeline.py`, `src/schema.py`, `src/config.py` | VLM → 라우팅 → 인물 슬롯 → 관절 → 검색을 연결 |
| 관절 추출·복구 | `src/pose.py`, `src/skeleton_extraction.py`, `src/pose_rescue.py`, `src/pose_cascade.py` | current-X 추출, crop 복구, Human-Art fallback 계약 |
| 포즈 검색 | `src/features.py`, `src/library.py`, `src/search.py` | BVH 다중 시점 투영과 같은 피처 공간의 기하 kNN, family 중복 제거 |
| 라이브러리 | `src/repo.py`, `src/library_source.py`, `src/pose_quarantine.py`, `scripts/build_db.py` | SQLite 색인, 외부 번들 공급, 불량 포즈 차단 |
| Refine | `src/refine.py`, `src/refine_v2.py`, `src/refine_selector.py`, [v2 기준](REFINE_V2_DESIGN.md) | 선택 후보 v2.5.3 safe-aggressive 조정과 base 복구 |
| 평가 | `standin_eval/`, `evaluation/`, [평가 하네스](EVAL_HARNESS.md) | HTTP/replay 비교, blind 라벨, refine·실메시 증거 계약 |
| Converter | `converter/`, `converter_api/`, [통합 인계](FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md) | 별도 Blender 5.2 서비스로 BVH→FBX, bundle, 썸네일 처리 |
| 배포·운영 | `.github/workflows/`, `Dockerfile*`, `.env.example`, [QA·릴리스](QA_SECURITY_RELEASE.md) | 추론·Converter CI/배포, production mock 차단, health/rollback |

## 현재 상태에서 주의할 점

- 구현된 추론 엔드포인트는 `api/app.py`가 정본이다. 자연어 검색 엔드포인트는 없다.
- Human-Art cascade 코드와 배포 기본값(`cascade`, `canary-100`)은 존재한다. 실제 배포 manifest의 라이선스 승인·모델 해시·현재 task 설정은 저장소만 보고 확정할 수 없다. production guard는 승인되지 않은 manifest를 거부한다.
- Refine v2.5.3은 코드 기본이지만 작가 blind·실메시 holdout을 계속 운영 근거로 관리해야 한다. `refined=false`는 실패 응답이 아니라 검증된 base 복구다.
- Converter의 palm-roll 안전 정책은 손바닥 방향 보정이다. 다양한 손 모양을 고르는 **손 포즈 라이브러리**와는 다른 기능이다.
- BVH·DB·모델·캐릭터 FBX는 Git에 들어 있지 않다. 실제 배포 버전과 개수는 artifact manifest, `/healthz`, task definition에서 확인한다.
- [ROADMAP](ROADMAP.md), [MVP 릴리스](MVP_RELEASE.md), [2026-07-14 검색 평가](SEARCH_EVAL_2026-07-14.md)의 숫자와 TODO에는 이후 구현으로 끝난 항목이 섞여 있다. 과거 근거로 읽고 현재 상태는 코드·테스트에서 확인한다.

## 앞으로의 우선 과제

| 순위 | 과제 | GitHub `main`에서 시작할 위치 | 완료 기준 |
|---|---|---|---|
| **1 · P0** | **검색 정확도 개선** | `standin_eval/`, `evaluation/datasets/selected12-v1/`, `src/search.py`, `src/skeleton_extraction.py`, `tests/fixtures/search_regressions.v1.jsonl` | 평가셋 라벨·provenance를 봉인하고 실제 모델 Top-K 활용 가능률과 수정 시간을 전후 비교. 기존 성공→실패와 p95를 함께 확인 |
| **2 · P0** | **라이브러리 포즈 증가** | `scripts/build_db.py`, `scripts/bvh_contact_sheet.py`, `scripts/render_bvh_thumbnails.py`, `scripts/deploy_pose_library.py`, `src/pose_quarantine.py` | 실패 유형별 gap을 메우는 포즈를 출처·권리·중복·리그·썸네일까지 검수하고, 고정 실패 컷의 적격 후보 공백 감소를 입증 |
| **3 · P1** | **손 라이브러리** | `converter/retarget.py`, `converter/palm_roll_policy.json`, `qa/retarget/CHAIN_TRANSPORT_V3_2_1_PALM_ROLL_QA/` | 기본 손과 기능성 손 프리셋의 적용 경계를 먼저 확정. 좌우·엄지·손목·관통 검사와 기본 손 fallback을 실제 캐릭터에서 통과 |
| **4 · P1** | **6~8컷 한번에 넣기** | `api/app.py`, `api/models.py`, `src/pipeline.py`, `scripts/run_batch_pipeline.py`, [API 계약](API_CONTRACT.md) | BFF와 여러 이미지 제출/페이지 자동 분할 중 입력 계약을 확정. 컷 순서·`cut_id`·부분 실패·재시도·진행 상태와 작업 p50/p95를 보존 |
| **5 · P1** | **자연어 검색** | `src/repo.py`, `src/schema.py`, `api/models.py`, `api/app.py` | 별도 검색 계약·색인·근거 상태를 설계하고 실제 사용자 문장 holdout에서 Top-K 품질과 `library_gap`을 검증. 기하 거리와 의미 점수를 같은 척도로 섞지 않음 |

검색 정확도와 라이브러리 확장은 함께 진행한다. 평가에서 후보가 있는데 순위가 틀리면 검색을 고치고, 필요한 자세 자체가 없으면 라이브러리를 늘린다. 두 원인을 분리하지 않고 포즈 수나 평균 거리만 올리면 완료로 보지 않는다.

## 유지보수 원칙

1. API 변경은 `api/models.py`, [API 계약](API_CONTRACT.md), BFF 매핑을 같은 PR에서 갱신한다.
2. 피처 변경은 쿼리와 라이브러리에 함께 적용하고 feature version·DB 재빌드·고정 회귀 평가를 남긴다.
3. 라이브러리는 DB·BVH·썸네일·quarantine을 한 버전으로 배포한다. 라이선스 제한 자산은 Git에 넣지 않는다.
4. 추론과 Converter는 별도 런타임이다. Converter에 Blender를 유지하고 추론 이미지에는 `bpy`를 넣지 않는다.
5. `main`은 직접 push하지 않는다. 기능 브랜치 → PR → `PR checks` → write 권한 리뷰 승인 순서를 따른다.

## 최소 검증

```bash
python -m compileall src api scripts
python -m pytest tests/ --ignore=tests/converter -q
python -m compileall converter converter_api tests/converter
python -m pytest -q \
  tests/converter/test_worker_contract.py \
  tests/converter/test_registry.py \
  tests/converter/test_runner_failures.py \
  tests/converter/test_api_contract.py \
  tests/converter/test_refined_bvh_e2e.py \
  tests/converter/test_deployment_contract.py \
  tests/converter/test_latency_probe.py
```

실 BVH·모델·Blender·ECS가 필요한 검증은 [QA·릴리스 문서](QA_SECURITY_RELEASE.md)와 각 배포 workflow를 따른다.
