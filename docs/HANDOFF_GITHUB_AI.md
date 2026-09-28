---
document_id: standin-server-github-main-ai-handoff
version: 1
owner: Standin technical team
audience: coding agents and Standin technical team
verified_at: 2026-09-28
source_repository: 04JUNG/Standin-server
source_branch: main
source_commit: b80dc29c6b6365d45877ef6418975a74ed0f8d09
scope: files tracked by the source commit only
entrypoint: src/pipeline.py::Pipeline.process_cut
inference_contract: api/app.py + api/models.py + docs/API_CONTRACT.md
converter_contract: converter_api/app.py + docs/FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md
companion: docs/HANDOFF_GITHUB_HUMAN.md
local_worktree_companion: docs/HANDOFF_AI.md
---

# Standin 기술팀 인수인계 — GitHub `main` 기준 AI 작업용

## 0. 해석 규칙

### 상태 값

- `TRACKED_RUNTIME`: 기준 커밋의 실행 코드와 테스트에 존재한다.
- `TRACKED_TOOLING`: 오프라인 빌드·평가·배포 도구로 존재하며 제품 API는 아니다.
- `EXTERNAL_STATE_UNKNOWN`: 코드와 배포 계약은 있으나 실제 AWS task·manifest·artifact 상태는 저장소만으로 확정할 수 없다.
- `HISTORICAL_DOC`: 저장소에 있으나 이후 코드보다 오래된 계획·수치·TODO다.
- `ABSENT`: 기준 커밋에 코드·계약이 없다. 로컬 작업 트리에만 있어도 GitHub `main` 기능으로 주장하지 않는다.

### 근거 우선순위

1. 기준 커밋의 실행 코드와 테스트.
2. `api/models.py` 및 실행한 OpenAPI, Converter의 `converter_api/app.py`.
3. 현재 CI·배포 workflow와 `.env.example`.
4. 기능별 계약 문서.
5. 날짜가 오래된 roadmap·평가·계획 문서.

이 문서의 기준 SHA는 [`b80dc29`](https://github.com/04JUNG/Standin-server/commit/b80dc29c6b6365d45877ef6418975a74ed0f8d09)다. 이후 `main`이 바뀌면 먼저 diff를 읽고 `source_commit`과 상태 표를 갱신한다.

`HANDOFF_HUMAN.md`와 `HANDOFF_AI.md`는 기술팀 로컬 작업 트리의 후속 실험까지 설명한다. 다음 경로는 기준 SHA에 없으므로 현재 GitHub 기능으로 import하거나 문서화하지 않는다.

```text
src/semantic_*.py
src/experimental/
src/camera.py
src/vlm/rough_slots.py
docs/NEXT_SPRINT/
docs/README.md
```

## 1. GitHub `main` 상태 행렬

| capability_id | 상태 | 기준 위치 | 판정 |
|---|---|---|---|
| `single_cut_geometry_topk` | `TRACKED_RUNTIME` | `api/app.py`, `src/pipeline.py`, `src/search.py` | `POST /analyze`가 컷 1장을 처리 |
| `pose_asset_delivery` | `TRACKED_RUNTIME` | `GET /pose/{pose_id}/bvh`, `GET /pose/{pose_id}/thumbnail` | base BVH와 4-view JPEG 번들 썸네일 제공 |
| `refine_v253` | `TRACKED_RUNTIME` | `src/refine_v2.py`, `src/refine_selector.py`, `POST /refine` | safe-aggressive 시도 후 conservative/base 복구, inline BVH/thumbnail |
| `humanart_cascade` | `TRACKED_RUNTIME` + `EXTERNAL_STATE_UNKNOWN` | `src/pose_cascade.py`, `src/pose_model_source.py`, `src/runtime_guard.py`, `.github/workflows/deploy.yml` | 코드·배포 기본은 존재. 실제 승인 manifest와 live task 설정은 외부 확인 필요 |
| `evaluation_harness` | `TRACKED_TOOLING` | `standin_eval/`, `evaluation/` | dataset/fixture/replay/blind/refine 증거 계약. `selected12-v1` GT는 미완성 |
| `library_build_deploy` | `TRACKED_TOOLING` | `scripts/build_db.py`, `scripts/render_bvh_thumbnails.py`, `scripts/deploy_pose_library.py` | 라이브러리 생성·검증·배포 지원. 원본 자산은 Git 외부 |
| `converter_v325` | `TRACKED_RUNTIME` | `converter/`, `converter_api/` | 별도 Blender 5.2 서비스: `/convert`, `/convert-bundle`, `/render-thumbnail` |
| `natural_language_search` | `ABSENT` | `/semantic-search`와 `src/semantic_*` 없음 | 신규 계약·색인·평가 필요 |
| `hand_pose_library` | `ABSENT` | palm-roll 정책만 존재 | 손바닥 회전 보정과 손 모양 라이브러리를 혼동하지 말 것 |
| `multi_cut_product_api` | `ABSENT` | `/analyze`는 단일 `UploadFile` | `scripts/run_batch_pipeline.py`는 오프라인 순회 도구 |
| `dynamic_image_router` | `ABSENT` | `src/vlm/rough_slots.py`, `src/experimental/rough_router.py` 없음 | 로컬 후속 실험을 main 동작으로 간주하지 말 것 |
| `part_retrieval_and_composition` | `ABSENT` | 전신·상체·하체 검색/조합 모듈 없음 | 신규 설계·lineage·안전 계약 필요 |

## 2. 런타임 경계

```text
Tauri/client
  → BFF: 인증, Job, 기록, /v1, rate limit, 최종 BVH 선택
    → inference service: 이미지 1장 → CutResult → optional refine
    → converter service: 최종 BVH → FBX 또는 BVH+FBX bundle
```

### Inference API

`api/app.py`에 구현된 경로만 현재 계약으로 취급한다.

```text
GET  /healthz
GET  /ops/metrics
POST /analyze
GET  /pose/{pose_id}/bvh
GET  /pose/{pose_id}/thumbnail
POST /refine
POST /export-order
```

- 동기·무인증·무상태 내부 서비스다.
- `/analyze` 입력은 multipart 이미지 한 장이다.
- `/refine` 성공 조정본은 `RefineResponse.bvh`에 인라인된다. 별도 refined URL/서버 캐시는 없다.
- `thumbnail` 렌더 실패는 조정 결과 전체 실패가 아니다.
- `/export-order`는 base-only legacy 주문서다. 실제 최종 base/refined 선택과 Converter 전달은 BFF가 소유한다.

### Converter API

`converter_api/app.py::create_app`의 현재 경로:

```text
GET  /characters
GET  /healthz
POST /convert
POST /convert-bundle
POST /render-thumbnail
```

- Blender는 child process로 실행하며 inference 이미지에는 `bpy`가 없어야 한다.
- `/convert-bundle`은 `final.bvh`, `final.fbx`, `manifest.json`을 원자적 ZIP으로 반환한다.
- BFF는 source/fbx/bundle SHA와 `conversion_id`를 보존한다.
- mirror는 Converter에서 한 번만 적용한다.

## 3. 기능별 파일 맵

| module_id | 코드·설정 | 같이 읽을 계약·테스트 |
|---|---|---|
| `inference_http` | `api/app.py`, `api/models.py` | [API 계약](API_CONTRACT.md), `tests/test_analyze_upstream_errors.py`, `tests/test_smoke.py` |
| `pipeline_policy` | `src/pipeline.py`, `src/schema.py`, `src/config.py`, `src/runtime_guard.py`, `src/tracing.py` | [파이프라인 개요](PIPELINE_OVERVIEW.md), `.env.example` |
| `vlm` | `src/vlm/client.py`, `src/vlm/prompts.py` | `tests/test_vlm_gemini_resilience.py`, `tests/test_vlm_gemini_fault_e2e.py` |
| `pose_slots` | `src/detect.py`, `src/pose.py`, `src/skeleton_extraction.py`, `src/pose_rescue.py` | [스켈레톤 계약](SKELETON_EXTRACTION_IMPROVEMENT.md), `tests/test_skeleton_extraction.py` |
| `pose_cascade` | `src/pose_cascade.py`, `src/pose_humanart.py`, `src/pose_contract.py`, `src/pose_model_source.py` | [cascade 설계](POSE_CASCADE_DESIGN.md), `tests/test_pose_cascade.py`, `tests/test_pose_contract.py`, `tests/test_pose_model_source.py` |
| `geometry_search` | `src/features.py`, `src/library.py`, `src/search.py` | `tests/test_search_family_grouping.py`, `tests/fixtures/search_regressions.v1.jsonl` |
| `library` | `src/repo.py`, `src/library_source.py`, `src/pose_quarantine.py`, `src/thumbnails.py` | `scripts/build_db.py`, `scripts/deploy_pose_library.py`, `tests/test_pose_quarantine.py` |
| `refine` | `src/refine.py`, `src/refine_v2.py`, `src/refine_selector.py`, `src/refine_policy.py`, `src/collision.py` | [Refine v2](REFINE_V2_DESIGN.md), [BFF 인계](REFINE_API_V25_BACKEND_BFF_HANDOFF.md), `tests/test_refine_v2.py` |
| `evaluation` | `standin_eval/`, `evaluation/datasets/selected12-v1/` | [평가 하네스](EVAL_HARNESS.md), `tests/test_eval_harness.py`, refine evidence/report tests |
| `converter` | `converter/`, `converter_api/`, `config/characters.example.json` | [통합 정본](FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md), `tests/converter/` |
| `deployment` | `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`, `.github/workflows/converter-*.yml`, `Dockerfile*` | [QA·릴리스](QA_SECURITY_RELEASE.md), `tests/test_inference_deployment_contract.py`, converter deployment tests |

## 4. 우선순위 작업 큐

### `W1_search_accuracy` — 검색 정확도 개선

- `priority`: 1 / P0
- `current`: 위치·각도·hybrid 기하 거리, mask-aware 피처, vectorized position search, pose-family dedup, quarantine가 구현돼 있다. 카메라 연속 탐색·동적 의미 라우터·부위 조합은 없다.
- `depends_on`: 봉인된 입력·사람 라벨·artifact lineage.
- `read`: `standin_eval/`, [평가 README](../evaluation/README.md), `evaluation/datasets/selected12-v1/`, `src/features.py`, `src/search.py`, `src/skeleton_extraction.py`, `tests/fixtures/search_regressions.v1.jsonl`.
- `next`:
  1. `selected12-v1`의 provenance, 사람 bbox, eligible, GT를 결과를 보지 않고 확정하고 seal한다.
  2. 실제 Gemini/RTMPose/현재 library SHA의 baseline run을 만든다.
  3. 실패를 `추출/소유권`, `검색 순위`, `라이브러리 공백`, `자산 품질`, `렌더/카메라`로 분리한다.
  4. 한 번에 한 요인만 바꾸고 fixture replay와 새 작품 holdout에서 비교한다.
- `accept`: 작가 blind Top-K 활용 가능률과 수정 시간이 개선되고 기존 성공→실패, 잘못된 인물 소유권, 후보 0건, p50/p95가 guardrail을 통과한다. 2D 거리만 좋아진 결과는 승인 근거가 아니다.

### `W2_pose_library_growth` — 라이브러리 포즈 증가

- `priority`: 2 / P0; `W1`의 실패 분류와 병행.
- `current`: BVH→SQLite, 4-view feature, JPEG thumbnail, bundle deployment, quarantine 경로가 있다. Git에는 실제 BVH/DB가 없다.
- `read`: `scripts/build_db.py`, `scripts/bvh_contact_sheet.py`, `scripts/render_bvh_thumbnails.py`, `scripts/deploy_pose_library.py`, `src/library.py`, `src/repo.py`, `src/library_source.py`, `src/pose_quarantine.py`, [결정 기록](DECISIONS.md).
- `deliver`: 실제 실패 빈도 × 수정 난이도 × 재사용성을 기준으로 한 gap batch; source/license/acquisition/conversion/keeper/reject/hash/rig metadata; 같은 snapshot의 DB·BVH·thumbs·quarantine.
- `accept`: bundle preflight, 중복·깨진 BVH·불량 메시·권리·썸네일 검사를 통과한다. 고정 실패 컷의 library-gap이 줄고 holdout에서 기존 성공을 해치지 않는다.
- `constraint`: [2026-07-14 평가](SEARCH_EVAL_2026-07-14.md)의 77개와 [MVP 문서](MVP_RELEASE.md)의 1511개는 시점 고정 숫자다. 현재 배포 개수는 `/healthz`, bundle manifest, DB query로 다시 측정한다.

### `W3_hand_library` — 손 라이브러리

- `priority`: 3 / P1
- `current`: `ABSENT`. `converter/palm_roll_policy.json`은 손바닥 roll 안전 정책이며 손 모양 검색/프리셋이 아니다.
- `depends_on`: canonical character rig, 최종 BVH와 Converter 사이의 책임 위치, 좌우 손 joint mapping.
- `read`: `converter/retarget.py`, `converter/bone_map.py`, `converter/palm_roll_policy.json`, `qa/retarget/CHAIN_TRANSPORT_V3_2_1_PALM_ROLL_QA/`, `tests/converter/test_convert.py`.
- `deliver`: 먼저 기본 손 pose와 side별 fallback을 정의하고, 적용 지점을 inference final BVH 또는 Converter retarget 중 하나로 고정한다. 그 뒤 검증된 소수 기능성 프리셋과 provenance를 추가한다.
- `accept`: left/right·엄지 방향·손목 제한·finger self-intersection·메시 관통·기존 palm-roll/retarget 비회귀를 실제 캐릭터로 검사한다. 전문 손 실패 시 해당 손만 기본 손으로 정확히 복구한다.
- `constraint`: COCO-17/RTMPose Body는 손가락을 관측하지 않는다. 손 모양 선택을 관측 기하라고 기록하지 않는다.

### `W4_multi_cut_6_to_8` — 6~8컷 일괄 입력

- `priority`: 4 / P1
- `current`: `ABSENT`. `/analyze`는 단일 `UploadFile`; `scripts/run_batch_pipeline.py`는 오프라인 파일 순회다.
- `depends_on`: BFF Job/폴링/인증 계약, 단일 컷 timeout, VLM/pose 동시성 예산.
- `read`: `api/app.py::analyze`, `api/models.py::CutResultOut`, [API 계약](API_CONTRACT.md), [BFF 결정](DECISIONS.md), `scripts/run_batch_pipeline.py`, `.github/workflows/deploy.yml`.
- `deliver`: `files[6..8]` 업로드와 페이지 이미지 자동 분할 중 1차 입력을 명시적으로 선택; `batch_id`, stable `cut_id`, order, per-cut status/result/error, idempotency, partial retry/cancel; bounded concurrency와 stage timing.
- `accept`: 한 컷 실패가 다른 컷 결과를 제거하지 않는다. 순서·인물·후보·최종 BVH lineage가 컷별로 유지되고, BFF가 진행률과 부분 결과를 재시작 후에도 복구한다. 단일 컷 계약과 결과는 회귀하지 않는다.
- `constraint`: 인증·외부 Job API를 inference 서비스에 중복 구현하지 않는다. 내부 batch primitive가 필요하면 BFF 경계와 별도로 정의한다.

### `W5_natural_language_search` — 자연어 검색

- `priority`: 5 / P1
- `current`: `ABSENT`. `/semantic-search`, semantic index, query parser, semantic API models가 없다.
- `depends_on`: `W2`의 pose provenance/metadata, 사용자 문장 평가셋, BFF UI 상태 계약.
- `read`: `src/schema.py`, `src/repo.py`, `src/library.py`, `api/models.py`, `api/app.py`, [API 계약](API_CONTRACT.md), [Controlled Vocabulary 결정](DECISIONS.md).
- `deliver`: 기존 `/analyze`와 분리된 versioned request/response; user query 원문, candidate evidence, exact/context/gap 상태, build/member lineage, 권리 적격성; deterministic constraints와 retrieval 평가.
- `accept`: 한국어 사용자 문장 holdout에서 Top-K 적합성, 방향/부정/정도/소품/관계 조건, `library_gap`, latency를 측정한다. BFF가 evidence와 build ID를 보존하고 적격 BVH/thumbnail만 전달한다.
- `constraint`: embedding score를 확률로 표시하지 않는다. 기하 distance와 semantic score를 무보정 합산하지 않는다. 의미 후보에 refine을 허용하려면 별도 관측·소유권·구조 안전 계약을 만든다.

## 5. 변경 불변식

1. VLM은 관절 좌표를 생성하지 않는다. COCO-17 좌표는 detector/pose model 소유다.
2. `detect.reconcile`의 count 신호, person ownership, skeleton coverage, search distance는 서로 다른 신호다. 하나로 다른 신호를 승격하지 않는다.
3. 쿼리와 라이브러리는 같은 `features.normalize_skeleton`과 feature version을 사용한다. 변경 시 DB를 재빌드한다.
4. `library.pose_to_feature`는 BVH projection과 refine가 공유하는 경로다. 한쪽만 변경하지 않는다.
5. 같은 pose family의 original/mirror/view가 Top-K를 독점하지 않도록 family dedup 계약을 유지한다.
6. quarantine pose는 search, BVH delivery, refine에서 fail-closed로 차단한다.
7. Refine은 selector와 안전 gate를 통과한 결과만 반환하고 실패 시 base/conservative로 정확히 복구한다.
8. inference는 최종 base/refined BVH를 결정하지 않는다. BFF가 선택하고 Converter에 SHA와 함께 전달한다.
9. Converter는 별도 프로세스이며 mirror를 한 번만 적용한다. inference Docker image에 Blender/bpy를 넣지 않는다.
10. production은 mock VLM/pose와 합성 라이브러리를 거부한다. Human-Art manifest는 license/runtime/hash 계약을 통과해야 한다.

## 6. 문서 충돌·외부 상태 처리

- `ROADMAP.md`, `MVP_RELEASE.md`, `SEARCH_EVAL_2026-07-14.md`는 유용한 배경이지만 현재 구현 상태 표가 아니다.
- `POSE_CASCADE_ROLLOUT_GATE_2026-09-01.md`의 당시 차단 판정과 현재 deploy workflow의 기본값이 다르다. 실제 운영 판정은 배포 task definition, immutable manifest, `/healthz`, CloudWatch/평가 결과를 함께 확인한다.
- `.github/workflows/deploy.yml`의 기본값은 실제 repository/environment variables가 없을 때 쓰는 설정이다. live 배포 상태를 단정하지 않는다.
- Git에 없는 `data/`, model ONNX, character FBX, evaluation outputs를 clean checkout에 있다고 가정하지 않는다.

## 7. 유지보수 매트릭스

| 변경 | 함께 수정·검증할 대상 |
|---|---|
| inference HTTP | `api/app.py`, `api/models.py`, [API 계약](API_CONTRACT.md), BFF DTO, API/smoke tests |
| feature/search | `src/features.py`, `src/library.py`, `src/search.py`, feature version, DB rebuild, fixed replay + holdout |
| library bundle | `src/repo.py`, `src/library_source.py`, `scripts/build_db.py`, thumbnails, quarantine, bundle/version/hash |
| pose/cascade | pose contract/manifest, runtime guard, cascade tests, deployment task env, rollback |
| refine | policy/config/selector, B0 exact fallback, refine evidence/report tests, inline BVH/thumbnail contract |
| Converter | `converter/`, `converter_api/`, protocol/registry, Blender image, converter CI, BFF SHA lineage |
| hand library | canonical rig/joint map, preset provenance, side/fallback, palm/mesh regression |
| multi-cut | BFF Job DTO, stable cut identity/order, partial failure/idempotency, load budget |
| natural language | new API contract, index/build ID, evidence schema, rights filter, golden/holdout queries |

## 8. 검증 명령

### Inference

```bash
python -m pip install -r requirements.txt pytest "scipy>=1.11"
python -m compileall src api scripts
python -m pytest -q tests/test_pose_cascade.py
python -m pytest tests/ --ignore=tests/converter -q
docker build --platform linux/amd64 --tag standin-inference:ci .
```

### Converter 계약

```bash
python -m pip install -r requirements-converter.txt
python -m pip install "numpy>=1.24" "Pillow>=10.0" httpx pytest
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

실 Blender 5.2 변환·컨테이너 격리·28/28 회귀는 `.github/workflows/converter-ci.yml`의 명령과 artifact mount 계약을 그대로 따른다.

## 9. PR·릴리스 규칙

- `main` 직접 push 금지. 기능 브랜치에서 PR을 만들고 `PR checks`와 write 권한 리뷰를 통과한다.
- API·계약·설정 변경은 관련 문서를 같은 PR에 포함한다.
- `.env`, API 키, ONNX, 라이선스 제한 BVH/FBX, 실행 결과는 커밋하지 않는다.
- 검색 품질 PR에는 동일 dataset/library/model SHA의 전후 결과와 악화 사례를 첨부한다.
- 배포 전 `/healthz`, runtime identity, pose library version, quarantine hash, Converter character/solver identity를 기록한다.
