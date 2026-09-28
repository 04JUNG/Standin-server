---
document_id: standin-server-ai-handoff
version: 2
as_of: 2026-09-24
owner: Standin technical team
audience: Standin technical team and coding agents
repository: Standin-server
source_snapshot: Standin technical-team working tree; referenced files absent from remote main are pending integration
entrypoint: src/pipeline.py::Pipeline.process_cut
external_contract: api/models.py + docs/API_CONTRACT.md
companion: docs/HANDOFF_HUMAN.md
github_main_companion: docs/HANDOFF_GITHUB_AI.md
---

# Standin 기술팀 인수인계 — AI 작업용

> 문서 소유: Standin 기술팀 · 기술팀원과 코드 작업 AI가 같은 구현 상태·우선순위·완료 기준을 사용하기 위한 문서

## 0. 해석 규칙

### 상태 값

- `CURRENT`: 이 작업 트리에서 실행되는 기존 동작. 코드·테스트로 확인한다.
- `SHADOW`: 코드가 있으나 기존 후보·응답을 바꾸지 않고 계획/진단만 기록한다.
- `DESIGN`: 합의된 목표 또는 구현 계획. 동작 중이라고 주장하지 않는다.
- `HISTORICAL`: 당시 실험·승격 전 기록. 현재 상태 판단에 직접 사용하지 않는다.
- `EXTERNAL_CONFIRMED`: 이 저장소 밖의 실제 운영 상태에 대한 최신 사용자 확인. 로컬 코드 기본값과 구분한다.

### 문서 우선순위

1. **현재 동작:** 실행 코드·테스트 → `api/models.py`/OpenAPI → [API 계약](API_CONTRACT.md).
2. **다음 image-only 검색 정책:** [라우터 규칙 v1.0](SEARCH_ROUTER_RULES_DRAFT_2026-09-22.md) → [검색 전략 v2](SEARCH_EDITABLE_POSE_STRATEGY_2026-09-22.md) → [P0–P7 구현 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md). 계획 본문에 남은 자연어·사용자 조건·잠금 예시는 이번 범위에서 제외한다.
3. **평가 사실:** [실제 모델 사용자 평가](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md)를 2D PCK 평가보다 우선해 작가 활용 가능성의 근거로 사용한다. 날짜와 표본을 함께 기록한다.
4. **과거 기록:** [archive](archive/)와 [9월 21일 통합 인수인계](archive/plans/HANDOFF_2026-09-21.md)는 현재 정책의 정본이 아니다.

문서가 `코드 미구현`이라고 쓰더라도 새 코드가 있을 수 있다. 2026-09-22 기준 `rough_router`의 phase-1 계획·shadow 연결은 존재한다. `SHADOW`를 후보 검색·조합의 구현 완료로 승격해 해석하지 않는다.

## 1. 범위와 현재 상태

| 항목 | 상태 | 확인 위치 |
|---|---|---|
| 이미지 → 인물별 기하 Top-K | `CURRENT` | `api/app.py`, `src/pipeline.py`, `src/search.py`, `src/schema.py` |
| 고른 후보 1개의 Refine v2.5 | `CURRENT` | `api/app.py`, `src/refine_v2.py`, `src/refine_selector.py` |
| 사용자 텍스트 `/semantic-search` | `CURRENT` opt-in; 이미지 라우터의 외부 입력 아님 | `src/semantic_service.py`, `src/semantic_search.py`, `api/app.py` |
| A/B1·의도 결합·카메라 탐색 | `SHADOW` 또는 오프라인 실험 | `src/experimental/`, `experiments/intent_fusion/`, `scripts/eval_camera_*.py` |
| 이미지 기반 의미 슬롯·SearchPlan | `SHADOW`; 계획 로그만 기록, 후보 불변 | `src/vlm/rough_slots.py`, `src/experimental/rough_router.py`, `rough_router_shadow.py`, `src/pipeline.py` |
| 부위별 실제 검색·U/L 조합·공통 최종 평가 | `DESIGN`; 아직 구현/승격 전 | [P3–P5 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md) |
| Semantic v3.3 | `DESIGN`; 현재 runtime을 v3.3로 가정하지 말 것 | [v3.3 기준](SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md), [실행안](SEMANTIC_V3_IMPLEMENTATION_PLAN.md) |
| 새 포즈의 제품 라이브러리 반영 | 배치별 확보·검수·배포가 필요한 `DESIGN` 작업. 로컬 BVH 수를 제품 적격 포즈 수로 세지 말 것 | [확장 가이드](POSE_LIBRARY_EXPANSION_GUIDE.md), `scripts/build_db.py`, `scripts/verify_pose_library_deployment.py` |
| 기본 손·전문 손 라이브러리 자동 적용 | `DESIGN`; RTMPose Body의 손가락 관측을 가정하지 말 것 | [손 포즈 설계](NEXT_SPRINT/HAND_POSE_PIPELINE.md), `src/bvh.py`, `qa/retarget/` |
| 6~8컷 제품 일괄 입력 | `DESIGN`; 현재 `POST /analyze`는 컷 1장, `scripts/run_batch_pipeline.py`는 오프라인 평가용 | [다중 컷 로드맵](NEXT_SPRINT/INFERENCE_ROADMAP.md), `api/app.py`, `api/models.py` |
| Human-Art 폴백의 최종 승격 | `EXTERNAL_CONFIRMED` — [최신 클로즈베타 자료](CLOSED_BETA_PRESENTATION_2026-09-22.md)의 사용자 확인. 승격 후 피드백은 아직 없음 | 현장 배포 manifest·환경 설정은 별도 확인. [9월 1일 gate](POSE_CASCADE_ROLLOUT_GATE_2026-09-01.md)는 `HISTORICAL` |

**새 라우터 입력:** 러프 이미지 하나. VLM 슬롯과 관절은 같은 이미지에서 시스템이 도출한다. 사용자 명령·독립 자연어·이전 후보 잠금은 이번 라우터의 입력이 아니다. 별도 `/semantic-search` API의 존재와 혼동하지 않는다.

## 2. 기능별 파일 맵

| module_id | 경로 | 변경 시 함께 볼 위치 |
|---|---|---|
| `http_contract` | `api/app.py`, `api/models.py` | [API 계약](API_CONTRACT.md), [BFF 필드 매핑](REFINE_API_V25_BACKEND_BFF_HANDOFF.md), `tests/test_api_boundaries.py`, `tests/test_semantic_api.py` |
| `pipeline_and_policy` | `src/pipeline.py`, `src/schema.py`, `src/config.py`, `src/runtime_guard.py`, `src/tracing.py` | [전체 흐름](PIPELINE_OVERVIEW.md), `.env.example`, `tests/test_smoke.py` |
| `vlm_and_routing` | `src/vlm/client.py`, `src/vlm/prompts.py`, `src/vlm/rough_slots.py`, `src/routing.py`, `src/detect.py`, `src/descriptor.py` | `tests/test_skeleton_extraction.py`, `tests/experimental/test_rough_router.py` |
| `pose_and_slots` | `src/pose.py`, `src/pose_contract.py`, `src/pose_cascade.py`, `src/pose_humanart.py`, `src/pose_model_source.py`, `src/pose_rescue.py`, `src/skeleton_extraction.py` | [cascade 설계](POSE_CASCADE_DESIGN.md), `tests/test_pose_cascade.py`, `tests/test_pose_model_source.py`, `pose_bench/` |
| `library_and_features` | `src/bvh.py`, `src/library.py`, `src/library_source.py`, `src/repo.py`, `src/features.py`, `src/camera.py` | `scripts/build_db.py`, `scripts/deploy_pose_library.py`, `scripts/verify_pose_library_deployment.py`, `tests/test_library_source.py`, `tests/test_repo_build.py` |
| `geometry_search` | `src/search.py`, `src/hybrid_search.py`, `src/pose_quarantine.py` | `config/library_exclusions.v1.json`, `config/refine_pose_quarantine.v1.json`, `tests/test_search_family_grouping.py`, `tests/test_pose_quarantine.py` |
| `experimental_search` | `src/experimental/a_minimal_support.py`, `src/experimental/b1_pose_facts.py`, `src/experimental/camera_search.py`, `src/experimental/intent_fusion.py`, `src/experimental/rough_router.py`, `src/experimental/rough_router_shadow.py`, `src/experimental/search_bundle.py` | `experiments/b1_pose_facts/`, `experiments/intent_fusion/`, `tests/experimental/`, `scripts/eval_camera_search.py`, `scripts/eval_camera_real_rough.py` |
| `semantic` | `src/semantic_service.py`, `src/semantic_search.py`, `src/semantic_catalog.py`, `src/semantic_documents.py`, `src/semantic_embedding.py`, `src/semantic_index.py`, `src/semantic_vocab.py`, `src/posecode.py` | `config/semantic_*.json`, `scripts/build_semantic_*.py`, `scripts/eval_semantic_search.py`, `tests/test_semantic_*.py`, [v3.3 설계](SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md) |
| `refine` | `src/refine.py`, `src/refine_v2.py`, `src/refine_selector.py`, `src/refine_policy.py`, `src/collision.py` | `scripts/eval_refine*.py`, `tests/test_refine_v2.py`, [Refine 기준](REFINE_V2_DESIGN.md) |
| `hand_library` | [손 파이프라인 설계](NEXT_SPRINT/HAND_POSE_PIPELINE.md), `src/bvh.py`, `qa/retarget/`, `assets/` | [캐릭터·리그 기준](CHARACTER_MODEL_SPEC_REVIEWED.md), [FBX 몸체 전략](NEXT_SPRINT/FBX_BODY_MODEL_STRATEGY.md); 전용 운영 모듈·프리셋 저장소는 아직 없음 |
| `multi_cut` | `api/app.py`, `api/models.py`, `src/pipeline.py`, `scripts/run_batch_pipeline.py` | [API 계약](API_CONTRACT.md), [다중 컷 로드맵](NEXT_SPRINT/INFERENCE_ROADMAP.md); 제품용 일괄 요청·결과 계약은 아직 없음 |
| `thumbnail_and_model` | `src/thumbnails.py`, `src/thumbnail_renderer.py`, `scripts/render_pose_library_thumbnails.py`, `qa/retarget/` | `tests/test_refined_thumbnail.py`, [캐릭터 요구사항](CHARACTER_MODEL_SPEC_REVIEWED.md) |
| `evaluation` | `standin_eval/`, `evaluation/datasets/`, `artifacts/camera_real_rough_20260921_v2/`, `scripts/audit_camera_human_review.py`, `scripts/build_camera_real_review.py`, `scripts/render_camera_model_review.py` | [9월 21일 사용자 평가](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md), `tests/test_camera_real_eval.py` |
| `deployment` | `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`, `Dockerfile`, `requirements.txt`, `docs/requirements-semantic.txt`, `.env.example` | [QA·릴리스](QA_SECURITY_RELEASE.md), [협업](COLLABORATION.md), `GET /healthz` |

비코드 디렉터리: `data/`는 로컬 DB/BVH/모델, `in/`은 러프 입력, `out/`·`artifacts/`·`outputs/`는 생성된 평가/발표 결과, `assets/`·`tripo_tpose_views*/`는 모델 참고 자산이다. 파일이 있다는 사실만으로 라이선스·운영 승격·재현성이 확보된 것은 아니다. `docs/archive/`는 대체된 정책·계획의 이력이다.

## 3. 우선순위 작업 큐

| workstream_id | 우선순위 | 제품 목표 | 구현 상태 | 상세 작업 |
|---|---|---|---|---|
| `W1_search_accuracy` | **1 · P0** | 실제 모델에서 쓸 수 있는 Top-K와 낮은 수정 시간 | 기하 검색 CURRENT, 라우터 SHADOW, 부위 검색·조합 DESIGN | `T0`~`T7` |
| `W2_pose_library_growth` | **2 · P0** | 실패 유형을 메우는 적격 BVH 증가 | 배치별 확장 작업 필요 | 아래 `W2` |
| `W3_hand_library` | **3 · P1** | 기본 손 보장과 검증된 전문 손 적용 | DESIGN | 아래 `W3` |
| `W4_multi_cut_6_to_8` | **4 · P1** | 6~8컷을 한 작업으로 입력·검토 | 제품 API DESIGN; 오프라인 배치 스크립트만 존재 | 아래 `W4` |
| `W5_natural_language_search` | **5 · P1** | 사용자 문장으로 포즈를 찾아 제품에서 사용 | `/semantic-search` CURRENT opt-in; v3.3 DESIGN | 아래 `W5` |

우선순위 숫자는 제품 과제의 순서다. `T0`~`T7`의 P0~P3는 `W1` 내부 구현 순서이며 다른 workstream의 우선순위를 낮추지 않는다. `W2`는 `W1`의 평가 계약과 병행한다. `W3`·`W4`·`W5`는 각각 리그, 단일 컷 계약, semantic build의 준비 상태를 확인하며 진행한다.

### `W1_search_accuracy` — 검색 정확도 개선: `T0`~`T7`

### `T0` — 평가 계약·입력 자산 고정

- `priority`: P0
- `status`: DESIGN / 첫 착수
- `depends_on`: 없음
- `read`: [P0 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), [사용자 평가](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md), `standin_eval/`, `evaluation/datasets/`
- `deliver`: SearchRequest/Plan/Candidate/Proposal 내부 타입 경계, DB·BVH·이미지·평가 label 해시, 개발 28명과 새 작품 holdout 분리, 성공·실패/수정 시간 기준.
- `accept`: 기존 라벨을 새 정의로 덮어쓰지 않고 재생 가능하다. 2D PCK, 실제 모델 활용 가능, 작가 수정 시간을 별도 지표로 저장한다.

### `T1` — 카메라·메시 방향 교정

- `priority`: P0
- `status`: DESIGN / 기존 A2 운영 승격 차단
- `depends_on`: `T0`의 고정 평가 입력
- `read`: [사용자 평가 §1](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md), `src/camera.py`, `src/library.py`, `scripts/render_camera_model_review.py`, `scripts/build_camera_model_sheet.py`, `qa/retarget/`
- `deliver`: 축 표시 비대칭 포즈로 앞/뒤·좌/우·깊이·가림·yaw/elevation/roll/원근을 투영·렌더·직렬화에서 일치시킨 별도 결과 run.
- `accept`: 물리적 앞쪽 표식과 occlusion이 예상대로 보이고, 기존 run/사용자 판정을 덮어쓰지 않는다. 새 카메라 규약은 version/hash로 구분한다.

### `T2` — image-only 슬롯·라우터 phase-1 검증

- `priority`: P1
- `status`: SHADOW 구현 존재; 검색 실행은 없음
- `depends_on`: `T0`; `T1`과 정책 검증은 병행 가능
- `read`: [확정 규칙 12개](SEARCH_ROUTER_RULES_DRAFT_2026-09-22.md), `src/vlm/rough_slots.py`, `src/experimental/rough_router.py`, `src/experimental/rough_router_shadow.py`, `src/pipeline.py`, `tests/experimental/test_rough_router.py`
- `current_flags`: `EXPERIMENTAL_ROUGH_ROUTER_MODE=off|shadow`, `EXPERIMENTAL_ROUGH_SLOTS_ENABLED=0|1`, `EXPERIMENTAL_ROUGH_ROUTER_LOG`. 기본 off, `on` 없음.
- `current_limit`: `rough-router-v1.1-phase1`은 그룹 관계·lower-only를 unsupported로 기록한다. 채널은 `not_executed_planning_only`; `refine_allowed=false`, `candidates_changed=false`.
- `deliver`: 고정 슬롯 replay, 실제 VLM 슬롯 검수, VLM 오류와 라우터 오류 분리, 인물 귀속·가시성 충돌·unknown·가설 trace. 미지원 경로를 성공으로 표시하지 않는다.
- `accept`: off에서 기존 출력 동일, shadow에서도 후보 동일; 숨은 관절의 G/F 미사용과 fail-closed 그룹/귀속 계약 통과.

### `T3` — 전신·상체·하체 재료 검색

- `priority`: P1
- `status`: DESIGN
- `depends_on`: `T0`, `T2`; 정량 품질 비교에는 `T1`
- `read`: [P3 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), `src/search.py`, `src/features.py`, `src/semantic_search.py`, `src/semantic_service.py`, `src/experimental/a_minimal_support.py`, `src/experimental/b1_pose_facts.py`, `src/experimental/intent_fusion.py`
- `deliver`: `retrieve(request, plan) -> materials`와 채널 trace; 같은 geometry DB/member의 full/upper/lower G·S·F·D 재료 합집합; 검색 출처·가설·mirror·자산 hash 보존.
- `accept`: 실제 상·하체 재료 시트와 누락 이유가 나온다. 부분 피처는 query/library 대칭, 결측/anchor 검증을 통과한다. 현재 v3.3 설계를 구현 runtime으로 오인하지 않는다.

### `T4` — 상·하체 조합

- `priority`: P2
- `status`: DESIGN
- `depends_on`: `T3`
- `read`: [P4 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), `src/bvh.py`, `qa/retarget/`, [검색 전략 v2](SEARCH_EDITABLE_POSE_STRATEGY_2026-09-22.md)
- `deliver`: 검색한 적격 U/L을 공통 골격·pelvis 기준으로 조합하는 생성 후보; 재료 ID/hash·rig version·BVH/handle lineage.
- `accept`: root/pelvis/spine·좌우·비틀림·접촉·메시 안전을 실제 모델에서 검사한다. 조합 실패 시 검증된 전신 후보를 유지한다. 원본 pose ID로 생성 후보를 내보내지 않는다.

### `T5` — 공통 카메라·제한 조정·최종 평가

- `priority`: P2
- `status`: DESIGN
- `depends_on`: `T1`, `T3`, `T4`
- `read`: [P5 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), `src/camera.py`, `src/refine_v2.py`, `src/collision.py`, `src/experimental/b1_pose_facts.py`
- `deliver`: 전신/조합에 같은 평가 인터페이스, 후보별 카메라 하나, 의미·상체·하체·관계·메시 검증과 안전 복구.
- `accept`: 상체 평균이 하체 실패를 숨기지 않는다. 숨은 부위에 기하 성공 점수를 주지 않는다. 기존 의미 후보의 refine 금지를 flag 하나로 해제하지 않는다. preview/export는 같은 최종 BVH를 참조한다.

### `T6` — 작가 블라인드 평가와 승격 결정

- `priority`: P2
- `status`: DESIGN
- `depends_on`: `T3`부터 경로별 중간 평가 가능; 최종 판정은 `T5`
- `read`: [P6 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), [이전 사용자 평가](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md), `standin_eval/`
- `deliver`: 기존 기하/A+B1, 후보 합집합, 부위 조합, 자동 조정의 분리 비교. 방법 이름을 가린 실제 모델·새 작품 holdout·작가 편집 과제.
- `accept`: Top-K 활용 가능률, 수정 없이 사용 가능, 실제 수정 시간, 기존 성공→실패, p50/p95를 동일 자산·예산에서 측정. 개선이 입증되지 않은 경로는 실험 상태 유지.

### `T7` — 이미지 API/BFF 점진 통합

- `priority`: P3
- `status`: DESIGN
- `depends_on`: `T6`의 승격 판단
- `read`: [P7 계획](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), `src/pipeline.py`, `api/app.py`, `api/models.py`, [API 계약](API_CONTRACT.md), [BFF 인계](REFINE_API_V25_BACKEND_BFF_HANDOFF.md)
- `deliver`: 버전 선택 검색 오케스트레이터, 결과의 observed/inferred/unspecified·출처·카메라·최종 BVH lineage, off/shadow/on 및 복구.
- `accept`: 의미/조합 실패·timeout·자산 mismatch가 기존 검증 경로로 복구된다. `/analyze`→선택 후보→preview/export가 같은 인물과 최종 BVH를 사용한다.

### `W2_pose_library_growth` — 라이브러리 포즈 증가

- `priority`: P0; `W1/T0`의 고정 실패 컷 분류와 병행
- `status`: 배치별 확장 과제. 저장소의 로컬 원본·DB 개수나 과거 3,000개 목표를 현재 제품 적격 개수로 간주하지 않는다.
- `read`: [라이브러리 확장 가이드](POSE_LIBRARY_EXPANSION_GUIDE.md), [실제 모델 실패 분석](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md), `scripts/build_db.py`, `scripts/deploy_pose_library.py`, `scripts/verify_pose_library_deployment.py`, `src/library_source.py`, `src/repo.py`, `src/pose_quarantine.py`, `config/library_exclusions.v1.json`.
- `deliver`: 실패 컷 수·수정 난이도·재사용성을 근거로 한 gap batch; 원본→변환→keeper BVH의 출처·권리·해시·리그·검수 manifest; geometry DB·semantic member·썸네일의 같은 자산 스냅샷.
- `accept`: 추가 포즈가 실제 모델·라이선스·제품 내보내기·중복/quarantine 검사를 통과한다. 고정 실패 컷의 적격 Top-K 공백과 기존 성공→실패를 전후 비교하고, 새 작품 holdout에서도 활용 가능률을 확인한다.
- `constraint`: 포즈 수 자체를 정확도 개선으로 보고하지 않는다. 의미 색인까지 적용할 때는 geometry member/hash와 semantic build를 함께 확인한다.

### `W3_hand_library` — 손 라이브러리

- `priority`: P1
- `status`: DESIGN. 기본 손·전문 손 자동 적용을 운영 기능으로 가정하지 않는다.
- `depends_on`: 최종 캐릭터 리그/손 관절 규약, 선택 후보의 최종 BVH·refine 결과.
- `read`: [손 포즈 파이프라인](NEXT_SPRINT/HAND_POSE_PIPELINE.md), [캐릭터 요구사항](CHARACTER_MODEL_SPEC_REVIEWED.md), [FBX 몸체 전략](NEXT_SPRINT/FBX_BODY_MODEL_STRATEGY.md), `src/bvh.py`, `src/refine_v2.py`, `qa/retarget/`.
- `deliver`: canonical left/right 기본 손 프리셋과 손목 안전 검사 → 별도 VLM 손 기능 feasibility probe → 통과한 기능에 한해 소규모 전문 손 프리셋 및 손별 fallback.
- `accept`: 양손 기본 손 제공, 좌우·엄지·손목·자체 관통 검사와 최종 메시 검수 통과. 전문 손은 기본 손 대비 블라인드 평가에서 악화가 없고 한쪽 실패가 다른 손을 폐기하지 않는다. 손 단계의 p50/p95를 기록한다.
- `constraint`: RTMPose Body에 손가락 관절이 없으므로 러프에서 세부 손가락을 관측했다고 표시하지 않는다. 설계의 12컷 probe만으로 자동 적용을 승격하지 않는다.

### `W4_multi_cut_6_to_8` — 6~8컷 일괄 입력

- `priority`: P1
- `status`: DESIGN. `scripts/run_batch_pipeline.py`는 여러 파일을 순회하는 오프라인 정성평가 스크립트이며 제품 API/Job 구현이 아니다.
- `depends_on`: 현재 한 컷 `/analyze` 계약의 안정성, BFF의 인증·Job·폴링 경계, 동시 실행/지연 예산.
- `read`: [다중 컷 로드맵 Stage 8](NEXT_SPRINT/INFERENCE_ROADMAP.md), [API 계약](API_CONTRACT.md), `api/app.py`, `api/models.py`, `src/pipeline.py`, `scripts/run_batch_pipeline.py`, `tests/test_api_boundaries.py`.
- `deliver`: BFF·클라이언트와 6~8개 컷 이미지 일괄 제출과 한 페이지 자동 분할 중 입력 계약을 확정; 안정적인 `cut_id`·순서·인물 ID·컷별 상태를 가진 작업 결과, 부분 실패·재시도·timeout·취소/중복 요청 처리, 컷 간 병렬성·서버 부하 계측.
- `accept`: 6~8컷 한 작업에서 컷 순서와 각 컷의 `CutResult`/BVH 출처가 유지된다. 한 컷 실패가 다른 컷 결과를 없애지 않고 BFF가 진행률·부분 결과를 안전하게 표시한다. 단일 컷 회귀와 작업 p50/p95를 확인한다.
- `constraint`: 현재 `/analyze`에 여러 파일을 임의로 넣어 기존 응답을 바꾸지 않는다. 페이지 이미지 자동 분할은 입력 계약에 포함하기로 결정한 경우에만 구현한다.

### `W5_natural_language_search` — 자연어 검색

- `priority`: P1
- `status`: `/semantic-search`와 v2 런타임은 CURRENT opt-in; 제품 연결·holdout 품질·v3.3은 미완료. image-only 러프 라우터와 별도 사용자 텍스트 경로다.
- `read`: [현행 API 계약](API_CONTRACT.md), [v2 BFF 인계](API_BFF_SEMANTIC_HANDOFF_2026-08-18.md), [v3.3 단일 설계](SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md), [구현 계획](SEMANTIC_V3_IMPLEMENTATION_PLAN.md), `api/app.py`, `api/models.py`, `src/semantic_service.py`, `src/semantic_search.py`, `src/semantic_index.py`, `scripts/eval_semantic_search.py`, `tests/test_semantic_api.py`.
- `deliver`: 현재 opt-in API의 실제 사용자 문장·BFF 흐름·권리 적격 후보 검증; v3.3 문서/색인·좌우·부정·조건 해석의 구현과 별도 버전 평가; pinned build/model, readiness, 오류·gap UI 계약.
- `accept`: 자연어 질의에서 적격 BVH·썸네일과 `status`·근거·`library_gap`을 BFF가 보존한다. 고정 질의와 새 작품 holdout에서 검색 품질·지연·이전 성공→실패를 비교하고, v3.3 승격은 별도 승인 기준을 통과한다.
- `constraint`: 의미 후보의 `refine_allowed=false`를 유지한다. 2026-08-18 BFF 문서는 v2 필드·운영 인계이며 v3.3 정책과 충돌하면 v3.3 설계와 새 API 계약을 따른다. 텍스트 검색의 존재를 이미지 라우터에 사용자 문장 입력이 추가됐다는 뜻으로 해석하지 않는다.

### 병행 유지 과제

| id | priority | 상태·다음 증거 | 참고 |
|---|---|---|---|
| `M1_pose_cascade` | P1 | Human-Art 최종 승격은 최신 사용자 확인. 실제 배포 설정/manifest와 승격 후 사용자 피드백·오류·지연을 관찰. 과거 gate를 현재 미승격 판정으로 복사하지 말 것 | [클로즈베타 자료](CLOSED_BETA_PRESENTATION_2026-09-22.md), [cascade 설계](POSE_CASCADE_DESIGN.md), `src/pose_cascade.py`, `scripts/run_pose_canary_eval.py` |
| `M2_refine` | P1 | v2.5.3 engineering closeout 후 작가 blind·실메시·lap-contact 실표본 및 quarantine 후 대체 Top-K 품질 확인 | [승격 전략](REFINE_V25_PROMOTION_STRATEGY.md), `src/refine_v2.py`, `tests/test_refine_v2.py` |
| `M3_release_integrity` | P1 | DB/BVH/thumbnail/quarantine provenance·hash와 BFF lineage/export/TTL 점검. `W2` 신규 배치에도 적용 | [라이브러리 가이드](POSE_LIBRARY_EXPANSION_GUIDE.md), [BFF 인계](REFINE_API_V25_BACKEND_BFF_HANDOFF.md), [QA](QA_SECURITY_RELEASE.md) |

## 4. 수정 불변식

1. VLM은 관절 좌표를 만들지 않는다. 새 의미 슬롯은 사람별 *해석*이며 관측·사용자 지정과 섞지 않는다.
2. `detect.reconcile`의 인물 수 일치 신호, 슬롯 소유권, skeleton coverage, 의미 신뢰도는 다른 신호다. 하나로 다른 신호를 승격하지 않는다.
3. 기하 쿼리와 BVH 색인은 같은 `features.normalize_skeleton`/`library.pose_to_feature` 규약을 쓴다. 피처 변경 시 version을 올리고 색인·테스트를 같이 갱신한다.
4. 숨긴/실패한 관절은 G/F 또는 refine 허용 근거가 아니다. D는 미관측 부위의 완성 대안이며 정답 주장도 아니다.
5. 새 검색의 G/S/F/D는 독립 재료 채널이다. 다른 metric의 원점수를 무보정 합산하거나 한 채널의 실패를 성공으로 채우지 않는다. 동일 DB/member/hash를 확인한다.
6. 기존 semantic 후보는 refine 금지다. 새 후보 조정 자격은 출처·귀속·관측·구조 안전을 별도로 검증해야 한다.
7. `REFINE_V2_ENABLED=1`의 selector는 공격적 결과의 구조 안전과 conservative 대비 비회귀를 확인하고 실패 시 conservative/base로 복구한다.
8. `src/schema.py`의 Controlled Vocabulary를 바꾸면 프롬프트·API 계약·색인 재태깅까지 함께 처리한다. 새 의미 슬롯은 versioned extension으로 둔다.
9. 오프라인 A2의 PCK 증가를 실제 모델 활용 가능률 개선으로 보고하지 않는다. 9월 21일 사용자 판정은 27명 paired, 단일 검토자·방식 노출 조건이다.

## 5. 유지보수 절차

### 변경 종류별 동시 갱신

| 변경 | 코드/자산 | 문서·검증 |
|---|---|---|
| HTTP field/endpoint | `api/app.py`, `api/models.py`, `src/schema.py` | [API 계약](API_CONTRACT.md), BFF mapping, smoke/API tests |
| VLM 슬롯/라우팅 | `src/vlm/{prompts,client,rough_slots}.py`, `src/experimental/rough_router*.py`, `src/pipeline.py` | [라우터 규칙](SEARCH_ROUTER_RULES_DRAFT_2026-09-22.md), slot replay + off/shadow 불변 테스트 |
| 피처·카메라·검색 metric | `src/features.py`, `src/library.py`, `src/search.py`, `src/camera.py`, `src/config.py` | feature/index/camera version, DB 재빌드, 고정 전후 평가·악화 사례 |
| DB·라이브러리 | `src/repo.py`, `src/library_source.py`, `scripts/build_db.py`, `config/` | provenance/license, BVH·DB·썸네일·semantic build hash, [QA](QA_SECURITY_RELEASE.md) |
| Refine/조합 | `src/refine*.py`, `src/collision.py`, 새 compose 코드 | 안전·메시·원본 복구 테스트, [Refine 기준](REFINE_V2_DESIGN.md), final BVH lineage |
| 손 프리셋·리그 | `src/bvh.py`, 신규 hand 모듈/프리셋, `qa/retarget/` | 좌우·손목·메시 안전, 기본 손 복구, [손 설계](NEXT_SPRINT/HAND_POSE_PIPELINE.md) |
| 다중 컷 요청 | `api/app.py`, `api/models.py`, `src/pipeline.py`, BFF Job DTO | 컷 ID·순서·부분 실패·재시도·지연 테스트, [API 계약](API_CONTRACT.md) |
| 자연어 검색 | `src/semantic_*.py`, `api/app.py`, `api/models.py`, semantic build | 고정 질의·holdout, build/member 해시·권리, [v3.3 기준](SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md), BFF 매핑 |
| 배포 플래그 | `src/config.py`, `.env.example`, `Dockerfile`, `.github/workflows/` | `/healthz`, startup guard, rollback 설정, [QA](QA_SECURITY_RELEASE.md) |

### 실행 기준

```bash
.venv/bin/python tests/test_smoke.py
.venv/bin/python tests/experimental/test_rough_router.py
.venv/bin/python tests/experimental/test_intent_fusion.py
.venv/bin/python -m compileall -q src api scripts
git diff --check
git status --short --branch
```

CI의 hermetic 목록은 `.github/workflows/ci.yml`에 있다. 실 BVH·geometry DB·semantic build/golden 검사는 [QA 문서](QA_SECURITY_RELEASE.md)의 artifact-backed 환경에서 별도로 실행한다. `python`이 없는 호스트는 `.venv/bin/python` 또는 `python3`를 사용한다.

### 자산·릴리스

- `.env`, 키, 라이선스 제한 BVH/모델·ONNX를 Git에 넣지 않는다. 로컬 `data/`, `in/`, `out/`, `artifacts/`를 clean checkout의 일부로 가정하지 않는다.
- 배포 전 mock/합성 fallback이 production에서 차단되는지, `/healthz`의 적격 pose와 semantic readiness, 선택한 manifest·quarantine·DB hash를 확인한다.
- 이 작업 트리는 현재 미커밋·미추적 변경이 많다. 다른 작업을 삭제하거나 일괄 stage하지 말고 변경 범위를 `git status`로 확인한다.
- 기준이 바뀐 문서는 [문서 안내](README.md)를 갱신하고 옛 계획은 [archive](archive/)에 보존한다. [협업 규칙](COLLABORATION.md)에 따라 계약과 문서를 같은 PR에서 고친다.
