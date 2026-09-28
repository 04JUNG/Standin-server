# Standin 기술팀 인수인계 — 사람이 읽는 요약

> 문서 소유: Standin 기술팀 · 갱신: 2026-09-24 · 대상: Standin 추론 서버를 이어받는 기술팀원
> 기준: 기술팀 작업 트리. 원격 `main`에 아직 없는 참조 파일과 기능은 후속 반영 대상이다.
> 상세 파일·변경 지침은 [AI 작업용 인수인계](HANDOFF_AI.md)에 있다. GitHub `main`만 기준으로 인계할 때는 [원격 기준 사람용](HANDOFF_GITHUB_HUMAN.md)과 [원격 기준 AI용](HANDOFF_GITHUB_AI.md)을 사용한다.

## 이 서버가 맡는 일

웹툰 러프 컷을 받아 인물별 포즈 Top-K를 찾고, 사용자가 고른 후보 한 개를 필요할 때 조정한다. 앱의 인증·작업 기록·CSP 전달은 이 저장소 밖의 BFF/클라이언트가 맡는다. 서버와 팀 사이의 기준은 `CutResult` 및 [HTTP API 계약](API_CONTRACT.md)이다.

현재 `/analyze`는 **컷 1장**의 관절을 추출해 기하 검색을 수행한다. `/semantic-search`는 별도 opt-in 텍스트 API이고, 의미 후보는 현재 refine 대상이 아니다. 새 러프 라우터는 **계획 기록용 shadow**까지 구현됐으며 검색 후보를 바꾸지 않는다. 상·하체 독립 검색과 포즈 조합, 손 라이브러리 자동 적용, 제품용 6~8컷 일괄 입력은 아직 구현 과제다.

## 기능별 위치

| 기능 | 먼저 열 파일·폴더 | 역할 |
|---|---|---|
| 요청·응답 | `api/app.py`, `api/models.py`, [API 계약](API_CONTRACT.md) | `/analyze`·`/refine`·BVH/썸네일·상태 응답 |
| 한 컷 처리·설정 | `src/pipeline.py`, `src/schema.py`, `src/config.py`, `src/runtime_guard.py` | 단계 연결, 결과 형식, 운영 안전 설정 |
| VLM·인물·관절 | `src/vlm/`, `src/detect.py`, `src/skeleton_extraction.py`, `src/pose*.py`, [스켈레톤 기준](SKELETON_EXTRACTION_IMPROVEMENT.md) | 사람 수·의미 분석, 소유권, 관절 추출과 복구 |
| 포즈 라이브러리 | `src/bvh.py`, `src/library.py`, `src/repo.py`, `src/features.py`, `config/`, [라이브러리 가이드](POSE_LIBRARY_EXPANSION_GUIDE.md) | BVH·SQLite·투영·공통 피처와 제외 정책 |
| 현재 검색 | `src/search.py`, `src/hybrid_search.py`, `src/pose_quarantine.py` | 기하 Top-K와 불량 포즈 차단 |
| 새 검색 실험 | `src/experimental/`, `src/vlm/rough_slots.py`, `experiments/`, [image-only 라우터 규칙](SEARCH_ROUTER_RULES_DRAFT_2026-09-22.md) | A/B1·카메라·의도 결합 및 라우터 shadow. 운영 후보 변경 전 단계 |
| 의미 검색 | `src/semantic_*.py`, `src/posecode.py`, [v3.3 설계](SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md) | 현행 텍스트 검색과 다음 버전의 설계 |
| 선택 후보 조정 | `src/refine.py`, `src/refine_v2.py`, `src/refine_selector.py`, [Refine 기준](REFINE_V2_DESIGN.md) | v2.5 안전 선택과 원본 복구 |
| 손 포즈 | [손 라이브러리 설계](NEXT_SPRINT/HAND_POSE_PIPELINE.md), `src/bvh.py`, `qa/retarget/` | 기본 손·전문 손 프리셋과 최종 리그 적용의 후속 과제 |
| 6~8컷 일괄 입력 | `api/app.py`, `api/models.py`, `src/pipeline.py`, [다중 컷 로드맵](NEXT_SPRINT/INFERENCE_ROADMAP.md) | 현재 API는 컷 1장 단위. `scripts/run_batch_pipeline.py`는 오프라인 평가용 |
| 실행·검증 | `scripts/`, `tests/`, `standin_eval/`, `evaluation/`, `.github/workflows/` | DB 빌드, 재현, 평가, CI |
| 캐릭터·리타게팅 | `qa/retarget/`, `assets/`, [캐릭터 요구사항](CHARACTER_MODEL_SPEC_REVIEWED.md) | FBX/메시 검수. 검색 정확도와 별도 판정 |
| 데이터·결과 | `data/`, `in/`, `artifacts/`, `out/`, `outputs/` | 로컬 원본·모델·DB와 평가 결과. Git의 코드와 구별 |

## 지금 중요한 판단

- [실제 모델 28명 평가](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md)에서 카메라 A2의 활용 가능률 개선은 확인되지 않았다. 같은 인물 27명에서 기존 7/27, A2 5/27이었다. 단일 검토자·방식 노출 평가이므로 일반화 수치는 아니다. 이전의 PCK 증가는 **입력 관절과의 2D 일치**였다.
- 가장 먼저 카메라의 실제 앞뒤·좌우·가림 규약을 맞춰야 한다. 현재 평가 후보 56개 중 몸통 기준 진단에서 50개가 등 방향으로 보였다. 하체 상태·접촉 오류도 별도로 해결해야 한다.
- 9월 22일 검색 목표는 **러프 이미지 하나**에서 얻은 관절과 인물별 VLM 슬롯으로 전신·상체·하체 재료를 찾고, 적격 재료를 조합하는 것이다. 사용자 문장·부위 잠금은 이번 라우터 입력 범위가 아니다. [검색 전략 v2](SEARCH_EDITABLE_POSE_STRATEGY_2026-09-22.md)의 오래된 확장 예시보다 [확정된 image-only 규칙](SEARCH_ROUTER_RULES_DRAFT_2026-09-22.md)을 우선한다.
- Human-Art 폴백은 [9월 22일 클로즈베타 자료](CLOSED_BETA_PRESENTATION_2026-09-22.md)의 최신 사용자 확인에 따르면 최종 승격됐다. [9월 1일 rollout gate](POSE_CASCADE_ROLLOUT_GATE_2026-09-01.md)는 승격 전 기록이다. 승격 후 사용자 피드백과 실제 배포 설정·manifest는 따로 확인한다.

## 다음 과제 — 우선순위

| 순위 | 과제 | 완료했다고 판단할 증거 | 먼저 볼 파일 |
|---|---|---|---|
| **1 · P0** | **검색 정확도 개선** — 카메라 앞뒤·가림 오류와 하체 불일치를 바로잡고, 전신·부위별 후보 검색과 조합을 검증 | 실제 모델을 본 작가의 Top-K 활용 가능률·수정 시간이 개선되고 기존 성공 사례가 악화되지 않음 | [실제 모델 평가](SEARCH_CAMERA_HUMAN_REVIEW_2026-09-21.md), [검색 구현 순서](SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md), `src/camera.py`, `src/search.py` |
| **2 · P0** | **라이브러리 포즈 증가** — 실제 실패 컷의 빈 자세를 묶어 BVH를 확보·검수·색인 | 새 포즈의 출처·권리·리그·품질을 확인하고, 고정 실패 컷에서 후보 공백이 줄어듦 | [라이브러리 확장 가이드](POSE_LIBRARY_EXPANSION_GUIDE.md), `scripts/build_db.py`, `src/library_source.py`, `scripts/verify_pose_library_deployment.py` |
| **3 · P1** | **손 라이브러리** — 좌우 기본 손을 보장하고, 검증된 기능성 손 프리셋을 선택 후보에 적용 | 손목·엄지·좌우 안전 검사 통과; 전문 손이 불확실하면 해당 손만 기본 손으로 복귀 | [손 포즈 설계](NEXT_SPRINT/HAND_POSE_PIPELINE.md), `src/bvh.py`, `qa/retarget/` |
| **4 · P1** | **6~8컷 한번에 넣기** — 한 작업에서 컷별 분석·순서·결과를 관리 | 한 컷 실패가 다른 컷을 막지 않고 컷별 인물·후보·진행 상태가 유지되며 지연시간을 측정 | [다중 컷 로드맵](NEXT_SPRINT/INFERENCE_ROADMAP.md), [API 계약](API_CONTRACT.md), `api/app.py`, `src/pipeline.py` |
| **5 · P1** | **자연어 검색** — 기존 opt-in `/semantic-search`를 검증·개선하고 BFF에서 사용 가능하게 연결 | 사용자 문장으로 적격 포즈를 찾고 `library_gap`·근거·BVH·썸네일을 정확히 전달; 새 v3.3은 별도 평가 후 승격 | [현행 API 계약](API_CONTRACT.md), [v3.3 설계](SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md), [BFF 인계](API_BFF_SEMANTIC_HANDOFF_2026-08-18.md), `src/semantic_service.py` |

검색 과제의 내부 순서는 **평가셋·카메라 규약 고정 → 라우터 shadow 검증 → 부위별 후보 회수 → 조합·공통 평가 → 작가 블라인드 평가 → API 적용**이다. 라이브러리 확장은 평가와 병행한다. 6~8컷 입력의 첫 계약은 컷 이미지 여러 장을 한 작업으로 묶는 방식과 페이지 한 장에서 자동 분할하는 방식을 BFF·클라이언트와 확정해야 한다. [AI 작업용 인수인계](HANDOFF_AI.md)에 작업별 의존 관계와 참고 경로를 적었다.

병행 유지 과제는 [Refine의 작가 blind·실메시 검증](REFINE_V25_PROMOTION_STRATEGY.md), Human-Art 승격 후 피드백, 라이브러리 출처·quarantine·썸네일 검수, 최종 BVH/썸네일 전달 확인이다.

## 유지보수할 때 지킬 것

1. API 필드가 바뀌면 `api/models.py`·[API 계약](API_CONTRACT.md)·BFF 매핑을 같이 고친다. `src/schema.py` 어휘 변경은 VLM 프롬프트와 색인 재태깅에 영향을 준다.
2. `src/features.py`의 피처 표현을 바꾸면 쿼리와 라이브러리 양쪽에 적용하고 feature version·DB 재빌드를 확인한다. 검색 품질 변경은 고정 입력의 전후 결과와 악화 사례를 남긴다.
3. 새 라우터의 VLM 의미는 가설이고, 숨은 관절은 관측 기하가 아니다. 의미 결과로 기존 confidence나 refine 자격을 올리지 않는다. 실험 플래그가 off일 때 기존 결과를 보존한다.
4. 라이브러리 DB·BVH·썸네일·quarantine 정책은 같은 버전으로 검증한다. `.env`, 키, 라이선스 제한 BVH/모델을 Git에 넣지 않는다. 배포는 `/healthz`와 [QA·릴리스 기준](QA_SECURITY_RELEASE.md)을 따른다.
5. 손은 최종 리그의 좌우·손목을 검증하고, 다중 컷은 `cut_id`·순서·부분 실패를 보존한다. 자연어 검색은 pinned build와 `/healthz`의 semantic 준비 상태를 확인한다.
6. 이 작업 폴더에는 미커밋·미추적 파일이 많다. 인계·PR 전에 `git status`와 필요한 자산의 manifest/hash를 확인한다. 예전 계획은 [archive](archive/)에서 당시 기록으로만 읽는다.

## 최소 확인 명령

```bash
.venv/bin/python tests/test_smoke.py
.venv/bin/python tests/experimental/test_rough_router.py
.venv/bin/python -m compileall -q src api scripts
git status --short --branch
```

실데이터·semantic build가 필요한 검증은 [QA 문서](QA_SECURITY_RELEASE.md)의 artifact-backed 환경에서 별도로 실행한다.
