# Semantic Search API · BFF 핸드오프

> 기준 브랜치: `codex/semantic-search-inference-v1`
>
> 작성일: 2026-08-18
>
> 대상: 추론 API 운영자, 앱 서버/BFF 구현자
>
> 현재 상태: API 계약 구현 완료 · 1,552개 후보 라이브러리는 staging이며 production 승격 전

## 1. 서비스 경계

이 저장소의 추론 서버는 내부 서비스다. 사용자 인증, 권한, 사용량 제한, 요청 ID,
외부 API 버전, 작업 저장은 BFF가 담당한다. 추론 서버를 인터넷에 직접 공개하지 않는다.

| 책임 | 추론 API | BFF |
|---|---|---|
| 자연어 포즈 검색과 근거 판정 | 담당 | 결과를 그대로 해석·표시 |
| BVH·썸네일 원본 제공 | 내부 상대 경로 제공 | 외부 URL로 프록시 |
| 인증·사용자·프로젝트·저장 | 미담당 | 담당 |
| rate limit·request ID·감사 로그 | 내부 동시성 제한만 담당 | 담당 |
| 외부 버전 경로 | 현재 없음 | 예: `/v1/semantic-search` |

추론 서버의 OpenAPI 원본은 `/docs`와 `/openapi.json`이다. BFF DTO를 바꿀 때는
문서 예시보다 이 스키마를 우선한다.

## 2. 핵심 HTTP 계약

### 검색

`POST /semantic-search`

```json
{
  "query": "왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세",
  "top_k": 5,
  "view_hint": "three_quarter"
}
```

- `query`: 필수, 공백 정규화 후 1~500자. 제어 문자는 거부한다.
- `top_k`: 기본 5. 요청 스키마 상 1~100이지만 서버 운영 상한은
  `SEMANTIC_TOP_K_MAX`이며 기본 20이다.
- `view_hint`: 선택, `front | three_quarter | side | back`.
  썸네일 시점만 정하며 검색의 exact 판정에는 관여하지 않는다.

응답 최상위의 중요한 필드는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `status` | 제품에서 처리할 검색 상태 |
| `exact_match_status` | exact 여부: `exact | library_gap | not_evaluated` |
| `semantic_build_id` | 결과를 만든 불변 인덱스 식별자 |
| `results` | 같은 mirror 단위를 중복 제거한 Top-K 후보 |
| `gap_reason` | 라이브러리 공백 사유 |
| `clarification_question` | 사용자에게 다시 물을 문장 |
| `cache_hit`, `service_time_ms` | 운영 진단용 |
| `warnings` | UI/BFF가 지켜야 할 안전 경고 |

`status`는 HTTP 오류가 아니라 정상적인 제품 상태를 포함한다.

| `status` | BFF/UI 처리 |
|---|---|
| `success` | 후보를 일반 검색 결과로 표시 |
| `contextual_candidates` | 정확 포즈가 아닌 **맥락 후보**임을 표시 |
| `library_gap` | 200 정상 응답. 정확 후보 없음 UI와 대체 검색 유도 |
| `clarification_required` | `clarification_question`을 사용자에게 표시 |

각 `results[]`에서 반드시 보존할 필드는 다음과 같다.

- `semantic_unit_id`: 원본과 미러를 묶는 방향 중립 그룹 ID
- `pose_id`: 실제 다운로드할 구체 포즈 ID
- `variant_kind`: `original | mirrored`
- `source_clip_id`: 출처 클립 추적용 ID이며 포즈의 고유 키가 아님
- `evidence_state`: `observed | contextual`
- `exact_pose_claim`: 관절 제약을 실제 관측해 exact라고 말할 수 있는지
- `retrieval_score`: dense+lexical RRF **순위 점수**이며 확률/신뢰도가 아님
- `bvh_url`: 선택한 구체 포즈의 내부 상대 경로
- `thumbnail_url`: 번들에 파일이 있을 때만 제공되며 `null`일 수 있음
- `refine_allowed`: semantic 후보에서는 항상 `false`

`observed + exact_pose_claim=true`도 동작명이나 전통 춤 같은 스타일이 사람 검수를
마쳤다는 뜻은 아니다. 관절·방향 제약이 라이브러리 포즈에서 관측됐다는 뜻이다.
`contextual + exact_pose_claim=false`는 파일명·출처 맥락 후보이므로 exact로 표시하면 안 된다.

### 상태 확인

`GET /healthz`

별도 `/healthz.semantic` 엔드포인트는 없다. 응답의 `semantic` 객체에서 다음을 확인한다.

- `enabled`, `required`, `ready`, `reason`
- `semantic_build_id`, `semantic_db_schema_version`
- `semantic_unit_count`, `pose_member_count`
- `embedding_version`, `embedding_model`, `embedding_revision`
- `max_concurrency`, `cache_size`, `stats`

`SEMANTIC_REQUIRED=0`이면 semantic이 준비되지 않아도 geometry 서비스 때문에 최상위
`ok=true`일 수 있다. 따라서 BFF의 semantic 기능 게이트는 반드시
`health.semantic.ready`를 사용해야 한다.

### BVH와 썸네일

- `GET /pose/{pose_id}/bvh`: 성공 시 `application/octet-stream`
  - `404`: DB에 없는 `pose_id`
  - `409`: DB에는 등록됐지만 BVH 파일이 없음
- `GET /pose/{pose_id}/thumbnail?view=front`
  - 성공 시 PNG, `Cache-Control: private, max-age=86400`
  - `400`: 지원하지 않는 view
  - `404`: 포즈 또는 썸네일 파일 없음

`pose_id`는 공백 등 특수 문자를 포함할 수 있는 opaque ID다. BFF는 이를 파싱하거나
정규화하지 말고 URL path segment로 인코딩해야 한다. 응답의 상대 URL은 BFF의 외부
주소로 변환하되, `thumbnail_url=null`이면 임의 URL을 만들지 않는다.

## 3. 오류와 재시도

| HTTP | `detail.code` | 처리 |
|---|---|---|
| 422 | `invalid_semantic_query` | 사용자 입력 오류 |
| 422 | `semantic_top_k_exceeded` | BFF 요청 상한 수정 |
| 503 | `semantic_not_ready` | 기능 비활성/배포 불일치. 재시도보다 기능 fallback |
| 503 | `semantic_busy` | 일시 과부하. `Retry-After: 1`, 제한된 재시도 가능 |

FastAPI/Pydantic 자체 422 형식도 처리해야 한다. `semantic_busy` 외에는 자동 재시도를
기본으로 두지 않는다. 추론 서버는 무제한 대기열 대신 기본 동시 실행 2개, 대기 250ms
후 요청을 거절하도록 설계됐다. BFF도 무제한 큐나 연쇄 재시도를 만들면 안 된다.

성공 응답에는 다음 진단 헤더가 포함된다.

- `Server-Timing: semantic;dur=...`
- `X-Standin-Timing-Kind: semantic-runtime`

## 4. BFF가 반드시 지킬 규칙

1. `retrieval_score`를 퍼센트나 신뢰도로 표시하지 않는다. 서로 다른 build 간 점수도
   비교하지 않는다.
2. `status`, `evidence_state`, `exact_pose_claim`, `warnings`를 버리지 않는다.
3. API가 한 `semantic_unit_id`당 구체 member 하나를 선택하므로 BFF에서 다시
   `source_clip_id` 기준 중복 제거하지 않는다. 같은 클립의 여러 포즈는 정상이다.
4. 미러 후보이면 `variant_kind=mirrored`와 반환된 `pose_id`를 그대로 보존하고 그
   `pose_id`의 BVH를 받는다.
5. semantic 후보는 `refine_allowed=false`다. `/refine`으로 전달하지 않는다.
   이미지 기반 `/analyze` 후보의 refine 가능성과 혼합하지 않는다.
6. geometry의 거리 점수와 semantic의 `retrieval_score`를 같은 척도로 합치지 않는다.
7. 검색 선택 기록에는 최소한 `semantic_build_id`, `semantic_unit_id`, `pose_id`,
   `variant_kind`, `evidence_state`, `exact_pose_claim`을 저장한다.
8. 사용자 인증·rate limit·request ID·감사 로그는 BFF에서 적용한다.

## 5. 배포자가 반드시 지킬 원자성

semantic 인덱스만 교체하면 안 된다. 아래 자산은 한 release 단위로 함께 배포한다.

1. geometry SQLite DB
2. DB가 가리키는 BVH 디렉터리
3. semantic build 디렉터리와 manifest
4. pinned E5 모델과 embedding profile

현재 runtime은 semantic manifest가 참조한 geometry DB와 실제 `DB_PATH`가 같은
release인지 자동으로 교차 검증하지 않는다. 서로 다른 버전을 섞으면 검색은 새
`pose_id`를 반환하지만 `/pose/{pose_id}/bvh`는 404가 될 수 있다. 배포 전 모든
semantic member의 `pose_id`가 geometry DB에 존재하고 BVH 경로가 실제 파일인지
검사해야 한다.

production 권장 설정은 다음과 같다.

```bash
APP_ENV=production
SEMANTIC_ENABLED=1
SEMANTIC_REQUIRED=1
SEMANTIC_BUILD_DIR=/absolute/release/path/semantic/build
DB_PATH=/absolute/release/path/poses.db
DATA_DIR=/absolute/release/path/data
POSE_LIBRARY_VERSION=<immutable-version>
```

- production에서는 `SEMANTIC_BUILD_DIR`를 명시해야 한다.
- semantic manifest의 `production_ready`가 `true`여야 한다.
- `SEMANTIC_REQUIRED=1`이면 준비 실패를 startup/readiness 실패로 처리한다.
- 부분 장애 격리가 필요한 환경만 `SEMANTIC_REQUIRED=0`을 사용한다.

## 6. 2026-08-18 라이브러리 상태

`data/`는 Git에서 제외된다. 브랜치를 checkout해도 BVH, SQLite, embedding은 내려오지
않으므로 API/BFF 코드 배포와 별도로 승인된 asset bundle을 전달해야 한다.

| 항목 | 현재 후보 값 |
|---|---|
| pose member | 1,552개 |
| mirror semantic unit | 776개 |
| text document/embedding | 3,480개 |
| semantic build ID | `sha256:ae3b9acac52d0b70546631a9f7aa09da61f8ff9e91400bccd2189d0f0c10e895` |
| pose library version | `sha256:458c80672e866506b260a4daebd19e77597874ca2e3bb76778a12eb7f94fafbf` |
| geometry DB SHA-256 | `sha256:a143a7c265b589c2495c6b2cc7cc16bba4929b25e85ee8e3bd8507dacb4e4112` |
| semantic DB SHA-256 | `sha256:4ddfd36e441c4c8634a104d1c5eb6ce4da328dff647a59de6e7217d7a8aaa82d` |
| 관련 회귀 테스트 | 56/56 통과 |
| 승격 상태 | `production_ready=false`, holdout 미실행 |

이 후보는 로컬 staging에만 있다. 현재 `data/poses.db`와 `data/bvh`를 교체하지 않았고,
후보 DB의 BVH 경로도 staging 경로를 가리킨다. production 배포 전 최종 asset 위치로
DB를 다시 빌드하거나 동일 디렉터리 구조를 보존해야 한다.

## 7. 연동 완료 조건

- [ ] BFF가 내부 `/semantic-search`를 인증된 외부 버전 경로로 프록시한다.
- [ ] `/healthz` 최상위 `ok`와 별도로 `semantic.ready`를 검사한다.
- [ ] 네 가지 정상 status를 각각 UI 상태로 처리한다.
- [ ] `semantic_busy`의 `Retry-After`를 보존하고 재시도를 제한한다.
- [ ] `semantic_not_ready`일 때 기존 geometry 기능을 유지한다.
- [ ] mirrored `pose_id`로 실제 BVH 다운로드가 성공한다.
- [ ] `thumbnail_url=null`과 thumbnail 404를 정상 처리한다.
- [ ] 선택 기록에 build/unit/member/evidence 정보를 남긴다.
- [ ] release의 semantic member → geometry DB → BVH 파일 전수 preflight가 통과한다.
- [ ] holdout 승인 후에만 manifest를 `production_ready=true`로 승격한다.
