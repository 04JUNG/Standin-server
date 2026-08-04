# PR #10 main 병합 전 통합 기능요구서

> 기준: `04JUNG/Standin-server` PR #10 `feat/skeleton-extraction-improvement → develop`  
> 작성일: 2026-08-05  
> 대상: 추론 서버, BFF(`Standin-app-server`), 데스크톱 클라이언트(`Standin-client`), AWS 인프라

## 1. 목적

PR #10은 두 기능을 함께 전달한다.

1. **스켈레톤 추출 보완**: VLM 슬롯, 전역 배정, 구조 품질 검사, 부분 마스킹, 슬롯별 crop 재추론, coverage별 거리 판정.
2. **refine**: 사용자가 고른 Top-5 후보를 유효한 사지만 러프에 맞춰 조정하고, 안전하지 않으면 베이스 BVH로 되돌림.

추론 서버 내부 구현과 테스트만으로는 제품 기능이 완성되지 않는다. BFF가 새 품질 신호를 버리거나 클라이언트가 조정 전 BVH를 저장하면 다음 문제가 생긴다.

- 정보가 적어 원시 거리가 작아진 스켈레톤을 UI가 `high`로 잘못 표시한다.
- 저신뢰 후보에 refine을 적용한다.
- refine은 성공했지만 사용자가 저장하는 파일은 기존 베이스 BVH다.
- 조정본이 추론 태스크 로컬 디스크에서 사라져 다운로드가 404가 된다.

이 문서는 위 문제 없이 `develop → main`으로 승격하기 위한 기능 요구사항과 수용 기준을 정한다.

---

## 2. 현재 결론

### 2-1. PR #10 자체를 develop에 병합

가능하다. `/analyze`의 변경은 대부분 필드 추가라 기존 소비자와 하위 호환되고, 현재 브랜치의 스켈레톤·스모크 테스트도 통과했다.

### 2-2. develop에서 main으로 승격

아래 두 단계로 나눈다.

- **스켈레톤 추출 보완 출시**: BFF의 신뢰도 매핑과 클라이언트의 soft/hard fallback 처리가 끝나야 한다.
- **refine 출시**: BFF 프록시, 조정본 영속화, 클라이언트 미리보기·저장 배선까지 끝나야 한다.

refine 통합이 늦어지면 추론 서버를 먼저 main에 넣을 수는 있으나, 배포 환경에서는 반드시 `REFINE_ENABLED=0`으로 둔다. 이 경우 스켈레톤 추출 보완만 출시한다.

> 최소 차단 조건: **현재 BFF의 거리만 이용한 `matchLevel` 계산은 수정하기 전 main 배포하면 안 된다.** 마스킹된 관절 수가 적을수록 평균 거리가 작아지는 특성 때문에 저정보 결과가 `high`로 보일 수 있다.

---

## 3. 추론 서버 계약 요약

### 3-1. `/analyze`에서 새로 소비해야 하는 `people[]` 필드

| 필드 | 값 | 소비 목적 |
|---|---|---|
| `confidence` | `high`, `low` | 최종 인물 단위 신뢰도. UI 신뢰도와 refine 허용의 최상위 기준 |
| `skeleton_state` | `valid`, `partial`, `suspect`, `missing`, `invalid` | 구조 품질과 fallback 사유 |
| `skeleton_source` | `full_image`, `crop_retry`, `none` | crop 복구 결과인지 구분 |
| `coverage_class` | `full`, `reduced`, `sparse`, `insufficient` | 거리 임계값이 적용된 관측 범위 |
| `slot_origin` | `vlm`, `rtm_provisional` | VLM 슬롯에 대응된 인물인지 임시 슬롯인지 구분 |
| `search_stability` | `stable`, `ambiguous`, `unstable`, `not_required`, `not_available` | 부분 관측의 검색 안정성 |
| `distance_metric` | 현재 `pose` 또는 `angle` 계열 | 거리 해석 기준 |
| `rank_distance` | number 또는 null | Top-1 실제 거리 |
| `confidence_threshold` | number 또는 null | 해당 coverage에 적용된 임계값 |
| `valid_limbs` | string[] | 검색에 남은 신체 부위 |
| `refinable_limbs` | string[] | refine이 움직여도 되는 사지 |
| `refine_allowed` | boolean | 현재 인물에 refine을 호출해도 되는가 |
| `keypoints`, `scores` | COCO-17 | refine 입력. `scores`는 구조 마스킹과 안전정책 반영값 |
| `raw_scores` | COCO-17 | 평가·디버깅 전용 원본 점수 |
| `quality_reasons`, `quality_trace` | string[], object | 진단·평가·임계값 보정 자료 |

인물 순서는 최종 `box.x1` 기준 왼쪽에서 오른쪽으로 고정된다. BFF와 클라이언트는 다른 기준으로 다시 정렬하지 않는다.

### 3-2. fallback 의미

| 상태 | 조건 | 제품 동작 |
|---|---|---|
| 정상 | `confidence=high`, 후보 있음 | 일반 Top-5, 조건을 만족하면 refine 가능 |
| soft fallback | `confidence=low`, 후보 있음 | 참고용 Top-5는 표시하되 저신뢰 안내, refine 금지 |
| hard fallback | 후보 없음 | 해당 인물은 자동 후보 없음. 다른 인물의 흐름은 계속 진행 |

`confidence=low`와 `candidates=[]`는 같은 상태가 아니다. BFF와 클라이언트 모두 이 둘을 구분해야 한다.

### 3-3. `/refine` 계약

요청에 다음 값을 그대로 전달한다.

```json
{
  "pose_id": "selected-pose-id",
  "view": "front",
  "keypoints": [[0.0, 0.0]],
  "scores": [0.0],
  "search_distance": 0.21,
  "refine_allowed": true,
  "refinable_limbs": ["left_arm"]
}
```

실제 `keypoints`는 17×2, `scores`는 17개다. 응답의 핵심 필드는 다음과 같다.

- `refined=true`: 조정본 생성 성공. `bvh_url`은 `/refined/{handle}/bvh`.
- `refined=false`: 오류가 아니라 안전 게이트에 의해 베이스를 유지한 정상 결과. `bvh_url`은 `/pose/{pose_id}/bvh`.
- `reason`: 적용·스킵 이유.
- `limbs`, `limb_decisions`: 실제 채택된 사지와 진단값. 기본 사용자 UI가 원문 전체를 노출할 필요는 없다.

### 3-4. 현재 계약의 공백

- 추론 서버의 `/export-order`는 아직 항상 `/pose/{pose_id}/bvh`를 반환한다.
- 실제 제품은 BFF의 `/v1/pose-candidates/:id/export`를 사용하므로, 이번 통합에서는 추론 서버 `/export-order`에 의존하지 않는다.
- 조정본은 현재 추론 컨테이너의 로컬 디스크에만 존재한다.

---

## 4. 추론 서버 기능 요구사항

### INF-01. OpenAPI와 문서 계약 고정 — 필수

- `PersonOut`, `RefineRequest`, `RefineResponse`의 허용값을 코드와 `docs/API_CONTRACT.md`에서 일치시킨다.
- `/analyze` 계약 fixture 또는 schema snapshot 테스트를 추가해 필드 삭제·이름 변경을 감지한다.
- `confidence`, `coverage_class`, `refine_allowed`, `refinable_limbs`는 BFF 계약 필드로 간주한다.
- `quality_trace`는 진단용이므로 내부 구조의 완전한 하위 호환을 보장하지 않아도 된다.

**수용 기준**

- FastAPI OpenAPI에서 위 필드와 `/refine`, `/refined/{handle}/bvh`를 확인할 수 있다.
- 계약 테스트가 `people[]`의 필수 안전 필드 누락을 실패로 잡는다.

### INF-02. 서버측 refine 재검증 — 필수

- BFF가 `refine_allowed=false`를 보내면 반드시 `refined=false`, `reason=skeleton_policy`로 반환한다.
- `keypoints`와 `scores` 길이, `pose_id`, `view`를 검증한다.
- 검색 실패·얽힘 세트·충돌·이동량·관절 제한 등 기존 안전 게이트를 약화하지 않는다.
- 동일 입력의 결과는 멱등적으로 처리한다.

### INF-03. 조정본 전달의 영속성 — refine 출시 전 필수

현재 로컬 파일 handle은 운영상 충분히 안전하지 않다. 인프라의 정상 `desiredCount`는 1이지만 `minHealthyPercent=100`인 롤링 배포 중에는 구·신 태스크가 동시에 존재할 수 있다. 태스크 교체나 헬스체크 재시작에도 파일이 사라진다.

다음 중 하나를 채택한다.

1. **권장: 추론 서버가 조정본을 private S3에 직접 저장**하고 object key를 BFF에 반환한다. 외부 공개 URL은 계속 BFF가 소유한다.
2. `/refine` 한 응답 안에 조정 BVH 바이트를 함께 전달하고 BFF가 즉시 private S3에 저장한다.
3. 임시 운영: `REFINE_ENABLED=0`으로 배포해 refine을 노출하지 않는다.

단순히 ECS 태스크 수를 1로 고정하는 것만으로는 롤링 배포와 재시작 문제를 해결하지 못한다.
또한 BFF가 `POST /refine` 뒤 상대 `bvh_url`을 Cloud Map으로 다시 GET하는 방식만으로는,
롤링 배포 중 두 요청이 서로 다른 태스크에 도달할 수 있어 완전한 해결이 아니다.

### INF-04. 배포 설정 명시 — 필수

- production에서 `REFINE_ENABLED`를 명시한다. 코드 기본값에 의존하지 않는다.
- P2 이동량 게이트의 현재 정책(`REFINE_MOVE_GATE=0`)과 P3 충돌 게이트(`REFINE_COLLISION_GATE=1`)를 명시한다.
- `POSE_BACKEND=rtmlib`, 실제 VLM, 실제 라이브러리 번들로 `/healthz`와 분석 smoke test를 실행한다.
- 기존 DB를 재빌드하지 않는 경우 `feature_version=1`, feature shape 34, BVH 경로·thumbnail 존재를 배포 전 검증한다.

---

## 5. BFF 기능 요구사항

### BFF-01. 추론 응답 타입 확장 — 필수

`src/inference.ts`의 upstream `CutResult` 타입에 §3-1 필드를 추가한다. 필드가 아직 없는 구버전 추론 서버와 순차 배포할 수 있도록 첫 배포에서는 optional로 받고 안전한 기본값을 사용한다.

안전한 기본값은 다음과 같다.

- `confidence` 누락: `low`
- `refine_allowed` 누락: `false`
- `coverage_class` 누락: `insufficient` 또는 미확인 상태
- `refinable_limbs` 누락: 빈 배열

### BFF-02. 거리 기반 `matchLevel` 보정 — main 전 필수·최우선

현재 BFF의 `matchLevelFromDistance`는 `distance <= 0.25`를 무조건 `high`로 만든다. 이 규칙은 coverage가 다른 쿼리에 단독 적용하면 안 된다.

최소 규칙:

```text
person.confidence != high  → 모든 후보 matchLevel=low
person.confidence == high → 기존 거리 구간을 표시용 세부 등급으로 사용 가능
```

권장 규칙:

- UI의 1급 신뢰도는 추론 서버의 `person.confidence`를 사용한다.
- `distance`, `confidence_threshold`, `coverage_class`는 개발자 진단용으로 유지한다.
- 서로 다른 `coverage_class`의 raw distance를 같은 절대 구간으로 비교하거나 정렬하지 않는다.

### BFF-03. 공개 분석 결과에 fallback 상태 보존 — 필수

클라이언트에 최소 다음 필드를 전달한다.

```ts
type FallbackMode = "none" | "soft" | "hard";

interface AnalysisPerson {
  personIndex: number;
  confidence: "high" | "low";
  skeletonState: "valid" | "partial" | "suspect" | "missing" | "invalid";
  skeletonSource: "full_image" | "crop_retry" | "none";
  coverageClass: "full" | "reduced" | "sparse" | "insufficient";
  fallbackMode: FallbackMode;
  refineAllowed: boolean;
  refinableLimbs: string[];
  candidates: PoseCandidate[];
}
```

`fallbackMode`는 BFF가 다음처럼 계산한다.

```text
candidates.length == 0                 → hard
candidates.length > 0 && confidence=low → soft
그 외                                  → none
```

`raw_scores`와 `quality_trace`는 기본 사용자 응답에 넣지 않는다. 평가·운영 로그나 관리자 API에만 남긴다.

### BFF-04. refine 입력을 서버측에 보관 — refine 출시 전 필수

클라이언트가 COCO-17 좌표와 안전정책을 임의로 되돌려 보내게 하지 않는다. BFF가 `/analyze` 응답에서 다음 값을 job/person별로 보관한다.

- `keypoints`, `scores`
- `confidence`, `skeleton_state`, `skeleton_source`, `coverage_class`
- `refine_allowed`, `refinable_limbs`
- 후보별 `pose_id`, `view`, `distance`

구현은 `analysis_people`의 JSONB 컬럼 또는 별도 `analysis_refine_context` 테이블 중 하나를 사용한다. 공개 `AnalysisResult`와 내부 refine context는 분리한다.

### BFF-05. 선택 후보 refine 프록시 — refine 출시 전 필수

권장 공개 API:

```http
POST /v1/analysis/jobs/{jobId}/people/{personIndex}/refine
Content-Type: application/json

{ "candidateId": "..." }
```

BFF 처리 순서:

1. 현재 installation/user가 job에 접근 가능한지 확인한다.
2. `candidateId`가 해당 person의 실제 Top-5인지 검증한다.
3. DB에 저장한 refine context와 후보 정보를 읽는다.
4. `refine_allowed=false`면 추론 호출 없이 정상 skip 결과를 만든다.
5. 허용된 경우에만 추론 서버 `POST /refine`을 호출한다.
6. `refined=false`를 오류로 처리하지 않는다.
7. 결과와 최종 export 대상의 연관관계를 저장한다.

권장 응답:

```json
{
  "jobId": "job-id",
  "personIndex": 0,
  "candidateId": "candidate-id",
  "refined": true,
  "reasonCode": "ok_partial",
  "adjustedLimbs": ["left_arm"],
  "exportUrl": "/v1/pose-candidates/pose-id/export?jobId=...&personIndex=0&candidateId=..."
}
```

### BFF-06. 조정 결과 저장과 export 일원화 — refine 출시 전 필수

- 현재 공개 export URL(`/v1/pose-candidates/:id/export`)은 유지한다.
- job/person/candidate에 유효한 조정본이 있으면 조정본을, 없으면 베이스 BVH를 반환한다.
- 조정본은 private S3에 저장한다. 기존 `betaData` 버킷을 사용한다면 입력 이미지와 다른 prefix를 사용하고 KMS 암호화·90일 lifecycle을 그대로 적용한다.
- BFF는 추론 서버의 로컬 handle이 아니라 S3 object key를 최종 artifact 식별자로 저장한다.
- 저장 레코드는 최소 `job_id`, `person_index`, `candidate_id`, `pose_id`, `refined`, `reason`, `object_key`, `created_at`을 가진다.
- 조정본 조회 실패 시 베이스로 안전하게 전환하되, 실제로 조정본을 저장했다고 거짓 성공 기록을 남기지 않는다.
- export analytics에는 `variant=refined|base`와 fallback 사유를 기록한다.

### BFF-07. 오류·타임아웃 정책 — 필수

- `/refine`은 약 1초 내외의 동기 호출로 시작하되 명시적 timeout을 둔다.
- timeout, 5xx, 조정본 저장 실패 시 사용자 작업 전체를 실패시키지 않고 베이스 BVH로 전환한다.
- 404 unknown pose, 409 BVH missing, 422 invalid contract는 운영 오류로 기록하고 사용자에게는 재시도/베이스 사용 가능한 메시지로 변환한다.
- 동일 job/person/candidate 요청은 멱등 처리해 중복 조정과 중복 S3 객체 생성을 막는다.

### BFF-08. DB·문서·관측성 — 필수

- DB migration은 재실행 가능해야 한다.
- `docs/API.md`와 타입 정의를 새 공개 계약에 맞춘다.
- 최소 지표: `soft_fallback_rate`, `hard_fallback_rate`, `crop_retry_rate`, `refine_requested`, `refine_applied`, `refine_skipped`, `refine_failed`, `refined_exported`.
- `quality_reasons`와 `reasonCode`는 개인정보가 아닌 코드형 값 중심으로 기록한다.

---

## 6. 클라이언트 기능 요구사항

### FE-01. 분석 타입 확장 — 필수

`PersonResult`에 `confidence`, `skeletonState`, `skeletonSource`, `coverageClass`, `fallbackMode`, `refineAllowed`, `refinableLimbs`를 추가한다. 신규 필드가 없는 구 BFF 응답은 low/hard 쪽으로 안전하게 해석한다.

### FE-02. soft/hard fallback UX — main 전 필수

- **soft fallback**: Top-5를 계속 보여주되 “스켈레톤 인식이 불확실해 참고용 후보입니다” 안내를 표시한다.
- soft fallback 후보는 선택·베이스 BVH 저장이 가능하지만 refine은 호출하지 않는다.
- **hard fallback**: 해당 인물에 자동 후보가 없음을 표시하고, 다른 인물의 후보 선택·저장은 막지 않는다.
- 사람 번호는 서버 `personIndex`를 그대로 사용한다. 화면에서 탐지 순서로 다시 번호를 매기지 않는다.
- raw `distance`와 `quality_trace`는 일반 사용자에게 노출하지 않는다.

### FE-03. 선택 후 refine 호출 — refine 출시 전 필수

- 사용자가 후보를 선택한 뒤, 저장 전에 BFF refine API를 호출한다.
- 여러 인물이면 `refineAllowed=true`인 선택만 제한된 병렬로 호출한다.
- `refined=false`는 정상 결과로 처리하고 베이스 흐름을 계속한다.
- 호출 실패·timeout에도 베이스 포즈를 유지하고 저장 흐름을 중단하지 않는다.
- 일반 앱과 플로팅 바가 같은 service/hook을 사용해 동작 차이가 없게 한다.

### FE-04. 조정본 확인 — refine 출시 전 필수

사용자가 선택한 포즈와 실제 저장되는 포즈가 달라질 수 있으므로 저장 전에 조정 결과를 확인할 수 있어야 한다.

- BVH/3D 미리보기가 있으면 BFF가 확정한 export URL로 다시 로드한다.
- 현재 화면이 정적 thumbnail만 지원한다면 조정본을 볼 수 있는 preview를 먼저 추가하거나 refine을 비활성화한다.
- 적용 시 “러프에 맞춰 조정됨” 배지를 표시한다.
- 스킵 시 raw reason 대신 “안전하게 원본 포즈를 유지했습니다”처럼 사용자용 문구를 표시한다.

### FE-05. 저장·드래그 앤 드롭 배선 — refine 출시 전 필수

- `useSaveFlow`는 BFF의 안정적인 export URL만 다운로드한다.
- 클라이언트가 추론 서버의 `/refined/{handle}`를 직접 저장하거나 장기 보관하지 않는다.
- 저장 성공은 로컬 파일 쓰기가 실제 완료된 뒤에만 기록한다.
- 저장된 조정본도 기존 BVH와 HIERARCHY가 같으므로 CSP 축 보정·네이티브 drag 로직은 변경하지 않는다.
- 조정본 다운로드 실패 시 BFF가 제공하는 베이스 fallback을 사용하고 사용자에게 결과를 알린다.

---

## 7. 인프라·배포 요구사항

### OPS-01. refine artifact 저장소 — refine 출시 전 필수

- 조정본은 공개 포즈 라이브러리 버킷에 넣지 않는다.
- 사용자 입력에서 파생된 private artifact로 취급한다.
- 기존 KMS 암호화 `betaData` 버킷을 재사용하거나 별도 private output bucket을 만든다.
- 선택한 전달 방식에 맞춰 inference task에는 write, BFF task에는 read 권한을 주거나, BFF가 저장한다면 BFF에만 read/write 권한을 준다. 모두 output prefix로 최소화한다.
- lifecycle 만료, 사용자 삭제 요청, job 삭제 시 artifact 처리 정책을 맞춘다.

### OPS-02. feature flag — 필수

추론 서버와 BFF에 별도의 rollout flag를 둔다.

- 추론: `REFINE_ENABLED`
- BFF 노출: 예시 `REFINE_FEATURE_ENABLED`

추론 endpoint가 존재해도 BFF flag가 꺼져 있으면 클라이언트에 refine을 노출하지 않는다.
BFF가 응답의 capability로 refine 사용 가능 여부를 알려 주고 클라이언트는 이를 따른다.
영속화와 미리보기가 끝나기 전 production 기본값은 off다.

### OPS-03. 배포 순서 — 필수

1. BFF를 구·신 추론 응답 모두 받을 수 있게 배포한다. refine flag는 off.
2. 클라이언트에 fallback UX와 optional 신규 타입을 배포한다. refine flag는 off.
3. PR #10 추론 서버를 배포한다. `REFINE_ENABLED=0`으로 스켈레톤 보완만 smoke test한다.
4. 조정본 영속화와 BFF export를 staging에서 검증한다.
5. preview·저장 흐름까지 확인한 뒤 추론/BFF refine flag를 켠다.

---

## 8. 통합 수용 테스트

아래 시나리오를 BFF 공개 API부터 클라이언트 저장까지 검증한다.

| ID | 입력/상태 | 기대 결과 |
|---|---|---|
| E2E-01 | `valid + full + high` | Top-5 표시, refine 허용, 조정본 또는 안전한 베이스 저장 |
| E2E-02 | refine 안전 게이트 탈락 | HTTP 성공, `refined=false`, 베이스 저장 |
| E2E-03 | `low`지만 후보 존재 | soft fallback 안내, 후보 선택 가능, refine 금지 |
| E2E-04 | `missing/invalid`, 후보 없음 | hard fallback 안내, 다른 인물 흐름 유지 |
| E2E-05 | crop 재추론으로 복구 | `skeletonSource=crop_retry`, low 유지, refine 금지 |
| E2E-06 | 여러 인물 | 원본의 왼쪽→오른쪽 personIndex와 UI·저장 결과 일치 |
| E2E-07 | 얽힘 세트 | refine 스킵, 베이스 제공, 오류로 표시하지 않음 |
| E2E-08 | 추론 태스크 교체 후 export | S3/BFF 소유 조정본이 계속 다운로드됨 |
| E2E-09 | 조정본 저장/조회 실패 | 베이스로 전환, analytics에 fallback 기록 |
| E2E-10 | 일반 앱과 플로팅 바 | 후보·경고·refine·저장 결과가 동일 |
| E2E-11 | thumbnail 없음 | 이미지 표시만 실패하고 BVH 선택·저장은 가능 |
| E2E-12 | 구 추론 ↔ 신 BFF 순차 배포 | 신규 필드 누락을 low/refine-off로 안전 처리 |

성능 기준은 고정 평가셋으로 측정한다.

- full-image 정상 경로는 포즈 추론 1회만 수행한다.
- crop retry는 실패 슬롯에만 최대 1회 수행한다.
- `/analyze` p50/p95, crop retry 비율, `/refine` p50/p95를 기록한다.
- 3컷 샘플뿐 아니라 고정 평가셋에서 사람 수, 슬롯 배정, fallback, Top-5를 회귀 비교한다.

---

## 9. main 승격 체크리스트

### 스켈레톤 추출 보완을 켜기 위한 최소 조건

- [ ] 추론 서버 전체 테스트와 production backend smoke test 통과
- [ ] BFF upstream 타입에 신규 필드 반영
- [ ] BFF `matchLevel`이 `person.confidence=low`를 high/medium으로 승격하지 않음
- [ ] BFF·클라이언트가 soft fallback과 hard fallback을 구분
- [ ] 다인물 personIndex가 왼쪽→오른쪽으로 유지됨
- [ ] API 문서와 실제 OpenAPI 일치

### refine까지 켜기 위한 추가 조건

- [ ] BFF가 refine context를 서버측에 보관
- [ ] BFF refine 프록시에서 후보·job 소유권 검증
- [ ] `refined=false`를 정상 베이스 fallback으로 처리
- [ ] 조정본을 private S3 또는 동등한 영속 저장소에 저장
- [ ] 기존 BFF export URL이 조정본/베이스를 올바르게 선택
- [ ] 클라이언트가 저장 전 실제 조정본을 확인
- [ ] 일반 앱·플로팅 바 모두 조정본을 저장하고 네이티브 drag 가능
- [ ] 롤링 배포·태스크 재시작 후에도 조정본 export 성공
- [ ] E2E-01~12 통과

체크가 끝나지 않았으면 production의 refine flag는 off로 유지한다.

---

## 10. 이번 통합에서 변경하지 않아도 되는 것

- CSP 쪽 BVH 축 보정·미러·drag 계약: 조정본은 원본 HIERARCHY와 채널 순서를 유지한다.
- 포즈 라이브러리의 action/view 태그 체계: 검색의 핵심은 동일 기하 피처다.
- 클라이언트의 raw COCO-17 렌더링: 일반 사용자 UI에는 필요 없다.
- refine을 별도 비동기 Job으로 만드는 작업: 현재 연산량에서는 동기 프록시와 timeout으로 충분하다.
- 검색이 실패한 컷에 refine을 강제로 적용하는 기능: 금지 상태를 유지한다.

---

## 11. 권장 작업 분할

| PR | 저장소 | 범위 | 의존성 |
|---|---|---|---|
| A | Standin-app-server | 신규 타입, confidence cap, fallbackMode, 문서·테스트 | 없음. 가장 먼저 가능 |
| B | Standin-client | fallback UX와 신규 타입 | A의 공개 계약 |
| C | Standin-app-server + infra | refine context, 프록시, private artifact 저장, export 선택 | PR #10 endpoint |
| D | Standin-client | 선택 후 refine, preview, 저장·플로팅 바 연결 | C |
| E | 통합 | staging E2E와 flag 활성화 | A~D |

가장 안전한 순서는 **A → B → PR #10 배포(refine off) → C → D → E**다.
