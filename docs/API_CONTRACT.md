# API 계약 — 도원 추론 서버 (FastAPI)

## 머리·어깨~머리 기본 모델 일시 중단 (2026-10-21)

추가 실험 전까지 `src/basic_closeup.py`의 `BASIC_CLOSEUP_ENABLED=False`로 고정한다.
새 분석에서 `head`/`shoulder_head` 기본 중립 모델로 우회하던 분기를 실행하지 않고,
기존 검색·차단·빈 결과·생성 자격 검사 흐름을 그대로 따른다. 반신으로 강제 대체하거나
검색/생성 성공을 보장하지 않는다. 범위 판별·사용자 수동 범위·Converter 크롭·안전 검사는 변경하지 않는다.
이미 저장된 작업의 기본 모델 미리보기 차단은 별도 BFF 변경이 필요하다.
실험 중인 수동 기본 모델 저장 기능·새 자산·환경변수·AWS 변경은 포함하지 않는다.


## 별도 v4 장면 보정 worker

기존 `/analyze` 및 기동 라우팅과 별개인 기본 꺼짐 서비스입니다.
내부 bearer 인증을 사용하는 `POST /scene/plan`과 `POST /scene/build`를 제공합니다.
BFF가 소유 원본·확인한 FBX를 plan으로 보내고, 제안된 등록 포즈를 같은 체형·손으로 변환한 full FBX를
build로 보냅니다. 최종 FBX·PNG와 SHA·인원 수·원본 SHA·plan ID가 반환됩니다.
공개 클라이언트에는 파일 경로·자유 관절·회전 명령 입력을 제공하지 않습니다.

실행 설정, calibration 제약, 배포 전 검증은
[v4 worker 계약](../src/scene_refinement/README.md)을 따릅니다.
현재 inference/Converter 배포 이미지나 ECS 서비스에 이 worker를 추가하는 변경은 포함하지 않습니다.

> 상태: 현재 계약 · 갱신일: 2026-10-17 · 기준 코드: `api/app.py`, `api/models.py`
>
> 이 문서는 **실제 구현된** HTTP 계약(`api/app.py`·`api/models.py`)을 문서화한다.
> `/export-order`의 상세는 별도 문서(`docs/EXPORT_CONTRACT.md`)에 있고, 여기서는 전체 엔드포인트와
> **앱 서버 팀 문서와의 경계·불일치**를 확정한다. 스키마의 단일 소스는 코드다 → FastAPI `/docs`(OpenAPI).

---

## 전신 동적 라우팅 연결 시 주의 (2026-10-16)

`ROUTING_FULLBODY_ENABLED=1`에서는 요청 `POST /analyze`의 multipart 형식은 유지하고,
인물별 새 응답 `people[].dynamic_search`를 읽는다. 세부 파이프라인·기동 조건은
[`FULLBODY_DYNAMIC_ROUTING.md`](FULLBODY_DYNAMIC_ROUTING.md)가 기준이다.

`PARALLEL_POSE_DESCRIPTION=1`(미설정 시 기본값 `1`)은 기존 인물별 자세 설명 호출을
관절 처리와 겹쳐 실행한다. 입력·전체 인물 순서가 정확히 일치할 때만 채택하며, 검색/검증 정책은
유지한다. 활성화한 HTTP 응답의 `search_routing.describe.prefetch`
(내부 `CutResult.routing_summary.describe.prefetch`)에는 `status`,
`call_elapsed_ms`(실제 설명 시간 또는 null), `join_wait_ms`(합류 대기 시간)가 추가된다.
`status=used|used_error`는 같은 요청 결과 채택(오류 포함), `discarded_request_changed`는
입력 변경으로 기존 경로 실행, `skipped_*`는 선행 실행 미사용, `join_failed|cancelled`는
합류 실패/취소다. 기존 `describe.status/error_reason/retryable` 실패 구분은 유지한다.
`PARALLEL_POSE_DESCRIPTION=0`은 기존 순차 경로로 되돌리는 롤백이며 선행 실행 진단 필드도 추가하지 않는다.
활성화해도 요청 형식·후보/내보내기 권한은 바꾸지 않는다. 환경에 명시된 `0`은 기본값보다 우선한다.
`ROUTING_FULLBODY_ENABLED`는 이 기능 브랜치의 API 팩토리에서 변수가 없을 때도 1이므로,
기존 mock/레거시 개발 환경은 명시적으로 0을 설정해야 한다. `.env.example`은 0으로 유지한다.
이는 기본값 1인 위 병렬 옵션과 별개다. PR 생성은 배포가 아니다.

- `selected`: 일반 경로에서는 핵심 동작 및 확인된 세부 조건을 통과한 전신 라이브러리 후보, 최대 5개.
  통합 체형 opt-in 경로에서 조건 통과 후보가 0개면 기존 생성 시도 이후 기하 참고를 반환할 수 있다.
  이 경우 `match_level=geometry_fallback`, `review_required=true`, `not_exact_match=true`,
  `constraints_satisfied=false`를 반드시 함께 읽어 **조건 통과 후보와 구분**한다.
  인물 소유권·유효 관절·기하 거리 상한과 명확한 자세(서기/앉기/눕기) 조건은 유지한다.
  팔/물건 양손 수 등 나머지 조건을 만족했다는 뜻이 아니다. 원래 조건 감사 결과는
  `core_conditions.whole_pose_validation`, 완화 범위는 `geometry_review_fallback`에 보존한다.
  자동 refine/export 권한은 없다. off/legacy 및 기존 조건 통과 후보 목록은 바뀌지 않는다.
- `near_candidates`: 핵심 동작은 맞지만 다리/무릎 세부가 다르거나 미확인인 별도 검수 참고.
  정답·확률·성공 5개 채우기용 후보가 아니며, 빈 `selected`의 생성 시도를 막지 않는다.
- `pose_detail_fit`: 관측된 2D 몸통 기울기·골반 이동·다리 간격·무릎 각도의 동일 카메라 비교.
- `generation`: 정상 검색이 비었을 때만 실행되는 검수용 자세 JSON→IK→BVH 생성. 검색/설명 오류는
  생성으로 덮지 않는다. 생성/근접 참고는 원본 라이브러리 ID나 자동 export 대상이 아니다.
- `basic_model`: 명시적 어깨~머리 컷의 기본 인체 출력 계획. 검색·생성 성공과 별도다.
- `candidate_validation`, `branches`, `description_evidence`, `geometry_evidence`: 조건·실제 검색문·
  관절 근거·분기별 실패/시간 기록. 기존 `candidates`/`semantic_candidates`와 임의 결합하지 않는다.

`candidate_validation.version=whole-pose-conditions-v3-critical`은 `requirements`에
`relations`(기존 팔짱/V자 관계)와 `knee_support`(`one`/`both`/null)를 추가한다.
기하·자연어 후보와 생성 BVH 재감사 모두 같은 필수 조건을 적용한다. 팔짱은 BVH 해시가
일치하는 검수 shape 프로필과 실제 가슴 앞 교차 형상이 모두 필요하며, 거친 팔 높이 분류는
팔짱/V자 관계를 덮어쓰지 않는다. 전신의 무릎 꿇기는 해시에 연결된 자세 태그와 낮은 굽힌
무릎의 지지 근사를 함께 확인한다. 한 무릎/양 무릎이 명시되면 해당 개수도 필수다.
근거 없음·stale 해시·관측 결측은 통과가 아니다. 상반신/클로즈업에서 안 보이는 무릎 조건은
추가하지 않는다. 이 지지 근사는 실제 바닥·피부 접촉이나 물리 검증을 인증하지 않는다.
선택적 `geometry_review_fallback`도 이 핵심 관계와 무릎 지지 조건을 완화하지 않는다.

이 경로는 Human-Art 관절의 조건부 VLM 검수·보완과 인물별 자세 설명을 사용한다.
아래 레거시 경로의 “VLM은 좌표를 생성하지 않음” 규칙과 달리, 보완 관절은 `vlm_repair` 출처를
명시하고 구조 검사 악화 시 되돌린다. 자동 high 신뢰도·refine 권한으로 승격하지 않는다.
상하체 분리 검색·런타임 BVH 합성은 하지 않는다. 2026-10-17 사용자 요청으로 API 팩토리의
RRF 자연어 비중은 0.60, 기하 비중은 0.40이다. 로컬 set4/set5 가중치 실험과 같은 비중이며,
조건 검사·거리 컷오프·단일 검색 경로의 가중치 1은 유지한다. 과거 비교를 재현하는 직접
`FullBodySearch(...)` 호출의 기본값 0.35는 보존하며, 서버 팩토리가 0.60을 명시한다.
모든 새 후보는 `release_allowed=false`, `refine_allowed=false`, `actual_mesh_validated=false`다.
동적라우팅 API는 실제 FBX 변환 후 반대 손 피부 삼각형 검사를 자동 호출한다.
전체 메시 안전 인증은 아니며, 검사 오류/미확인은 정상 후보와 분리한다(아래 자동 검사 계약).

---

## 전신 후보의 손 피부 자동 검사 (2026-10-17)

전신 API 팩토리는 검색 엔진에 `HandSkinGate`를 연결한다. `/analyze`의 기하/자연어
검색과 조건·순위 계산을 먼저 끝낸 뒤, 원래 순서대로 실제 FBX 검사를 수행한다.
필요한 내부 주소는 기존 `REFINE_THUMBNAIL_CONVERTER_URL`을 재사용한다.

- converter의 `POST /screen-hand-contact`에 BVH 바이트·원문 SHA256·대상 캐릭터 ID/SHA256·
  인물별 실제 손 프리셋을 보낸다. 러프 이미지나 Gemini 설명은 보내지 않는다.
- converter는 기존 동결 solver로 full FBX를 만들고, 요청한 손 프리셋을 굽고, **그 실제 FBX를
  재임포트**하여 변형된 반대 손 피부 삼각형 교차를 검사한다. crop·카메라·미리보기 이전 단계다.
- 입력 BVH·캐릭터·solver·손 프리셋·손 라이브러리·검사한 FBX SHA를 검증한 receipt만 사용한다.
  별도 캐시는 저장하지 않으며 동일 요청 안에서 같은 BVH/손 설정만 재사용한다.
- `clear_screen`: 반대 손 교차 미검출. 해당 후보에 `hand_skin_screen` 근거를 붙여 유지한다.
- `intersection`: 정상 `selected`에서 제외하고 예산 안에서 다음 순위 후보를 확인한다.
- `unavailable`: 누락/변경 BVH, 구버전 converter, HTTP/시간 초과, 스킨 부위 누락,
  receipt 불일치 등. `hand_skin_screen.unverified`에 분리하고 정상 후보로 반환하지 않는다.
  안전 후보가 남으면 `partial_error`, 없으면 `error`이며 **오류를 생성으로 덮지 않는다**.
- 검사 가능한 검색 후보가 모두 교차로 탈락했고 다른 오류가 없다면 기존 빈 검색 생성 분기를
  실행할 수 있다. 생성 후보도 관절·BVH 검증 후 같은 피부 검사를 통과해야 `preview_ready`에 남는다.
- 인물별 검색/생성이 **최대 10회·누적 검사 기본 45초(HTTP 1회 최대 15초)**를 공유한다.
  예산 때문에 확인하지 못한 후보는 미확인이며 통과나 자산 부재로 취급하지 않는다.
  `HAND_SKIN_SCREEN_BUDGET_SECONDS`로 누적 예산을 설정한다(유한한 `(0,60]`초,
  이전 30초 예산으로 복귀 가능). 추가 변환 지연이 있으므로 호출 측 시간 예산에 이 후처리를 포함해야 한다.

`dynamic_search.hand_skin_screen`에는 검사 횟수/예산/제외/미확인 진단이 남는다.
2026-10-19: 단일 Blender 용량을 유지한 필수 검사 우선순위·선택 미리보기 선점,
HTTP disconnect 취소·큐/실행 기한과 안전 timing 진단을 추가한다.
`checks_diagnostics`는 검사 번호·phase/cause·대기/실행·남은 예산의 안전 근거다.
선택적 mirror 검사는 낮은 우선순위이며 이미 통과한 본후보의 검사 스냅샷은 유지한다.
`/convert-framed-export`는 동일 계보 검증의 일반 출력 경로이며 기존 `preview_format=png|model`의
PNG+FBX 또는 GLB+FBX pair를 그대로 지원한다.
기존 `/convert-framed`는 구형 scoped FBX 호출도 포함해 선점 가능하므로 BFF 최종 FBX
호출만 새 경로로 전환해야 한다. Converter→BFF 적용 순서와 오류/재시도/호환 한계는
[HAND_SCREEN_AVAILABILITY.md](HAND_SCREEN_AVAILABILITY.md)를 따른다.

HTTP 실패(`hand_screen_http_error`, 상태 코드만), 요청 시간 초과(`hand_screen_request_timeout`),
네트워크 오류(`hand_screen_network_error`), receipt 형식 오류(`hand_screen_invalid_receipt`),
입력 계보 불일치(`hand_screen_receipt_lineage_mismatch`, 불일치 필드명만),
공통 예산/횟수 소진(`hand_screen_budget_exhausted`/`hand_screen_check_limit_exhausted`)을 구분한다.
내부 URL·예외 원문·receipt의 불일치 값은 진단에 포함하지 않는다.
제외·미확인 생성 진단에 BVH 본문을 새로 넣지 않는다. `near_candidates`는 기존 조건 차이의
검토 참고로 유지하며 피부 검사 통과 후보라고 표시하지 않는다. 기본 머리·어깨 인체도 검사 대상이 아니다.
손·몸통·소품·같은 손 내부, 완전 포함, 공면 접촉을 포함하는 완전한 충돌 검사가 아니므로
`actual_mesh_validated=false`, `refine_allowed=false`, `release_allowed=false`를 유지한다.
피부 검사 단계 자체는 IK/손목 목표/손 프리셋을 수정하지 않는다.

이 연결은 전신 동적라우팅 `/analyze` 대상이다. 레거시 검색, 기존 `/convert*` export 승인,
미병합 직접 문장 API #96은 변경하지 않는다. 다른 캐릭터·손 프리셋·BVH로 바꾸면 기존 receipt를
재사용할 수 없으며 해당 출력의 재검사가 필요하다.

배포는 **converter의 새 검사 endpoint부터 준비한 뒤 inference를 적용**해야 한다.
두 자동 배포가 병렬로 진행되어 converter가 구버전이면 잠시 `error/partial_error`가 날 수 있다.
구버전/장애 시 검사를 생략하고 정상 후보를 내보내는 폴백은 없다. S3/IAM 변경이나 실제 배포는
이 기능 PR 범위 밖이다. `/healthz.dynamic_routing.hand_skin_screen_required`는 연결 여부이며
converter의 실시간 가용성 인증이 아니다.

### 생성 양손 휴대폰의 제한된 손 간격 보정 (2026-10-18)

`/analyze` API 팩토리는 자연어 생성기에 `GeneratedHandSpacingRepair`를 연결한다.
**기존 관절/BVH/관찰 조건을 통과한 새 생성 후보**에서 실제 반대 손 피부 교차가
확인된 경우에만 실행한다. 라이브러리 원본, 참고 후보, 검사 불가, 한손/불명확한
물건 조건은 보정하지 않는다. 관찰 조건에 `hand_counts.phone=two`가 필요하다.

- 마지막 통과한 자세 제안과 저장 BVH의 동일성을 다시 생성한 SHA-256으로 확인한다.
- 양손 `phone`·가슴 프레임 목표만 지원한다. 원래 제안에서 좌우 각 2/4/6/8cm를
  독립적으로 시도한다. 이전 수정에 누적해서 20cm 이동시키지 않는다.
- 모델 재호출 없이 별도 종료 가능한 IK 프로세스를 사용한다. 전체 요청 최대 4회·
  보정 벽시계 30초, #97의 누적 메시 검사 설정 예산(기본 45초)/10회 한도도 함께 지킨다.
  기존 자연어 생성 60초와 별도 후처리 예산이므로 전체 응답은 60초 상한이 아니다.
- 뼈 길이/관절/저장 왕복/독립 안전 검사, 원래 관찰 조건, 실제 손모양을 적용한
  변환 후 피부 검사를 모두 다시 통과한 최소 시도안을 검수용으로 남긴다.
- 뿌리·몸통·머리·다리·손가락의 BVH 채널과 뼈대는 불변이다. 양쪽 어깨/팔/손목의
  IK 채널만 바뀔 수 있다. 명시적 손바닥 목표나 원본 재현 실패는 수정하지 않는다.
- 실패/시간 초과는 성공으로 바꾸지 않는다. 메시 검사 불가는 기존 error 정책을 유지한다.

`generation.diagnostics.hand_spacing_repair`에는 예산·시도·관찰 조건·피부 receipt가,
성공 후보의 `hand_spacing_repair`에는 원본 BVH 해시·`delta_cm`이 남는다.
후보 ID/미리보기 해시는 보정된 새 BVH 기준이다. `pose_presentation.generated_candidates`
의 `validation.hand_spacing_repair`에도 같은 계보를 전달한다. 실패 진단에 새 BVH 본문은 넣지 않는다.
양손 간 교차 미검출은 휴대폰 소품이나 실제 그립·완전한 충돌 인증이 아니다.
`actual_mesh_validated=false`, 자동 적용·refine·export 금지는 유지한다.

## 0. 경계 — 이 서버는 어디에 있나 (반드시 먼저 읽을 것)

전체 제품은 **두 개의 서버 계층**이다. 이 저장소는 그중 **안쪽 한 층**만 담당한다.

```
[Tauri 앱] ──HTTP──> [앱 서버(친구들)] ──HTTP──> [도원 추론 서버 = 이 저장소]
                     · 인증(토큰)                · POST /analyze (동기, 무인증)
                     · Job 큐·상태·폴링           · GET  /pose/{id}/bvh
                     · 사용자·기록 저장            · POST /export-order
                     · /v1 버전 프리픽스           · GET  /healthz
```

핵심: **이 서버는 동기·무인증·무상태 추론 API다.** 인증·Job 비동기·버전 프리픽스(`/v1`)·사용자 관리는
**앱 서버가 감싸서** 제공한다. 그래서 클라이언트 문서(`Standin-client/docs/08_API_CONTRACT.md`)가 그리는
API와 이 서버의 API는 **의도적으로 다르다**(§7에서 대조·확정).

⚠ **이 경계가 어느 문서에도 명시돼 있지 않았던 것이 통합 리스크였다.** 앱 서버가 실제로 이 층을 감싸는지,
아니면 Tauri가 이 서버를 직접 부를지는 팀 확인 항목(§8-1).

---

## 1. 기본 원칙

- 응답은 JSON. base BVH는 바이트 스트림으로 받고, refine 결과 BVH와 PNG preview는
  다중 인스턴스에서도 유실되지 않도록 `/refine` JSON에 인라인한다.
- 시간은 ISO 8601(`/export-order`의 `created_at`).
- ID는 string(`pose_id`, `cut_id`).
- **좌표는 검출기·포즈 모델만 생성**한다(VLM은 개수·의미 태그만) — 태그 어휘는 §4.
- 인덱스·모델은 **기동 시 1회 로드**(lifespan) → 요청마다 재로딩하지 않는다.
- `/analyze`는 동기(`def`) 실행 → FastAPI가 threadpool에서 돌려 이벤트 루프를 막지 않는다.
- **API 키·무거운 모델 없이도 돈다**: mock VLM + 합성 인덱스가 기본값(`CFG.vlm_provider="mock"`).

---

## 2. 엔드포인트 요약

| 메서드 | 경로 | 입력 | 출력 | 소비자 |
|---|---|---|---|---|
| `GET` | `/healthz` | — | `{ok, env, provider, pose_backend, pose_count}` | 앱 서버 기동 확인(비정상 시 503) |
| `POST` | `/analyze` | multipart PNG (+`hint`, `rescue`) | `CutResult` | 앱 서버 → 뷰어 Top-K 표시 |
| `GET` | `/pose/{pose_id}/bvh` | 경로 파라미터 | `application/octet-stream` | 동원 내보내기 |
| `POST` | `/export-order` | `ExportOrderRequest` | `ExportOrder` | base-only legacy 주문서 (→ `EXPORT_CONTRACT.md`) |
| `GET` | `/docs` | — | OpenAPI UI | 사람(계약 확인) |
| `POST` | `/refine` | `RefineRequest` | `RefineResponse` | 앱 서버 → 조정된 BVH와 후보 매칭 시점 preview 이미지 본문(응답 `bvh`, `thumbnail`) |

> `PersonOut`에 **`keypoints`(17×2, 이미지 픽셀)** · **`scores`(17)** 가 포함된다.
> `/analyze`가 이미 추출하는 값이라 연산 추가는 0이며, `/refine`을 순수 함수로 만들기 위한 것이다.
> 클라이언트는 이 두 값을 **그대로 `/refine`에 되돌려주면** 된다 — 러프 재전송·포즈 재추론 없음.

### 2-1. `POST /refine` — 고른 후보 1개를 러프에 맞춰 조정

작가가 Top-K 중 하나를 고른 **직후** 호출한다. 연산은 커밋된 포즈에만 든다.

```jsonc
// 요청
{
  "pose_id": "Sitting Idle_01",
  "view": "front",
  "keypoints": [[120.5, 88.0], ...],   // /analyze의 PersonOut.keypoints 그대로
  "scores":    [0.91, 0.87, ...],      // /analyze의 PersonOut.scores 그대로
  "search_distance": 0.21,             // v1 게이트, v2 진단 lineage
  "refine_allowed": true,              // PersonOut.refine_allowed 그대로
  "refinable_limbs": ["left_arm"],     // PersonOut.refinable_limbs 그대로
  "skeleton_state": "valid",
  "coverage_class": "full",
  "slot_origin": "vlm",
  "skeleton_source": "full_image",
  "search_stability": "not_required",
  "distance_metric": "pos",
  "confidence_threshold": 0.45,
  "gap_type": "unknown",               // 평가 라벨이며 실행 게이트가 아님
  "refine_mode": null                   // 생략/null 권장. 서버 기본은 safe aggressive
}
// 응답
{
  "pose_id": "Sitting Idle_01", "view": "front",
  "refined": true, "reason": "ok_partial",
  "bvh_url": "/pose/Sitting Idle_01/bvh",   // 항상 베이스. 조정본에는 URL이 없다
  "bvh": "HIERARCHY\nROOT Hips\n...\nMOTION\nFrames: 1\n...",  // 조정본 본문(LF)
  "thumbnail": {
    "view": "front",
    "media_type": "image/png",
    "encoding": "base64",
    "data": "iVBORw0KGgoAAA...",
    "width": 256,
    "height": 256,
    "renderer_version": "fbx-anatomical-v1/standin-master-v2"
  },
  "loss_base": 0.599, "loss_final": 0.004, "gain": 0.993,
  "backend": "scipy+numpy",
  "refine_version": "v2.5.3",
  "refine_outcome": "improved",
  "limbs": ["right_arm"],
  "limb_decisions": {
    "left_arm": {
      "accepted": false, "reason": "self_collision",
      "mean_move": 0.083, "endpoint_move": 0.153,
      "collision": {
        "checked": true, "status": "new_penetration",
        "pair": "left_hand:torso", "part": "hand",
        "base_depth": 0.0, "solved_depth": 0.055, "final_depth": 0.0,
        "sample_fraction": 0.75,
        "collision_point": [10.59, -4.66, 3.36]
      }
    },
    "right_arm": {
      "accepted": true, "reason": "ok",
      "mean_move": 0.361, "endpoint_move": 0.549,
      "collision": {
        "checked": true, "status": "clear", "pair": null, "part": null,
        "base_depth": 0.0, "solved_depth": 0.0, "final_depth": 0.0,
        "sample_fraction": null, "collision_point": null
      }
    }
  },
  "diagnostics": {
    "mode_requested": "aggressive",
    "mode_applied": "aggressive",
    "aggressive_attempted": true,
    "aggressive_reason": "ok_partial",
    "hybrid_loss_base": 0.72,
    "hybrid_loss_solved": 0.10,
    "hybrid_loss_adopted": 0.14,
    "block_alphas": {"left_arm": 0.0, "right_arm": 0.75},
    "torso": {"attempted": false, "accepted": false},
    "aggressive_attempted": true
  }
}
```

**`refined: false`는 오류가 아니다.** 안전 게이트가 조정을 버리고 베이스를 준 것이며,
이때 `bvh_url`은 `/pose/{pose_id}/bvh`가 된다. 클라이언트는 `bvh_url`만 따라가면
두 경우를 구분하지 않고 동작한다("좋아지거나, 그대로"). `reason` 값 목록은
`REFINE_DESIGN.md`의 안전 처리 기준.

`thumbnail`은 최종 결과를 **후보 썸네일과 같은 렌더러**로 요청의 `view`에서 그린 256×256
이미지다. 즉 사용자가 고른 후보 썸네일의 방향을 유지하고, 그림체도 같다. 2026-09-04부터
그 렌더러는 converter 서비스의 `POST /render-thumbnail`이다 — 조정 BVH를 V3.2.5
(`chain-transport-v3.2.5`)로 `standin-master-v2`(FBX 남성 모델)에 리타게팅한 뒤
라이브러리 썸네일 빌드와 같은 anatomical 카메라·재질·조명으로 Blender 렌더한다.

- `refined=true` → 조정 BVH를 converter로 그린다. `renderer_version`은
  `fbx-anatomical-v1/<character_id>`.
- `refined=false` → 베이스는 라이브러리 BVH 그대로이므로 번들의 후보 썸네일
  (`thumbs/<pose_id>__<view>.jpg`, 같은 렌더러로 이미 구운 파일)을 돌려준다.
  `renderer_version`은 `fbx-anatomical-v1/library`. 번들에 그 view가 없으면 converter로 그린다.
- `media_type`은 기본 `image/png`이고 서버 설정(`REFINE_THUMBNAIL_FORMAT=jpeg`)으로 JPEG가
  될 수 있다. UI는 `data:{thumbnail.media_type};base64,{thumbnail.data}`로 조립한다
  (`image/png`를 하드코딩하지 말 것).
- `renderer_version`이 `warm-mannequin-v1`이면 옛 2D 마네킹이다. 서버가
  `REFINE_THUMBNAIL_RENDERER=mannequin`으로 비상 복구 중이거나 개발 모드에서 converter
  주소 없이 뜬 경우다. production은 converter 주소가 없으면 기동하지 않는다.

추론 서버에는 이미지나 조정 BVH를 영속 저장하지 않는다. converter 왕복(변환 ~3s + 렌더)은
`REFINE_THUMBNAIL_TIMEOUT_SECONDS`(기본 20초) 안에 끝나야 하며, 넘으면 아래 규칙대로
`thumbnail: null`이다. 즉 `/refine` 전체 지연은 refine 예산(`REFINE_TIMEOUT_SECONDS`)에
이 시간이 더해질 수 있다 — BFF의 `/refine` read timeout은 그 합보다 커야 한다.

**렌더 실패는 오류가 아니다.** 그릴 수 없으면 `thumbnail`을 `null`로 두고 나머지는 그대로 준다
(`refine_thumbnail_failed` 로그만 남는다). 썸네일은 확인 화면이 쓰는 부가 산출물이므로, 그림
때문에 응답을 실패시키면 소비자가 그걸 상류 장애로 읽고 **방금 계산한 조정 결과까지 버린다** —
사용자는 그림 한 장 대신 더 나쁜 포즈를 저장하게 된다. 소비자는 `thumbnail` 없음을 정상으로
다루고 후보 썸네일로 폴백한다.

**렌더 실패는 오류가 아니다.** 그릴 수 없으면 `thumbnail`을 `null`로 두고 나머지는 그대로 준다
(`refine_thumbnail_failed` 로그만 남는다). 썸네일은 확인 화면이 쓰는 부가 산출물이므로, 그림
때문에 응답을 실패시키면 소비자가 그걸 상류 장애로 읽고 **방금 계산한 조정 결과까지 버린다** —
사용자는 그림 한 장 대신 더 나쁜 포즈를 저장하게 된다. 소비자는 `thumbnail` 없음을 정상으로
다루고 후보 썸네일로 폴백한다.

- v1에서 `search_distance`는 베이스 불일치 게이트다. `REFINE_V2_ENABLED=1`에서는 거리·순위만으로
  실행을 막지 않고 진단에 남긴다. 대신 `refine_allowed`, 스켈레톤 상태·coverage·소유권 lineage와
  `refinable_limbs`를 모두 그대로 보내야 하며, 누락·불일치하면 fail-closed한다.
- **조정본은 응답 `bvh` 본문으로만 나간다.** 사용자 preview는 같은 응답의 `thumbnail`이다.
  `bvh_url`은 `refined` 여부와 무관하게 항상 베이스
  `/pose/{pose_id}/bvh`이며 조정본 다운로드 URL은 존재하지 않는다. 추론 서버는 조정본을 저장하지
  않으므로 무상태이고, 재계산을 막는 멱등성은 BFF의 `refined_artifacts` PK가 담당한다
  (`docs/REFINE_HANDOFF.md` §3 4단계).
- `refine_mode`를 생략하거나 `null`로 보내면 서버 기본값을 따른다. 현재 제품 기본은
  **safe aggressive**(`REFINE_DEFAULT_MODE=aggressive`)다. `aggressive`는 같은 hard safety gate
  아래 보수적 단계를 먼저 실행하고 hand/lap/lower pair와 제한적 Foot counter-rotation을 추가
  시도한다. 공격적 단계가 실패하면 보수적 artifact를, 보수적 단계도 실패하면 베이스를 반환한다.
  `REFINE_V2_ENABLED=0`이면 요청값과 무관하게 effective mode는 `conservative`다.
- `refine_outcome`은 `improved | unchanged | reverted | not_attempted`이며 `gap_type`과 섞지 않는다.
  v2의 `diagnostics`에는 전체·부위별 base/solved/adopted direction·position·hybrid 손실,
  3D 이동량, 안전 판정, 부분 채택 alpha, 몸통 local rotation과 버전 lineage가 들어간다.
- `limbs`에는 P3까지 통과해 실제로 조정된 사지만 담긴다. `limb_decisions`는 고려한
  모든 사지의 채택 여부·탈락 사유·몸통 길이로 정규화한 3D 이동량을 담는다.
- P1 solve까지 도달한 결과의 팔에는 `collision`이 추가된다. 동결된 팔도 진단하되
  게이트로 수정하지는 않는다. `base_depth`와 P1 전체 조정본의 `solved_depth`를 비교해
  refine이 새로 만든 깊은 관통만 `new_penetration`으로 판정한다. P3가 해당 팔만 복구하면
  `final_depth`는 베이스 깊이로 돌아가고, 충돌하지 않은 반대팔 조정은 유지된다.
- 전체 `reason`에는 `collision_gate`(충돌 팔 복구 후 남은 사지 없음),
  `collision_unresolved`(복구 후 깊이 불변식 실패)가 포함된다. 사지 reason의
  `self_collision`은 해당 사지만 베이스로 복구됐다는 뜻이다.
- 오류: 없는 `pose_id` → 404 · BVH 파일 미존재·파싱 불가 → 409 · 잘못된 shape 또는
  NaN/Inf·음수 score → 422. solver timeout은 오류 응답 대신 `refined=false`, `reason=timeout`,
  베이스 `bvh_url`로 복구한다.

기본 주소: `uvicorn api.app:app --reload` → `http://127.0.0.1:8000`. **버전 프리픽스 없음**(`/analyze`, `/v1/analyze` 아님).

---

## 3. `POST /analyze` — 컷 1장 → 인물별 Top-K 후보

동기 호출. **한 번의 응답에 최종 결과가 다 온다**(Job·폴링 없음).

### 요청

```http
POST /analyze
Content-Type: multipart/form-data
```

| 폼 필드 | 필수 | 설명 |
|---|:---:|---|
| `file` | ✅ | 러프 콘티 컷 이미지(PNG). `UploadFile`로 수신, PIL로 RGB 로드 |
| `hint` | — | **mock provider 전용 dev 편의**. `VLM_PROVIDER=mock`일 때만 이미지 대용 힌트 문자열로 사용, 실모델이면 무시 |
| `rescue` | — | `auto`/빈 값은 실패 슬롯 자동 rescue, `all`은 모든 슬롯 재검토, `0,2`는 최종 좌→우 인물 index 선택. 형식 오류는 `422 invalid_rescue_selector` |

⚠ 현재 서버는 파일 크기·MIME을 강하게 검증하지 않는다(PIL 로드 실패 시 512×768 더미로 폴백). 업로드 검증은 앞단(앱 서버/Tauri) 책임 — 보강 항목은 §8-4.

### 응답 — `CutResult`

```json
{
  "route": "core",
  "count_confidence": "high",
  "detector_count": 1,
  "vlm_count": 1,
  "people": [
    {
      "index": 0,
      "box": [120.0, 80.0, 360.0, 720.0],
      "tags": { "shot": "full_half", "action": "standing", "view": "front", "relationship": "solo" },
      "skeleton": {
        "schema_version": "coco17-v1",
        "keypoints": [
          [640.0, 120.0], [630.0, 110.0], [650.0, 110.0], [615.0, 120.0],
          [665.0, 120.0], [590.0, 210.0], [690.0, 210.0], [560.0, 320.0],
          [720.0, 320.0], [540.0, 420.0], [740.0, 420.0], [610.0, 410.0],
          [670.0, 410.0], [605.0, 560.0], [675.0, 560.0], [600.0, 700.0],
          [680.0, 700.0]
        ],
        "scores": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
      },
      "confidence": "high",
      "candidates": [
        {
          "pose_id": "stand_solo",
          "view": "front",
          "distance": 0.168,
          "tags": { "shot": "full_half", "action": "standing", "relationship": "solo", "view": "front" },
          "rerank_score": 0.91,
          "bvh_url": "/pose/stand_solo/bvh"
        }
      ]
    }
  ],
  "notes": [],
  "image": { "width": 1280, "height": 720 },
  "inference_metadata": {
    "deployment_version": "git-sha",
    "vlm_provider": "gemini",
    "vlm_model": "gemini-2.5-flash",
    "pose_backend": "rtmlib",
    "pose_model_version": "runtime-default",
    "pose_library_version": "lib-20261002-91b57d0f",
    "feature_version": 1,
    "pose_library_sha256": "91b57d0f…"
  }
}
```

`pose_library_version`은 번들 루트의 `library_manifest.json`에서 온다(내용 해시에서 만든
`lib-YYYYMMDD-<hash8>`, `docs/POSE_LIBRARY_BUNDLE.md`). manifest가 없는 옛 번들이면
env `POSE_LIBRARY_VERSION`(기본 `v1`)이고 `pose_library_sha256`은 `null`이다. 프로덕션에서
manifest가 `poses.db`와 맞지 않으면 기동하지 않는다. `GET /healthz`의 `pose_library`
(`version`·`source`(manifest\|env)·`content_sha256`·`db_sha256`)로 떠 있는 번들을 확인한다.

**최상위 필드**

| 필드 | 타입 | 의미 |
|---|---|---|
| `route` | string | `core`(전신·반신·흉상, 관측 검색) \| `bust`(구 서버의 흉상 검색 스킵) \| `skip`(두상만 있는 컷, 조기 종료) |
| `count_confidence` | string | `high`(검출기 개수 = VLM 개수) \| `low`(불일치 → 저신뢰 폴백) \| `n/a` |
| `detector_count` | int | 검출기가 센 사람 수 |
| `vlm_count` | int | VLM이 센 사람 수 (둘의 일치가 신뢰도 신호 — `CLAUDE.md` 불변식 §2) |
| `people` | Person[] | 인물별 결과. `skip`에도 인물·범위 메타데이터 유지, 후보는 빈 배열. 혼합 컷은 인물별 검색 가능 여부가 다를 수 있음 |
| `notes` | string[] | 폴백 사유 등 사람이 읽는 메모 |
| `image` | object | 분석 기준 원본의 `width`, `height` |
| `inference_metadata` | object | 배포·VLM·포즈 backend/model·포즈 라이브러리·feature schema 버전. `vlm_prompt_version`은 답한 VLM 프롬프트(`p1-scope` \| `p2-person-tags`) |
| `vlm_tags` | object \| null | VLM이 컷에 대해 실제로 말한 `shot`·`action`·`view`·`relationship`. 어휘 밖이거나 없으면 그 칸이 `null`이다(`people[].tags`는 기본값으로 채운다) |

**`people[]` (PersonOut)**

`output_scope`는 `{ "detected": "full" | "half" | "bust" | "head" | null,
"source": "vlm_person" | "legacy_shot" | "observed_legs" | "unknown" }`이다.
`observed_legs`는 VLM이 half/bust로 판별했지만 화면 안에서 무릎이 관측돼 `full`로 올린 값이다
(원래 값은 `quality_trace.output_scope_promotion`). 기존 `shot`과 독립적인
출력 구도 메타데이터이며 검색/refine 정책을 바꾸지 않는다. 판별·BFF 저장·앱 계약은
[BODY_SCOPE.md](BODY_SCOPE.md)에 있다. 실제 부분 미리보기/FBX 크롭은 후속 단계다.

`person_tags`는 `{ "action": Action | null, "view": View | null,
"source": "vlm_person" | "legacy_cut" | "unknown" }`이다(§4 어휘). 그 인물의 행동과 몸이
향한 방향을 VLM이 인물마다 따로 판단한 값이며, 컷 단위 `tags`와 달리 모르면 `null`이다.
라이브러리 공백 분석(`docs/POSE_GAP_LOOP.md`)에 쓰는 기록용이고 검색·라우팅·refine에는
쓰지 않는다(`CLAUDE.md` 불변식 1).

- `vlm_person`: `p2-person-tags` 프롬프트가 인물별로 답했다. VLM이 만든 인물 슬롯에만 붙는다.
- `legacy_cut`: 인물별로 묻지 않은 프롬프트(`p1-scope`)에서 사람이 1명인 컷. VLM이 실제로 말한
  컷 값을 그 인물의 값으로 쓴다. 기본값으로 채운 컷 값은 쓰지 않는다.
- `unknown`: 그 밖의 경우. 여러 명인 컷의 컷 값은 각 인물을 설명하지 못하므로 복사하지 않는다.

프롬프트는 env `VLM_PROMPT_VERSION`으로 고른다(기본 `p1-scope`). 모르는 값이면 기동이 멈춘다.
`p2-person-tags`는 p1에 인물별 배열 두 개와 판단 기준만 더하며, 켜기 전에 같은 평가셋에서
route·인원수·박스가 흔들리지 않는지 비교한다.

| 필드 | 타입 | 의미 |
|---|---|---|
| `index` | int | 컷 안 인물 인덱스(0부터) |
| `box` | float[4] \| null | `[x1,y1,x2,y2]` 픽셀 |
| `tags` | object | 이 인물의 의미 태그(§4) |
| `skeleton` | object \| null | `coco17-v1`의 정확히 17개 `[x,y]` keypoints와 17개 관절 confidence. 순서는 COCO-17 표준을 따른다. |
| `confidence` | string \| null | 인물 검출·매칭 신뢰도 |
| `candidates` | Candidate[] | Top-K 후보(기본 `top_k_final=5`, `config.py`) |
| `keypoints` / `scores` | float[17][2] / float[17] \| null | 원본 좌표와 구조 마스킹·refine 정책을 반영한 유효 score |
| `raw_scores` | float[17] \| null | 평가용 RTMPose 원본 score |
| `confidence` | string | `high` 또는 `low` |
| `skeleton_state` | string | `valid` · `partial` · `suspect` · `missing` · `invalid` |
| `skeleton_source` | string | `full_image` · `crop_retry` · `none` |
| `coverage_class` | string | `full` · `reduced` · `sparse` · `upper_only` · `insufficient` |
| `slot_origin` | string | `vlm` · `rtm_provisional` |
| `search_stability` | string \| null | `stable` · `ambiguous` · `unstable` · `not_required` · `not_available` |
| `valid_limbs` / `refinable_limbs` | string[] | 검색에 남은 부위와 refine 허용 사지 |
| `refine_allowed` | bool | v1은 검색+구조 정책, v2는 스켈레톤·소유권·coverage 안전 정책의 실행 허가 |
| `quality_reasons` / `quality_trace` | string[] / object | 구조 판정 사유와 배정·coverage·retry·검색 진단값 |
| `output_scope` | object | 출력 구도(위 설명) |
| `person_tags` | object | 인물별 action·view와 출처(위 설명). 검색에 쓰지 않는다 |

**`candidates[]` (CandidateOut)**

| 필드 | 타입 | 의미 |
|---|---|---|
| `pose_id` | string | 라이브러리 포즈 식별자 |
| `view` | string | 매칭된 투영 각도(§4 View) |
| `distance` | float | 기하 kNN 거리. **낮을수록 유사**(좋은 매칭 ~0.15, 앉기-서기 ~0.36, 추출 실패 ~0.6+) |
| `tags` | object | 후보 포즈의 의미 태그 |
| `rerank_score` | float \| null | rerank 점수(`USE_RERANK=1`일 때). 높을수록 좋음 |
| `bvh_url` | string | `GET /pose/{pose_id}/bvh` 다운로드 경로(동원이 소비) |

> ⚠ `distance`(원시 점수)는 클라이언트 `08_API_CONTRACT.md` §6의 `matchLevel`(high/medium/low 라벨)과 **다른 축**이다.
> 서버는 원시 `distance`/`rerank_score`만 준다 → **`matchLevel` 라벨 매핑은 앱 서버 또는 앱 어댑터가 수행**한다(원시 점수와 UI 라벨 분리는 클라이언트 문서의 명시적 설계 결정).
> 서버가 이미 계산하는 `count_confidence`/폴백(`person_confidence` 'low')을 이 라벨 산출의 입력으로 넘길지 §8-5.

---

## 4. Controlled Vocabulary (태그 어휘) — 단일 소스 = `src/schema.py`

`tags` 객체의 허용값. VLM 프롬프트(`src/vlm/prompts.py`)와 **반드시 일치**해야 한다(값 추가·변경 시 라이브러리 재태깅 필요).

| 키 | 허용값 |
|---|---|
| `shot` | `full_half` · `bust` · `face` |
| `action` | `standing` · `sitting` · `walking` · `running` · `reaching` · `lying` · `other` |
| `view` | `front` · `side` · `back` · `three_quarter` |
| `relationship` | `solo` · `talking` · `hugging` · `holding_hands` · `fighting` |

> 매칭은 **순수 기하 kNN**이라 `action`/`view`/`relationship`는 **필터로 쓰지 않는다**(기하와 중복 — `CLAUDE.md` 불변식 §1). 태그는 라우팅·분기·표시·로깅용.

---

## 5. `GET /pose/{pose_id}/bvh` — 라이브러리 base BVH 원본

BFF base 핸드오프. 선택된 후보의 BVH 파일을 **가공 없이** 반환한다. refine 성공 시 최종 BVH는
이 URL이 아니라 `RefineResponse.bvh` inline 본문이다. BFF가 최종 바이트를 확정한 뒤 내부 Converter
bundle API가 V3.2 retarget과 요청된 mirror를 한 번 적용하고, 입력 최종 BVH와 생성 FBX를 함께
반환한다. CSP는 같은 mirror를 반복하지 않고 소재 등록·배치와 다인 상대 위치 조정을 담당한다.

| 상태 | 조건 | 본문 |
|---|---|---|
| `200` | 등록됨 + 파일 존재 | `application/octet-stream`, `filename={pose_id}.bvh` |
| `404` | `pose_id` DB에 없음 | `{"detail": "unknown pose_id: ..."}` |
| `409` | 등록됐으나 BVH 파일 미존재(합성 인덱스 단계) | `{"detail": "pose '...' 등록됨(경로=...)이나 BVH 파일 미존재. 실 라이브러리 빌드 전 단계."}` |

> `409`는 **합성 인덱스로 계약만 확인하는 단계**의 정상 신호다(실 BVH 폴더 `BVH_DIR` 빌드 전). 실 라이브러리에서는 나오지 않는다.

---

## 6. `GET /healthz` · `POST /export-order`

**`GET /healthz`**

```json
{ "ok": true, "env": "development", "provider": "mock", "pose_backend": "mock", "pose_count": 20 }
```

`ok`는 파이프라인 로드 **그리고** 라이브러리가 비어 있지 않은지(`pose_count > 0`). `provider`/`pose_backend`로 **현재 mock인지 실모델인지**, `env`로 개발/프로덕션 여부를 앱 서버가 확인할 수 있다.

`ok: false`일 때는 **HTTP 503**으로 응답한다 — 후보를 하나도 낼 수 없는 상태라 로드밸런서·오케스트레이터가 트래픽을 보내지 않아야 한다.

**`POST /export-order`** — 작가가 고른 base 포즈의 legacy 주문서(`ExportOrder`). 상세 계약·예시는
**`docs/EXPORT_CONTRACT.md`** 참조. DB에서 base `bvh_url`·`set_id`·`tags`를 채우지만 inline refined
artifact는 알지 못한다. 따라서 Phase 3 제품 export는 BFF가 final base/refined bytes를 먼저 확정한 뒤
별도 내부 `POST /convert-bundle`을 호출한다. FBX 단일 응답인 `/convert`는 하위 호환용이다.

---

## 7. 오류 형식 — 현재 FastAPI 기본값 (⚠ 클라이언트와 불일치)

이 서버는 FastAPI 기본 `HTTPException`을 쓴다:

```json
{ "detail": "unknown pose_id: stand_xyz" }
```

클라이언트 `08_API_CONTRACT.md` §2는 **다른 형식**을 가정한다:

```json
{ "error": { "code": "INVALID_CREDENTIALS", "message": "...", "details": null, "requestId": "req_123" } }
```

**정리(권장):**
- **앱 서버가 `{error:{code,...}}` 봉투를 소유**한다 — 이 추론 서버의 `{detail}`을 받아 code로 매핑·requestId 부여. 도원 서버는 내부 계층이라 기본 형식 유지로 충분.
- 만약 Tauri가 이 서버를 **직접** 부르는 구조로 확정되면, 여기에 `{error:{code}}` 봉투를 도입해야 함(§8-1의 결론에 종속).

### 7-1. 상류 VLM 오류 원인 — `POST /analyze` → `503 VLM_UNAVAILABLE`

VLM(Gemini/OpenAI)의 인증·권한·결제 한도·429·5xx·시간 초과를 서버 버그와 구분한다. 기존 BFF 호환을 위해
HTTP **503**과 `code=VLM_UNAVAILABLE`은 유지하고, `reason`과 `retryable`을 추가한다.
503 자체만으로 혼잡이나 사용자 사용량 초과를 단정하면 안 된다.

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 30
```

```json
{
  "detail": {
    "code": "VLM_UNAVAILABLE",
    "message": "AI 분석 요청 한도에 일시적으로 도달했습니다. 잠시 후 다시 시도해 주세요.",
    "reason": "rate_limited",
    "retryable": true,
    "retry_after_seconds": 30
  }
}
```

| `reason` | 근거 | `retryable` / 다음 행동 |
|---|---|---|
| `rate_limited` | 429, 명확한 일일 한도 근거 없음 | true / 안내된 시간 후 재시도 |
| `daily_quota_exhausted` | 429 + QuotaFailure의 quotaId 또는 quotaMetric에 명시된 per-day/daily | false / 한도 초기화 또는 운영자 확인 |
| `authentication_failed` | 상류 401 | false / 운영자가 API 키 확인 |
| `permission_denied` | 상류 403 | false / 운영자가 프로젝트·모델 접근 권한 확인 |
| `billing_exhausted` | 상류 402 또는 OpenAI 429의 명시적 `insufficient_quota`/`billing_hard_limit_reached` 코드 | false / 운영자가 결제·크레딧 확인 |
| `upstream_unavailable` | 503 | true / 상류 혼잡·일시 이용 불가, 잠시 후 재시도 |
| `timeout` | 요청 시간 초과 | true / 잠시 후 사용자 재시도; 서버 내부 자동 재시도 없음 |
| `upstream_error` | 기타 5xx | true / 상류 오류, 잠시 후 재시도 |

- `retryable`은 **사용자/BFF의 후속 재시도 가능 여부**이며 서버가 재시도했다는 뜻이 아니다.
  일일 한도 소진은 즉시 반복 호출하지 않는다. 초기화 시각은 추정하지 않는다.
- 일일 한도 소진 시 `retry_after_seconds=null`, `Retry-After` 헤더 없음.
  그 외에는 기본 30초와 상류 `RetryInfo.retryDelay`/`Retry-After` 중 큰 값을 초 단위로 올림한다.
- 서버 내부 재시도도 상류 대기 시간을 줄이지 않는다. 대기 후 다음 시도가 전체 예산에
  들어가지 않으면 즉시 오류를 반환한다. SDK 내부 중첩 재시도는 끄고 기존 횟수·시간 예산으로 관리한다.
- 단순 `RESOURCE_EXHAUSTED`나 무료 요금제라는 이유만으로 일일 한도로 분류하지 않는다.
  원인이 불명확한 429는 `rate_limited`로 남긴다. 키·원본 응답·입력 이미지는 안내/로그에 넣지 않는다.

- 이 경로에서만 `detail`이 문자열이 아니라 객체다. 기계가 읽을 코드가 필요해서다(BFF가
  자기 오류 봉투의 code로 옮겨 담는다).
- **앱 서버/클라이언트 연동 필요:** 기존 code는 유지하되 `reason`, `message`, `retryable`,
  `retry_after_seconds`도 오류 봉투로 전달한다. `retryable=false`에는 반복 재시도 대신 한도 확인을
  안내하고, true에는 대기 시간을 반영한다. 이 서버 PR만으로 BFF/UI 안내가 자동 변경되지는 않는다.
  `INFERENCE_FAILED`나 하나의 혼잡 문구로 덮어쓰지 않는다.
- 400·422(잘못된 요청)와 500(우리 버그)은 그대로다. 이 구분이 알림 분류의 기준이기도 하다
  (503은 `VLM_UNAVAILABLE`, 500은 `UNHANDLED_ERROR`).

> 2026-08-21 프로덕션에서 이 구분이 없어 Gemini 503이 500 + P2 `UNHANDLED_ERROR`로
> 보고됐다(24시간 분석 13건 중 4건). 알림은 "우리 버그"로, 사용자 안내는 "다른 이미지로"로
> 잘못 나갔다.

---

## 8. 확인 필요 (서버 관점, 팀 합의 항목)

클라이언트 `08_API_CONTRACT.md` §11 질문 목록의 **서버측 대응**이다.

1. **계층 구조** — ✅ **얇은 앱 서버(BFF) 분리로 결정**(`DECISIONS.md` 결정 4): 인증·Job·오류봉투·`/v1`은 BFF 소유, 이 서버는 순수 추론 유지. 남은 팀 확인은 "BFF를 누가/어느 레포에 만드나"와 도입 시점(결정 4 ⚠).
2. **동기 vs Job** — 클라이언트는 `POST /v1/analysis/jobs → 폴링` 비동기를 가정. 서버는 `POST /analyze` 동기 즉시 반환. 추론이 짧으면 앱 서버가 **동기 호출을 Job으로 감싸** 클라이언트에 폴링을 제공(권장). 추론이 길어지면 이 서버에도 Job/진행 단계 도입 재검토.
3. **단계별 진행률** — 클라이언트 status enum(`detecting`/`skeleton`/`pose_search`/`rendering`…)은 세분화돼 있으나, 서버는 동기라 **중간 단계를 스트리밍하지 않는다**. 서버가 만들지 않는 진행률을 앱이 임의 생성하지 않는다(클라이언트 원칙과 일치).
4. **업로드 검증** — 현재 서버는 크기·MIME 약검증(PIL 폴백). 최대 크기·허용 형식·손상 이미지 처리를 어느 층에서 강제할지(권장: 앱 서버 + 서버 방어적 재검증).
5. **후보 개수·신뢰도 라벨** — `top_k_final=5` 기본이나 폴백 시 후보가 적거나 없을 수 있다(`route:"skip"`은 빈 배열). `matchLevel` 라벨 산출에 서버의 `count_confidence`/폴백 신호를 넘길지.
6. **BVH 전달 방식** — ✅ 확정: base는 `/pose/{id}/bvh` 응답 바이트, refined는
   `RefineResponse.bvh` inline UTF-8 바이트다. BFF가 둘 중 하나를 확정해 내부 Converter API로
   multipart 업로드하고, 제품 export에서는 최종 BVH와 생성 FBX가 든 검증 가능한 bundle을 받는다
   (`FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`).
7. **인증 헤더 전파** — 앱 서버가 이 서버를 호출할 때 내부 인증(서비스 토큰/네트워크 격리)을 둘지. 현재 무인증이라 **공개 노출 금지**(내부망/로컬 전제).

---

## 9. 스키마 소스 (코드 = 계약)

| 계약 | 소스 파일 |
|---|---|
| `/analyze` 응답 | `api/models.py` : `CutResultOut` · `PersonOut` · `CandidateOut` |
| `/export-order` 요청·응답 | `api/models.py` : `ExportOrderRequest` · `ExportOrder` · `ExportItem` |
| 태그 어휘(Controlled Vocabulary) | `src/schema.py` : `Shot` · `Action` · `View` · `Relationship` |
| 내부 결과 타입 | `src/schema.py` : `CutResult` · `PersonDescriptor` · `PoseCandidate` |
| 검색 파라미터(`top_k_final` 등) | `src/config.py` |

OpenAPI 자동 문서: 서버 기동 후 **`http://127.0.0.1:8000/docs`**. 이 문서와 `/docs`가 어긋나면 **코드가 정본**이고, 이 문서를 갱신한다(`05` 문서 도입 시 "API 변경은 문서 동시 수정" 규칙 적용).

## Internal converter: numeric body baking (2026-10-15, registry v2 2026-10-16)

`POST /convert-framed` accepts an optional multipart `shape` JSON string:
`{schemaVersion:"body-shape.v1", controls:{...}, shapeSha256:"..."}` and, with it,
`expected_shape_source`: `{rules, blendSha256, controllerSha256}` — the authoring source
the BFF catalog promises for this body. The BFF validates/normalizes controls first; the
converter independently checks keys, finite values, ranges, step grid and the same digest.
Request-selected source paths or Python code are never accepted.

**Approval is data, not code (`shape-bake.v2`).** `CONVERTER_SHAPE_REGISTRY` names a
registry shipped inside the immutable, manifest-pinned body release:

```json
{"schema": "shape-bake.v2",
 "sources": {"standin-body-0.7.1": {"blend": "/app/body-staging/sources/standin-body-0.7.1.blend",
                                     "controller": "/app/body-staging/sources/standin-body-0.7.1.py",
                                     "blend_sha256": "…", "controller_sha256": "…"}},
 "characters": {"male-bulk-v0-7-1": {"rules": "standin-body-0.7.1", "preset": "BULK", "asset": "…",
                                      "asset_sha256": "<neutral preset FBX>", "defaults": {…},
                                      "controls": [{"key": "height", "min": 160, "max": 195, "step": 1, "default": 178}, …]}}}
```

At start-up (`/healthz` → `checks.shape_sources`) and again inside every Blender job the
converter re-hashes each source, requires the controller's `VERSION` literal to equal its
`rules`, and requires every preset default to lie on its own slider grid (`(default-min)/step`
integral — the build shifts `min` up by less than one step when needed). `shape-bake.v1`
registries (hard-coded 0.6.7/0.2.2 pins) are rejected. A new body version therefore needs a
new release built by `scripts/build_shape_release.py`, not a code change.

Errors: different `expected_shape_source` → `409 SHAPE_SOURCE_MISMATCH` before any Blender
work; other shape problems → `400 INVALID_SHAPE`.

The approved controller bakes neutral mesh and bone proportions, the existing
V3.2.5 solver transfers the BVH, and approved pose/contact correction is baked
before framing. PNG and final review GLB are made from that exact final FBX.
Response adds `shape_sha256` and `shape_bake` lineage (`rules`, `blend_sha256`,
`controller_sha256`, `controls`, roundtrip/contact QA); the BFF requires the lineage to equal
the promised source. `CONVERTER_FORCE_EXACT_V324` rejects shape baking rather than claiming
exact-parent compatibility. New shape FBXs preserve all 52 real terminal bones; only synthetic
unweighted `_end` markers are removed during the isolated shape job.

**Latency.** A shaped job opens the source .blend twice and runs the controller's pose
correctives. Measured 30–110 s per job (2 vCPU Linux, pip bpy 5.0.1; heaviest: male BULK),
dominated by `correct_pose` (21–70 s). Shaped jobs use `CONVERTER_SHAPE_TIMEOUT_SECONDS`
(default 300) instead of the 30 s conversion budget; the BFF uses `CONVERTER_SHAPE_TIMEOUT_MS`
(default 320000). Load balancer idle timeouts on this path must exceed these.

BFF `BODY_SHAPE_BAKE_ENABLED` defaults off. Enable with `BODY_SHAPE_EDIT_ENABLED`
only after deploying matching converter/BFF code and immutable source/catalog data.
Modified-body Top-K uses PNG fallback instead of an unrelated precomputed GLB;
final review GLB is generated from the shaped FBX. New catalogs remain draft for
inference matching; supported BVHs are explicit and do not imply full-library QA.

## 원근 검색 런타임 연결 (2026-10-15)

`POSE_SEARCH_MODE=perspective`(새 코드 기본값)는 단일 인물·정상 full coverage·
full_image·전신 pos 검색에서 117개 조합/195개 시작점 조정/Top20 윤곽 재정렬을
실행한다. Z 모델은 사용하지 않는다. `legacy` 설정과 재시작으로 기존 검색으로
복귀하며, 부분·다인물·복구·비-pos·원본 BVH 미확보는 기존 경로를 유지한다.
운영 배포 설정은 이 코드 변경에 포함하지 않는다.

새 경로는 점수 미보정 때문에 `confidence=low`, `confidence_threshold=null`을 유지하지만,
refine v2는 기존 구조/소유권/관측 게이트로 허용한다. 일괄 refine 차단과 scores=0은 제거했다.
refine v1 설정은 기존 검색/보정 경로를 유지한다. 새 후보의 `camera.version`은
`candidate-camera-v2`이며 rotation 외에 projection, distance_torso, labels_swapped,
pose_id, query_sha256, observed_joint_mask를 포함한다. 클라이언트는 이 객체 전체와
`distance_metric=perspective_pos`를 `/refine`에 그대로 재전송해야 한다. 새 metric에
camera v2가 없거나 필드/포즈/XY/원본 SHA가 맞지 않으면 422(분석 중 원본 교체는 409)다.
v1/null camera 계약은 유지한다. v2의 원근/좌우가 전방계산과 최종 selector에 적용되고,
보정 결과는 원래 BVH 좌표로 복원한다. 안전한 보정이 불가능하면 기존 게이트나
`camera_projection_invalid` 사유로 원본을 유지한다. 표시 썸네일은 아직 직교다.
`quality_trace.perspective_search`에는 applied/reason/contour 복귀/비용/검색 카메라
진단값을 남긴다. 후보 `distance`는 카메라 혼합+윤곽 점수이며 legacy 거리와 직접
비교하면 안 된다. 외부 BFF/앱의 v2 전달/원근 화면 재현은 별도 통합 확인 대상이다.
세부 적용 범위와 제한: [PERSPECTIVE_RUNTIME.md](PERSPECTIVE_RUNTIME.md).

## 관측 상체 검색 추가 (2026-10-02)

- 흉상도 추출·검색을 수행하며 응답 `route=core`. 골반을 관측하지 못했으나
  어깨·팔 기하가 충분하면 `coverage_class=upper_only`, `distance_metric=upper_pos`.
- 전신과 별도 정규화를 사용하므로 항상 `confidence=low`, `confidence_threshold=null`,
  `refine_allowed=false`, `refinable_limbs=[]`, `scores=0`인 참고 후보다.
- 두상 인물은 `quality_reasons`에 `head_search_unsupported`, 후보는 빈 배열.
  전원 두상인 컷은 `route=skip`; 혼합 컷은 다른 인물을 정상 처리한다.
- `output_scope`는 출력 메타데이터다. 수동 설정으로 관측/검색/refine 정책을 바꾸지 않는다.
- 기존 full-body 피처 버전/DB/BVH/export 계약은 그대로다. 자세한 기준과 제한은
  [BODY_SCOPE.md](BODY_SCOPE.md)의 2단계를 참조한다.


## 선택적 체형 관측·자동선택 결과

2026-10-16 P0·P1: `body-match.v1`에 decision/policy/suggestion을 선택적 진단으로 추가한다. 약한 체형 근거는 자동 ID를 null로 반환하고, 기존 BFF의 수동 선택·사용자/카탈로그 기본값 처리를 따른다. 명확한 디자인 계열의 `auto_presentation_default`는 유지한다. [리뷰 및 최신 20종 draft 명세](BODY_SELECTION/P0_P1_REVIEW.md).

`BODY_MATCHING_MODE=off`(기본값)에서는 `/analyze.body_matching={}`이다. `shadow|auto`에서는 인물별 관측·자동선택 결과를 추가한다. 기존 people/포즈 순위/검색·refine 정책은 그대로다. 스키마·오류·catalog 승인 조건은 [체형 감지·선택 API 계약](BODY_PIPELINE_API_CONTRACT.md)을 따른다. 이 필드는 기존 클라이언트의 수동 선택을 덮어쓰지 않으며, 렌더·BFF 상태 저장·Export는 실행하지 않는다.

# 후보 표시 카메라 (2026-10-07 추가)

`CandidateOut.camera`는 선택적 `candidate-camera-v1` 메타데이터다. 기존 `view`는
검색 투영 식별자로 유지된다. 화면에는 `camera.display_view`를 사용하고, 회전행렬은
BFF가 저장한 값을 Converter로 전달한다. `rotation`은 원본 BVH Y-up 좌표에서 고정
정면 카메라 좌표로 가는 3×3 proper rotation이다. 원본은 `source_bvh_sha256`으로 묶인다.

`POST /refine`에 같은 `camera`를 전달하면 카메라 좌표에서 조정한 후 BVH를 원래
좌표로 복구한다. SHA가 달라지면 422로 거절한다. 카메라 유무와 관계없이 BVH 다운로드는
원래 방향이며, 새 카메라를 구 4방향 정적 썸네일로 대신 표현해서는 안 된다.
관측이 부족하거나 정면이 모호하면 camera=null이다. 구 작업은 기존 동작을 유지한다.
세부 내용: [후보 카메라](CANDIDATE_CAMERA_ALIGNMENT.md).

## Precomputed candidate previews

### Existing PNG poster compatibility for native references

`GET /pose/{pose_id}/poster?source_sha256=...&scope=...` keeps the existing URL,
`image/png` response, source/quarantine checks and `default-body-poster-v1` protocol.
For a complete native bundle, full/half/bust/shoulder_head use the verified
`thumbs/scope8/{source}__{scope}__yaw_000.jpg` (fixed front, not the search camera).
The server verifies source, character, frame 0, solver, crop, camera, approved
native renderer revision and JPEG hash, then losslessly encodes the decoded RGB
pixels as PNG. No Blender, pose generation or source/bundle writes occur.

The existing HTTP `frozen-retarget-v1` method describes this frozen native-retarget
family, **not** reference skinning. `X-Standin-Preview-Revision` remains the real
native renderer revision; `X-Standin-Native-Render-Method` preserves
`frozen-native-scope8-v1`, and `X-Standin-Native-Artifact-SHA256` identifies the
original JPEG. `X-Standin-Artifact-SHA256` identifies the returned PNG. Native
metadata is never relabelled. Missing/corrupt advertised native assets fail closed
with 404. Old bundles and excluded head-only scope keep the existing poster path.
App/BFF code, candidate ordering, templates, selected body/hands and export grants
are unchanged. Browser in-memory caches may retain an older successful poster
until the app is restarted or its existing cache expires.

The internal converter additionally serves read-only
`GET /pose-preview/{source_sha}?character_id=...` (`model/gltf-binary`).
It never runs inference or Blender. Configuration, identity, failure behavior
and client/BFF integration are specified in [PRECOMPUTED_POSE_PREVIEWS.md](PRECOMPUTED_POSE_PREVIEWS.md).

## 저장 전 모델 미리보기 (2026-10-08)

`POST /refine`의 선택 필드 `render_thumbnail`은 기본 `true`다. `false`이면
보정·수락/기각·BVH·안전 정책을 그대로 실행하고 `thumbnail=null`을 반환한다.
BFF/앱이 최종 출력의 모델 미리보기를 준비할 때만 사용한다.

내부 Converter `POST /convert-framed`에 `preview_format=model`을 보내면
PNG 렌더 없이 최종 FBX와 그 FBX를 재임포트한 정적 GLB를 함께 반환한다.
기본값 `png`는 기존 동작이다. 모델 형식은 `preview_view=front`만 허용한다.
`preview_base64`/`preview_sha256`은 이 경우 GLB 바이트/해시이며,
`preview_format=model`, `preview_model_revision`이 추가된다. 기존 카메라·체형·
BVH SHA·범위·FBX SHA 필드도 유지된다. GLB의 `asset.extras`에는
`version=framed-mesh-v1`, `revision`, `scope`, `camera_rotation`,
`base_fbx_sha256`, `source_bvh_sha256`, `character_id`, `character_sha256`가 있다.
모델에는 카메라 회전/출력 범위가 이미 적용되어 있으므로 다시 회전시키지 않는다.
메시만 정적이며 FBX의 뼈대 구조나 frozen solver를 변경하지 않는다.

Converter `/healthz`의 `preview_model_revision`과 `character_hashes`로 지원 여부와
캐시 세대를 확인한다. exact V3.2.4 모드에서는 모델 capability가 꺼진다.
정적 GLB는 단일 mesh·내부 buffer를 유지하며 최대 8개의 triangle primitive를 허용한다.
큰 아동 메시처럼 UInt32 인덱스 표가 8MiB 한도/50만 accessor 항목을 넘으면,
원본 정점·split normal·삼각형과 winding을 보존해 최대 65,535정점의 UInt16 파트로 나눈다.
LOD·decimation·양자화는 적용하지 않으며 원본 FBX/BVH와 뼈대는 유지한다.
8MiB 파일·50만 accessor 항목 제한은 그대로이고 각 bufferView는 4바이트 정렬을 유지한다.
Exporter revision과 사전 생성 GLB 캐시 키는 새 소스로 갱신되므로 재생성되지 않은 캐시는
기존 missing-model 경로로 처리한다. **다중 primitive를 허용하는 앱을 먼저 배포한 뒤 Converter를 배포한다.**

### 대분류 체형 추천 (opt-in)

`BODY_CLASSIFICATION_STRATEGY=coarse`는 인물별 `coarse_classification`과 `coarse_recommendation`을 추가한다. 남녀 계열을 먼저 판독하고 해당 계열 BASE를 기본 추천으로 두며 명확한 특징만 근육형·IDEAL83 등으로 변경한다. 모든 인물에 `recommended_body_id`와 추천 근거를 기록한다. 자동 적용과 달리 `recommended_body_id`는 진단용 프리셋 ID이며 자산 QA 권한이 아니다. `auto_body_id/applied_body_id/selected_asset/suggested_asset`는 null, `render_required=false`이다. 기존 기본 전략은 legacy이고 기존 수동 선택·기본값 계약은 유지한다. 상세: [대분류 추천](BODY_SELECTION/COARSE_RECOMMENDATION.md). 판독 입력은 머리 여백을 포함한 인물 crop과 같은 crop의 대상 표시본이다. `observations.crop_box`는 실제 확장 crop 좌표이며 crop을 만들지 못한 경우 null이다. 남녀 판독이 tentative인 경우 주 추천과 같은 계열 BASE를 보존하고 반대 계열 BASE도 대안에 포함한다(최대 3개). clear이면 같은 계열 대안만 제공한다.

선택적 `BODY_COARSE_FAMILY_REVIEW_MODEL`은 coarse의 잠정·미확정 계열만 이미지로 재검토한다. `body_matching.family_review_model`은 설정된 재검토 모델 또는 null이며, `coarse_classification.family_review`는 미실행이면 null/생략, 실행 시 최초 presentation·reading(face/clothing/body_design/family/certainty/reason)·status(resolved/unresolved/conflict/error)·error를 제공한다. 같은 확신도의 반대 계열은 기존 추천을 유지한다. 반대 계열 BASE 대안은 잠정/미확정 계열에서만 HTTP 검증을 통과한다. 실패·시간 소진은 최초 추천을 보존하며 자동 선택은 만들지 않는다. 기본 비활성, 기존 BODY_TIMEOUT_SECONDS 공유. [평가·제한](BODY_SELECTION/FAMILY_REVIEW.md).
## 러프함 기반 검색 경로 필드 (04J-286, 2026-10-13)

`SEARCH_ROUTING=1`일 때만 채워지는 **추가 필드**다. 기존 필드의 의미·값은 바뀌지 않는다.
끈 상태(기본)에서는 `people[i].search_routing=null`, `people[i].semantic_candidates=[]`, 최상위 `search_routing=null`.

| 필드 | 위치 | 내용 |
|---|---|---|
| `search_routing` | `people[i]` | `route`(geometric·semantic·undetermined·error), `reason`, 관절/의미 판정 수치와 기준값, `search.status`(ok·no_result·error·not_run), `elapsed_ms` |
| `semantic_candidates` | `people[i]` | route=semantic일 때 `[{pose_id, score, matched, pose}]`. `score`는 의미 점수라 `candidates[].distance`와 비교 불가. refine 대상 아님 |
| `search_routing` | 최상위 | 자세 설명 호출 상태(`describe.status`, `error_reason`, `retryable`, `retry_after_seconds`, `elapsed_ms`) |

route가 geometric이 아닌 인물은 `candidates=[]`, `confidence=low`, `refine_allowed=false`다(기존에도 관절 근거가
부족한 인물은 후보 0개였다). 상세: `docs/SEARCH_ROUTING.md`.

몸 관찰 개선(d2): semantic_evidence.description에 기본값이 있는 body_observations와
context_evidence를 추가했다. body_observations는 knees·legs·arms·torso의 정성 분류이며
관절 좌표·각도가 아니다. semantic_candidates[].matched는 body_matches·body_conflicts·
score_components를 추가로 포함할 수 있다. 미관측 속성은 중립이며 semantic은 계속 refine 금지다.
점수 정책 버전은 search.backend에 기록된다. 서로 다른 backend 버전의 score나 기하 distance를
직접 비교하지 않는다. `valid`는 검색 근거 조건 통과이며 정확도 보증이 아니다.

선택적 BVH 몸 색인(v3): `SEMANTIC_BODY_INDEX_PATH`를 지정하면 backend는
`tag-v3-bvh-body-interim`이다. `matched.body_source`는 `bvh` 또는 `library_text`,
`body_index_usage`는 쿼리 사용 조건·폴백 이유·행동 가중치를 기록한다.
`body_metrics`가 있으면 **후보 라이브러리**에서 미리 계산한 무릎 각도·정규화 손 높이 등이다.
러프에서 좌표·각도를 생성한 값이 아니며 기하 피처/distance/refine에는 입력하지 않는다.
불충분한 intrinsic 관찰은 기존 문장 검색을 유지하고, 색인 누락·버전/코퍼스 불일치·손상은
경고 로그와 함께 v2로 폴백한다. 새 색인은 기본 비활성이고 외부 응답 형태는 그대로다.

선택적 핵심 관계 색인(v4): `SEMANTIC_SHAPE_INDEX_PATH`를 명시하면
backend=`tag-v4-critical-shapes-interim`이다. 명확한 `fold_arms` 또는 양팔 V자
쿼리는 해당 관계가 검증된 후보만 반환한다. 명시한 비일반 자세(예: sitting)도 일치해야 한다.
검증 후보 없음은 `no_result`, reason=`critical_pose_combination_unverified:<관계+자세>`이며
라이브러리 전체의 실제 부재를 증명하는 값은 아니다. 색인 손상·누락은 `error`이고
핵심 조건을 놓친 이전 검색기로 조용히 폴백하지 않는다.

`matched.critical_relations`는 required·verified·근거 source를 담는다.
`matched.camera_comparison`은 24개 정사영 시점 중 관계의 가시성을 비교한 진단이며
`kind=qualitative_visibility_not_pixel_fit`이다. `pose.preview_camera`의 yaw_deg·pitch_deg·roll_deg는
**라이브러리 BVH 미리보기 카메라** 설정이고 러프에서 추정한 관절 각도나 포즈 수정값이 아니다.
기존 `pose.view`는 yaw 라벨만 나타내므로 기존 4방향 JPG만으로 pitch 적용 결과를 표시하면 안 된다.
BVH·기하 후보·distance·refine은 바뀌지 않는다. 몸 기준 V 관계와 화면에서 보이는 V 관계는
구별한다. 위/아래에서 보는 단서가 문장에 명시되지 않으면 러프와 같은 시점이라고 보장하지 않는다.

### 실험적 자연어 생성 hook (2026-10-14, 기본 비활성)

`Pipeline(..., pose_generator=<explicit research adapter>)`에 직접 주입한 경우만,
유효한 semantic 설명의 검색이 정상 수행되어 `no_result`이고 hits가 0개일 때 실행한다.
API 앱의 기본 Pipeline factory에는 주입하지 않는다. 별도 운영 env/endpoint/자동 export 없음.
provider/색인 오류, 모호한 설명, geometric 정상 경로, head/얽힘 기존 차단 및 명시적 half/bust에는 실행하지 않는다.

선택적으로 `people[i].search_routing.generation`에 다음을 기록한다:

- `status`: preview_ready / unsupported / rejected / timeout / error / not_requested.
- `backend`, `reason`, `elapsed_ms`, `diagnostics`, `candidates`(별도 합성 미리보기).
- 모든 상태에 `release_allowed=false`, `refine_allowed=false`, `actual_mesh_validated=false`.
- 미리보기는 `pose_id=generated_<hash>`, `source=text_constraints_analytic_ik`,
  `kind=synthetic_research_preview`, `bvh`(1프레임 본문), `keypoints_3d`, `constraints`,
  기본 리그 출처 hash 및 기구학 검사 결과를 포함한다. VLM 관절 좌표가 아니다.

`route=semantic`, `search.status=no_result`, 기존 `candidates=[]`, `semantic_candidates=[]`,
`confidence=low`, refine 금지를 유지한다. 생성 미리보기 3개를 검색 성공 3개로 세지 않는다.
생성 ID는 라이브러리 ID가 아니며 `/pose/{id}/bvh`, `/refine`, 자동 export 대상으로 보내면 안 된다.
기본 앱 응답은 기존과 동일하고 generation 필드가 없다.
기본 뼈대→자연어 조건→별도 IK 생성은 기존 refine과 다르다.
실제 피부 메시, 손목 twist/손가락/관절 해부학 축/의자 접촉/3D GT는 미검증이다.
세부 지원 범위·검증은 `docs/TEXT_POSE_GENERATION_PROBE.md`를 따른다.

### 04J-285 시맨틱 검색 연결 (2026-10-14, opt-in)

`SEMANTIC_BACKEND=semantic285`를 명시하면 285의 직접 문장 검색 서비스가
286의 기존 SemanticQuery/SemanticSearchResult 계약 뒤에서 실행된다. 기본값 tag와
`SEARCH_ROUTING=0`은 유지한다. `/semantic-search` HTTP endpoint는 이 연결 작업에서 추가하지 않았다.
`route`, `search.status`, `candidates`, `semantic_candidates`의 응답 구조는 동일하다.

- backend는 `semantic-285-text+bvh-validation-v1`. score는 285의 0..1 정책으로,
  기존 tag-v4 의미 점수나 geometric distance와 직접 비교하지 않는다.
- `semantic_candidates[].pose`는 285가 검증한 `standin-pose-record-v1` 포즈 JSON이며 ID·DB·태그 정합성 검사를 유지한다.
- `matched.semantic285`에 원래 점수 성분·문장/태그 근거·색인 정체성을 기록한다.
- 전체 기준 통과 후보군에 BVH 팔짱/V·요청 자세·미리보기 카메라 및 알려진 몸 관찰 충돌 검사를 적용한 뒤 family Top-K를 선택한다.
  상위 5개 또는 20개만 필터링해 적합 후보를 놓치지 않는다. 미관측 값은 중립이다.
- 문장별 선호/거절 기록은 기본 비활성. 수동 검수 결과는 일반화 성능으로 집계하지 않는다.
- 기존 격리 목록과 가져온 시맨틱 격리 목록의 합집합을 시맨틱 색인에 적용한다.
  이 작업은 기존 global 기하/refine 정책 파일을 변경하지 않는다.
- 명시한 DB/색인/encoder 실패는 error. 팔 관계 색인이 없는 핵심 요청도 error이며 이전 태그 검색기로 조용히 폴백하지 않는다.
- 기하 후보/추출/쿼리 좌표/refine solver 불변. 모든 시맨틱 후보는 계속 refine 금지다.

코드·30개 저장 실험·검수 인수인계는 `docs/SEMANTIC_285_INTEGRATION.md`를 따른다.

## 직접 문장 검색 — POST /semantic-search

이미지 없이 작가가 입력한 문장으로 **현재 로드된 전신 라이브러리**를 검색한다.
#87의 개발용 검수 UI·피드백 저장·별도 검색 엔진은 가져오지 않았다.
기존 `/analyze`의 관절 추출·VLM 검수·동적 라우팅·RRF 비중·응답 계약은 변경하지 않는다.

```http
POST /semantic-search
Content-Type: application/json

{"query":"의자에 앉아 양손으로 휴대폰을 보는 자세","topK":5}
```

- `query`: 문자열 1~500자, 공백/기호만 있는 입력 금지. 앞뒤 공백 제거.
- `topK` 또는 `top_k`: 양의 정수 또는 `null`. **생략하면 기존대로 5개**, `null`이면
  검증을 통과한 전체 후보를 반환한다. 20개 상한은 없다. 숫자 문자열·bool·0·음수는 거절한다.
  유한 라이브러리의 유효 결과만 반환하며 요청 수만큼 억지로 채우지 않는다.
- 다른 필드(`source`, `tags`, 임계값 등)는 422. 출처는 항상 `user`이며 운영 엔진의
  사용자 임계값(현재 기본 0.40)을 사용한다. 이미지/VLM 경로의 임계값은 바꾸지 않는다.
- 기동 시 로드한 `FullBodySearch.semantic` 어댑터·색인·BVH 조건 검증기를 재사용한다.
  전신 경로가 꺼져 있거나 색인/검증기가 준비되지 않았으면 503이며, 다른 엔진·mock로 폴백하지 않는다.
- 기존 어휘에서 명확한 자세/행동/도구 개념과 문장 속 몸 상태를 구조화한다.
  기존 핵심 관계·양손 조건·자세 검사와 격리 정책, 저장 합성 포즈 포함 설정을 적용한다.
  어휘 기반 해석이지 새 VLM 판독은 아니며, 임의 문장의 완전한 이해나 좌우·접촉 정답을 보장하지 않는다.
- 전체 기준 통과 후보군에 조건 검사·제외 정책을 적용하고 `near_match`를 분리한 **뒤** TopK를 자른다.
  동작 중 상하체 분리 검색·합성, 117시점 기하 검색, 이미지 분석, 포즈 생성은 호출하지 않는다.

응답(필드명은 snake_case):

| 필드 | 의미 |
|---|---|
| `status` | `ok` / `no_result` / `error` |
| `error` | 정상은 null, 오류는 `{code,message}` |
| `query` | 입력 `text`, 고정 `source:user`, 구조화한 `parsed`, 적용한 `conditions` |
| `candidates` | 정식 후보 `[{rank,pose_id,score,matched,pose,...}]` |
| `near_candidates` | 일부 조건만 만족한 검토 참고 후보. 정식 결과로 세지 않음 |
| `backend`, `threshold`, `top_k` | 실제 엔진, 사용자 의미 점수 기준, 요청 개수(`null`은 전체 결과) |
| `total_candidates`, `total_near_candidates` | TopK로 자르기 전의 정식·근접 후보 수. 전체 라이브러리 크기와 다름 |
| `pose_library`, `elapsed_ms` | 현재 번들 정체성, 처리 시간 |
| `score_is_accuracy`, `generation_requested` | 항상 false |

후보 `pose`는 기존 `standin-pose-record-v1` JSON이며 `pose_id`가 바깥 ID와 같아야 한다.
BVH는 기존 `pose.bvh_url`(`/pose/{id}/bvh`)로 받는다. `matched`에 점수 성분·검색 근거·
BVH 조건 판정을 보존한다. 의미 점수는 정확도가 아니며 기하 거리와 비교/합산하지 않는다.
모든 후보의 `refine_allowed`, `release_allowed`, `actual_mesh_validated`는 false다.
`near_candidates`의 rank는 정식 후보와 별도다.

| HTTP | 의미 |
|---|---|
| 200 + `ok` | 정식 후보 있음 |
| 200 + `no_result` | 검색 정상 완료, 정식 후보 없음. near 후보가 있을 수 있음 |
| 422 | 입력 형식/범위 오류(FastAPI validation 응답) |
| 503 + `error` | 엔진 미준비 또는 색인/encoder 등 검색 실패 |
| 500 + `error` | 후보 ID/점수/조건 판정 불일치 또는 예외 |

후보 없음은 라이브러리 자산 부재의 증명이 아니며 **이 API는 자동 생성하지 않는다**.
오류를 후보 없음으로 바꾸지 않는다. 사용자 문장·전체 검색 근거·백엔드 예외 원문은
요청 로그에 남기지 않는다. 응답에는 호출자 자신의 문장과 검색 근거를 반환한다.
기존 추론 서버와 동일한 내부 전용 경계이며, 앱/BFF 입력창·인증·중계 연결은 별도 작업이다.
새 환경변수·모델·라이브러리 업로드는 필요 없다.

### 전체 결과 탐색과 UI/BFF 인계 (2026-10-19)

```json
{"query":"공을 던지는 자세","topK":null}
```

- 전체 결과는 점수 기준·핵심 몸 조건·BVH 검증·격리 정책·family 중복 제거를 통과한
  후보만 포함한다. 개수 제한 해제가 점수 기준 완화나 자산/미러 중복 표시를 의미하지 않는다.
- `/analyze`의 러프 입력 Top5와 공유 검색기의 기본/최대 개수 설정은 변경하지 않는다.
  새 요청은 직접 문장 검색 API의 마지막 응답 개수만 제어한다.
- BFF는 `null`을 생략하거나 `topK ?? 5`로 바꾸지 말고 그대로 중계하고, 반환 배열을
  다시 5/20개로 자르지 않는다. 응답 `top_k` 타입을 `number | null`로 처리한다.
- 현재 전체 결과 옵션은 **한 응답에 JSON 전체를 반환**한다. 서버 페이지네이션이나
  cursor/offset, 검색 스트리밍은 추가하지 않았다. UI는 응답을 저장해 스크롤할 때
  20개씩 점진 표시하고 마지막 결과에서 멈출 수 있다. 실제 서버 페이징은 별도 계약이다.
- 검색창은 상단에 유지하고 입력 변경 시 debounce(예: 300ms) 후 재검색할 수 있다.
  이전 요청은 취소하거나 요청 ID로 무시해 늦게 도착한 옛 결과가 새 검색을 덮지 않게 한다.
  검색어가 바뀌면 이전 결과·표시 개수·스크롤 상태를 초기화한다.
- 카드 미리보기는 `pose.thumbnail_urls`의 저장 이미지를 lazy-load하고 크기 공간을 미리
  확보한다. `GET /pose/{id}/thumbnail`은 번들 파일 반환·ETag 캐시 경로이며 새 렌더가 아니다.
  목록의 모든 후보에 BVH→FBX 변환/3D 렌더를 선행하지 말고 선택·상세 보기 때만 수행한다.
  실제 앱의 느린 미리보기 원인은 이 변경에서 측정/확정하지 않았다.
- `near_candidates`는 ‘비슷한 자세/일부 조건 불일치’로 별도 표시한다.
  `no_result`는 정식 후보가 없다는 뜻이며 근접 후보가 있을 수 있다. `error`는 실패로 표시한다.
- UI/BFF의 배포 전 준비 여부는 새 API 스키마를 기준으로 맞춘다. 이전 서버는 `null`을
  422로 거절하므로 서버 변경 배포 없이 UI 요청만 먼저 전환하지 않는다.

## 손 모양·손 후보 (04J-290, 2026-10-14)

> refine 화면·BFF 저장/중계·오류 규칙까지 포함한 전체 설계: [`HAND_SELECTION_DESIGN.md`](HAND_SELECTION_DESIGN.md)

`POST /analyze`의 `people[].hands`에 인물별 좌/우 손 자동 선택과 후보가 실린다
(`VLM_HAND_SHAPES=1`일 때 VLM 별도 호출로 채움, 꺼져 있으면 `status=not_requested`와 기본값).
`left`/`right`는 그 인물 자신의 손이다. 각 손: `preset`(자동으로 구울 값), `source`(`vlm`|`default`),
`confidence`(`high`|`low`|null), `visible`, `candidates`(최대 4개, 앞이 유력).
검색·refine에는 쓰지 않는다.

FBX에 손을 구우려면 converter `POST /convert`·`/convert-bundle`·`/convert-framed`에 form 필드
`hand_left`, `hand_right`(프리셋 id)를 보낸다. 보내지 않으면 기존과 바이트 단위로 같은 FBX다.
프리셋 목록·불변식·제약: `docs/HAND_LIBRARY.md`.


### 첫 VLM 체형 통합 (선택적)

`BODY_CLASSIFICATION_STRATEGY=integrated`와 `BODY_MATCHING_MODE=shadow|auto`를 함께 설정하면 최초 `VLM.analyze`가 기존 사람 수·shot·박스·가시성에 `person_bodies`를 추가한다. `coarse_classification`과 `coarse_recommendation`의 외부 형태는 coarse와 같고, `classification_strategy=integrated`, `prompt_version=body-integrated.v7.parallel-review`로 출처를 구분한다. `inference_metadata.vlm_body_prompt_version`은 활성화한 통합 프롬프트 버전 또는 null이다.

Gemini 통합 모드의 `BODY_INTEGRATED_REVIEW_ENABLED=1`(기본)에서는 첫 판독의 남녀 계열이 unknown/androgynous이거나 certainty=unknown인 인물만 crop을 독립 재검토한다. 여러 대상은 최대 1회 batch 호출하고, 첫 판독의 결론·정답·파일 ID를 보내지 않는다. 이미 binary 계열을 판독한 인물은 tentative여도 변경하지 않는다. 재검토는 짧은 얼굴·의상·몸 근거를 출력하고 명시된 `VLM_BODY_THINKING_LEVEL`을 공유한다. 재검토는 `BODY_TIMEOUT_SECONDS`와 주 VLM의 남은 총 예산 중 작은 값으로 제한하며 재시도하지 않는다. 오류·미확정이면 원래 추천을 유지하고 `family_review`에 기록한다. tentative 추천은 그대로 tentative이며 자동 적용 근거가 아니다. 중성/unknown 계열의 해소만으로 기존 명확한 근육·체격·비율 근거를 낮추지 않는다. 실제 기존 binary 계열을 반대로 교정하는 경우의 특징 강등 가드는 유지한다. `BODY_INTEGRATED_REVIEW_ENABLED=0`이면 추가 체형 호출이 없다. OpenAI 통합은 첫 호출만 지원한다. 체형은 원본 VLM 슬롯 인덱스로 전달하여 출력 인물의 좌우 정렬 뒤에도 연결을 보존한다. 누락·잘못된 체형 응답은 해당 체형만 unknown으로 처리하며 기존 제어 필드를 변경하지 않는다. 주 추천은 항상 기록하되 관측 불명 시 기본 추천이라는 근거를 함께 반환한다. 약한 근거로 자동 적용하지 않으며 `auto_body_id/applied_body_id/selected_asset`는 null이다. 손 모양, 인물 태그, 전신 포즈 설명·재검토·생성은 기존 기능의 독립 호출이므로 전체 파이프라인의 VLM 호출이 한 번이라는 뜻은 아니다.

`Pipeline.process_cut`의 Gemini 통합 경로에서는 첫 판독 직후 미확정 체형 재검토를 시작하여 검출·포즈 추출·검색과 병렬로 실행한다. 포즈 경로가 끝난 뒤 최초에 정한 남은 예산만 기다리고, 원본 VLM 슬롯 인덱스로 descriptor에 합친다. 대기 시점에 새 예산을 부여하지 않는다. 기한 내 완료된 응답은 포즈 작업이 더 늦게 끝나도 사용하며, 시간 초과·실패는 최초 추천을 보존한다. 작업은 요청별 이미지·분석 복사본을 사용하므로 늦게 끝난 응답이 반환 결과를 바꾸지 않는다. SDK의 실행 중 HTTP 요청은 강제 취소할 수 없으며 기존 HTTP timeout이 유지된다. 직접 `GeminiVLMClient.analyze()`를 호출하는 경로는 기존 동기 계약을 유지한다.

재검토를 시작한 응답에만 `body_matching.parallel_review` 진단이 추가된다. `mode=parallel`, `status=completed|error|deadline_exceeded`, `target_count`, `merged_count`, `error`, `review_finished`, `review_elapsed_ms`, `overlap_ms`, `join_wait_ms`, `pose_path_ms`를 제공한다. `review_finished=false`이면 경과 시간은 수집 시점까지의 값이며 완료 시간이 아니다. 이 시간들은 내부 병렬 구간 진단이고 전체 요청 지연이나 제품 SLA가 아니다. [병렬화 검증과 제한](BODY_SELECTION/PARALLEL_REVIEW.md).

기본은 `legacy`/`off`이며 실제 환경 설정을 바꾸지 않았다. 평가 모델·28명 기준·전신 경로의 시간과 회귀 관측은 [통합 평가](BODY_SELECTION/INTEGRATED_FIRST_CALL.md)를 따른다.
## 동적라우팅 BFF·앱 표시 연결 (dynamic-pose-presentation-v1)

`POST /analyze`에 **추가**된 `people[i].pose_presentation`이 동적라우팅의 표시용 계약이다.
기존 `dynamic_search` 원문과 legacy `candidates`는 변경하지 않는다. 전신 동적라우팅에서는
`candidates=[]`가 의도된 값이므로 이것만으로 후보 없음으로 판정하면 안 된다.
legacy/라우팅 미실행이면 `pose_presentation=null`이며 기존 앱 경로를 그대로 사용한다.

| 필드 | BFF·앱 처리 |
| --- | --- |
| `version`, `person_index`, `route` | `dynamic-pose-presentation-v1`; `person_index=people[].index`; 선택된 인물별 라우팅 경로 |
| `status` | 표시 상태. `ready`, `review_ready`, `basic_reference`, `no_result`, `generation_failed`, `partial_error`, `error`, `blocked`, `undetermined`, `not_run` |
| `search_status`, `search_reason` | 검색 엔진 원래 상태·사유. HTTP 200이어도 검색 오류일 수 있음 |
| `generation_status`, `generation_reason` | 생성 원래 상태·사유. 참고 후보가 있어도 생성 실패 배너는 유지 |
| `ranking` | 실제 RRF 가중치. API 팩토리 `weighted_rrf_k60_geometry_0.4_semantic_0.6`. 배포 후 `/healthz.dynamic_routing`에서도 실제 ranking·두 가중치 확인 가능 |
| `output_scope` | 변환 출력 범위 `full/half/shoulder_head/head/null`. 미관측을 임의로 full로 확정하지 않음 |
| `unobserved_body_is_default` | 생성 진단의 명시적 기본값 여부만 전달. true=화면 밖 부위 기본값; null=근거 없음 |
| `preview_status` | `ready`=표시용 파일 입력 있음, `unavailable`=파일/계약 오류로 없음, `not_requested`=미리보기 입력 없음. BFF 조회 실패 시 별도 오류 표시 필요 |
| `library_candidates` | qualified 원본 검색 후보(최대 5개). 통과한 검색 조건 범위의 후보이며 완전한 메시 안전 인증 아님 |
| `review_candidates` | `geometry_review`, `near_match`, `framing_reference`, `mirror_reference` 분리 표시(최대 5개; 검색·생성 후보와 합산하여 총 5개). 정답 검색 성공으로 집계 금지 |
| `generated_candidates` | `preview_ready` 생성만 `generated_review`로 제공(최대 3개, 전체 5개 중 남은 자리만 사용). 실패/근접 생성 진단의 BVH는 포함하지 않음 |
| `basic_model` | 어깨~머리 기본 모델의 **출력 계획**. 미렌더 상태이며 BVH/FBX 경로를 지어내지 않음 |
| `issues` | 미리보기 파일 누락·격리·해시 변경 등의 코드. 내부 파일 경로·예외 원문 미노출 |

러프 검색의 `library_candidates`·`review_candidates`·`generated_candidates` 합계는 인물별 최대 5개다.
원래 selected 후보를 우선 유지하고, 검증된 미러 참고는 기존처럼 한 자리를 사용한다.
이후 남은 자리에 근접 참고 → 구도 참고 → 생성 검수 순서로 채운다. 미리보기 파일 검증 실패는
자리를 차지하지 않으며, 후보의 원래 rank·ID와 검색 보고서는 유지한다. 이 제한은
`dynamic-pose-presentation-v1`에 적용하며 자연어 검색의 전체 결과 계약은 변경하지 않는다.

`ready`는 표시할 파일이 준비되었다는 뜻이지 자동 적용·내보내기 승인이라는 뜻이 아니다.
모든 항목은 `review_required=true`, `auto_apply_allowed/refine_allowed/export_allowed/
release_allowed/actual_mesh_validated=false`다. 각 항목의 `validation`은 원래 검사 근거이며
피부 검사 receipt가 없는데 통과한 것으로 채우지 않는다. 별도 피부 자동 검사 PR의 적용을
이 계약 변경이 대신하지 않는다. 표시용 투영은 원래 점수·조건 게이트·생성 호출/재시도를
바꾸지 않는다. 별도 사용자 요청에 따른 서버 RRF 비중 0.6 적용은 위 ranking 항목을 따른다.

### 3D 미리보기 파일 전달

각 항목은 `candidate_id`, `pose_id`, `source`, `library_source`, `kind`, 원래 그룹 내 `rank`, `asset`, `camera`, `camera_kind`, `render_camera`,
`validation`을 가진다. `rank_score`는 검색 후보의 RRF 점수이며 정확도가 아니다.
실패한 항목을 건너뛰어도 원래 rank·순서는 유지한다. 그룹 간 임의 통합 재정렬은 하지 않는다.
`candidate_id=preview_<sha256>`는 버전 v1의 인물 index·종류·라이브러리 ID(생성은 null)·
BVH SHA-256·정규화된 카메라·출력 범위에 바인딩한다. 점수/순위가 바뀌어도 같은 미리보기는
같은 ID이며, 바이트/인물/시점/범위가 바뀌면 달라진다. 전역 인증 토큰이 아니므로 BFF의
job/사용자 소유권 확인을 대체하지 않는다. `lookup_allowed=true`는 미리보기 입력 조회만 뜻한다.
`score_kind=weighted_rrf` 또는 null이며 `score_is_accuracy=false`다.

* 라이브러리: `asset.transport=library_bvh`, `bvh_url=/pose/{URL-encoded-id}/bvh`,
  `bvh_sha256`, `library_lookup_supported=true`. 기존 GET과 같은 레지스트리/격리 정책을
  확인하고, 가능하면 검색 인덱스의 원본 해시도 비교한다. BFF는 이 **내부 상대 경로**를
  고정된 추론 서버에서 가져오는 인증된 프록시로 바꾸고, 받은 바이트 해시를 재확인한다.
  임의 외부 URL을 가져오는 프록시를 만들지 않는다. 저장된 과거 결과의 파일이 바뀌면
  다른 포즈로 조용히 대체하지 말고 재분석/미리보기 불가로 표시한다.
* 생성: `asset.transport=inline_bvh`, `bvh` 본문, `bvh_sha256`,
  `library_lookup_supported=false`, `bvh_url=null`, `encoding=utf-8`, `pose_id=null`, `source=generation`.
  본문을 그대로 미리보기 입력으로 사용하고 `/pose/generated_.../bvh`를 요청하지 않는다.
  새 공개 파일 저장소·임시 경로·자동 라이브러리 등록은 만들지 않는다.
* `thumbnail_url`은 실제 번들 파일이 있을 때만 제공한다. 정면 썸네일이므로
  `thumbnail_view=front`, `thumbnail_is_search_view=false`다. 없으면 null이며 3D 파일은
  정상일 수 있다. 이 API에서 converter 렌더를 추가 실행하지 않는다.
* `camera_kind=fixed_117`의 `camera={yaw,pitch,distance}`는 `src/camera_search.py::project`
  규약(도 단위, 몸통 길이 정규화, hip 중심, Ry 후 Rx 회전, distance null은 직교)을 유지한다.
  legacy `CandidateCamera` v1/v2와 다른 형식이므로 그 스키마로 캐스팅하지 않는다.
  `semantic_preview`는 기존 색인의 프리뷰 카메라, `viewer_default`는 시점 미확정이다.
  서버가 새로운 시점/좌우/관절을 추정하거나 BVH에 회전을 굽지 않는다.

완전한 렌더 규약은 추가된 `render_camera`를 사용한다. `pose-preview-camera-v1`은 Y-up
오른손 좌표계, hip midpoint 중심, hip→shoulder midpoint 길이 정규화를 사용한다. 회전행렬은
열벡터 기준 `Rz @ Rx @ Ry`이며 실제 점은 `q=R*(p-hip)/torso_length`다. 정사영은
`xy=(q.x,-q.y)`, 원근은 `xy=(q.x,-q.y)/(q.z+distance_torso)`이며 깊이>.1이어야 한다.
각도는 도, 거리는 몸통 길이 단위다. 이는 정규화 투영면 수식이므로 FOV는 제공/추정하지 않는다.
기성 원근 렌더러를 쓰면 focal length=1인 대칭 투영면을 구현해야 하며, 이미지 채우기용 확대·
이동은 별도다. `fit_to_rough_pixels=false`: 러프 픽셀 구도까지 복원했다는 뜻이 아니다.
`projection_matches_search=true`는 117시점 기하의 방향·투영 수식만 같다는 뜻이다.
`semantic_preview`는 정사영 참고, `viewer_default`는 중립 정사영 표시 기본값(관측 시점 아님).
`bvh_camera_rotation_applied=false`이므로 BVH 원래 루트/관절 회전은 유지한 채 이 **표시 회전만
한 번** 적용한다. BVH 단위는 원본 단위이며 미터로 단정하지 말고 위 torso 정규화를 사용한다.

BVH 경로는 만료 시각이 있는 signed URL이 아니며 별도 TTL은 없다. 다만 같은 라이브러리·
파일·접근 정책이 유지될 때만 재시작 후에도 유효하다. 기본 추론 라우트에 새 공개 인증/토큰을
추가하지 않았으므로 기존 내부 네트워크/서비스 인증 경계를 유지하고 앱에 직접 노출하지 않는다.
생성 본문은 반환 후 서버 임시 파일 없이도 사용할 수 있다. BFF는 기존 분석 결과 보관 기간·
job 소유권·접근제어·사용자 삭제/만료 정책 안에서 미리보기용으로 저장/재사용할 수 있다.
본문·해시·후보 ID·검사 근거를 함께 보존하고 공개 저장소/무기한 보관/라이브러리 등록/export로
확장하지 않는다. 보관 기간을 이 계약에서 새로 늘리지 않는다.

우선순위: 지원하는 `pose_presentation.version`이 있으면 표시 목록/상태는 이 필드가 기준이다.
`dynamic_search`는 근거 확인용이며 두 결과를 합치지 않는다. 미포함/null인 구버전은 팀원의
기존 dynamic_search 연결을 유지할 수 있고, 그것도 없을 때 legacy candidates를 사용한다.
알 수 없는 버전은 지원 불가로 표시한다. 오류를 보고 다른 목록을 정상 후보로 승격하지 않는다.

검색 오류·차단이면 주입된 생성/검색 항목을 표시용 목록으로 승격하지 않는다. 정상 빈 검색의
생성만 허용하며, 이후 별도 기하 참고 목록이 붙더라도 qualified 결과가 없었던 생성은 유지한다.
파일 접근 실패는 `error/partial_error + issues`이며 `no_result`로 바꾸거나 재생성하지 않는다.
어떤 표시용 항목도 legacy `candidates`, `/refine`, `/export-order`에 복사하지 않는다.

### 한손 반대손 참고 (`mirror_reference`)

한쪽 **캐릭터 해부학적 손**만 관측되고 일반 원본 후보가 기존 전신 조건·손 피부 검사를
통과한 경우, 같은 명시적 `pose_family_id`의 반대 `is_mirror` 실제 라이브러리 에셋을
최대 1개 추가로 검토할 수 있다. 필수 팔 상태·고신뢰도 visible-chain 관계·소지품 손
중 양손 또는 상충하는 측면 근거가 있으면 요청하지 않는다. 보이지 않는 팔은 채우지 않는다.
파일명으로 가족/측면을 추정하거나 BVH 생성·수정·CSS 반전을 하지 않는다.

* 원본과 참고 모두 태그 메타데이터 `bvh_sha256`, 카메라 인덱스 해시,
  body 인덱스 `source_sha256`가 같아야 한다. 참고는 현재 allowed/nonquarantine
  라이브러리의 같은 family, 반대 **boolean** `is_mirror`여야 한다.
* 관측 조건의 좌우 필드만 교환하여 `WholePoseValidator`·기존 의미 조건·실제 손 피부
  검사를 다시 통과해야 한다. 그 밖의 자세/소지품/손 개수 조건은 유지하며 unknown이나
  conflict를 성공으로 승격하지 않는다. 원본 검색 순서·점수는 유지한다.
* `review_candidates[].kind=mirror_reference`, `rank_score=null`, `score_kind=null`.
  `validation.not_exact_match=true`, `validation.constraints_satisfied=false`이며
  `review_required=true`. 모든 표시 후보와 같이 내보내기·refine·release·자동 적용·
  실제 메시 승인 권한은 false다. `lookup_allowed=true`는 검토용 파일 조회만 뜻한다.
* `mirror_comparison`에는 `family_id`, `original_pose_id`, `original_bvh_sha256`,
  `original_candidate_id`, 참고의 boolean `is_mirror`, `observed_side`, `reference_side`,
  `reason=opposite_hand_reference`, `matches_observed_side=false`를 담는다. 두 side는
  `left/right` 중 서로 반대이며 화면 좌우가 아니다. API가 현재 표시 원본의 candidate ID와
  BVH 해시를 연결하고 참고의 `source_bvh_sha256`를 실제 registry 에셋 해시와 재검증한다.
  원본이 사라졌거나 바이트/카메라가 바뀌면 참고를 버린다.
* 카메라는 원본과 동일한 수치·회전 행렬을 한 번만 적용한다. 참고의
  `render_camera.projection_matches_search=false`는 참고가 원래 관측의 검색 일치가
  아님을 표시한다. 메타데이터 측면이나 카메라를 다시 반전하지 않는다.
* 참고가 준비되면 일반 원본 최대 4개 + 참고 1개, 전체 최대 5개다. near/framing/geometry
  review도 이 예산을 공유하며, 연결된 원본은 유지된 첫 4개 안에 있어야 한다.
  선택적 참고 검증 실패는 이미 검증된 일반 후보와 그 순위·권한을 바꾸지 않는다.
* 상위 `presentation.mirror_comparison.status`는 `ready/unavailable/not_requested`다.
  `ready`에는 유효 참고가 정확히 하나 있어야 한다. API 재투영에서 참고가 탈락하면
  `unavailable`, `reason=mirror_preview_unavailable`로 낮추고
  `issues=mirror_reference_preview_unavailable`를 남긴다. 이것만으로 일반 후보의
  `ready`를 오류로 바꾸지 않는다. 필수 일반 후보 손 검사 실패 정책은 그대로다.

엔진의 `mirror_hand_skin_screen`은 선택적 검사 진단이다. 일반 후보에 이미 확정된
`hand_skin_screen` 스냅샷과 분리하며, 참고 검사 timeout/미확인을 일반 후보 성공이나
생성 허용으로 바꾸는 근거로 쓰지 않는다. BFF·앱은 `반대손 참고`와 원본 연결, 관측 손과
참고 손을 표시하고 참고에 export/refine 동작을 제공하지 않는다.

재현 가능한 실제 에셋 검사는 `tests/test_mirror_reference_assets.py`의 opt-in 테스트다.
`MIRROR_ASSET_ROOT`에 기존 릴리스 data 디렉터리, `MIRROR_CHARACTER_FBX`에 등록된
`standin-master-v2` 파일을 지정하고 실행한다. 실행 시 구성된 누적 예산·최대 10회 손 검사와 실제 Blender
worker·손 피부 HTTP 계약·현재 API registry/download 검증을 사용한다. 원본 라이브러리는
읽기만 하고 API용 경로는 임시 DB 복사본에서 재배치한다. `MIRROR_EVIDENCE_DIR`을 지정하면
로컬 presentation/receipt 증거를 남긴다. 입력은 라이브러리 포즈의 결정적 2D 투영과 명시적
한팔 관측이므로, 이 테스트 통과를 실제 러프/VLM/스테이징 통합 검증으로 보고하지 않는다.

### 팀원 연결 요청과 완료 기준

추론 서버 소유 변경은 이 필드·파일 무결성·OpenAPI·회귀 테스트까지다. BFF/앱 구현·배포는 별도다.

1. BFF는 새 필드를 인물별로 저장·전달하고 library/inline 전송을 구분한다. body matching의
   인물/체형 선택·수동 설정·손 프리셋 등 기존 계약은 그대로 보존한다. 검수/변환 조건을
   바꾸거나 프리셋을 다시 구웠다면 이전 검사 근거를 새 결과의 인증으로 재사용하지 않는다.
2. 앱은 새 필드가 있으면 그 목록을 읽고 검색/참고/생성 그룹을 구분한다. `search_status`,
   `generation_status`, `issues`를 보존해 실패를 단순 후보 없음과 구분한다. 구버전 서버만
   기존 candidates 경로를 사용한다. 두 목록을 임의로 합치거나 export 제한을 해제하지 않는다.
3. 같은 러프의 인물별 목록 수와 표시 수가 일치하고 실제 BVH 미리보기가 열리는지 E2E로 확인한다.
   검색 1개/legacy 0개, 정상 빈 검색, 생성 검수 성공, 생성 실패, 부분 오류, 근접 참고,
   어깨~머리 기본 계획, 파일 누락·변경, 다인물 및 legacy 응답을 각각 검사한다.

로컬 HTTP 회귀 예제: `tests/test_pose_presentation.py`. `pose_presentation` 로그는 그룹별 수와
오류 코드만 기록한다(러프·설명·BVH 본문은 기록하지 않음). BFF·앱이 새 필드를 읽기 전까지는
추론 서버만 배포해도 기존 앱의 ‘후보 0개’ 표시는 해결되지 않는다.

`GET /healthz.semantic_text.ready`는 직접 문장 검색에 쓰는 기존 전신 엔진의 실제 index·validator·startup 상태를 나타낸다. BFF는 이 값으로 문장 검색 준비 여부를 확인한다. 환경 플래그만으로 준비됨을 표시하지 않는다. 이 조회는 VLM·검색·생성을 호출하지 않는다.

### 자세 설명 응답 형식과 오류 진단
Gemini describe 호출은 `d5-structured-description`부터 response_json_schema를 함께 보낸다. people 배열의 minItems/maxItems는 요청 슬롯 수와 같고, 두상·가림으로 읽을 수 없어도 슬롯을 생략하는 대신 visible=false/certainty=low/빈 관찰로 기권한다. 좌표·숨은 팔다리·정상 검색 결과를 만들어 채우지 않는다. 기존의 관찰 및 의미 근거 게이트가 계속 최종 유효성을 판단한다.

로컬 검증은 missing_text/missing_json/invalid_json/people_count_mismatch/invalid_person을 구분한다. gemini_pose_describe 로그의 validationCode/validationField에는 고정 코드와 필드 이름만 기록한다. VLM 응답 원문·프롬프트·이미지·인증값은 기록하지 않는다. 형식 실패는 기존 invalid_response/nonretryable을 유지하며 후보 없음이나 생성 성공으로 바꾸지 않는다. transient 오류의 기존 제한 재시도 정책도 유지한다.

`tests/test_description_contract.py`는 실제 google-genai SDK와 localhost HTTP 서버로 요청의 schema 직렬화 및 HTTP 200 뒤 잘못된 응답 처리를 검증한다. 합성 이미지·가짜 키를 쓰며 외부 Gemini는 호출하지 않는다. 올바른 기권 응답은 빈 관찰을 유지하고, 형식 실패는 고정 code/field만 남긴 채 1회 호출로 종료한다. 이 회귀 통과와 실제 Gemini/스테이징 입력 재현은 별도 검증이다.

스테이징 제보의 과거 로그는 ValueError만 남아 실패 필드를 복원할 수 없다. 이 변경은 응답 구조 예방과 다음 실패의 진단을 보강하며, 원본 입력의 스테이징 재실행 성공을 의미하지 않는다. 표시 scope의 불일치는 별도 BFF 범위 resolver 변경에서 처리한다.

### 아동 자산 추천 제한 (2026-10-20)

이미지의 비율·얼굴·스타일·가림·웅크림 또는 VLM clear 표시로 아동 자산을 자동 추천하지 않는다. 레거시 랭커도 순위·제안·자동 선택에서 child 자산을 제외한다. 해당 경우 `child_recommendation_policy`와 `child_requires_explicit_user_selection`을 반환하며, 다른 기존 기본 후보가 없으면 선택을 요구한다. 실제 나이 필드/나이 확률을 생성하지 않는다. 사용자 수동 선택·저장한 기본값·last-used 아동은 유지한다. [기준·경로·검증 한계](BODY_SELECTION/CHILD_RECOMMENDATION_POLICY.md).

### 확실한 두상 출력의 손 단계 생략

추론의 최종 인물 범위가 `head`이고 출처가 `vlm_person`, `vlm_description` 또는 단일
인물의 `legacy_shot`일 때만 손 작업을 생략한다. 누락 관절·일반 설명·출처 불명은
근거가 아니며 `bust`, `shoulder_head`, `half`, `full`은 기존 추론 손 정책을 유지한다.
사용자가 이후 바꾸는 수동 범위는 BFF의 최종 범위 resolver와 손 선택 계약이 우선한다.

- 전신 동적 경로는 인물 설명과 범위 보정을 끝낸 뒤 손 VLM을 시작한다. 남은 인물만
  요청하며 원 VLM 슬롯 인덱스로 응답을 복원한다. 생략 인물은 `hands.status=not_requested`,
  요청 실패는 요청한 인물만 `unavailable`이다. 전원이 head이면 손 호출·Future 대기가 없다.
- head의 검색 후 손 피부 검사·생성 손 검사·미러 손 검사 세션을 만들지 않는다.
  `dynamic_search.hand_skin_screen`은 `method=head-output-hand-omission-v1`,
  `status=not_applicable`, `reason=head_only_output`, `scope=head`, `checks=0`을 기록한다.
  후보에도 같은 N/A marker를 붙인다. `clear_screen`이 아니고 실제 메시 검증 성공도 아니다.
- 모든 기존 `release_allowed`, `refine_allowed`, `actual_mesh_validated`와 최종 검수 정책은
  그대로다. 두상 생성/geometry fallback도 기존 참고 자격만 유지한다. 다른 범위에서의
  검사 오류는 기존대로 fail-closed이며 head 요청이 공유 gate 상태를 바꾸지 않는다.
- 두상 미러도 원본/참고 해시·family·원문 조건·카메라·첫 4개 안의 원본 연결을 검증한다.
  손 검사를 대체하는 N/A는 API 투영의 범위가 head이고 엔진 summary와 양쪽 marker가
  정확히 일치할 때만 인정한다. 다른 범위의 N/A, 손 검사 성공으로 바꾼 marker,
  원본 연결 오류 또는 권한 승격은 거부한다.

전신 동적 경로의 손 VLM 시작이 설명/범위 판정 뒤로 이동하므로 손이 있는 인물은 이전의
초기 포즈 추출과 손 호출 중첩을 사용하지 않는다. 실제 지연 변화는 배포 후 별도 측정한다.
실제 모델 호출·피부 렌더 없이 회귀 계약은 `tests/test_head_only_hands.py`에서 검증한다.

## 동적 라이브러리 후보의 선택 후 refine 복구 (2026-10-21)

`people[].refine_candidates`는 선택적 후보 바인딩 목록이다. 기존 `CandidateOut` 형식으로
pose_id, view, distance, camera를 제공한다. `camera`는 원본 BVH SHA-256·입력 관절 SHA-256·
관측 마스크·원래 117시점 검색 회전/원근을 담은 `candidate-camera-v2`다. 새 카메라 fitting이나
검색 순위 변경은 없다. 기존 legacy `candidates`와 `pose_presentation` 목록은 그대로 유지한다.

full_image / VLM 소유 슬롯의 valid 또는 partial 스켈레톤이 full/reduced 범위와 보정할 사지를
갖고, 실제 기하 검색 카메라에 연결된 정식 라이브러리 후보가 있을 때 기존 구조 정책으로
`refine_allowed=true`와 마스킹된 scores/refinable_limbs를 보존한다. 동적 후보가 있다는 이유로
`has_candidates=False`를 넣어 scores를 0으로 만들던 처리를 제거한다. low 검색 신뢰도는
v2 refine 금지 사유가 아니며, 하체 미관측 동결·충돌·해부학·selector·5초 제한은 그대로다.

semantic-only, geometry fallback, near/framing/mirror 참고, 생성 후보 및 기존 구조 정책에서
차단된 인물은 이 목록에 포함하지 않는다. `REFINE_ENABLED=0` 또는 `REFINE_V2_ENABLED=0`도
목록을 비운다. `pose_presentation`은 계속 미리보기/수동 확인 계약이며 그 내부의 자동 권한은
승격하지 않는다. BFF는 명시적 바인딩을 같은 library_match 후보의 자산·카메라와 대조해서
선택 후보의 refine 권한만 연결해야 한다. 바인딩이 없는 구 응답은 기존 차단을 유지한다.

배포 순서: 추론 서버 → BFF → 클라이언트. BFF의 `REFINE_FEATURE_ENABLED`도 true여야 한다.
과거 저장 Job의 0점 관절을 사후 복원하지 않는다. 새 분석부터 적용한다.


## Top5 수평 8방향 미리보기 2026-10-21

기존 `GET /pose/{pose_id}/poster`에 선택적으로 `variant=rough-yaw8-v1&view=yaw_045`를 전달하면 검색 좌표계 기준 45도 간격의 검증된 정적 참고 이미지를 제공합니다. BFF가 후보별 카메라로 view를 결정합니다. `source_sha256`·scope·등록/격리 검증은 유지하며 variant 없는 호출은 기존 native PNG 호환 경로를 유지합니다.

실제 served yaw·폴백·투영 불일치는 버전 있는 헤더로 전달합니다. 없는 방향이나 머리 단독 자산을 임의로 채우거나 원본 포즈·정밀 카메라·내보내기 권한을 바꾸지 않습니다. 자산 좌표 변환, JPEG 형식, 누락/손상 구분 및 헤더는 [ROUGH_YAW8_PREVIEWS.md](ROUGH_YAW8_PREVIEWS.md)를 참조합니다.
