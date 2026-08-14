# API 계약 — 도원 추론 서버 (FastAPI)

> 상태: 현재 계약 · 갱신일: 2026-08-11 · 기준 코드: `api/app.py`, `api/models.py`
>
> 이 문서는 **실제 구현된** HTTP 계약(`api/app.py`·`api/models.py`)을 문서화한다.
> `/export-order`의 상세는 별도 문서(`docs/EXPORT_CONTRACT.md`)에 있고, 여기서는 전체 엔드포인트와
> **앱 서버 팀 문서와의 경계·불일치**를 확정한다. 스키마의 단일 소스는 코드다 → FastAPI `/docs`(OpenAPI).

---

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

- 응답은 JSON. 무거운 산출물(BVH)은 **바이트 스트림 다운로드**(base64 인라인 아님).
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
| `POST` | `/analyze` | multipart PNG (+`hint`) | `CutResult` | 앱 서버 → 뷰어 Top-K 표시 |
| `GET` | `/pose/{pose_id}/bvh` | 경로 파라미터 | `application/octet-stream` | 동원 내보내기 |
| `POST` | `/export-order` | `ExportOrderRequest` | `ExportOrder` | 동원 내보내기 (→ `EXPORT_CONTRACT.md`) |
| `GET` | `/docs` | — | OpenAPI UI | 사람(계약 확인) |
| `POST` | `/refine` | `RefineRequest` | `RefineResponse` | 앱 서버 → 조정된 BVH (→ `REFINE_DESIGN.md`) |
| `GET` | `/refined/{handle}/bvh` | 경로 파라미터 | `application/octet-stream` | 동원 내보내기(조정본) |

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
  "refine_mode": "aggressive"           // v2.4: 기본 conservative, 명시 선택 aggressive
}
// 응답
{
  "pose_id": "Sitting Idle_01", "view": "front",
  "refined": true, "reason": "ok_partial",
  "bvh_url": "/refined/7d3ebff90f5064ec92ab3401/bvh",
  "loss_base": 0.599, "loss_final": 0.004, "gain": 0.993,
  "backend": "numpy",
  "refine_version": "v2.4.0",
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
    "cache_hit": false
  }
}
```

**`refined: false`는 오류가 아니다.** 안전 게이트가 조정을 버리고 베이스를 준 것이며,
이때 `bvh_url`은 `/pose/{pose_id}/bvh`가 된다. 클라이언트는 `bvh_url`만 따라가면
두 경우를 구분하지 않고 동작한다("좋아지거나, 그대로"). `reason` 값 목록은
`REFINE_DESIGN.md`의 안전 처리 기준.

- v1에서 `search_distance`는 베이스 불일치 게이트다. `REFINE_V2_ENABLED=1`에서는 거리·순위만으로
  실행을 막지 않고 진단에 남긴다. 대신 `refine_allowed`, 스켈레톤 상태·coverage·소유권 lineage와
  `refinable_limbs`를 모두 그대로 보내야 하며, 누락·불일치하면 fail-closed한다.
- 같은 입력은 같은 `bvh_url`을 돌려준다. 캐시 키에는 query/mask, 베이스 BVH content hash,
  feature·pose-library·refine code/config version, view·허용 부위·`refine_mode`가 포함된다. 실제 sidecar/BVH가
  존재할 때만 cache hit로 반환하고 설정·라이브러리가 바뀌면 자동 무효화된다.
- `refine_mode` 기본값은 `conservative`다. `aggressive`는 같은 hard safety gate 아래 보수적 단계를
  먼저 실행하고 hand/lap/lower pair와 제한적 Foot counter-rotation을 추가 시도한다. 공격적 단계가
  실패하면 보수적 artifact를, 보수적 단계도 실패하면 베이스를 반환한다.
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
    "pose_library_version": "v1",
    "feature_version": 1
  }
}
```

**최상위 필드**

| 필드 | 타입 | 의미 |
|---|---|---|
| `route` | string | `core`(전신·반신, 검색 수행) \| `bust`(흉상, 검색 스킵) \| `skip`(얼굴, 조기 종료) |
| `count_confidence` | string | `high`(검출기 개수 = VLM 개수) \| `low`(불일치 → 저신뢰 폴백) \| `n/a` |
| `detector_count` | int | 검출기가 센 사람 수 |
| `vlm_count` | int | VLM이 센 사람 수 (둘의 일치가 신뢰도 신호 — `CLAUDE.md` 불변식 §2) |
| `people` | Person[] | 인물별 결과. `route:"skip"`이면 빈 배열 |
| `notes` | string[] | 폴백 사유 등 사람이 읽는 메모 |
| `image` | object | 분석 기준 원본의 `width`, `height` |
| `inference_metadata` | object | 배포·VLM·포즈 backend/model·포즈 라이브러리·feature schema 버전 |

**`people[]` (PersonOut)**

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
| `coverage_class` | string | `full` · `reduced` · `sparse` · `insufficient` |
| `slot_origin` | string | `vlm` · `rtm_provisional` |
| `search_stability` | string \| null | `stable` · `ambiguous` · `unstable` · `not_required` · `not_available` |
| `valid_limbs` / `refinable_limbs` | string[] | 검색에 남은 부위와 refine 허용 사지 |
| `refine_allowed` | bool | v1은 검색+구조 정책, v2는 스켈레톤·소유권·coverage 안전 정책의 실행 허가 |
| `quality_reasons` / `quality_trace` | string[] / object | 구조 판정 사유와 배정·coverage·retry·검색 진단값 |

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

## 5. `GET /pose/{pose_id}/bvh` — 라이브러리 BVH 원본

동원 핸드오프. 선택된 후보의 BVH 파일을 **가공 없이** 반환한다. CSP 미러링·축 보정은 이 서버가 아니라 **동원 내보내기 단계** 책임(`DECISIONS.md` 결정 3).

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

**`POST /export-order`** — 작가가 고른 하나 → 동원 주문서(`ExportOrder`). 상세 계약·예시(1인/2인/얽힘/스킵)는 **`docs/EXPORT_CONTRACT.md`** 참조. 요약: `/analyze`(Top-K 보여주기)와 **별개 계약**이며, DB에서 `bvh_url`·`set_id`·`tags`를 채워 완성한다.

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

---

## 8. 확인 필요 (서버 관점, 팀 합의 항목)

클라이언트 `08_API_CONTRACT.md` §11 질문 목록의 **서버측 대응**이다.

1. **계층 구조** — ✅ **얇은 앱 서버(BFF) 분리로 결정**(`DECISIONS.md` 결정 4): 인증·Job·오류봉투·`/v1`은 BFF 소유, 이 서버는 순수 추론 유지. 남은 팀 확인은 "BFF를 누가/어느 레포에 만드나"와 도입 시점(결정 4 ⚠).
2. **동기 vs Job** — 클라이언트는 `POST /v1/analysis/jobs → 폴링` 비동기를 가정. 서버는 `POST /analyze` 동기 즉시 반환. 추론이 짧으면 앱 서버가 **동기 호출을 Job으로 감싸** 클라이언트에 폴링을 제공(권장). 추론이 길어지면 이 서버에도 Job/진행 단계 도입 재검토.
3. **단계별 진행률** — 클라이언트 status enum(`detecting`/`skeleton`/`pose_search`/`rendering`…)은 세분화돼 있으나, 서버는 동기라 **중간 단계를 스트리밍하지 않는다**. 서버가 만들지 않는 진행률을 앱이 임의 생성하지 않는다(클라이언트 원칙과 일치).
4. **업로드 검증** — 현재 서버는 크기·MIME 약검증(PIL 폴백). 최대 크기·허용 형식·손상 이미지 처리를 어느 층에서 강제할지(권장: 앱 서버 + 서버 방어적 재검증).
5. **후보 개수·신뢰도 라벨** — `top_k_final=5` 기본이나 폴백 시 후보가 적거나 없을 수 있다(`route:"skip"`은 빈 배열). `matchLevel` 라벨 산출에 서버의 `count_confidence`/폴백 신호를 넘길지.
6. **BVH 전달 방식** — URL 다운로드(`/pose/{id}/bvh`) vs 응답 바이트 인라인(작은 파일). `EXPORT_CONTRACT.md` §4와 동일한 미결 항목.
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
