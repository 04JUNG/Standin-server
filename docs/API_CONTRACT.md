# API 계약 — 도원 추론 서버 (FastAPI)

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
| `GET` | `/healthz` | — | `{ok, provider, pose_backend}` | 앱 서버 기동 확인 |
| `POST` | `/analyze` | multipart PNG (+`hint`) | `CutResult` | 앱 서버 → 뷰어 Top-K 표시 |
| `GET` | `/pose/{pose_id}/bvh` | 경로 파라미터 | `application/octet-stream` | 동원 내보내기 |
| `POST` | `/export-order` | `ExportOrderRequest` | `ExportOrder` | 동원 내보내기 (→ `EXPORT_CONTRACT.md`) |
| `GET` | `/docs` | — | OpenAPI UI | 사람(계약 확인) |

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
  "notes": []
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

**`people[]` (PersonOut)**

| 필드 | 타입 | 의미 |
|---|---|---|
| `index` | int | 컷 안 인물 인덱스(0부터) |
| `box` | float[4] \| null | `[x1,y1,x2,y2]` 픽셀 |
| `tags` | object | 이 인물의 의미 태그(§4) |
| `candidates` | Candidate[] | Top-K 후보(기본 `top_k_final=5`, `config.py`) |

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
{ "ok": true, "provider": "mock", "pose_backend": "mock" }
```

`ok`는 파이프라인 로드 여부. `provider`/`pose_backend`로 **현재 mock인지 실모델인지** 앱 서버가 확인할 수 있다.

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

1. **계층 구조 확정** — Tauri가 (a) 앱 서버를 거쳐 이 서버를 호출하나, (b) 이 서버를 직접 호출하나? → 인증·Job·오류봉투·`/v1`을 **누가 소유하는지**가 여기서 갈린다. (현 구현은 (a) 전제)
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
