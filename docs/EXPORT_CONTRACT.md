# 동원 Export 계약 — 선택 → BVH 주문서

> 전체 흐름: 사용자 → **Tauri** → **FastAPI(도원)**: VLM·RTMPose·Retrieval → `/analyze`로 Top5 반환
> → Tauri가 뷰어에 표시 → **사용자가 Top5 중 선택** → **`/export-order`** → **동원 Export**:
> `pose_id → GET /pose/{id}/bvh` → CSP 미러링·축 보정 → **Clip Studio 내보내기**.

> ⚠ **BVH는 1인만 지원한다(위치도 미반영).** 그래서 **export item 1개 = 1인 BVH 1개**다.
> 다인 컷은 인물 수만큼 item(=1인 BVH)이 나오고, 얽힘(포옹·격투)도 2인 BVH가 아니라
> **1인 BVH 여러 개를 `set_id`로 묶어** 표현한다. 상대 위치는 BVH가 안 실으므로 작가가 CSP에서 맞춘다.

**두 계약은 별개다:**
- `/analyze` = "Top5 **보여주기**"(인물별 후보 리스트). → `CutResult`
- `/export-order` = 작가가 **고른 하나**를 동원에게 넘기는 **주문서**. → `ExportOrder`  ← 이 문서

FastAPI가 DB에서 `bvh_url·set_id·tags`를 채워 완성된 주문서를 만든다(라이브러리 메타 단일 소스).

---

## 1. 요청 — Tauri → 도원 (`POST /export-order`)

작가 선택만 보낸다. 각 선택은 뷰어에서 클릭한 후보의 `pose_id`+`view`.

```json
{
  "cut_id": "ep12_cut03",
  "source_image": "ep12_cut03.png",
  "selections": [
    { "person_index": 0, "pose_id": "stand_solo", "view": "front" }
  ]
}
```

| 필드 | 필수 | 설명 |
|------|:---:|------|
| `cut_id` | ✅ | 컷 식별자(회차_컷번호 등) |
| `source_image` | — | 원본 러프 파일명(추적용) |
| `selections[].person_index` | ✅ | 컷 안 인물 인덱스(0부터) |
| `selections[].pose_id` | ✅ | 선택한 후보 pose_id |
| `selections[].view` | ✅ | 선택한 후보 view(매칭된 투영 각도) |

---

## 2. 응답 — 도원 → 동원 (`ExportOrder`)

동원이 실제로 소비하는 최종 JSON.

```json
{
  "schema_version": "1.0",
  "cut_id": "ep12_cut03",
  "source_image": "ep12_cut03.png",
  "created_at": "2026-07-13T09:02:12+00:00",
  "items": [
    {
      "person_index": 0,
      "pose_id": "stand_solo",
      "bvh_url": "/pose/stand_solo/bvh",
      "view": "front",
      "set_id": null,
      "set_role": null,
      "tags": { "shot": "full_half", "action": "standing", "relationship": "solo", "view": "front" }
    }
  ],
  "notes": []
}
```

| 필드 | 설명 | 동원이 쓰는 곳 |
|------|------|----------------|
| `items[].bvh_url` | BVH 원본을 읽는 경로 | `GET {base}/pose/{id}/bvh` 로 다운로드 |
| `items[].view` | 매칭 시점 | 페이싱(좌우) 판단 참고 |
| `items[].set_id` | 얽힘 그룹 id(nullable) | 같은 값끼리 한 상호작용 |
| `items[].set_role` | 세트 내 역할(A/B) | solo면 null |
| `items[].tags` | 의미 태그 | 로깅·검증·파일명 |

---

## 3. 예시 모음

### 3-1. 1인 전신 (기본)
위 §1·§2 그대로. `items` 1개.

### 3-2. 2인 · 비얽힘(대화) — 인물별 BVH 2개
작가가 두 인물을 각각 선택 → `items` 2개. 동원은 BVH 2개를 각각 배치(위치는 작가가 CSP에서 최종 조정).

요청:
```json
{
  "cut_id": "ep12_cut07",
  "selections": [
    { "person_index": 0, "pose_id": "stand_solo", "view": "three_quarter" },
    { "person_index": 1, "pose_id": "reach_solo",  "view": "front" }
  ]
}
```
응답 `items`:
```json
[
  { "person_index": 0, "pose_id": "stand_solo", "bvh_url": "/pose/stand_solo/bvh", "view": "three_quarter", "set_id": null, "tags": {"shot":"full_half","action":"standing","relationship":"solo","view":"three_quarter"} },
  { "person_index": 1, "pose_id": "reach_solo",  "bvh_url": "/pose/reach_solo/bvh",  "view": "front", "set_id": null, "tags": {"shot":"full_half","action":"reaching","relationship":"solo","view":"front"} }
]
```

### 3-3. 2인 · 얽힘(포옹/격투) — 1인 BVH 여러 개를 set_id로 묶음
⚠ **2인 BVH는 없다.** 얽힘 포즈는 라이브러리에 **1인 BVH 2개(각 배우)**로 저장하고 `set_id`로 연결한다.
작가가 상호작용을 고르면 인물별로 **각자의 1인 BVH**가 선택되고, export는 **item 2개**를 내되 둘 다 같은 `set_id`를 가진다.

요청(인물0=배우A, 인물1=배우B):
```json
{
  "cut_id": "ep12_cut11",
  "selections": [
    { "person_index": 0, "pose_id": "hug_01_A", "view": "front" },
    { "person_index": 1, "pose_id": "hug_01_B", "view": "front" }
  ]
}
```
응답:
```json
{
  "schema_version": "1.0",
  "cut_id": "ep12_cut11",
  "created_at": "…",
  "items": [
    { "person_index": 0, "pose_id": "hug_01_A", "bvh_url": "/pose/hug_01_A/bvh", "view": "front", "set_id": "hug_01", "set_role": "A", "tags": {"shot":"full_half","action":"other","relationship":"hugging","view":"front"} },
    { "person_index": 1, "pose_id": "hug_01_B", "bvh_url": "/pose/hug_01_B/bvh", "view": "front", "set_id": "hug_01", "set_role": "B", "tags": {"shot":"full_half","action":"other","relationship":"hugging","view":"front"} }
  ],
  "notes": ["set_id='hug_01': 한 상호작용의 1인 BVH들 → 상대 위치는 작가가 CSP에서 조정."]
}
```
동원은 1인 BVH 2개를 각각 CSP에 얹고, 둘의 상대 위치는 작가가 맞춘다.
> 참고: 얽힘 세트 **검색·제작**(어떻게 매칭하고 라이브러리에 만들어 넣을지)은 별도 과제 — 설계문서 §10대로 태그+작가 미세조정. 비얽힘 2인이 MVP 우선.

### 3-4. 얼굴/스킵 컷
`/analyze`에서 `route:"skip"` → 후보 없음 → 이런 컷은 애초에 `/export-order`를 호출하지 않는다(작가 직접).

---

## 4. 동원 단계에서 하는 일(계약 밖, 참고)
받은 주문서로:
1. 각 `bvh_url`에서 라이브러리 BVH 다운로드
2. **CSP 좌표계 보정** — 좌우 반전(02 실험), 다리/루트 축(05 실험) → 이건 **도원이 아니라 동원 책임**
3. 필요 시 파일명·소재 폴더 규칙 적용 후 Clip Studio 소재로 내보내기

> ⚠ 확인 필요(동원): ① BVH를 URL 다운로드 vs 응답에 바이트 인라인 ② 미러링을 동원 단계에서(권장) vs 도원이 라이브러리에 미리 구움 ③ 다인 위치/앞뒤순서를 주문서에 넣을지(현재 미포함 — 작가가 CSP에서 조정 전제)

---

## 5. 스키마 참조
`api/models.py`의 `ExportOrderRequest` / `ExportOrder` / `ExportItem`가 소스. FastAPI `/docs`(OpenAPI)에 자동 노출된다.
