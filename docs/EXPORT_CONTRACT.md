# Export 계약 — 선택 → 최종 BVH → FBX

> 현재 제품 흐름: 사용자 선택 → 선택적 `/refine` → **BFF가 최종 BVH 바이트 확정** → 내부
> Converter bundle API → V3.2 retarget FBX → BFF가 최종 BVH와 FBX를 함께 저장·다운로드 →
> CSP 소재 배치.
>
> `POST /export-order`는 DB의 base `bvh_url`만 채우는 legacy 원본 주문서다. refined artifact나
> 최종 FBX를 복원하지 못하므로 Phase 3 제품 export의 단일 소스로 사용하지 않는다.

> ⚠ **BVH는 1인만 지원한다(위치도 미반영).** 그래서 **export item 1개 = 1인 BVH 1개**다.
> 다인 컷은 인물 수만큼 item(=1인 BVH)이 나오고, 얽힘(포옹·격투)도 2인 BVH가 아니라
> **1인 BVH 여러 개를 `set_id`로 묶어** 표현한다. 상대 위치는 BVH가 안 실으므로 작가가 CSP에서 맞춘다.

**세 계약은 별개다:**
- `/analyze` = "Top5 **보여주기**"(인물별 후보 리스트). → `CutResult`
- `/refine` = 고른 후보 하나의 조정본 inline `bvh` 또는 정상 base fallback. → `RefineResponse`
- `/export-order` = base 선택만 표현하는 legacy 주문서. → `ExportOrder`
- `/convert-bundle` = 최종 BVH 원문과 그 바이트로 만든 V3.2 FBX를 무결성 manifest와 함께 반환.
- `/convert` = 하위 호환용 FBX 단일 응답.

FastAPI inference가 DB에서 legacy 주문서의 `bvh_url·set_id·tags`를 채운다. 최종 base/refined
선택과 artifact 영속화는 BFF가 소유한다.

---

## 1. Legacy 요청 — Tauri/BFF → inference (`POST /export-order`)

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

## 2. Legacy 응답 — inference → BFF (`ExportOrder`)

base 라이브러리 선택을 표현하는 JSON. refined 결과 이후의 최종 export에는 이 응답만 사용하지 않는다.

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

## 4. Phase 3 제품 export — BFF → Converter → CSP

BFF는 인물별 최종 BVH를 먼저 확정한다.

```python
if refine_response is not None and refine_response.refined:
    assert refine_response.bvh
    final_bvh_bytes = refine_response.bvh.encode("utf-8")
    artifact_kind = "refined"
else:
    final_bvh_bytes = GET(base_bvh_url).content
    artifact_kind = "base"
```

`RefineResponse.bvh_url`은 항상 베이스이므로 `refined=true`에서 사용하면 안 된다. BFF는 선택한
바이트의 SHA256을 lineage에 기록하고 인물별로 내부 `POST /convert-bundle`을 한 번 호출한다.

```text
multipart bvh           = final_bvh_bytes
artifact_kind           = base | refined (위 분기 결과를 명시)
expected_bvh_sha256     = sha256(final_bvh_bytes)
character_id            = standin-master-v2 등 registry ID
frame                   = 0
mirror                  = false 또는 사용자의 명시값
output_mode             = rigged_rest
apply_root_translation  = false
```

성공 응답은 `application/zip`이며 고정된 세 엔트리만 가진다.

```text
final.bvh
final.fbx
manifest.json
```

BFF는 ZIP을 publish하기 전에 다음을 모두 검증하고, 통과한 `final.bvh`와 `final.fbx`를 같은 Job의
최종 산출물로 저장한다.

```text
X-Standin-Artifact-SHA256          == sha256(response ZIP bytes)
X-Standin-Bundle-SHA256            == sha256(response ZIP bytes)
X-Standin-Source-BVH-SHA256        == sha256(final_bvh_bytes)
X-Standin-FBX-Artifact-SHA256      == sha256(final.fbx)
manifest.artifacts.bvh.sha256      == sha256(final.bvh)
manifest.artifacts.fbx.sha256      == sha256(final.fbx)
manifest.artifact_kind             == BFF가 선택한 artifact_kind
X-Standin-Solver-Version           == chain-transport-v3.2
```

mirror는 Converter가 한 번만 적용한다. CSP는 같은 좌우 반전을 다시 하지 않고 소재 등록·배치와
다인 상대 위치 조정을 담당한다. 다인 컷은 item 수만큼 독립 Converter 요청과 독립
`BVH + FBX` 쌍이 생긴다. `set_id`는 묶음 메타일 뿐 하나의 다인 BVH/FBX가 아니다.

상세 BFF 구현·lineage·오류 계약은
[`FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`](FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md)를 따른다.

---

## 5. 스키마 참조

- legacy `/export-order`: `api/models.py`의 `ExportOrderRequest` / `ExportOrder` / `ExportItem`
- inference/refine → converter Phase 3: `FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`
- 내부 `/convert-bundle` 및 호환 `/convert`: `converter_api/app.py`와 converter OpenAPI `/docs`
