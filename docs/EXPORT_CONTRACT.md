# Export 계약 — 선택 → 최종 BVH → FBX

> 현재 제품 흐름: 사용자 선택 → 선택적 `/refine` → **BFF가 최종 BVH 바이트 확정** → 내부
> Converter bundle API → V3.2 retarget FBX → BFF가 최종 BVH와 FBX를 함께 저장·다운로드 →
> CSP 소재 배치.
>
> `POST /export-order`는 DB의 base `bvh_url`만 채우는 legacy 원본 주문서다. refined artifact나
> 최종 FBX를 복원하지 못하므로 Phase 3 제품 export의 단일 소스로 사용하지 않는다.

> ⚠ **BVH는 1인만 지원한다(위치도 미반영).** 그래서 **export item 1개 = 1인 BVH 1개**다.
> 다인 컷은 인물 수만큼 item(=1인 BVH)이 나오고, 얽힘(포옹·격투)도 2인 BVH가 아니라
> **1인 BVH 여러 개를 `set_id`로 묶어** 표현한다. BVH는 상대 위치를 담지 않는다.
> 04J-289의 추가 내부 API `/compose-scene`은 최종 인물별 FBX와 명시적 3D 배치를 받아
> **1컷 다인 FBX**를 만든다. 기존 인물별 BVH·FBX 흐름 뒤에 선택적으로 연결한다.

**세 계약은 별개다:**
- `/analyze` = "Top5 **보여주기**"(인물별 후보 리스트). → `CutResult`
- `/refine` = 고른 후보 하나의 조정본 inline `bvh` 또는 정상 base fallback. → `RefineResponse`
- `/export-order` = base 선택만 표현하는 legacy 주문서. → `ExportOrder`
- `/convert-bundle` = 최종 BVH 원문과 그 바이트로 만든 V3.2 FBX를 무결성 manifest와 함께 반환.
- `/convert` = 하위 호환용 FBX 단일 응답.
- `/compose-scene` = 최종 1인 rigged-rest FBX 여러 개 + Y-up 미터 배치 → `scene.fbx`와 manifest ZIP.

`/analyze`의 VLM 실패 원인·재시도 필드는 `API_CONTRACT.md` §7-1을 따른다.
이는 분석 실패 안내만 확장하며, 정상 스켈레톤·선택·BVH/FBX export 계약은 바꾸지 않는다.

체형 P0·P1의 decision/policy/suggestion은 `/analyze` 진단이다. suggestion을 export 체형으로 사용하지 않는다. 최종 체형은 기존 BFF 선택 결과를 사용하며 수동 선택·fixed_default·기본값 우선순위와 export 형식은 유지한다. [체형 계약](BODY_PIPELINE_API_CONTRACT.md).

FastAPI inference가 DB에서 legacy 주문서의 `bvh_url·set_id·tags`를 채운다. 최종 base/refined
선택과 artifact 영속화는 BFF가 소유한다.

### 전신 동적 라우팅 검수 후보의 경계 (2026-10-16)

`/analyze.people[].dynamic_search`의 원본 후보·`near_candidates`·`generation`·`basic_model`은
기존 자동 export 흐름으로 직접 연결하지 않는다. 새 경로는 `refine_allowed=false`,
`release_allowed=false`이며, 실제 인체 변환 후 검수와 사용자 선택이 필요하다.
생성 BVH는 `generation.candidates[].bvh` 본문으로만 제공되고 `/pose/{id}/bvh`에 등록되지 않는다.
로컬 손 피부 교차 검사 통과도 손-몸통/소품/동일 손 내부 전체의 충돌 인증은 아니다.
출력 범위는 `output_scope.mesh_scope`를 변환기의 `output_scope`에 전달한다
(`full`, 팔·손을 보존하는 `half`, 목·어깨선·머리만 남기는 `shoulder_head`).
이 PR은 기존 `/export-order` 요청 스키마나 운영 export 승인 정책을 확장하지 않는다.

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
`BVH + FBX` 쌍이 생긴다. `set_id` 자체는 묶음 메타다. 이 최종 FBX들을 명시적 3D 배치와
함께 `/compose-scene`에 넘기면 한 컷 다인 FBX를 만들 수 있다. 스켈레톤/2D bbox만으로
깊이를 복원하지 않으며 BFF/뷰어가 배치값을 확정한다. 호출 규격과 검증은
[`FBX_MULTI_PERSON_SCENE.md`](FBX_MULTI_PERSON_SCENE.md)를 따른다.

상세 BFF 구현·lineage·오류 계약은
[`FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`](FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md)를 따른다.

---

## 5. 스키마 참조

- legacy `/export-order`: `api/models.py`의 `ExportOrderRequest` / `ExportOrder` / `ExportItem`
- inference/refine → converter Phase 3: `FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`
- 내부 `/convert-bundle` 및 호환 `/convert`: `converter_api/app.py`와 converter OpenAPI `/docs`
# 후보 카메라와 FBX (2026-10-07 추가)

`POST /convert-framed`는 선택적 multipart `camera_rotation`을 받는다(JSON 3×3
proper rotation, Y-up). 사용 시 `preview_view=front`만 허용한다. 동결 solver 후
완성 모델 전체를 회전하고 저장한 FBX를 다시 읽어 preview를 만든다. 응답은 실제 적용한
`camera_rotation`을 그대로 되돌려준다. BFF는 요청과 다르거나 누락되면 실패 처리한다.

카메라는 FBX와 preview에만 적용한다. BVH는 회전하지 않는다. Mirror와 카메라를
동시에 요청하는 경로는 지원하지 않으며, exact V3.2.4도 회전을 허용하지 않는다.
전신·반신·흉상·두상 모두 같은 행렬을 사용한다. 관절·손가락의 상대 회전은 바꾸지 않는다.

### 대분류 추천과 Export 경계

`BODY_CLASSIFICATION_STRATEGY=coarse`의 `coarse_recommendation.recommended_body_id`는 캐릭터 대분류에 따른 프리셋 추천이다. 이를 승인된 `selected_asset` 또는 Export 주문으로 간주하지 않는다. 이 전략은 `applied_body_id=null`, `render_required=false`이며 기존 수동 선택·자산 QA·Export 검증 절차를 유지한다. 추천 출력과 자동 적용을 구분하는 필드는 [대분류 추천 계약](BODY_SELECTION/COARSE_RECOMMENDATION.md)에 기록한다.

coarse의 tentative 계열에 제공하는 반대 계열 BASE도 진단용 대안이다. `crop_box` 확장이나 추천 후보 변경은 Export 자산 권한·주문·자동 적용을 만들지 않는다.

선택적 `family_review`는 이미지 판독 근거와 추천에만 영향을 준다. 재검토로 계열이 교정되어도 `applied_body_id/selected_asset`은 null이며 Export 주문이나 자산 승인을 생성하지 않는다. 기존 수동 선택·기본값 처리와 Export 입력은 유지한다. [이미지 재검토 계약](BODY_SELECTION/FAMILY_REVIEW.md).
