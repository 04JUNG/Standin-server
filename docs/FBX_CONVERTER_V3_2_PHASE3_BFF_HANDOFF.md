# FBX Converter V3.2 Phase 3 — BFF 핸드오프 계약

> 상태: Standin-server 내부 계약 및 E2E 기준
> 정본: `FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md` Phase 3
> BFF 구현 위치: 별도 저장소 `04JUNG/Standin-app-server`

이 문서는 BFF가 선택·refine 결과에서 최종 BVH 바이트를 하나로 확정해 내부 Converter API로
전달하는 경계를 고정한다. 이 저장소는 BFF 코드를 소유하지 않으며, inference의 legacy
`POST /export-order`를 refined artifact 저장소로 확장하지 않는다.

## 1. 최종 BVH 선택

인물별로 아래 분기를 정확히 한 번 실행한다.

```python
if refine_response is not None and refine_response.refined:
    assert refine_response.bvh
    final_bvh_bytes = refine_response.bvh.encode("utf-8")
    artifact_kind = "refined"
else:
    final_bvh_bytes = GET(base_bvh_url).content
    artifact_kind = "base"

final_bvh_sha256 = sha256(final_bvh_bytes).hexdigest()
```

- `RefineResponse.bvh_url`은 refined 여부와 무관하게 항상 베이스 `/pose/{pose_id}/bvh`다.
- `refined=true`의 유일한 조정본은 inline `bvh`다. 이를 버리고 `bvh_url`을 GET하면 refine이
  조용히 사라진다.
- `refined=true`인데 inline `bvh`가 비어 있으면 계약 오류다. 베이스로 몰래 바꾸지 않는다.
- `refined=false`는 정상 안전 fallback이다. 이 경우에만 베이스 URL의 응답 바이트를 사용한다.
- 조정본 문자열은 UTF-8로 인코딩한다. inference writer가 LF를 고정하므로 BFF가 개행을 다시
  직렬화하거나 정규화하지 않는다.

## 2. Converter 호출과 SHA 대조

BFF는 최종 바이트를 확정한 뒤 내부 `POST /convert`를 multipart로 호출한다.

```text
bvh                    final_bvh_bytes
character_id           registry ID
frame                   0
mirror                  false 또는 사용자의 명시값
output_mode             rigged_rest
apply_root_translation  false
```

Converter는 입력 BVH SHA를 독립 계산해 worker report와 구조화 로그에 남기고 성공 응답에 다음
헤더를 보낸다.

```text
X-Standin-Conversion-Id
X-Standin-Source-BVH-SHA256
X-Standin-Artifact-SHA256
X-Standin-Solver-Version
```

BFF는 성공 응답을 publish하기 전에 다음을 검증한다.

```text
X-Standin-Source-BVH-SHA256 == final_bvh_sha256
sha256(response FBX bytes)  == X-Standin-Artifact-SHA256
X-Standin-Solver-Version    == chain-transport-v3.2
```

불일치는 lineage/integrity 오류다. 해당 FBX를 저장·배포하지 않는다. BFF 로그에는 자기 Job 식별자와
converter의 `conversion_id`를 함께 기록해 양쪽 로그를 연결한다.

최소 BFF lineage:

```text
job_id
person_index
candidate_id 또는 pose_id
artifact_kind = base | refined
final_bvh_sha256
base_bvh_url
refined, refine reason, refine_version
character_id
mirror
conversion_id
source_bvh_sha256
fbx_artifact_sha256
solver_version
```

## 3. mirror 소유권

MVP에서 표준 BVH의 mirror는 Converter가 한 번만 적용한다.

- 기본은 `mirror=false`다.
- 사용자가 명시적으로 요청한 경우 BFF가 `/convert`의 `mirror=true`를 한 번 보낸다.
- BFF는 BVH rotation을 직접 미러링하지 않는다.
- Converter가 만든 FBX를 받는 CSP 단계는 같은 좌우 반전을 다시 적용하지 않는다.
- CSP는 소재 등록·배치·상대 위치 조정을 소유한다.

이 규칙은 Converter 도입 전 `DECISIONS.md`와 `EXPORT_CONTRACT.md`에 남은 “CSP 단계 mirror”
설명을 대체한다.

## 4. 다인 컷

다인 컷은 인물마다 최종 BVH 선택과 Converter 호출을 독립 수행한다.

```text
person 0 -> final BVH 0 -> POST /convert -> FBX 0
person 1 -> final BVH 1 -> POST /convert -> FBX 1
```

- 두 인물의 BVH 바이트를 이어 붙이거나 하나의 Blender job에 넣지 않는다.
- 각 item은 고유 `conversion_id`, source BVH SHA, output FBX SHA를 가진다.
- `set_id`와 `set_role`은 상호작용 묶음 메타일 뿐 다인 BVH를 뜻하지 않는다.
- 얽힘 세트는 현재 refine을 스킵하므로 각 인물의 베이스 BVH를 독립 변환한다.
- 상대 위치와 앞뒤 배치는 CSP/작가가 조정한다.

## 5. 오류와 fallback

| 상황 | 처리 |
|---|---|
| `refined=false` | 정상 베이스 선택 후 변환 |
| `refined=true`, inline `bvh` 없음 | BFF 계약 오류, silent base fallback 금지 |
| base BVH GET 404/409 | 다음 후보 선택 또는 원본 데이터 장애 처리 |
| converter 400/413/422 | BFF 입력·선택 계약 오류로 기록 |
| converter 503/504/500 | FBX publish 금지, 재시도/사용자 오류 정책 적용 |
| source BVH SHA 불일치 | integrity 오류, FBX 폐기 |
| output FBX SHA 불일치 | integrity 오류, FBX 폐기 |

Converter 실패 뒤 base와 refined 사이를 임의로 바꿔 재시도하지 않는다. 입력 종류를 바꾸는 fallback은
BFF가 사용자 결과와 lineage를 명시적으로 갱신하는 별도 결정이어야 한다.

## 6. 이 저장소의 실행 가능한 Phase 3 증거

`tests/converter/test_refined_bvh_e2e.py`가 다음을 고정한다.

1. inference의 실제 `/pose/{pose_id}/bvh` base 경로
2. `refined=true` inline 본문 우선 및 base GET 금지
3. `refined=false` exact base fallback
4. `refined=true`인데 inline 본문 누락 시 fail-closed
5. Converter 단일 mirror 전달
6. 다인 item별 독립 변환
7. BFF 계산 SHA = 응답 header = converter structured log = worker report

실제 BFF PR에서도 같은 시나리오를 공개 export URL과 영속 저장소까지 확장해 검증한다.
