# 파츠 v4 서버 연결 — BFF 계약

`refine-v4-parts-http-v1`은 전용 scene worker의 **실제 native 파츠 실행 API**다.
기존 장면 보정의 `scene-refinement-v4`와 구분한다. `/analyze`와 기존 `/refine`의
정책·검색·기본 동작은 유지한다. [04J-318](https://linear.app/04jung/issue/04J-318).

## 호출 위치와 배포

- 전용 Blender worker의 기존 내부 주소/포트(8002), 기존 `SCENE_REFINEMENT_TOKEN` 사용.
- `GET /parts/capabilities`, `POST /parts/refine` 모두 Bearer 인증이 필요하다.
- scene plan/build와 파츠 실행은 같은 단일 작업 슬롯을 쓴다. 중복 실행은
  `503 "scene worker busy"` + `Retry-After: 5`. 업로드 중에도 슬롯을 예약한다.
- `SCENE_PARTS_ENABLED=1`이어야 실행된다. 기본은 비활성이며 capability의
  `available=false`, 실행 요청은 `503 PARTS_DISABLED`다.
- worker의 기존 asset/Blender/model 기동 검사도 통과해야 한다. Blender preflight는
  파츠 runner와 팔 어댑터, camera transport를 실제 import한다.
- converter와 scene-worker는 별도 이미지다. **둘 다 이 변경을 포함한 커밋으로
  빌드·배포**해야 저장 모듈 누락과 파츠 API가 함께 반영된다. 추론 서비스 배포만으로
  Blender worker는 바뀌지 않는다. production 활성화는 이 변경에서 하지 않는다.

코드: `src/scene_refinement/api.py` → `Engine.refine_parts` →
`src/refine_v4/service.py` → 고정 Blender subprocess `worker.py` →
CLI와 공유하는 `runner.execute`. 기존 CLI는 이 공유 구현을 호출한다.
외부 요청은 경로·실행 파일·Python 코드·donor FBX를 지정할 수 없다.

## capability

```json
{
  "contract_version": "refine-v4-parts-http-v1",
  "engine_version": "refine-v4-parts.1",
  "available": true,
  "operations": ["advance_foot", "recipe"],
  "catalog_sha256": "<catalog SHA256 or null>",
  "recipes": [{"id": "left-side-extension", "label": "왼다리 옆으로 뻗기", "description": "<검토된 정확한 보정 문장>"}],
  "automatic_image_extraction": false,
  "requires_explicit_intent": true,
  "requires_artifact_review": true
}
```

카탈로그가 없으면 `advance_foot`만 제공한다. `available=true`는 기동한 worker의
지원 여부이며 특정 입력의 피부/접촉 검사 통과나 보정 성공을 뜻하지 않는다.
기존 scene capability의 `readiness=configuration_only`로 이 API 실행 가능 여부를
대신하지 않는다.

## 요청

아래 예시의 placeholder는 실제 파일/출처/SHA로 대체한다. 전송 대상은 BFF가 소유권과
현재 선택을 검증한 **현재 full native FBX**다. 원래 평가 baseline을 대신 보내지 않는다.
크롭된 반신 FBX·다른 topology/가중치/길이·미지원 리그는 보정 성공을 보장하지 않는다.

```json
{
  "contract_version": "refine-v4-parts-http-v1",
  "engine_version": "refine-v4-parts.1",
  "request_id": "opaque-request-123",
  "current_asset": {
    "fbx_base64": "<현재 검수한 binary FBX>",
    "sha256": "<그 파일의 SHA256>",
    "review_reference": "<소유 작업/선택 revision의 검수 출처>"
  },
  "intent": {
    "source": "user_confirmed",
    "reference": "<명시적 수정 지시 또는 검토된 파츠 출처>",
    "description": "왼발을 인물 앞쪽으로 5cm 더 내민다"
  },
  "operation": {"type": "advance_foot", "side": "left", "distance_m": 0.05},
  "context": {
    "job_id": "job_example",
    "person_index": 0,
    "candidate_id": "selected-candidate",
    "input_sha256": "<원래 러프 SHA256>",
    "vlm_description": "<처음 러프 입력 시 VLM이 읽은 상세 설명 원문>",
    "user_instruction": "<후속 사용자 지시 또는 null>"
  }
}
```

- `request_id`: 영숫자/`_`/`-`, 1~128자. BFF가 소유 job과 매핑한다.
- 현재 FBX는 32MiB 이하, binary FBX magic과 SHA를 검증한다.
- `intent.source`: `user_confirmed` 또는 `reviewed_library`. VLM의 불확실한 좌우를
  자동 확정한 출처는 허용하지 않는다.
- `context`는 선택 필드다. 보내면 위 6개 필드를 모두 포함한다. 설명/지시는 null이
  가능하다. 원문과 후속 지시는 구분해서 request/context SHA에 결합한다. context를
  기하 목표로 임의 변환하지 않는다.
- `advance_foot`: 인물 자신의 `left`/`right`, 유한한 `0 < distance_m <= 0.1`.
  골반 앞방향과 현재 발끝 방향이 일치해야 한다. 고정 뼈 길이 IK·발목 높이·피부·저장
  재수입 검사를 통과한 경우만 적용된다. 접지력/체중 균형 검증을 뜻하지 않는다.
- 카탈로그 파츠: `operation`을 `{"type":"recipe","recipe_id":"<지원 ID>"}`로 보낸다.
  이때 `intent.description`은 capability에 나온 해당 recipe의 정확한 설명과 같아야 한다.
  다른 좌우/교차 지시를 같은 recipe에 붙이면 `409 PARTS_INTENT_MISMATCH`다.

이 API는 **명시적으로 검토한 파츠 선택을 실행**한다. 러프에서 파츠 ID/인물 좌우를
자동 추출하는 VLM 기능은 제공하지 않는다. BFF는 기존 VLM 원문을 보존하고 불확실한
교차나 좌우를 추정해서 실행하지 않아야 한다. 추가 팔 후속 커밋 `752fe0a`는 이 구현에
자동 포함되지 않는다. 현재 파츠 엔진의 지원 범위는 #151과 동일하다.

## 응답과 정확한 파일 유지

```json
{
  "contract_version": "refine-v4-parts-http-v1",
  "engine_version": "refine-v4-parts.1",
  "engine_executed": true,
  "request_id": "opaque-request-123",
  "request_sha256": "<원 요청 전체의 SHA>",
  "context_sha256": "<context SHA 또는 null>",
  "catalog_sha256": null,
  "status": "applied",
  "applied": true,
  "reason": null,
  "input_sha256": "<현재 원본 SHA>",
  "artifact_sha256": "<보정된 FBX SHA>",
  "fbx_base64": "<검사한 최종 FBX의 실제 바이트>",
  "intent": {"source":"user_confirmed","reference":"...","description":"..."},
  "operation": {"type":"advance_foot","side":"left","distance_m":0.05},
  "adjusted_limbs": ["left_leg"],
  "saved_verification": {"matrix_error":0.00001,"surface_error_m":0.000001},
  "artifact_contract": "verified-parts-fbx-v1",
  "requires_artifact_review": true,
  "release_allowed": false,
  "score_status": "not_scored"
}
```

`applied=true`는 actual Blender 실행, 저장 FBX 재수입, 관절 행렬·실제 피부 정점 오차
각 0.02mm 이하, 최종 피부/필요 접촉 검사 통과를 요구한다. 관절 오차는 행렬 성분 오차다.
피부 검사는 capped weighted triage의 비회귀 검사이며 완벽한 무충돌 인증은 아니다.

보정 거부/변화 없음은 HTTP 200 + `status=retained`, `applied=false`,
`reason=PARTS_REJECTED` 또는 `PARTS_UNCHANGED`, `adjusted_limbs=[]`,
`artifact_contract=retained-current-fbx-v1`. **이때 파일은 입력 현재 FBX와 바이트 단위로
같다.** 거부 결과를 과거 baseline으로 바꾸거나 보정 성공으로 세지 않는다.

`release_allowed=false`는 보정/유지 후보에 새로운 BFF 검수가 필요하다는 뜻이다.
서버가 라이브러리/rig의 자산 검수를 대신 승인하거나 사용자의 다운로드 권한을 부여하지 않는다.

## BFF 연결 순서

1. 소유 job/인물/후보와 현재 full FBX·체형·손·출력 범위 revision을 고정한다.
2. capability와 엔진 버전을 확인하고, 명시적 보정 의도로 이 API를 호출한다.
3. 반환 request/입력/출력 SHA와 실제 엔진 버전, `applied` 상태를 확인한다.
4. **반환 FBX를 기준으로** 필요한 출력 크롭/장면 조립·렌더만 수행한다. 이 파일을
   다시 원본 BVH에서 만들거나 기존 scene plan/build의 팔 IK로 덮어쓰지 않는다.
   기존 장면 자동 보정과 native 파츠 보정은 BFF에서 구분해서 라우팅해야 한다.
5. 후처리 후 최종 FBX의 SHA에 미리보기·카메라·review receipt를 묶고, 사용자가 확인한
   동일 바이트만 다운로드한다. 후처리로 SHA가 바뀌면 native 입력/출력 SHA도 계보에 남긴다.
6. 선택/설명/파츠/체형/손/범위/버전이 바뀌면 검수·캐시를 무효화한다. 실패는 복구 가능한
   원선택 저장과 구분한다. 기존 `refineAllowed=false`를 이 연결을 위해 강제로 바꾸지 않는다.

파츠 API에 BFF 호출을 추가하는 것은 별도 앱서버 작업이다. 이 서버 변경만 배포해서
기존 앱의 자동 보정이 새 엔진을 사용한다고 판단하지 않는다.

## 검토된 파츠 카탈로그 배치

`SCENE_PARTS_CATALOG`와 `SCENE_PARTS_CATALOG_SHA256`를 함께 설정한다. 경로는 운영자
설정에서만 받는다. 카탈로그는 검토된 immutable release의 파일을 참조해야 한다.
기존 scene release 아래에 넣는 경우 scene inventory/manifest SHA도 함께 갱신한다.
원본 러프·라이선스 모델·실제 FBX는 Git에 커밋하지 않는다.

```json
{
  "version": "refine-v4-parts-http-v1",
  "review": {"status":"approved", "reference":"<실제 자산 검토 기록>"},
  "files": {"left-extension.fbx":"<실제 SHA256>"},
  "recipes": [{
    "id":"left-side-extension",
    "label":"왼다리 옆으로 뻗기",
    "description":"오른다리로 지지하고 왼다리는 옆으로 길게 뻗는다",
    "request": {
      "method":"native_geometry",
      "part": {
        "fbx":"left-extension.fbx", "sha256":"<실제 SHA256>", "sides":["left","right"],
        "intent":{"source":"reviewed_library", "reference":"<검토 기록>",
                  "sentence":"오른다리로 지지하고 왼다리는 옆으로 길게 뻗는다"}
      }
    }
  }]
}
```

`directions`는 pin된 `legs.part_file`, `arms`는 pin된 `rig_profile`을 요구한다.
recipe의 구조는 [기존 CLI 문서](REFINE_V4_PARTS.md)를 따른다. 단, 파일은 release-relative
경로와 inventory SHA가 필요하다. symlink/경로 탈출/파일 변조를 거부하고, 실행마다 필요한
자산을 다시 해시 확인한 후 임시 디렉터리로 복사한다. 프로필·donor 경로는 capability나
응답으로 내보내지 않는다. 임시 입력/출력은 성공·거부·시간 초과 후 모두 정리한다.

## 오류

- 409 `PARTS_CONTRACT_UNSUPPORTED`, `PARTS_EXPLICIT_INTENT_REQUIRED`,
  `PARTS_RECIPE_UNAVAILABLE`, `PARTS_INTENT_MISMATCH`
- 422 `PARTS_INVALID_INPUT`
- 503 `PARTS_DISABLED` / `PARTS_UNAVAILABLE` / `PARTS_RELEASE_CHANGED`
- 504 `PARTS_TIMEOUT`
- 502 `PARTS_WORKER_FAILED` / `PARTS_INTEGRITY`

오류 본문은 `{"detail":{"code":"..."}}`. 기동 전에는 기존
`SCENE_WORKER_NOT_READY`, 바디 크기 제한은 기존 413 계약을 사용한다. 내부 예외·경로·Blender
출력·원본 이미지는 응답/로그에 내보내지 않는다. 시간 초과는 subprocess를 종료·대기한 뒤
슬롯을 해제한다. HTTP 취소도 실제 worker thread가 끝날 때까지 슬롯을 유지한다.
