# FBX converter V3.2.5 운영 승격 계약

## 범위

운영 변환기의 기본 solver는 `chain-transport-v3.2.5`다. 동결된 단계는 다음 순서로
실행된다.

1. V3.2.1 palm roll (`mu=0.5`)
2. V3.2.2 ankle swing
3. V3.2.3 ankle clearance
4. V3.2.4 contact foot-plant
5. V3.2.5 airborne plantar selector

V3.2.5가 바꿀 수 있는 최종 본은 `foot.L`, `toe.L`, `foot.R`, `toe.R`뿐이다.
측정 불가, 안전 게이트 실패, 결합 상태 실패, export 예외에서는 먼저 생성한
V3.2.4 FBX를 byte-exact로 복구한다.

## 런타임 격리와 무결성

- 운영 코드는 `qa/`를 import하지 않는다.
- `converter/SHA256SUMS.v325`가 solver 코드·정책과 dispatcher 13개를 동결한다.
- worker는 manifest 자체와 모든 항목의 SHA-256을 변환 전에 검증하고 불일치 시
  FBX를 내보내지 않는다.
- Docker image는 명시적 allowlist로 운영 파일만 복사한다.

## 썸네일 렌더 단계 (`POST /render-thumbnail`, 2026-09-04)

변환 뒤에 선택적으로 붙는 단계다. runner가 job에 `thumbnail` 블록(view·resolution·samples·
engines·tempdir 안의 PNG 경로)을 넣으면 worker는 **내보낸 FBX 바이트를 빈 씬에 다시
임포트**해 `converter/thumbnail_render.py`로 렌더한다(라이브러리 썸네일 빌드와 같은
anatomical 카메라·재질·조명). report의 `thumbnail`(sha256·size·engine)을 runner가 다시
검증한 뒤 API가 256px PNG/JPEG로 줄여 준다.

- solver 동결 범위(`SHA256SUMS.v325`) 밖이다. `convert.py`·`retarget.py`는 건드리지 않는다.
- `thumbnail: null`이면 기존 `/convert`·`/convert-bundle`과 바이트 단위로 같은 동작이다.
- job schema는 `3`이다(`thumbnail` 키 필수, null 허용).
- 렌더 실패는 `thumbnail_render_failed` → API `500 THUMBNAIL_RENDER_FAILED`. FBX는 버린다.

## 100STYLE 리그 프로파일 (2026-10-02)

`converter/bone_map.py`에 `100style` 프로파일(`STYLE100`)을 더했다. 100STYLE(CC BY 4.0)에서
뽑은 정리 포즈를 운영에서 FBX로 내보낼 수 있게 하기 위한 **매핑 데이터 추가**이고,
solver 단계(V3.2.1~V3.2.5)와 기존 프로파일은 바뀌지 않았다. `bone_map.py`가 동결 범위에
있으므로 `SHA256SUMS.v325`의 해당 줄과 `protocol.SOLVER_MANIFEST_SHA256`를 함께 갱신했다.

- 이름 체계가 다르다: 100STYLE의 `LeftShoulder`는 상완, `LeftElbow`는 전완, `LeftCollar`는
  쇄골이다(CMU·Mixamo의 `LeftShoulder`는 쇄골). 상부 척추는 Chest3를 건너뛰고 Chest4에 매핑한다.
- 판별 회귀: 운영 번들 BVH 1,248개(mixamo 635·cmu_bvh 353·mixamo_noprefix 260)의
  `resolve_profile` 결과가 변경 전후 동일. 정리 배치의 100STYLE BVH 296개는 판별 불가 →
  `100style`. 순수 파이썬 계약은 `tests/test_converter_profiles.py`.
- 실변환: Blender 5.2 운영 worker(동결 lineage 검증 포함)로 100STYLE 포즈를 Standin Master V2에
  변환 — 자동 판별 `100style`, 매핑 22본, `pose_fidelity_rmse` 0.150(골격 기준 0.142, Δ0.008),
  degenerate·missing·chain fallback 없음. 경고는 어깨 rest swing 17.8°(런타임 보정으로 흡수).
  변경 전 converter는 같은 입력을 "알려진 리그 프로파일과 매칭 실패(최대 일치 5본)"로 거부했다.
- 손가락: 다른 프로파일과 같이 canonical 22본만 변환한다(손가락은 캐릭터 기본 자세).
- refine(`src/refine.py`)은 `LeftArm`/`LeftForeArm` 이름으로 사지를 찾으므로 100STYLE 포즈는
  조정 없이 베이스를 돌려준다(`no_solvable_joints`). refine 지원은 별도 평가가 필요하다.

## 비상 복구

`CONVERTER_FORCE_EXACT_V324=true`를 converter 서비스 환경변수로 설정하면 runner가
서버 작성 job에 `force_exact_v324=true`를 잠가 전달한다. 사용자 HTTP 입력으로는
이 값을 변경할 수 없다. 이 모드에서도 V3.2.4 산출물을 임시 경로에 먼저 만들고
최종 경로로 byte-for-byte 복사한 뒤 해시 동치를 report에 남긴다.

## 승격에서 제외되는 문제

V3.2.5는 발/발가락 회전 selector다. 여성 메시의 허리선·가랑이 찢어짐, 골반
스키닝, DQ/LBS 선택, corrective shape는 이 solver의 변경 범위가 아니다. 해당
문제는 캐릭터 리그/웨이트 승격 게이트에서 별도로 다룬다.

## 필수 검증

- V3.2.1~V3.2.5 QA snapshot SHA-256
- 운영 `SHA256SUMS.v325`
- Blender converter 회귀 28/28
- V3.2.5 기본 경로와 V3.2.4 kill switch 실변환
- base/refined/mirror CLI E2E
- worker/API 계약, Docker runtime 격리
