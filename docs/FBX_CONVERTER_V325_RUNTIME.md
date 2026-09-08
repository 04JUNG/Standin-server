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
