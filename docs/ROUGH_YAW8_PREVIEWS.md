# Top5 수평 8방향 미리보기

`GET /pose/{pose_id}/poster?source_sha256=<sha>&scope=full&variant=rough-yaw8-v1&view=yaw_045`

기존 등록·격리·BVH SHA 검사를 통과한 원본에 대해, 로드한 라이브러리 번들의 `thumbs/scope8/` 정적 JPEG를 제공합니다. 요청 중 Blender·Converter·검색을 실행하지 않습니다. variant 없는 호출은 기존 PNG 계약을 유지합니다. 원본 검색과 내보내기의 yaw·pitch·roll·원근은 보존합니다.

## 기존 자산의 좌표 변환

`view`는 검색 `pose-preview-camera-v1` 좌표계의 45도 간격입니다. 기존 native 자산은 azimuth A일 때 물리 카메라 yaw N을 `180-A`로 만들었습니다. 카메라의 screen-right는 `-Ry(N)[0]`, 검색의 screen-right는 `Ry(Y)[0]`이므로 검색 yaw Y에는 **asset azimuth A = -Y mod 360**을 조회합니다. 검색 45도 → 기존 `yaw_315.jpg`. 이미지 반전이나 원본 포즈 회전은 수행하지 않습니다.

승인 renderer revision 두 개와 기본 캐릭터·solver SHA만 허용합니다. `directional_posters.py`가 source/scope/view·실제 JPEG 크기·해시·프레이밍·번들 선언을 검사하고, `rough_posters.py`가 좌표 변환과 폴백을 담당합니다. 제작기나 카메라 기준을 바꾸면 새 버전으로 검증해야 합니다.

## 응답과 폴백

- `X-Standin-Poster-Variant: rough-yaw8-v1`
- `X-Standin-Poster-Yaw`: 실제 제공한 검색 좌표계의 각도
- `X-Standin-Poster-Fallback`: `none`, `missing_angle`, `missing_directional_asset`, `unsupported_scope`
- `X-Standin-Projection-Matches-Search: false`
- native JPEG의 `X-Standin-Preview-View`는 실제 파일 azimuth를 보존합니다.

방향 파일 누락만 동일 source·scope의 검증된 0도 또는 기존 정면 PNG로 폴백합니다. 해시·메타데이터 손상은 404입니다. head는 8방향 자산이 없어 검증된 기존 head PNG만 정면 참고로 제공하며, 없으면 404입니다. 종이인형·다른 source·다른 scope로 대체하지 않습니다.

## 적용

기존 S3 라이브러리 다운로드 경로를 재사용합니다. 별도 S3 prefix 직접 조회나 IAM 변경은 없습니다. 번들에는 승인된 `directional_thumbnails` 선언과 `thumbs/scope8/` 이미지·개별 JSON이 필요합니다. 앞서 공유한 256 PNG/5범위/별도 prefix 제안과는 다른 기존 자산 형식입니다.

서버 지원과 자산을 먼저 준비하고 BFF·클라이언트를 연결합니다. 구 클라이언트는 PNG 경로를 유지합니다. 생성 포즈·정밀 미리보기·손/체형·refine·FBX의 기존 경로와 권한은 변경하지 않습니다.

검증: `tests/test_rough_posters.py`, `tests/test_directional_posters.py`, `tests/test_pose_posters.py`, `python tests/test_smoke.py`.
