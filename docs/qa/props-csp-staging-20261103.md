# 소품 검색·출력과 기존 체형 커스텀의 스테이징 통합

## 적용 범위

- 통합 코드: `d6e5026` (소품 PR #166, 체형 PR #155, 썸네일 PR #165).
- 검수된 170개 장면의 무기·일상 소품을 미리보기와 FBX에 함께 전달한다.
- 문장 검색은 기존 점수·적합도 판정을 유지하면서 같은 적합도 그룹 안에서 검수된 소품 장면을 우선한다.
- 기존 CSP 체형 레지스트리, 소스 모델, 저장 카드와 BFF는 변경하지 않는다.
- 소품 태그만 있고 검수된 장면이 없는 나머지 포즈에는 소품이 자동 생성되지 않는다.

## 재현 가능한 이미지

`Dockerfile.props-converter-overlay`와 `Dockerfile.props-inference-overlay`는 기존 스테이징 이미지의 의존성을 고정하고 필요한 런타임만 교체한다. 일반 배포 Dockerfile을 대체하지 않는다.

| 서비스 | 이미지 태그 | 이미지 SHA-256 |
| --- | --- | --- |
| converter | `props-csp-d6e5026` | `38c74e1b9151b80c71cbedc7b4a206bb5599e35419ebebe45c66c39d0c28ba4b` |
| inference | `props-search-d6e5026` | `bfddec96752b105a984cba815aba5648d42ae27617857480e7777b8a9dc79624` |

ECR 저장소는 스테이징 계정의 `standin/converter`, `standin/inference`이다. 변환기의 `DEPLOYMENT_VERSION`은 `d6e5026-props-csp-staging`, 검색 서버는 `d6e5026-props-search-staging`이다.

## 미리보기 캐시

- 스테이징 버킷의 `pose-previews/releases/props-r7-20261103/`에 5,862개 GLB와 대응 manifest를 업로드했다.
- 그중 검수된 소품 장면 170개의 소스 BVH, 레시피, FBX 계보를 대조했다.
- 업로드 전 전체 파일 해시와 GLB 내부 FBX 해시를 검증하고, 업로드 완료 후 `release.json`을 기록했다.
- 기존 경로를 덮어쓰지 않았다. `POSE_PREVIEW_URI`로 이 릴리스를 선택한다.
- 모델 revision: `d8c91a6125977e30822306af945b4832415cb7b3f7912710b0198aeaac1e4136`.
- 문장 검색에 `SEMANTIC_REVIEWED_PROPS_FIRST=1`을 적용한다.

## 배포 전 확인

- 관련 계약·체형·검색·썸네일 테스트: 504 passed, 1 skipped.
- 실제 Linux 배포 이미지에서 검, 좌우 반전 검, 소총, 창, 컵, 전화기, 책, 반신 컵의 FBX·GLB 소품 메시와 계보 검사 통과.
- 동일 이미지에서 등록된 `male-base-v0-7-6`에 키 185cm·근육 0.65를 적용한 검 출력 통과.
- `female-base-v0-3-9`에 키 165cm를 적용한 컵 출력 통과.
- 실제 shape recipe를 사용했다. 등록된 커스텀 체형은 기본값에서도 shape 경로가 필요하므로 고정 FBX만 가져오는 테스트로 대체하지 않는다.
- 위 검사는 구조·전달 검사이며, 모든 커스텀 체형에서의 손 접촉과 관통에 대한 시각적 보증은 아니다.

## 운영 및 롤백

배포는 기존 ECS task definition을 복사해 이미지와 명시한 환경 변수만 바꾼다. 기존 body-assets-init, 볼륨, shape registry, CSP 기능 플래그, 자격 증명 참조, 네트워크 설정을 보존한다.

변환기 이전 task revision은 `StandinStagingAppConverterTask5DE04741:30`, 검색 서버는 `StandinStagingAppInferenceTask76CF0AC4:107`이다. 문제가 발생하면 해당 스테이징 서비스의 task definition을 이 revision으로 되돌린다. 이때 기존 캐시 경로도 함께 복원된다. 운영 서비스에는 적용하지 않는다.

사용자가 앱 테스트를 직접 진행하기로 했으므로 배포 이후 체형 저장·재조회, 문장 검색 결과, 소품 배치와 FBX 가져오기의 최종 사용자 검수는 별도이다.
