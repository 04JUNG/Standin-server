# 전신 라우팅 스테이징 기동 복구

## 원인과 적용 범위

#92의 API factory는 미설정 시 전신 라우팅을 켜지만, 기존 배포 workflow는
`POSE_MODEL_VARIANT=cascade`를 렌더했다. CI의 코드/이미지 검사는 통과해도
실제 ECS 환경에서는 `humanart-m` 필수 검사에서 기동이 중단됐다.

이 수정은 전신 라우팅을 끄지 않는다. develop 배포 기본은 명시적 fullbody=1이며,
main 배포는 해당 environment가 따로 켜기 전에는 legacy=0이다. main에 이 PR을
자동 병합/배포하지 않는다. 명시적 `ROUTING_FULLBODY_ENABLED=0` 롤백도 보존한다.
상하체 분리 검색·합성은 추가하지 않았고, 손·다인 장면·원근 검색 모듈은 삭제하지 않는다.

## 배포 순서

1. 현재 ECS task definition을 기존 GitHub OIDC deploy role로 읽는다.
2. `prepare_fullbody_deployment.py`가 기존 역할·secret 참조·다른 설정을 보존하면서
   fullbody=1, rtmlib/humanart-m, canary-100, strict, asset 준비, manifest calibration을 설정한다.
   environment 변수에 명시된 variant/stage가 이 계약과 충돌하면 **ECS 변경 전에 실패**한다.
3. infra가 제공하는 `POSE_MODEL_URI`·`POSE_LIBRARY_URI`, 실제 VLM provider와 해당 key의
   secret 참조 존재를 검사한다. secret 값을 가져오거나 로그에 남기지 않는다.
4. 배포할 **동일 amd64 이미지**에서 `check_fullbody_startup.py`를 실행한다.
   실제 S3 라이브러리/모델을 준비하고 API와 같은 resource builder를 통과해야 한다.
   통과하지 못하면 새 task definition 등록/서비스 교체를 실행하지 않는다.
5. 검사 통과 후 서비스 cold-start grace 900초, 기존 container health check가 있다면
   startPeriod 300초를 사용한다. health probe 명령·실패 임계·rollback breaker는 유지한다.
6. 기존 ECS deploy action으로 배포하고 안정화를 기다린다.

이 PR 제출 자체는 5~6단계를 실행하지 않는다. develop 머지 뒤 workflow가 실행한다.

## 자산·보정 준비 계약

- Human-Art ONNX 2개/manifest: 기존 S3 provisioner의 SHA·크기·정확한 runtime version
  검사를 그대로 사용하고 실제 RTMPose startup probe도 실행한다.
- `POSE_CALIBRATION_FROM_MANIFEST=1`: 검증된 모델의 **전체 보정 프로필**을 함께 적용한다.
  현재 promoted manifest의 keypoint threshold는 0.35다. 모델만 바꾸고 current-X의
  0.3을 유지하는 두 번째 기동 실패를 막는다. HumanArtPoseModel의 exact calibration
  검사를 삭제/완화하지 않는다. cascade 및 미설정 legacy에는 적용하지 않는다.
- DB/BVH/태그/어휘 파일은 기존 배포 번들을 사용하며 다시 업로드하거나 수정하지 않는다.
- `ROUTING_PREPARE_ASSETS=1`: 암묵적 body/shapes 파일이 없다면 같은 DB/태그/BVH에서
  `<DATA_DIR>/fullbody-cache/<content-key>/`에 두 색인을 만든다. 원본 DB·태그·BVH 불변,
  태그/DB/실제 BVH 해시와 알고리즘 버전으로 키를 고정하고 완성된 쌍만 원자 공개한다.
- 명시한 cache 경로가 없거나 기존 cache가 낡았거나 손상됐으면 조용히 재생성하지 않고
  실패한다. 모든 비격리 검색 포즈의 coverage와 BVH hash가 일치해야 한다.
- 117시점 색인과 생성 rig 고정 SHA/어휘 파일도 확인한다. 격리 정책은 유지한다.
- Gemini/OpenAI API 호출이나 실제 러프 검색은 preflight에서 하지 않는다. VLM secret
  참조 확인은 key 유효성/모델 API 가용성 검사와 다르며, 실제 API 기동과 별도 canary가 필요하다.

## 필요한 권한과 확인

사용자의 로컬 IAM에 관리자/배포 권한을 추가하는 방식이 아니다. GitHub의 기존 deploy
role이 ECS 읽기·등록·UpdateService·PassRole·ECR 권한으로 배포한다. 새 이미지 preflight에는
**그 role의** 해당 라이브러리/모델 S3 prefix `s3:GetObject`도 필요하다. 없으면 preflight에서
막히며 IAM을 자동 수정하거나 다른 자격 증명으로 우회하지 않는다. 실제 Fargate task role의
S3 읽기/secret 주입 권한은 별도다. workflow는 VLM secret 값을 조회할 권한이 필요 없다.

현재 task definition의 VLM/URI 참조는 로컬 `ecs:DescribeTaskDefinition` 권한이 없어
확인하지 못했다. workflow가 환경별 실제 task를 읽어 검사하며, 참조 부재/잘못된 값은
운영 담당자가 해당 GitHub environment/infra에서 수정해야 한다. 코드 통과만으로 이를 보증하지 않는다.

배포 뒤 `/healthz.dynamic_routing.enabled=true`, `camera_count=117`, 실제 포즈 수와
`pose_backend=rtmlib`·실제 model identity/library version을 확인한다. 이 필드는
환경변수 문자열이 아니라 초기화된 engine을 보고한다. 이후 승인된 러프 canary로
`people[].dynamic_search`를 확인해야 최종 제품 연결을 검증한 것이다.

## 로컬 실파일 확인 (2026-10-17)

- staging S3 `pose-library/v1.tar.gz`의 고정 VersionId
  `y0HHT3EUqwKK5tgNZnoX6jWTSjmREcpN` 사본을 임시 폴더로만 내려받았다.
- library version `lib-20261016-ca99615b`, DB 5,795개, 기존 격리 11개를 제외한
  5,784개 cache 준비, semantic285 로딩 및 117시점 camera 자산 검사를 통과했다.
- promoted/approved Human-Art build `20260828.codex1-fp32-cpu-raw-simcc`의 실제 모델을
  S3에서 읽어 SHA 검증·rtmlib 0.0.15/onnxruntime 1.28.0·CPU startup probe를 통과했다.
- 생성 rig SHA와 taxonomy 파일도 통과했다. paid VLM 호출·S3 업로드·ECS 변경은 0회다.
- 로컬은 macOS arm64이므로 실제 Linux amd64/Fargate·VLM secret·서버 가동 성공을
  주장하지 않는다. amd64 이미지 preflight는 GitHub 배포 단계에 추가했다.

```sh
python -m pytest -q tests/test_fullbody_deployment.py tests/test_inference_deployment_contract.py
# 실제 모델/번들을 준비한 동일 배포 이미지에서, 위 rollout env와 함께:
python scripts/check_fullbody_startup.py
```
