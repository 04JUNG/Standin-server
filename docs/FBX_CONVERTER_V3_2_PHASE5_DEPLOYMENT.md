# FBX Converter V3.2 Phase 5 배포 준비

> 정본: `docs/FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md` Phase 5. 이 문서는 AWS 배포에
> 필요한 converter 전용 계약과 아직 확정되지 않은 운영 결정을 분리한다.

## 현재 판정

- `standin-master-v2.fbx` SHA256
  `7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d` 확인.
- 2026-08-27 실제 모델의 `g1-move1`과 Mixamo control 통합 FBX를 사용자 육안 확인 완료.
- converter image·API·실제 모델 로컬 Linux/amd64 검증 완료.
- 인프라 배선은 `Standin-infra#35`에 있으나 아직 merge·배포되지 않았다.
- 실측에 따라 초기 task 크기는 1 vCPU / 2 GiB로 정한다. 실제 캐릭터의 staging 측정 전까지
  `CONVERTER_AUTO_DEPLOY_ENABLED=true`로 바꾸지 않는다.

## 런타임 분리

| 항목 | inference | converter |
|---|---|---|
| ECR | `standin/inference` | `standin/converter` |
| ECS service | 기존 inference service | 별도 `standin-converter` service |
| task container | `inference`, port 8000 | `converter`, port 8001 |
| image | `Dockerfile` | `Dockerfile.converter` |
| Blender/bpy | 금지 | Blender 자식 프로세스에서만 허용 |
| 외부 노출 | 기존 BFF 경계 | BFF·inference에서만 접근 가능한 내부망 |

`deploy/ecs/converter-task-definition.example.json`은 converter 전용 Fargate task의 기준이다.
`deploy/ecs/converter-service-network.example.json`은 public IP를 금지하는 service network 기준이다.
inference task definition이나 service에 converter container를 추가하지 않는다.

## 내부망과 보안그룹

ECS service는 `awsvpc`와 private subnet을 사용하고 `assignPublicIp=DISABLED`로 만든다.

- inbound: BFF security group과 **inference security group**에서 converter port `8001`만 허용
  (inference는 2026-09-04부터 `POST /refine` preview를 `POST /render-thumbnail`로 그린다 —
  `docs/API_CONTRACT.md` §refine thumbnail)
- internet-facing ALB와 public DNS 생성 금지
- 호출 주소: private Cloud Map 또는 internal ALB
- registry의 `s3://` URI를 ECS task role로 내려받아 SHA-256 검증 후 전용 temp cache에 보관
- `standin-master-v2.fbx`는 Git·image·CloudWatch log에 넣지 않음
- task role은 등록된 character object를 읽는 데 필요한 `s3:GetObject`만 부여
- execution role은 ECR pull과 `/ecs/standin/converter` log write만 허용

private subnet에서 S3를 읽으려면 S3 gateway endpoint 또는 승인된 egress 경로가 필요하다.
`converter_api.registry`는 HTTP(S)를 받지 않고, 환경변수로 지정한 S3 object만 lazy download한다.
다운로드 결과가 registry hash와 다르면 cache와 응답 artifact로 승격하지 않는다.

## inference → converter (refine preview)

`POST /render-thumbnail`은 BFF가 아니라 **inference 서비스**가 부른다. 조정본 BVH를
V3.2.5로 변환한 뒤 라이브러리 썸네일과 같은 anatomical 카메라로 렌더해 256px 이미지를
돌려준다(FBX는 반환하지 않는다). 배포에 필요한 것:

1. converter security group inbound에 inference SG → `8001` 추가
2. inference task definition environment에 `REFINE_THUMBNAIL_CONVERTER_URL`
   (BFF가 쓰는 converter 내부 주소와 같은 값, 예 `http://standin-converter.internal:8001`)
   — `POSE_MODEL_URI`처럼 infra가 넣고 `deploy.yml`은 존재만 확인한다
3. converter 이미지는 `Pillow`가 필요하다(`requirements-converter.txt`)

렌더 엔진은 `CONVERTER_THUMBNAIL_ENGINES`(기본 `BLENDER_EEVEE,CYCLES`) 순서로 시도한다.
Fargate에는 GPU가 없으므로 EEVEE는 Mesa 소프트웨어 GL로 돌거나 실패할 수 있다. 실패하면
같은 프로세스에서 Cycles(CPU)로 넘어가고 응답 헤더 `X-Standin-Thumbnail-Engine`과
`converter_thumbnail_complete` 로그의 `engine`/`engine_attempts`에 무엇을 썼는지 남는다.
EEVEE가 프로세스를 죽이는 환경이면 `CONVERTER_THUMBNAIL_ENGINES=CYCLES`로 고정한다.
`converter-ci.yml`의 HTTP smoke가 실제 Blender 5.2 이미지에서 `/render-thumbnail`을
호출해 256×256이고 회색 배경 위에 어두운 실루엣이 있는지 검사한다(단색이면 실패).

## Health와 timeout

- image와 ECS health check: `GET /healthz`, 30초 간격, 15초 timeout, start period 120초
- health는 API·Blender `5.2.0/fbe6228777e7`·tempdir·기본 character SHA를 모두 검사
- conversion timeout: 30초
- timeout 시 Blender process group 종료 grace: 2초
- ECS stop timeout: 45초
- task당 Blender 동시 실행: 1
- 서비스 health-check grace period: 최소 120초

## CloudWatch 구조화 로그

`Dockerfile.converter`는 `CONVERTER_JSON_LOGS=1`을 기본 설정한다. `standin.converter`의 각
메시지는 JSON 한 줄이며 awslogs가 `/ecs/standin/converter`로 보낸다.

성공 이벤트 `converter_complete`의 검색 키:

```text
service
version
event
conversion_id
source_bvh_sha256
artifact_sha256
task_cold_start
queue_wait_ms
execution_ms
request_total_ms
report
```

입력 BVH 본문, character 경로·바이트, 사용자 파일 내용은 기록하지 않는다. 실패 이벤트도
`conversion_id`, HTTP status, 안전한 error code만 기록한다.

## 배포 workflow

`.github/workflows/converter-deploy.yml`은 converter 경로가 `main` 또는 `develop`에서 바뀔 때
후보가 된다. `main`은 GitHub environment `beta`와 `:latest`, `develop`은 `staging`과
`:develop`을 사용한다. 실제 job은 repository variable
`CONVERTER_AUTO_DEPLOY_ENABLED=true`일 때만 실행된다.

repository-level GitHub variables:

```text
CONVERTER_AUTO_DEPLOY_ENABLED
CONVERTER_AWS_DEPLOY_ROLE
AWS_REGION
```

`beta`와 `staging` environment에 각각 필요한 variables:

```text
CONVERTER_ECS_CLUSTER
CONVERTER_ECS_SERVICE
```

초기 부트스트랩에서는 `CONVERTER_ECS_SERVICE`를 비워 둔다. 이때 워크플로는 ECR image만
빌드·push하고 ECS 단계는 건너뛴다. 그 image tag로 infra가 converter service를 만든 뒤 환경별
service 변수를 채우고 다시 실행한다. 이후에는 현재 task definition의 `converter` container
image만 새 Git SHA로 교체한다. `inference` container가 발견되면 배포 전에 실패한다. ECR
repository와 ECS service 자체는 infra가 준비한다.

## warm/cold latency 측정

private endpoint에 접근 가능한 환경에서 새 task 직후 probe를 시작한다.

```bash
python scripts/measure_converter_latency.py \
  --base-url http://standin-converter.internal:8001 \
  --bvh /path/to/approved.bvh \
  --warm-iterations 20 \
  --json-out converter-latency.json
```

측정값은 다음을 분리한다.

- probe 시작부터 `/healthz` 준비까지의 `readiness_ms_from_probe_start`
- task의 첫 변환 여부 `first_request.task_cold_start`
- semaphore 대기 `queue_ms`
- Blender 실행·검증 `execution_ms`
- HTTP 전체 `wall_ms`
- warm 20회 p50·p95·max

초기 목표는 ECS warm 20회 `p95 <= 3000ms`다. Apple Silicon의 `linux/amd64` 에뮬레이션
측정은 기능 참고값이며 ECS SLO 승격 근거로 쓰지 않는다.

### 2026-08-27 로컬 기준값

- image: `standin-converter:phase5-local`
- image ID: `sha256:8210db52f8ae92fe84038cc9160dff202ae8ecccf871610a61826a80cb448875`
- 환경: Docker Desktop on Apple Silicon, `linux/amd64` 에뮬레이션
- character: 승인 `standin-master-v2` SHA `7c648b97…01f3d`
- BVH: `g1-move1` SHA `f5a1ff1a…ed835`
- readiness from probe start: `253.549ms`
- first request: wall `3583.480ms`, execution `3070.909ms`, task cold=`true`
- warm 20회 wall: p50 `3292.149ms`, p95 `3448.647ms`, max `3605.355ms`
- warm 20회 execution: p50 `2785.769ms`, p95 `2922.771ms`, max `3007.394ms`
- warm queue p95: `0.014ms`
- report SHA256: `c96982687385e3d8364e3351f0e03f0a20960d13281dd9a8caca87a747235e30`

에뮬레이션 wall p95는 목표보다 `448.647ms` 느리다. execution p95는 목표 안이지만 ECS에서
network transfer를 포함한 wall 20회를 다시 측정하기 전에는 SLO PASS로 판정하지 않는다.

## 아직 필요한 운영 결정

1. beta/prod private subnet·security group·Cloud Map/internal ALB 선택
2. production/staging assets bucket에 승인 character 업로드 및 S3 endpoint 확인
3. 최소 task `0` 또는 `1` 비용·UX 결정
4. CPU 1 vCPU / memory 2 GiB에서 실제 캐릭터 ECS 20회 측정 후 조정
5. CloudWatch retention·alarm: 5xx, timeout, p95, unhealthy task
6. BFF converter base URL과 connect/read timeout

이 여섯 항목이 확정되기 전에는 자동 배포 변수를 활성화하지 않는다.
