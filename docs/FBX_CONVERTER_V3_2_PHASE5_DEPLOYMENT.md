# FBX Converter V3.2 Phase 5 배포 준비

> 정본: `docs/FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md` Phase 5. 이 문서는 AWS 배포에
> 필요한 converter 전용 계약과 아직 확정되지 않은 운영 결정을 분리한다.

## 현재 판정

- `standin-master-v2.fbx` SHA256
  `7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d` 확인.
- 2026-08-27 실제 모델의 `g1-move1`과 Mixamo control 통합 FBX를 사용자 육안 확인 완료.
- converter image·API·실제 모델 로컬 Linux/amd64 검증 완료.
- AWS 리소스 생성과 배포는 아직 하지 않았다. 아래 변수와 비용 정책이 확정될 때까지
  `CONVERTER_AUTO_DEPLOY_ENABLED`를 설정하지 않는다.

## 런타임 분리

| 항목 | inference | converter |
|---|---|---|
| ECR | `standin/inference` | `standin/converter` |
| ECS service | 기존 inference service | 별도 `standin-converter` service |
| task container | `inference`, port 8000 | `converter`, port 8001 |
| image | `Dockerfile` | `Dockerfile.converter` |
| Blender/bpy | 금지 | Blender 자식 프로세스에서만 허용 |
| 외부 노출 | 기존 BFF 경계 | BFF에서만 접근 가능한 내부망 |

`deploy/ecs/converter-task-definition.example.json`은 converter 전용 Fargate task의 기준이다.
`deploy/ecs/converter-service-network.example.json`은 public IP를 금지하는 service network 기준이다.
inference task definition이나 service에 converter container를 추가하지 않는다.

## 내부망과 보안그룹

ECS service는 `awsvpc`와 private subnet을 사용하고 `assignPublicIp=DISABLED`로 만든다.

- inbound: BFF security group에서 converter port `8001`만 허용
- internet-facing ALB와 public DNS 생성 금지
- 호출 주소: private Cloud Map 또는 internal ALB
- character EFS access point를 `/characters`에 read-only mount
- `standin-master-v2.fbx`는 Git·image·CloudWatch log에 넣지 않음
- task role은 character EFS mount에 필요한 권한만 부여
- execution role은 ECR pull과 `/ecs/standin/converter` log write만 허용

EFS 대신 S3 bootstrap을 선택하려면 task 시작 전에 해시가 고정된 artifact를 내려받는 별도 init
계약이 필요하다. 현재 예시는 read-only EFS를 정본으로 둔다.

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

`.github/workflows/converter-deploy.yml`은 converter 경로가 `main`에서 바뀔 때만 후보가 된다.
실제 job은 repository variable `CONVERTER_AUTO_DEPLOY_ENABLED=true`일 때만 실행된다.

필수 GitHub variables:

```text
CONVERTER_AUTO_DEPLOY_ENABLED
CONVERTER_AWS_DEPLOY_ROLE
CONVERTER_AWS_REGION
CONVERTER_ECS_CLUSTER
CONVERTER_ECS_SERVICE
```

워크플로는 기존 converter ECS service의 현재 task definition을 내려받고 `converter` container
image만 새 Git SHA로 교체한다. `inference` container가 발견되면 배포 전 실패한다. ECR repository와
ECS service를 자동 생성하지 않으므로 infra 소유자가 먼저 준비해야 한다.

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
2. EFS filesystem/access point와 character 배포 책임자
3. 최소 task `0` 또는 `1` 비용·UX 결정
4. CPU 2 vCPU / memory 4 GiB에서 실제 ECS 20회 측정 후 조정
5. CloudWatch retention·alarm: 5xx, timeout, p95, unhealthy task
6. BFF converter base URL과 connect/read timeout

이 여섯 항목이 확정되기 전에는 자동 배포 변수를 활성화하지 않는다.
