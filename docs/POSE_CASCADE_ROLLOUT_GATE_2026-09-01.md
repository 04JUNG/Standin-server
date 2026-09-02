# Pose cascade rollout gate — 2026-09-01

## 결론

`current-X → Human-Art M rescue cascade`는 고정 19컷 shadow 품질/비용 게이트와
로컬 endpoint 동시성 smoke를 통과했다. 그러나 이 결과는 배포 하드웨어의 worker SLO
증거가 아니며, Human-Art M 상용 사용 권한도 승인되지 않았다. 따라서 manifest는
`status: candidate`, `source.license_review: pending`을 유지하고 **live shadow와
`canary-5`는 시작하지 않는다.**

진행 순서는 다음과 같이 고정한다.

1. 모델·학습 데이터 권리자로부터 아래 범위를 포함한 서면 허가를 확보하고 내부 승인자가 서명한다.
2. 실제 배포 이미지/CPU·메모리 제한에서 worker 1/2/4 및 요청 동시성 매트릭스를 재측정한다.
3. 승인된 manifest로 production startup guard를 통과시킨다.
4. live shadow를 먼저 실행해 채택률 50% 이상, wrong-owner 0, 오류율/SLO 비회귀를 확인한다.
5. 그 뒤 별도 deployment에 트래픽 5%만 라우팅해 `POSE_CANARY_STAGE=canary-5`를 실행한다.

## 로컬 동시성 결과

측정 조건:

- 호스트: Apple Silicon macOS, ONNX Runtime `CPUExecutionProvider`
- API worker: 1개 프로세스 안의 `TestClient`
- 각 셀: warm-up 후 8요청
- 입력: 1인 `124629.png`, 5인 `131112.png`
- cascade: detector session/bbox 재사용, `POSE_CANARY_STAGE=shadow`
- 범위: **local smoke only; deployment hardware SLO evidence가 아님**

### current-X

| 인원 | 요청 동시성 | 오류 | p50 | p95 | 처리량 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 631.1ms | 661.2ms | 1.58 rps |
| 1 | 2 | 0 | 1,074.4ms | 1,088.8ms | 1.86 rps |
| 1 | 4 | 0 | 2,251.5ms | 2,308.8ms | 1.76 rps |
| 5 | 1 | 0 | 2,512.1ms | 2,701.1ms | 0.398 rps |
| 5 | 2 | 0 | 4,665.0ms | 4,985.9ms | 0.427 rps |
| 5 | 4 | 0 | 9,058.2ms | 9,415.0ms | 0.442 rps |

### cascade shadow

| 인원 | 요청 동시성 | 오류 | p50 | p95 | 처리량 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 608.6ms | 654.0ms | 1.63 rps |
| 1 | 2 | 0 | 1,106.9ms | 1,134.2ms | 1.81 rps |
| 1 | 4 | 0 | 2,172.9ms | 2,285.8ms | 1.83 rps |
| 5 | 1 | 0 | 2,649.2ms | 2,813.0ms | 0.377 rps |
| 5 | 2 | 0 | 6,010.1ms | 6,901.5ms | 0.336 rps |
| 5 | 4 | 0 | 12,721.8ms | 14,401.4ms | 0.314 rps |

5인 cascade는 동시성 증가 시 처리량까지 떨어진다. 이 호스트에서는 CPU 과구독과
fallback pose lock 경합이 있으므로, 배포 하드웨어 결과가 나오기 전 임시 안전값은
**API worker 1, 프로세스당 pose 요청 동시 실행 1**이다. worker 수를 늘려 해결된다고
가정해서는 안 된다. 프로세스마다 ONNX session/RSS가 복제되므로 실제 container memory
limit에서 OOM과 tail latency를 함께 측정해야 한다.

원본 산출물:

- `out/pose-cascade-load-20260901-v1/current-x/load_report.json`
- `out/pose-cascade-load-20260901-v1/cascade-shadow/load_report.json`

## 배포 하드웨어 worker 재측정 계약

### 현재 ECS 확인 결과

읽기 전용 `DescribeServices`로 확인한 운영 대상은
`StandinApp-ClusterEB0386A7-YtBcZrnPfn06`의
`StandinApp-InferenceService1C7A7625-KPZkcW87EjUE`다.

- launch type: Fargate / platform `LATEST`
- task definition: `StandinAppInferenceTask679E29CF:21`
- desired/running/pending: 1/1/0, rollout `COMPLETED`
- load balancer: 없음

현재 IAM 사용자에는 `ecs:DescribeTaskDefinition`, `ecs:ListTasks`,
`cloudformation:ListStackResources` 권한이 없다. 이 때문에 task vCPU·memory limit·container
command와 running task endpoint를 확인할 수 없고, 격리된 worker 1/2/4 task도 시작할 수 없다.
기존 task는 1개뿐이므로 운영 서비스를 수정해 부하 시험하지 않는다.

재개에 필요한 최소 권한/입력은 다음과 같다.

- 읽기: `ecs:DescribeTaskDefinition`, `ecs:ListTasks`, `ecs:DescribeTasks`
- 격리 테스트 task 생성·종료 권한 또는 동일 task image를 실행할 staging 서비스
- staging endpoint 및 요청 인증 방식
- 테스트 허용 시간대와 최대 부하

필수 기록:

- 배포 이미지 digest, CPU 종류/할당 vCPU, memory limit, execution provider
- ORT intra/inter-op thread 수와 프로세스별 session 수
- API worker 1/2/4 각각의 cold start, warm p50/p95/max, 처리량, 오류율, peak RSS/OOM
- 요청 동시성 1/2/4와 운영 peak, 1인 및 혼잡 컷
- current-X와 cascade shadow를 같은 이미지/설정/트래픽으로 분리 측정

통과 기준:

- 오류/OOM 0
- cascade shadow p95 및 peak RSS가 같은 worker/current-X 대비 각각 +20% 이내
- worker 증가가 처리량을 유의하게 높이지 않거나 p95를 악화하면 가장 작은 worker 수 선택
- live shadow 시작값은 측정에서 통과한 최소 worker/동시성 조합

## Human-Art 라이선스 검토

### 확인된 사실

- MMPose 코드 저장소 자체는 Apache-2.0이다.
- Human-Art 공식 저장소는 데이터셋을 CC 라이선스로 제공한다고 설명하면서,
  authorization form 작성과 **non-commercial purposes** 사용을 명시한다.
- 현재 사용 중인 `rtmpose-m_8xb256-420e_humanart-256x192` checkpoint의 상용 서비스
  추론·파생 ONNX·재배포 범위를 명시적으로 허용하는 별도 문구는 확인되지 않았다.
- MMPose issue #3271도 HumanArt 학습 checkpoint의 상용 사용/재배포 조건을 묻고 있으나
  2026-09-01 현재 공개 답변이 없다.

### 판정

**미승인 / production 차단.** Apache-2.0 코드 라이선스를 모델 가중치와 학습 데이터의
권리까지 확장해 해석하지 않는다. 저장소의 runtime guard가 production에서
`license_review != approved`를 거부하는 것이 현재 올바른 동작이다.

### 승인에 필요한 증거

권리자 또는 법무 승인 문서에는 최소한 다음이 포함되어야 한다.

- 이 서비스의 상업적/비상업적 성격과 사용 주체
- HumanArt로 학습된 checkpoint를 서버 추론에 사용하는 권리
- PyTorch checkpoint를 ONNX로 변환·보관하는 권리
- container/배포 artifact에 가중치를 포함하는 권리
- 고객 또는 제3자에게 가중치를 배포하지 않는 경우와 배포하는 경우의 구분
- attribution, 고지, 사용 제한, 만료/철회 조건
- 승인자, 승인 일시, 근거 문서 위치와 해시

승인 후에만 manifest를 새 사본으로 발행한다. 그 사본은
`source.license_review: approved`, rollout 단계에 맞는 `status`,
`verification.approved_by/approved_at`을 포함하고 manifest hash를 다시 고정한다.

참고 링크:

- <https://github.com/idea-research/humanart>
- <https://github.com/open-mmlab/mmpose>
- <https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/humanart/rtmpose_humanart.md>
- <https://github.com/open-mmlab/mmpose/issues/3271>

## live shadow → canary-5 실행 조건

live shadow는 실제 운영 트래픽을 cascade deployment에 복제하되 결과 슬롯을 교체하지 않는다.
다음 조건을 모두 만족해야 canary-5로 간다.

- license manifest와 deployment hardware matrix 승인
- 채택률 `would_accept / unresolved_before >= 50%`
- 수동 블라인드 검수 wrong-owner 0
- fallback 초기화/추론 오류율 및 hard fallback 비율 비회귀
- current-X 대비 p95/RSS +20% 이내

`canary-5`는 프로세스 내부 랜덤 샘플링이 아니라 배포 라우터가 안정적으로 5% 트래픽을
고정 배정한다. wrong-owner 1건이면 즉시 `POSE_MODEL_VARIANT=current-x`,
`POSE_CANARY_STAGE=off`로 롤백하고 승격을 중단한다.
