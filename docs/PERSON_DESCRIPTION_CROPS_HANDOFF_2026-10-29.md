# 다인컷 자연어 설명 개선 — 서버 반영 인계

작성: 2026-10-29 · AI-assisted 검토, 사람 검수 전. 배포하지 않은 로컬 통합 브랜치다.

## 무엇을 바꿨고 왜 필요한가

| 구간 | 이전 | 이번 변경 | 기대하는 효과 / 한계 |
|---|---|---|---|
| 설명 입력 | 원본 전체 이미지와 여러 인물 박스를 한 요청으로 전달 | 설명 대상이 2명 이상이면 인물별 박스를 20% 확장(최소 8px)해 크롭. 크롭 좌표로 대상 박스를 환산하고 1명 스키마로 각각 호출 | 작은 인물에 더 집중하고 주변 인물과의 혼동을 줄이는 목적. 최초 박스 밖의 손·신체는 놓칠 수 있음 |
| 요청 지시 | 여러 인물의 설명을 함께 생성 | 크롭한 대상 1명만 설명하고 주변 인물·그림 속 인물을 섞거나 잘린 몸을 복원하지 말라는 지시 추가 | 의미 정확도 전부를 보장하지 않음. M03의 물건 오인은 남음 |
| 실행 | 다인 설명 요청 단위로 결과 처리 | 크롭 호출 최대 2개 동시 실행, 인물별 성공/오류 분리, 원래 슬롯에 연결 후 화면 순서 정렬 | 1명 실패가 다른 사람의 성공을 지우지 않음. 프로세스마다 동시 2개이므로 인스턴스/worker 수에 따라 전체 호출 수 증가 |
| 복구 | 설명 최초/재시도 각각 15초 | 최초 최대 15초, 일시 오류만 1회 최대 30초 재시도. backoff·좌우 검수·큐 대기를 포함해 설명 전체 45초 공유 | 45초 안에 항상 두 번을 온전히 실행하는 것은 아님. 남은 시간에 맞춰 단축. 긴 cooldown·일일 한도·비일시 오류는 재시도하지 않음 |
| 전체 기한 | 설명·검색 단계가 같은 요청 기한을 일관되게 공유하지 않음 | `/analyze` 시작부터 120초 기한을 API/파이프라인/설명 worker에 전달. 설명은 남은 요청 시간에서 검색용 15초를 뺀 범위로 제한 | HTTP 시간 초과는 504 `ANALYSIS_TIMEOUT`. 동기 CPU/native 작업을 강제 종료하는 기능은 아님. 15초 검색 예약도 검색 완료 보장은 아님 |

단일 설명 대상은 기존 원본 이미지 입력을 유지한다. 모델·라이브러리·검색 문턱을 변경하지 않았지만, 새 설명 내용에 따라 검색 대상/순위/후보 수는 달라질 수 있다. 스켈레톤 검출·FBX 출력 범위·팔/다리 리파인은 별도 과제다.

## 실제로 확인된 결과

동일 실패 원본 3장으로 실제 Gemini 3.5 Flash + Human-Art CPU + 스테이징과 동일한 라이브러리 `lib-20261021-21b1b3d9`를 사용해 새 로컬 요청을 실행했다. 체형 선택과 포즈 생성은 OFF, 포즈 추출과 설명 겹침은 ON이었다. 원본 Job은 수정하지 않았다.

| 컷 | 과거 스테이징 설명 | 새 설명 | 새 전체 요청 | 설명 단계 | 정상 후보 수: 최종 출력 인물 0 / 1 |
|---|---|---|---|---|---|
| M01 · Job 67e844f6… | 504 두 번 실패 | 2/2명 첫 호출 성공 | 33.9초 | 21.4초 | 0 / 0 |
| M02 · Job 3b49e047… | 재시도 성공 | 2/2명 첫 호출 성공 | 50.3초 | 18.2초 | 5 / 0 |
| M03 · Job dde715a5… | 504 두 번 실패 | 2/2명 첫 호출 성공 | 35.2초 | 10.7초 | 0 / 5 |

6/6명 설명 반환 및 semantic-valid는 확인했다. 이는 자세 정확도나 검색 성공률 100%라는 뜻이 아니다. M01의 앞쪽 얼굴 박스가 들어 올린 손을 포함하지 못했고, M03에서는 명확하지 않은 물건을 들었다고 해석했다. 과거 로그와 새 로컬 실행 비교라 동시 A/B 실험이 아니며, 크롭 하나만의 인과 효과나 속도 개선율을 산출할 수 없다. 실제 3컷은 모두 첫 시도에 성공하여 이번 실행으로 재시도 복구율을 측정하지 않았다.

실호출 코드는 `fix/describe-analysis-budget-120`의 `3f5c278`이다. 아래 최신 develop 통합 브랜치에서는 이 유료 실호출을 다시 실행하지 않고 오프라인 회귀 검증을 추가했다.

## 적용 가능한 서버 브랜치

- 통합 브랜치: `fix/person-description-crops-develop`
- 기준: `origin/develop`의 `029a50c` (2026-10-29 확인 시점)
- 통합 커밋 1: `6d5999b` — 요청 기한 공유·설명 재시도
- 통합 커밋 2: `2dc8310` — 인물별 크롭·개별 실패 격리
- 배포 설정 연결 커밋: `120c4e8`
- 원본 실험 커밋: `67155fc`, `3f5c278`
- 최신 develop의 refine v3 evidence/shadow와 scene-camera 변경을 보존했다.
- `.env.example` 충돌은 develop의 prefetch 기본 OFF를 유지하고 새 예산 설정을 추가했다. API 문서 충돌은 양쪽 설명을 모두 보존했다.
- feature → develop → main 순서로 검토한다. 실험 브랜치 전체를 develop에 merge하면 main 전용 기본값 등 불필요한 변경이 섞일 수 있으므로 이 통합 브랜치 또는 첨부 patch를 사용한다.

원격 브랜치 `origin/fix/person-description-crops-develop`에 푸시했다. PR 생성은 연결 도구 권한 부족으로 미완료이며 merge와 스테이징/운영 배포는 수행하지 않았다.

[develop 대상 PR 작성 화면](https://github.com/04JUNG/Standin-server/compare/develop...fix/person-description-crops-develop?expand=1)에서 변경 내용을 검토할 수 있다. reviewer 최소 1명의 확인 후 develop에 반영한다. 통합 후 develop이 더 진행됐다면 담당자가 최신 develop에 rebase/충돌 해결하고 CI를 다시 확인한다.

### 변경 파일

| 파일 | 반영 내용 |
|---|---|
| `src/vlm/person_crops.py` (신규) | 크롭/좌표 환산, 제한된 병렬 호출, deadline/cancel 전파, 결과 수집 |
| `src/vlm/client.py` | Gemini 설명 분기, 대상 한 명 지시, 시간 제한 재시도와 좌우 검수 |
| `src/search_routing.py` | `DescribeOutcome.person_errors/person_reports` |
| `src/pipeline.py` | 인물별 판정, 원래 슬롯 대응, 개별 오류/크롭 보고 |
| `src/analysis_budget.py` (신규) | 요청별 monotonic 기한 공유 |
| `src/description_prefetch.py` | 동일 요청 검증에 크롭 옵션 포함, 원래 기한 내 join |
| `api/hand_skin_middleware.py`, `api/app.py` | 모든 `/analyze`에 전체 기한/cancel 적용 |
| `src/config.py`, `.env.example` | 예산·크롭 옵션 |
| `.github/workflows/deploy.yml`, `scripts/prepare_description_deployment.py` | 환경별 명시적 rollout 값 연결, 미지정 값/기존 task 보존 |
| `tests/test_person_crops.py`, `tests/test_description_recovery.py`, `tests/test_analysis_budget.py` 등 | 실제 SDK를 로컬 장애 서버에 연결하는 테스트와 좌표/부분 오류/기한 회귀 |
| `CLAUDE.md`, `docs/API_CONTRACT.md` | 동작·설정·한계 문서 |

`person_reports`와 `person_errors`의 key는 **설명 요청 배열 인덱스**다. 최종 화면의 `people[].index`와 동일하다고 가정하면 안 된다. pipeline은 원래 slot에 연결한 뒤 최종 출력 순서를 정렬한다. 크롭 좌표는 신규 전신 좌표나 직접 관측 권한이 아니다.

## 스테이징 반영 설정

새 코드가 포함된 inference 이미지를 먼저 빌드해야 한다. 예전 이미지에 환경변수만 추가하면 구현이 생기지 않는다. 기존 VLM/모델/라이브러리/secret 설정은 유지하고 inference 컨테이너에 다음 값을 명시한다.

```dotenv
ROUTING_DESCRIBE_TIMEOUT_MS=15000
ROUTING_DESCRIBE_BUDGET_MS=45000
ROUTING_DESCRIBE_RESERVE_MS=15000
ROUTING_DESCRIBE_PERSON_CROPS=1
ROUTING_DESCRIBE_PERSON_WORKERS=2
PARALLEL_POSE_DESCRIPTION=1
```

`PARALLEL_POSE_DESCRIPTION=1`은 실험과 같이 포즈 추출과 설명을 겹치는 opt-in이다. 인물 크롭 분리 자체는 이 값이 0이어도 Gemini 설명 대상이 2명 이상이면 실행된다. develop 코드의 기본값은 0을 유지했다. `VLM_PROVIDER=gemini` 및 기존 전신 동적 검색 설정이 필요하며, mock/다른 provider에는 이 Gemini 크롭 구현을 적용했다고 주장하지 않는다.

**배포 환경변수 연결:** 이 PR은 `.github/workflows/deploy.yml`에 `Prepare optional description rollout overrides` 단계를 추가한다. GitHub staging 환경의 위 6개 Variables를 `scripts/prepare_description_deployment.py`로 전달해 다운로드한 task definition의 inference 환경변수에 적용한 뒤 새 이미지를 render한다. 비어 있는 변수는 기존 task 설정을 유지하고, 명시적 `0`은 롤백 값으로 적용한다. production 플래그를 workflow 기본값으로 켜지 않는다. 이미지·role·health check·다른 컨테이너·model/library URI·secret은 보존한다. 키 이름 충돌·중복 환경변수·잘못된 숫자/플래그는 AWS task 등록 전에 실패한다. 실제 GitHub Variables나 ECS 설정은 이번 작업에서 변경하지 않았다.

workflow는 main/develop push 외에 수동 실행도 지원하고 환경별 `AUTO_DEPLOY_ENABLED` 조건을 사용한다. develop merge가 조건에 따라 스테이징 배포를 유발할 수 있으므로 reviewer가 실제 환경/이미지 SHA를 확인한다. BFF/ALB/클라이언트가 더 먼저 끊으면 추론 서버의 120초 응답 전에 실패할 수 있다. 전체 120초는 inference `/analyze` 수신 시점 기준이므로 BFF 큐·통신 시간은 별도로 관측한다.

### 반영 후 확인

1. 배포 버전 SHA와 실제 컨테이너 환경변수를 확인한다. 기존 3개 Job 화면을 보는 대신 동일 원본을 **새 요청**으로 분석한다.
2. M01/M02/M03 각각에서 인물 수와 원본 대상 대응을 확인하고, `gemini_person_describe_start/summary`, `gemini_pose_describe_attempt/summary`, `routing_summary.describe`를 요청별로 연결한다.
3. 인물별 `body_details/search_text`가 저장/반환되는지와 설명 단계/전체 요청 시간을 확인한다. 동의·보관 정책에 의해 DB에 설명이 저장되지 않는 경우를 VLM 추출 실패와 구분한다.
4. 반환 설명뿐 아니라 semantic-valid 여부, 정상 후보 수, 참고 후보 수, 원본과의 시각 비교를 따로 확인한다. 설명은 나왔지만 후보 0인 경우는 다른 문제로 기록한다.
5. 부분 오류·전체 실패·120초 초과를 구분한다. 인물 한 명만 실패하는 회귀는 자동 테스트로 검증했으며 실제 Gemini 요청에 임의 장애를 주입할 필요는 없다.
6. 단일 인물/다인 순서 변경/초기 박스가 작은 컷을 함께 확인한다. 머리·상체 박스 밖의 손을 자동으로 복원하지 않는다. 원문/이미지 로그 보관은 기존 동의 정책을 따른다.

## 검증 결과와 롤백

최신 develop 통합본: 관련 pytest **360 passed + 6 subtests**, 배포 설정 회귀 **33 passed**, 스모크 **54/54**, compileall 및 diff check 통과. 기존 Starlette/httpx deprecation 경고 1건. 테스트는 로컬 장애 서버/모의 응답을 포함하며 실제 Gemini 유료 호출은 위 3컷 별도 실험이다. 전체 저장소 CI 및 실제 ECS 환경 검증은 아직 실행하지 않았다.

- 크롭만 끄기: `ROUTING_DESCRIBE_PERSON_CROPS=0` → 기존 전체 이미지/다인 batch 입력으로 복귀, 새 재시도/120초 기한은 유지.
- 포즈 추출 겹침만 끄기: `PARALLEL_POSE_DESCRIPTION=0` → 크롭 기능 자체는 유지.
- 수정 전체 롤백: 반영 전 승인된 이미지/task revision으로 되돌린다. 환경변수만으로 새 미들웨어의 전체 기한을 제거하는 옵션은 없다.
- 이 변경은 모델·라이브러리 재빌드나 데이터 마이그레이션이 필요하지 않다. 과거 실패 Job/저장 결과는 재작성하지 않는다.

갱신된 공유 패키지의 `develop-integration.patch`는 `029a50c` 기준 두 기능 커밋, 인계서, 배포 설정 연결을 포함한다. 해당 기준의 깨끗한 작업 브랜치에서 `git am develop-integration.patch`로 적용하고 CI를 실행할 수 있다. 다른 기준이면 충돌을 수동 검토한다. patch에 API 키·원본 러프·모델·DB는 포함하지 않는다.
