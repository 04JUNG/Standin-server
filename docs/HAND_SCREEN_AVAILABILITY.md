# 필수 손 검사와 선택적 미리보기의 Blender 경합

2026-10-19 수정. 포즈 순위·BVH·카메라·skin lineage·권한을 바꾸지 않는다.
`clear_screen`만 유지하고 교차/미확인 후보는 기존대로 제외한다. 인물별 누적
45초·최대10회, HTTP1회 최대15초는 늘리지 않는다.

## 관측과 해결 범위

스테이징의 단일 Converter(2vCPU/4GiB, uvicorn worker1, Blender slot1)에서
필수 검사와 선택적 미리보기가 같은 세마포어를 썼다. 관측된 6번째 검사는
미리보기 뒤에서6.452초 대기하고6.361초 실행됐다. 검색은 누적45.01초에
종료했고 Converter의 정상 clear 응답은 그2.16초 뒤에 도착했다. 앞선 검사에는
실제 교차도 있었으므로 검사를 생략하거나 실패를 빈 검색 성공으로 바꾸면 안 된다.

새 `PrioritySlots`는 같은 프로세스 수 제한을 사용한다. 필수 검사 > 일반 출력 >
선택적 미리보기 순으로, 각 등급 안에서는 FIFO다. 대기열 기본8개로 sync worker를
무제한 점유하지 않는다. 큐가 미리보기로 가득 찼다면 필수 검사는 대기 중인
미리보기 한 건을 제거하고 진입한다. 필수 검사가 실행 중인 선택적 미리보기보다
늦게 도착하면 그 미리보기의 Blender process group을 종료한다. 실제 종료와 임시
디렉터리 정리 뒤에만 slot을 넘긴다. 중단된 결과의 부분 FBX/PNG/GLB는 반환하지 않는다.
필수 검사 완료 후200ms의 짧은 이어 실행 구간으로 연속된 검사 HTTP 호출 사이에
새 미리보기가 끼어드는 일을 줄인다. CPU/메모리 용량이나 동시 Blender 수는 늘리지 않는다.

이 정책은 **선택적 미리보기 경합**을 해결한다. 이미 실행 중인 일반 내보내기/scene은
선점하지 않는다. 긴 일반 출력이나 다른 필수 검사가15초를 점유하면 필수 검사는
여전히 unavailable일 수 있다. 원인과 대기/실행 시간을 기록하고 fail-closed를 유지한다.
큐는 runner/worker별이다. 현행 worker1 배포를 기준으로 하며 worker 수를 늘려
호스트 전체 동시 실행 제한을 우회하는 방식으로 사용하면 안 된다.

## 요청 수명과 로딩 담당 접점

ASGI middleware가 body backpressure를 보존하는 단일 receive pump로 disconnect를
관찰한다. thread-safe cancellation을 큐 대기와 실행 중 communicate polling에
전달한다. 취소/기한 소진 요청은 큐에서 제거하고, 실행 중이면 process group을
종료하고 slot/tempdir를 정리한다. scene도 같은 communicate 제어를 사용한다.

| 인터페이스 | 계약 |
|---|---|
| `X-Standin-Work-Budget-Ms` | HTTP 수신 시점부터의 남은 작업 예산. 양의 정수, screen 최대15000ms, 그 외 최대300000ms. 초과/비정상 입력400. 생략 시 각 상한 사용. 큐 대기도 포함 |
| `X-Standin-Screen-Optional: 1` | 반대손 참고 검사를 선택적 우선순위로 낮춘다. 미리보기를 필수 검사로 올리는 헤더는 없다 |
| `503 PREVIEW_PREEMPTED` | 더 중요한 검사 때문에 선택 작업 중단. `Retry-After: 1`. 기본 포즈 카드를 보존하고 사용자가 원할 때 제한된 재시도 가능 |
| `503 WORK_QUEUE_FULL` | 제한된 대기열 포화. `Retry-After: 1`. 무제한 즉시 재시도 금지 |
| `504 WORK_BUDGET_EXHAUSTED` | queue/execution 기한 소진. 안전 통과나 자산 부재로 취급하지 않음 |
| `499 REQUEST_CANCELLED` | 호출 측 취소/disconnect. 이미 끊긴 연결에서는 응답 수신을 기대하지 않고 서버 진단에 기록 |

Inference는 기존 per-call 제한 안에서150ms 응답 여유를 남긴 server 작업 예산을
보낸다. 구형 Converter는 새 헤더를 무시할 수 있으며 기존 fail-closed가 유지되지만,
그 조합에는 새 큐/취소 보장이 없다. 본후보 검사 client와 선택적 mirror client 상태는
세션별로 분리한다. optional mirror 실패가 본후보의 확정 검사 스냅샷을 무효화하지 않는다.

## 최종 scoped FBX와의 호환

기존 `/convert-framed`는 선택적 PNG/GLB와 최종 scoped FBX 양쪽에 사용된다.
동일 요청/응답만으로 호출 목적을 알 수 없으므로 **기존 경로는 선점 가능**하며
구형 scoped FBX 호출도 `PREVIEW_PREEMPTED`를 받고 재시도가 필요할 수 있다.

새 `/convert-framed-export`는 일반 출력 등급이다. `/convert-framed`와 동일한
BVH/인체/shape/손/scope/카메라/preview revision 및 출력 계보 검증을 실행하고,
`preview_format=png|model`을 동일하게 지원하며 PNG+FBX 또는 GLB+FBX pair를 반환한다.
최종 FBX도 GLB 확인 pair를 요구할 수 있으므로 형식을 임의로 좁히지 않는다. 경로는 scheduling 의도만
구분하며 export/release/refine 권한을 부여하지 않는다. 사용자 권한 검증은 기존 BFF에 남는다.

배포 시 Converter 새 경로를 먼저 준비한 뒤 BFF의
`src/output-scope/framingRoutes.ts` **최종 `format=fbx` 호출만** 새 경로로 연결한다.
선택적 detail/PNG/GLB 호출은 기존 경로를 유지한다. 이 BFF 변경은 로딩 PR 담당이다.
현재 PR은 server만 포함하며 PR 생성 단계에서 병합/배포하지 않는다.
새 경로를 알지 못하는 서버로 BFF를 먼저 배포하면404가 나므로 순서를 지킨다.

## 안전 진단

모든 작업 응답의 `X-Standin-Work-Class/Phase/Reason`,
`X-Standin-Queue-Wait-Ms`, `X-Standin-Execution-Ms`는 고정 enum/유한 수치다.
`/healthz.checks.work_queue`에는 capacity/active/등급별 queued 개수만 추가된다.
HTTP 오류 본문에는 고정 code와 같은 안전 진단만 있으며 내부 URL·예외 원문을 내보내지 않는다.

Inference의 `hand_skin_screen_check`/`hand_skin_screen_summary` 로그는 검사 번호,
필수/선택 여부, 대기/실행/전체 시간, 남은 예산, 실패 phase/cause, 제외/미확인 수를 남긴다.
검사별 `conversion_id`를 Converter의 동일 UUID/응답 헤더와 연결할 수 있다.
기존 `req_<UUID>` 형식의 X-Request-Id만 Converter로 전달해, 응답 수신 전 연결이
끊겨도 양쪽 로그를 연결한다. 임의 header 문자열은 tracing 값으로 반영하지 않는다.
`dynamic_search.hand_skin_screen.checks_diagnostics`에도 검사별 안전 근거를 보존한다.
원시 VLM·러프 이미지·BVH 본문·토큰·서명 URL을 넣지 않는다. HTTP 응답의 임의
실패 원문/비정상 timing 헤더도 그대로 로그에 전달하지 않는다.

## 검증과 한계

- 이벤트로 순서를 고정한 검사+대기 미리보기 경합, 큐 포화, 큐 취소/기한, FIFO,
  연속 필수 검사, 일반 출력 비선점, optional mirror 격리 및 disconnect 회귀.
- 실제 Python child가28초 실행하도록 구성한 fixture에서, 이미 실행 중인 미리보기를
  종료하고15초 예산의 필수 검사를2초 이내 완료. 이는 실제 Blender/스테이징 검증과 구분한다.
- 고정 Blender5.2.0/fbe6228777e7 + 실제 등록 BVH/FBX: 알려진 교차477쌍 검출,
  clear 포즈, 손 프리셋 적용 clear 및 artifact/입력 계보 검증 통과.
- 같은 실제 child FBX 반신 GLB와 실제 BVH/master FBX 손 검사로, 동일 worker/solver에서
  이전 세마포어 정책과 새 정책을 로컬 비교했다:

| 로컬 실측 | 이전 세마포어 | 새 정책 |
|---|---:|---:|
| 필수 검사 큐 대기 | 5911.8ms | 37.1ms |
| 필수 검사 실행 | 1409.7ms | 1348.9ms |
| 필수 요청 도착→결과 | 7.32s | 1.39s |
| 두 작업의 child-process CPU 누계 | 10.89s | 3.31s |
| 선택적 child 미리보기 | 완료 | 중단·process 종료 확인 |
| 변환 동시 실행 용량 | 1 | 1 |

표는 최종 코드의 재실측이다. 앞선 로컬 측정에서도 큐8.94초→0.025초,
요청10.99초→1.95초로 같은 방향을 확인했다(캐시/호스트 부하 차이는 통제하지 않음).
이 로컬 이전 정책 실행에서는 검사도15초 안에 성공했다. 스테이징2vCPU의 실패를
이 실측이 그대로 재현했다고 주장하지 않는다.28초 중간 도착의 실패 경계는 별도
fixture가 검증하며 실제 child 실행은 CPU 부하/종료/대기 절감의 근거다.
샌드박스 안에서는 입력 없는 Blender 기동도139로 종료되어, 실제 Blender 검증은
설치된 로컬 바이너리를 제한 밖에서 실행했다. 코드 변경 없이 배포 환경이 수정된 것이 아니다.
수정 코드는 아직 스테이징에 배포하지 않았으므로 동일 원본3종/native205의 최신 E2E는 남는다.

실제 자산 검증 도구는 `python -m tests.converter.verify_hand_skin_screen` 및
`python -m tests.converter.verify_screen_contention --help`를 따른다. 공유 로컬
벤치마크는 coordination의 atomic heavy-check lock을 잡은 동안만 실행한다.
