# 이미지 기반 라우터 1차 구현 로그와 롤백

> 2026-09-22 · 정책 rough-router-v1.1-phase1 · 슬롯 rough-slots-v1.
> 1차 제외: 그룹 관계 검색/조합, 하체만 보이는 입력.
> 현재 구현은 **SearchPlan 생성·검증·shadow 기록**이다. 실제 새 검색/조합/카메라 보정/승격은 실행하지 않는다.

## 1. 진행 상태

| 계획 단계 | 이번 결과 | 남은 작업 |
|---|---|---|
| P0 공통 계약 | 인물별 의미 슬롯, 관측, 검색 채널, SearchPlan 타입·정책 버전 구현 | P3/P4의 실제 재료/생성 포즈 계약 |
| P1 기준 정리 | 현재 DB·카메라·투영·refine·프로토콜 SHA256 기록, DB 변경 없음 | 물리적 앞뒤/깊이 규약 재검증·교정, 동일 DB의 의미 빌드 |
| P2 슬롯·라우터 | 슬롯 validator, 양 VLM provider의 opt-in 프롬프트, 결정적 라우터, pipeline shadow observer | 실제 러프에 새 VLM 호출 후 슬롯 품질 판정, bust에서 실제 추출/검색 연결 |
| P3~P7 | 미구현 | 부위 색인·검색·조합·공통 보정·작가 평가·실제 API 적용 |

P2의 순수 정책 구현은 P1과 독립적으로 진행 가능하다는 계획에 따라 먼저 구현했다. P1이 완료됐다고 간주해 검색/조합 승격을 진행하지 않는다.

## 2. 이번 코드

- `src/vlm/rough_slots.py`: versioned 인물별 의미 슬롯 파서. 누락→unknown, 중복/잘못된 person_index→확장 슬롯 거부, 가시성·근거·대안 검증. 기존 action을 새 의미로 복제하지 않는다.
- `src/experimental/rough_router.py`: image-only 계획. G/S/F/D 채널, A/B1 관측 팩트, 기하·의미 합집합 계획, 조건부 조합 계획. 실제 retrieval/compose 실행 없음.
- `src/experimental/rough_router_shadow.py`: 기존 결과를 변경하지 않고 JSONL 기록. person_index로 descriptor와 VLM 슬롯 연결, 관계 대상 인물도 제외. 실제 이미지 해시/정책/출처/의미 스냅샷/경로/보류 이유 기록.
- `src/config.py`, `src/vlm/client.py`, `src/pipeline.py`, `.env.example`: 기본 off와 opt-in 연결. Gemini/OpenAI 모두 기존 1회 이미지 분석 요청에 확장 슬롯을 요청할 수 있다.
- `scripts/replay_rough_router.py`: 봉인된 실제 러프 관절 또는 명시적인 합성 fixture로 계획 재현. 실제 데이터는 이미지/응답 해시를 확인한다.
- `scripts/rollback_rough_router.py`: 작업 전 preimage와 작업 후 해시를 비교해 이번 변경만 복원. 기본 dry-run.

새 Controlled Vocabulary는 확장 객체 안에서만 사용하며 기존 action/view enum이나 라이브러리를 재태깅하지 않았다. 기존 A/B1의 운영 게이트·순위 정책, 검색 거리·confidence·refine 자격도 변경하지 않았다.

### 확정 제외 경로

- `unsupported_group_relation`: 관계를 선언한 인물과 지목된 상대를 제외. 그룹 자산이 없는 상태에서 solo 조합으로 대체하지 않는다.
- `unsupported_lower_only`: 이미지에 하체만 보이는 경우. 전신이 보이지만 상체 추출만 실패한 사례는 구분해 의미 경로를 유지한다.
- 두 상태는 신규 SearchPlan의 결과다. **shadow는 기존 API 응답 자체를 차단/변경하지 않는다.** 새 라우팅의 실제 실행 모드는 아직 없다.

## 3. 실행 모드

기본값 / 즉시 운영 동작 복구:

```bash
EXPERIMENTAL_ROUGH_ROUTER_MODE=off
EXPERIMENTAL_ROUGH_SLOTS_ENABLED=0
```

환경변수를 반영해 프로세스를 재시작한다. off에서는 기존 프롬프트 그대로이며 observer import/파일 기록도 없다.

계획 기록만 활성화:

```bash
EXPERIMENTAL_ROUGH_ROUTER_MODE=shadow
EXPERIMENTAL_ROUGH_SLOTS_ENABLED=0
EXPERIMENTAL_ROUGH_ROUTER_LOG=out/rough-router/events.jsonl
```

확장 VLM 슬롯도 요청하려면 `EXPERIMENTAL_ROUGH_SLOTS_ENABLED=1`을 별도로 설정한다. 추가 VLM 요청은 하지 않지만 기존 한 요청의 프롬프트가 확장되므로 토큰 비용과 기존 VLM 필드 출력이 달라질 수 있다. 기존 응답 동일성 테스트는 **같은 VLM 출력/프롬프트를 사용하는 off↔shadow** 계약을 검증한다. 새 프롬프트를 사용한 라이브 호출 결과까지 동일하다고 주장하지 않는다.

`on` 모드는 거부한다. slots=1/router=off 조합도 거부한다. 슬롯 파싱/로그 기록 장애 시 기존 API 결과는 보존하고 서버 로그에 실패 종류를 남긴다. bust/face 조기 종료는 현 단계에선 유지하며 해당 로그에 `not_run_legacy_bust/skip`으로 관측 미실행을 명시한다.

## 4. 검증 결과

| 검증 | 결과 |
|---|---|
| 신규 규칙·소유권·파서·복구 테스트 | 30/30 통과 |
| 기존 smoke 계약 | 51/51 통과 |
| 기존 A/B1 계약 | 14/14 통과 |
| 문법/수정 파일 whitespace 검사 | 통과 |
| 합성 정책 fixture replay | 8건, 제외 두 경로·얼굴·반신·의미 전용 등 재현 |
| 봉인 실제 러프 replay | 28건: full_observed 15, upper_observed 3, unresolved_person 10 |

실제 replay의 10건은 프로토콜 `owner_verified`가 true가 아닌 사례이며 새 모델에서 오배정을 새로 확인한 수치가 아니다. 28건 모두 확장 의미 슬롯이 없는 예전 기록이다. 따라서 실제 의미 검색 0회, 조합 0회이며 정확도 향상 수치를 산출하지 않았다.

개발 중 첫 신규 테스트는 로컬 `.env`의 실제 pose backend를 테스트가 의도치 않게 사용해 실패했다. 테스트에 MockPoseModel을 명시 주입한 후 재검증했다. 최종 실패는 없다. 의도적으로 발생시킨 로그 저장 실패도 기존 결과가 유지되는지 검증했다.

### 산출물

모두 `artifacts/rough_router_phase1_20260922/` 아래에 있다.

- `asset-baseline.json`: P1 자산 해시와 미해결 카메라 상태.
- `router-test.log`, `smoke-test.log`, `ab1-test.log`: 테스트 기록.
- `fixtures.json`, `fixture-replay/`: 합성 정책 검증 자료. 실제 품질 근거 아님.
- `real-replay-final/plans.jsonl`, `real-replay-final/summary.json`, `real-replay-final/REPORT.md`: 실제 관절 기반 경로 재현.
- `before/`, `checkpoint.json`, `changes.patch`, `implementation-events.jsonl`: 변경·복구 기록.
- `rollback-dry-run.json`: 실제 체크포인트에 대한 복구 사전 검사.

현재 DB SHA256: `d30c39736fe7aea76f2a28212635905866124030a65e3db77bc320b0f7bc8360`.

## 5. 코드 롤백

이 작업을 시작할 때 이미 존재하던 수정분을 preimage로 보관했다. HEAD로 되돌리는 방식이 아니다. 신규 파일만 삭제하고 수정 파일은 이번 작업 직전 내용으로 복원한다.

사전 확인:

```bash
.venv/bin/python scripts/rollback_rough_router.py \
  artifacts/rough_router_phase1_20260922/checkpoint.json
```

실제 복원:

```bash
.venv/bin/python scripts/rollback_rough_router.py \
  artifacts/rough_router_phase1_20260922/checkpoint.json --apply
```

작업 이후 다른 수정이 있으면 **전체 복원 전에 중단**한다. 백업 해시도 검증한다. 롤백은 서버를 재시작하거나 실행 환경변수를 바꾸지 않으므로 위 두 flag를 끄고 재시작하는 운영 복구와 구분한다. 복구가 끝나도 로그/보고서 artifact는 보존한다. 실제 작업 트리에는 롤백을 적용하지 않고 사전 검사만 수행했으며, 실제 복원/충돌 보호는 임시 디렉터리 테스트로 검증했다.

## 6. 다음 구현 지점

후속 실행 기록: [phase2 카메라·의미 자산 검증](SEARCH_ROUTER_PHASE2_2026-09-22.md). 카메라 v2 로컬 검증과 현재 BVH 측정 갱신을 완료했고, 실제 VLM 슬롯은 Gemini 이미지 전송 승인 대기다.

1. 같은 러프에 새 슬롯을 추출해 인물 귀속·가시성·자세 가설을 검수한다. 기존 28건의 action 태그로 대체하지 않는다.
2. P1 카메라 앞뒤/깊이 규약과 현재 DB용 의미 빌드를 검증한다.
3. 확정 SearchPlan에 P3 전신/부위 G/S/F/D 실행 어댑터를 붙인다. 채널 실행과 plan_only를 로그에서 구분한다.
4. P4 조합·P5 공통 보정 후 동일 실제 모델 시트로 평가한다. 그룹 관계·하체 단독 제외는 유지한다.
