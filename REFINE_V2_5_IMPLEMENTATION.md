# Refine v2.5 Safe Aggressive Selector 구현 설계

작성일: 2026-08-17  
상태: 구현 전 검토본  
기준 코드: Refine v2.4.0  
기준 평가:

- `REFINE_V2_5_AGGRESSIVE_PROMOTION.md`
- `REFINE_V2_5_CURRENT_ROUGH_EVAL.md`
- `out/eval/v25_current_rough_near_gap_d0_20260817/REPORT.md`

---

## 1. 결정

v2.5는 aggressive의 변형 강도를 낮추는 버전이 아니다. **근접 공백을 줄이는 raw aggressive 후보는
유지하고, 실제 반환 artifact를 aggressive/conservative/base 중에서 안전하게 선택하는 버전**이다.

```text
REFINE_DEFAULT_MODE=aggressive

의미:
  aggressive 후보를 우선 시도한다.
  raw 후보를 그대로 반환한다는 뜻이 아니다.
  최종 selector가 aggressive / conservative / exact base 중 하나를 반환한다.
```

raw aggressive는 평가·진단용 중간 산출물이며 제품 URL·cache·export가 직접 가리켜서는 안 된다.

### 구현 판단

- `IMPLEMENT_V2_5_SAFE_AGGRESSIVE_SELECTOR`
- 구현은 진행한다.
- raw aggressive의 config 직접 승격은 하지 않는다.
- v2.5 final 정책이 D0 회귀와 별도 승격 테스트를 통과한 뒤에만 기본 mode를 aggressive로 바꾼다.

---

## 2. D0에서 확인된 사실

### aggressive의 보존 대상

Raw aggressive는 conservative 대비 다음 이득을 보였다.

| 지표 | 상대 오차 감소 | better/tie/worse |
|---|---:|---:|
| Joint NME | 7.89% | 19/4/4 |
| Endpoint NME | 9.65% | 19/4/4 |
| Hand-pair | 23.61% | 5/1/0 |
| Lower-pair | 22.56% | 4/4/0 |

작가 블라인드 비교는 aggressive 9승, conservative 3승, tie 12, both-bad 3으로
`NetPreference_A=+22.2%p`였다. 손·팔 모으기와 다리 모으기는 v2.5에서 약화하면 안 된다.

### raw candidate에서 발견된 구조 실패

- `131112:p0`: 원본 base 대비 누적 회전 `50.449782°`, trust region `45°` 초과
- `131112:p2`: 신규 `right_arm:torso` 충돌
- `131112:p3`: 신규 `right_arm:torso` 충돌
- `171734:p0`: 신규 `left_arm:left_leg` 충돌

### 사람 평가 major-worse

- `2.16.52:p2`: 손을 모으는 과정에서 한 팔은 앞, 다른 팔은 뒤를 향하는 관절 붕괴

현재 v2.4 final cascade는 이 사례를 conservative로 복구했다. 이 복구를 회귀 fixture로 고정한다.

### 현재 final cascade의 의미

D0 artifact 재검사에서 `A_v24_aggressive_final`의 구조적 hard violation은 `0/27`이었다. final은
conservative 대비 Joint NME 약 5.83%, Endpoint NME 약 6.81% 개선을 유지했다. 따라서 현재의
block alpha·부분 rollback 구조는 폐기하지 않고, 그 바깥에 final selector를 추가한다.

### 아직 증거가 없는 부분

- lap-contact 활성 표본은 0개다.
- 27개 모두 near-gap으로 라벨됐지만 반신·중복 검출·검색 누락이 섞였다.
- 사람 평가는 raw candidate와 conservative 비교이며 v2.5 final 평가는 아니다.

lap-contact 가중치를 D0만 보고 더 강화하지 않는다. 기존 구현과 안전 band를 유지하고 별도 fixture가
생길 때 조정한다.

---

## 3. 범위

### v2.5 P0 범위

1. 원본 base 기준 final 구조 안전 검사
2. aggressive final과 conservative의 공통 metric non-regression selector
3. 안전 실패 시 conservative, 그마저 없으면 exact base 복구
4. candidate 미생성·timeout·fallback의 명시적 lineage
5. 단계별 deadline과 aggressive 조기 미시도
6. env 기본 mode와 요청 override
7. cache/capability/version 계약 갱신
8. D0 실패·성공 fixture 회귀 테스트

### v2.5 P1 범위

1. 보이지 않는 하체에 대한 shot·crop-aware refine mask
2. duplicate/crop/ownership 불확실 시 aggressive 미시도
3. 평가 라벨을 `unit × arm × alert_type`으로 확장
4. front/back 평가 renderer 표기 수정

### 범위 밖

- pose-family 자동 분류
- 검색 Top-5·view 선택 개선
- `4.56.21:p0` 야구 투구 검색 누락 수정
- 새로운 body mesh 또는 collision engine 교체
- torso 기본 활성화
- 다인 entangled set 공동 refine
- BFF 비동기화·영속 저장 구조 변경

검색 구조 공백을 aggressive의 더 큰 회전으로 메우지 않는다. `gap_type`은 계속 평가 라벨이며 제품
실행 gate로 직접 사용하지 않는다.

---

## 4. 불변식

v2.5 구현은 다음을 깨면 안 된다.

1. `refined=False`면 base 원본 URL을 반환한다.
2. aggressive 실패는 HTTP 오류가 아니라 conservative/base fallback이다.
3. v1은 `REFINE_V2_ENABLED=0`에서 byte-for-byte 기존 동작을 유지한다.
4. 몸통은 `REFINE_V2_TORSO=0` 기본이다.
5. search distance·rank만으로 v2 solve를 차단하지 않는다.
6. 유효 skeleton·coverage·ownership 실패에서는 solver를 실행하지 않는다.
7. raw aggressive artifact는 제품 cache/export에서 접근할 수 없다.
8. 최종 반환 BVH만 응답의 `bvh` 본문으로 나간다. 추론 컨테이너 로컬 디스크에는
   아무것도 남기지 않는다(§15).
9. fallback의 content/geometry는 선택한 conservative 또는 base와 정확히 같아야 한다.
10. selector 판정은 solver 내부 loss가 아니라 고정 query mask의 공통 metric과 구조 안전으로 한다.

---

## 5. 최종 실행 흐름

```text
입력 정책 검증
  ├─ invalid skeleton / ownership / coverage → exact base
  └─ valid
       ↓
Conservative phase C
  ├─ 결과 없음 → B0를 selector reference로 사용
  └─ 결과 있음 → C를 selector reference로 사용
       ↓
시간 예산 확인
  ├─ aggressive를 시작할 시간이 없음 → C 또는 B0
  └─ 충분함
       ↓
Aggressive phase A
  ├─ low_observability / already_matched / timeout / no-op → C 또는 B0
  └─ final adopted candidate A'
       ↓
Final structural post-check
  ├─ 실패 → C 또는 B0
  └─ 통과
       ↓
C 대비 common-metric selector
  ├─ 회귀 또는 실익 없음 → C 또는 B0
  └─ positive gain + non-regression → A'
       ↓
선택본 인라인 반환 + diagnostics
```

제품 반환 대상은 `A'`, `C`, `B0`뿐이다. solver의 gate 전 full solve `A_raw`는 이 흐름에 포함되지 않는다.

---

## 6. 원본 base와 phase base 분리

현재 aggressive phase는 conservative BVH를 새로운 `base_bvh`로 사용한다.

```python
aggressive_base = conservative_path if conservative.refined else base_bvh
```

따라서 phase-local trust region은 conservative 기준으로는 안전해도 **원래 선택한 base 기준 누적 회전**이
45°를 넘을 수 있다. `131112:p0`의 50.449782°가 이 경로다.

v2.5는 다음 두 기준을 분리한다.

```text
solve reference:
  aggressive 최적화를 시작하는 자세. C 또는 B0.

policy reference:
  사용자가 선택한 원본 B0. 실행 전체에서 불변.
```

모든 최종 trust-region·채널·신규 충돌·fallback identity 검사는 policy reference인 B0를 기준으로 한다.
common-metric non-regression만 C와 비교한다.

```text
구조 안전: B0 → A'
제품 이득: C  → A'
절대 이득: B0 → final   # 진단
```

---

## 7. selector 설계

### 7-1. 입력

```text
policy_base_path       # 원본 B0
conservative_path      # C. 없으면 B0
aggressive_final_path  # 기존 block gate 이후 A'
target_keypoints
target_scores
target_valid_mask
allowed_limbs/joints
view
deadline
```

### 7-2. 출력

```text
selected_mode: aggressive | conservative | base
selected_path
candidate_available
candidate_accepted
fallback_stage
fallback_reason
structural_checks
common_metrics
time_budget
```

### 7-3. 구조 안전 gate

최종 A′를 다시 parse/FK한 뒤 B0 대비 검사한다.

Hard gate:

- BVH read/parse/FK 실패
- NaN/Inf 또는 frame count 오류
- hierarchy·root translation·비허용 channel 변경
- 허용 joint의 wrapped 누적 회전 상한 초과
- 신규 arm-torso, arm-leg, leg-leg, leg-torso 관통
- 신규 elbow/knee joint-limit 위반
- 검사 불가능 상태가 B0 대비 새로 발생

회전량은 raw Euler 뺄셈이 아니라 `[-180°, 180°]` wrapped delta 또는 회전행렬 상대각으로 계산한다.
상한은 기존 per-joint/axis 제한과 전역 `REFINE_MAX_DELTA_DEG` 중 작은 값을 사용한다.

### 7-4. proxy alert 정책

다음 외부 evaluator 경보를 v2.5에서 새로 일괄 hard gate로 추가하지 않는다.

- `foot_direction_regression`
- `ground_contact_regression`
- `lap_contact_regression`

D0에서 false positive가 많았고 `124702:p0`처럼 aggressive 이득이 큰 사례도 경보에 걸렸다. 기존
solver 내부의 발 방향·지면·contact band 검사는 유지하되, 외부 proxy 하나만으로 final 전체를 버리지
않는다. proxy는 diagnostics와 다음 holdout의 `arm + alert_type` 사람 라벨에 남긴다.

### 7-5. common-metric non-regression

A′와 C를 같은 target mask로 투영해 다음을 계산한다.

- `joint_nme`
- `endpoint_nme`
- target evidence가 활성화된 `hand_pair_error`
- target evidence가 활성화된 `lower_pair_error`
- target evidence가 활성화된 `lap_contact_error`

채택 규칙:

```text
all active metrics:
  metric_A <= metric_C + epsilon_metric

and at least one active metric:
  metric_A < metric_C - epsilon_metric
```

`epsilon_metric`은 metric 단위별로 둔다. 전역 solver `refine_gain_epsilon`을 contact 제곱 손실에 그대로
재사용하지 않는다. 반복 evaluator가 deterministic이었던 D0에서는 수치 delta가 0이었지만, float·rig
차이를 위해 고정 작은 tolerance를 사용한다.

권장 초기값:

| metric | selector epsilon |
|---|---:|
| joint NME | `1e-6` |
| endpoint NME | `1e-6` |
| hand/lower pair | `1e-6` |
| lap-contact | `1e-8` |

환경변수로 임의 튜닝하기보다 v2.5 code/config hash에 포함되는 명시적 config 필드로 둔다.

### 7-6. selector 실패 처리

| 실패 | selected mode | reason |
|---|---|---|
| A′ 구조 안전 실패 | C 또는 B0 | `candidate_structural_gate` |
| A′ common metric 회귀 | C 또는 B0 | `candidate_non_regression` |
| A′ positive gain 없음 | C 또는 B0 | `candidate_no_gain` |
| aggressive timeout | C 또는 B0 | `aggressive_timeout` |
| aggressive low observability | C 또는 B0 | `aggressive_low_observability` |
| C 없음/불안전 | B0 | 기존 conservative reason |

top-level `RefineResponse.reason`은 실제 반환 artifact의 성공/실패 의미를 유지한다. aggressive 탈락 사유는
`diagnostics.selector.fallback_reason`에 기록한다.

---

## 8. 기존 block alpha/rollback과의 관계

`src/refine_v2.py::_refine_bvh_v2_phase`의 다음 구조는 유지한다.

- `REFINE_V2_BLEND_ALPHAS=1.0,0.75,0.5,0.25`
- hand-pair 공동 채택 후 per-arm fallback
- lower-pair 공동 채택 후 per-leg fallback
- 각 block hybrid non-regression
- arm-torso/arm-leg/leg-leg/leg-torso safety
- 제한적 ankle counter-rotation
- 몸통 별도 default-off phase

v2.5 selector는 이 내부 부분 채택이 끝난 **실제 final frame**을 다시 검사한다. raw full solve가 구조
위반을 가질 수 있다는 이유만으로 solver를 약화하지 않는다.

향후 final post-check가 특정 block 때문에 반복 실패한다면 다음 순서의 v2.5.1 개선을 허용한다.

```text
문제 block alpha backoff → 문제 block rollback → 전체 C fallback
```

v2.5 최초 구현에서는 기존 block rollback을 신뢰하고 outer selector 실패 시 C로 정확히 복구하는 단순한
fail-closed 경로를 우선한다.

---

## 9. candidate 미생성 계약

D0의 4건은 다음과 같이 분류한다.

| unit | 원인 | v2.5 처리 |
|---|---|---|
| `131056:p2` | low observability | 정상 no-op/fallback |
| `131145:p0` | low observability | 정상 no-op/fallback |
| `131211:p0` | timeout | 시간 예산 개선 대상 |
| `3.08.13:p0` | base/no-op | 정상 unchanged/base |

목표는 candidate 100% 생성이 아니다. 모든 미생성을 명시적으로 계측하고 분모에 포함하는 것이다.

```text
candidate_status:
  generated | not_attempted | timeout | no_gain | already_matched | error
```

평가용 `diagnostic_candidate_out_path`가 없을 때 제품 실행이 raw candidate 파일을 만들지 않는 기존 계약은
유지한다.

---

## 10. 시간 예산

D0 측정:

| 단계 | p50 | p95 | max |
|---|---:|---:|---:|
| conservative | 303ms | 1,265ms | 1,553ms |
| aggressive request 전체 | 634ms | 2,699ms | 5,001ms |
| 외부 평가 | 32ms | 34ms | 35ms |

초기 제품 목표는 cache-off p95 3초 이하, deadline 도달 0건이다.

### 단계별 정책

```text
total request deadline: 기존 REFINE_TIMEOUT_SECONDS, 기본 5초
final selector reserve: 최소 150ms
aggressive start rule: 남은 시간이 최소 750ms 이상
```

권장 신규 config:

```env
REFINE_V25_FINAL_CHECK_RESERVE_SECONDS=0.15
REFINE_V25_AGGRESSIVE_MIN_REMAINING_SECONDS=0.75
REFINE_V25_INACTIVE_SKIP_BUDGET_FRACTION=0.25
```

- conservative가 끝난 뒤 남은 시간이 부족하면 aggressive를 시작하지 않고 C를 반환한다.
- aggressive cooperative timeout은 C를 오류 없이 반환한다.
- final selector reserve를 침범하면 A′를 채택하지 않는다.
- completed latency에서 timeout을 제외하지 않고 별도 비율로 보고한다.

---

## 11. 가시성·소유권 정책

### 11-1. 현재 존재하는 gate 재사용

`src/refine_policy.py::structural_refine_allowed`와 pipeline의 다음 lineage를 단일 소스로 유지한다.

- `skeleton_state`
- `coverage_class`
- `refinable_limbs`
- `slot_origin`
- `skeleton_source`
- `ownership_valid`

assignment ambiguity·duplicate·crop retry는 refine 전에 차단되어야 한다. D0 runner는 frozen RTMPose를
직접 사용해 제품 pipeline의 일부 gate를 우회했으므로, D0 오염 3건만 보고 중복 검출기를 refine solver
안에 새로 구현하지 않는다.

### 11-2. 보이지 않는 하체

RTMPose가 반신 컷의 보이지 않는 무릎·발목을 높은 score로 추정할 수 있어 score만으로는 부족하다.

P1에서 `refinable_limbs` 생성에 다음 증거를 추가한다.

- VLM `shot`
- owner/VLM box 안의 관절 위치와 crop 경계
- skeleton source와 crop retry 여부
- distal joint가 실제 이미지/owner 영역에 존재하는지
- 동일 인물 내 limb 길이·방향 오염 검사

정책:

```text
face/bust:
  lower-body refine 금지

full_half + 하체 비가시/경계 잘림:
  해당 leg를 refinable_limbs에서 제거

full/reduced + 완전한 leg evidence:
  기존 observability/foreshortening gate로 전달
```

보이지 않는 하체는 “마음에 들지 않는 결과”가 아니라 평가 대상 밖으로 숨기지 않는다. diagnostics에
`visibility_excluded_limbs`와 reason을 남긴다.

---

## 12. 코드 변경 지도

| 파일 | 변경 |
|---|---|
| `src/refine_selector.py` | 신규. B0/C/A′ 공통 metric·구조 안전·선택 결정 |
| `src/refine_v2.py` | conservative/aggressive 결과를 selector에 전달, fallback과 diagnostics 병합 |
| `src/refine.py` | v2.5 code version·호출 인자 전달, v1 동작 유지 |
| `src/config.py` | default mode·selector epsilon·시간 reserve config 추가 |
| `src/refine_policy.py` | ownership/visibility lineage를 fail-closed로 검증 |
| `src/skeleton_extraction.py` | P1 shot/crop-aware limb visibility와 refine mask |
| `api/models.py` | mode 생략 지원, 응답 reason 설명 갱신 |
| `api/app.py` | effective mode 해석, capability/diagnostics에 v2.5 lineage 반영, 조정본 인라인 반환 |
| `standin_eval/refine_evaluator.py` | 최종 BVH 독립 재검사 유지, proxy arm 연결 보강 |
| `standin_eval/refine_render.py` | front/back 표기 회귀 수정 |
| `scripts/eval_v25_current_rough_d0.py` | final v2.5 vs C/B0 재실행 지원 |
| `tests/test_refine_v2.py` | selector·fallback·D0 fixture 단위/통합 테스트 |
| `tests/test_refine_three_arm.py` | 기본 mode·capability·cache·ITT 계약 테스트 |
| `tests/test_v25_current_rough_d0.py` | D0 artifact·라벨·보고서 회귀 |

`src`에서 `standin_eval`을 import하지 않는다. 공통 collision/FK primitive는 `src`에 두고 evaluator는 최종
artifact를 독립적으로 다시 parse/FK해 감사한다.

---

## 13. Config 계약

### 신규 config

```env
REFINE_DEFAULT_MODE=aggressive
REFINE_V25_SELECTOR_ENABLED=1
REFINE_V25_JOINT_NME_EPSILON=0.000001
REFINE_V25_ENDPOINT_NME_EPSILON=0.000001
REFINE_V25_PAIR_EPSILON=0.000001
REFINE_V25_CONTACT_EPSILON=0.00000001
REFINE_V25_JOINT_NME_WEIGHT=0.15
REFINE_V25_PARTIAL_ALPHAS=0.75,0.5,0.25
REFINE_V25_SKIP_INACTIVE_AGGRESSIVE=1
REFINE_V25_INACTIVE_SKIP_BUDGET_FRACTION=0.25
REFINE_V25_FINAL_CHECK_RESERVE_SECONDS=0.15
REFINE_V25_AGGRESSIVE_MIN_REMAINING_SECONDS=0.75
```

2026-08-18 제품 결정으로 배포 기본값은 다음과 같다.

```env
REFINE_V2_ENABLED=1
REFINE_V25_SELECTOR_ENABLED=1
REFINE_DEFAULT_MODE=aggressive
REFINE_V2_LOWER_BODY=1
REFINE_V2_TORSO=0
```

### validation

- `REFINE_DEFAULT_MODE`는 `conservative|aggressive`만 허용한다.
- v2가 꺼졌으면 effective mode는 conservative다.
- production에서 default aggressive인데 selector가 꺼진 구성은 startup fail 또는 unhealthy로 처리한다.
- selector config 전체는 `/healthz.refine.config_sha256`과 cache identity에 포함한다.

### 비상 복구

```env
REFINE_DEFAULT_MODE=conservative
```

또는

```env
REFINE_V2_ENABLED=0
```

selector를 꺼서 raw aggressive를 우회 반환하는 복구 방식은 허용하지 않는다.

---

## 14. API 계약

### 요청

현재 `RefineRequest.refine_mode`의 literal은 유지하되 mode 생략을 허용한다.

```json
{
  "refine_mode": null
}
```

해석:

```text
명시적 conservative/aggressive → 요청값 우선
생략/null → CFG.refine_default_mode
v2 disabled → conservative
```

cache handle을 만들기 전에 `effective_mode`를 확정한다. 요청의 null과 명시적 aggressive가 같은 결과를
사용하더라도 capability/config hash와 effective mode를 cache identity에 넣는다.

### 응답

기존 필드를 유지한다. 신규 정보는 `diagnostics`에 추가한다.

```json
{
  "diagnostics": {
    "mode_requested": "default|conservative|aggressive",
    "mode_effective": "aggressive",
    "mode_applied": "aggressive|conservative|base",
    "candidate_status": "generated|not_attempted|timeout|no_gain|error",
    "selector": {
      "version": "v2.5.1",
      "accepted": true,
      "selected_mode": "aggressive",
      "fallback_stage": null,
      "fallback_reason": null,
      "structural_checks": {},
      "metrics": {
        "conservative": {},
        "aggressive": {},
        "delta": {}
      }
    },
    "time_budget": {
      "total_ms": 5000,
      "conservative_ms": 0,
      "aggressive_ms": 0,
      "selector_ms": 0,
      "remaining_ms": 0
    }
  }
}
```

`mode_applied=conservative`인데 `refined=true`일 수 있다. `refined`는 base와 geometry가 달라졌는지를
나타내며 aggressive 채택 여부가 아니다.

### BFF/export

- 새 endpoint를 만들지 않는다.
- `bvh_url`은 **항상 베이스** `/pose/{pose_id}/bvh`다. 조정본에는 URL이 없다.
- 조정본을 얻는 유일한 경로는 응답의 `bvh` 본문이며, 보관은 BFF가 한다.
- refined export 주문은 BFF가 자기 저장소 기준으로 조립한다. 추론 서버의
  `/export-order`는 베이스 pose 전용 legacy 경로로 남는다.
- raw/conservative 임시 파일 경로는 응답하지 않는다.

---

## 15. 무상태 인라인 전달

v2.5는 조정본을 파일로 게시하지 않는다. `POST /refine` 응답의 `bvh` 본문이 조정본을 얻는
유일한 경로이고, 추론 컨테이너의 로컬 디스크에는 아무것도 남지 않는다
(`docs/REFINE_HANDOFF.md` §3 4단계).

```text
refined=true   bvh_url = /pose/{pose_id}/bvh   # 항상 베이스(폴백 위치)
               bvh     = 조정된 BVH 본문(LF)    # 실제 최종 결과

refined=false  bvh_url = /pose/{pose_id}/bvh
               bvh     = null
```

### 왜 파일 게시를 버렸나

`GET /refined/{handle}/bvh` + 로컬 sidecar 캐시는 두 가지를 요구했다. 롤링 배포 중
`POST /refine`과 이어지는 GET이 **같은 태스크**에 도달해야 하고, 쌓이는 조정본을 누군가
비워야 한다. 인라인 전달은 두 번째 요청 자체가 없으므로 두 요구가 사라진다.

같은 선택을 다시 눌러도 재계산하지 않는 멱등성은 서버 캐시가 아니라 **BFF의
`refined_artifacts` PK `(job_id, person_index, candidate_id)`**가 담당한다. 원래의 로컬
디스크 캐시는 인스턴스 단위라 ALB 뒤 다중 태스크에서 히트율이 이미 낮았고, BFF 캐시는
공유·영속이라 더 잘 듣는다.

### selector는 여전히 파일을 쓴다 — 단 요청 수명 안에서만

final selector와 FINAL collision/extension post-check는 후보를 실제 BVH로 다시 parse/FK해
검증한다. 따라서 내부적으로는 파일이 필요하다. `refine_bvh_v2`는 `out_path=None`이면 그
파일들을 요청 수명의 임시 디렉터리에 만들고, 끝나면 통째로 지운 뒤 채택본의 본문만
`RefineResult.bvh_text`로 돌려준다.

```text
scratch/            (tempfile.TemporaryDirectory, 요청 끝나면 삭제)
  conservative temp
  aggressive temp
  selector decision
  final.bvh         → 본문만 읽어 bvh_text로 반환
```

`out_path`를 명시한 호출(평가·진단 스크립트, D0 harness)은 기존처럼 파일을 받는다. 제품
경로와 평가 경로가 같은 solver·selector를 공유하고 전달 계층만 갈린다.

### fallback

`refined=false`면 `bvh`는 `null`이고 `bvh_url`은 베이스다. 복사본을 만들지 않는다.
conservative가 선택되면 conservative bytes가 그대로 `bvh`로 나간다.

---

## 16. 테스트 계획

### 16-1. selector 단위 테스트

- A′가 C보다 모든 metric에서 non-regression이고 하나 이상 개선 → aggressive
- Joint NME 회귀 → conservative
- Endpoint NME 회귀 → conservative
- 활성 hand/lower/lap pair 회귀 → conservative
- 모든 metric tie → conservative
- metric unavailable이 C에는 없고 A′에 새로 생김 → conservative
- 구조 안전 검사 실패 → conservative
- C가 base와 같으면 fallback mode base
- selector timeout/exception → fail-closed C/base

### 16-2. 원본 base 누적 trust-region

- conservative 30° + aggressive 추가 20°가 phase-local로는 허용돼도 B0 대비 50°면 거부
- `131112:p0` 50.449782° fixture가 `candidate_structural_gate`로 C를 선택
- ±180° wrap 경계에서 실제 작은 회전은 오탐하지 않음
- equivalent Euler 표현은 회전행렬 기준으로 동등 처리

### 16-3. 충돌 회귀

- `131112:p2` raw right-arm/torso 충돌
- `131112:p3` raw right-arm/torso 충돌
- `171734:p0` raw left-arm/leg 충돌
- B0 기존 관통을 보존한 결과와 신규/악화 관통을 구분
- block-level 채택 뒤 최종 합성 자세 전체 collision 재검사
- final selected artifact의 구조 violation 0

### 16-4. positive fixture

- `124702:p0`의 손·팔·무릎 모으기 aggressive 이득 유지
- `131000:p0` 팔꿈치 이동 유지
- `131040:p1` 지지 다리 방향 이득 유지
- `131112:p2` 겨드랑이·팔 모아짐 이득을 final 안전 범위에서 유지

외부 foot/ground proxy alert 하나만으로 `124702:p0`을 버리지 않는지 확인한다.

### 16-5. 사람 major fixture

- `2.16.52:p2` raw candidate는 저장 가능
- 제품 final은 conservative/base
- raw candidate hash가 final hash와 다름
- response `mode_applied=conservative`
- fallback reason이 `diagnostics.selector.fallback_reason`에 존재

### 16-6. 미생성·deadline

- low observability → C/base, `candidate_status=not_attempted`
- already matched → C/base, `candidate_status=already_matched`
- aggressive 시작 전 시간 부족 → C, solver 미호출
- aggressive timeout → C exact content
- final selector reserve 내 완료
- deadline 경계에서 부분 aggressive 파일 게시 금지

### 16-7. visibility·ownership

- bust/face lower-body refine 0
- full-half에서 distal leg가 경계 밖이면 해당 leg 제외
- 완전한 전신 leg는 기존 lower-pair 대상 유지
- duplicate/assignment ambiguity/crop retry는 aggressive 미시도
- ownership 실패 시 v1/v2 모두 fail-closed 정책 유지

### 16-8. API/config/cache

- mode 생략 + default conservative
- mode 생략 + default aggressive
- 명시적 conservative가 default aggressive를 override
- 명시적 aggressive가 default conservative를 override
- v2 disabled에서 aggressive effective 금지
- mode/config 변경 시 cache handle 변경
- `/healthz`가 default/effective capability와 v2.5 version 노출
- cached response도 동일 selector lineage 유지

### 16-9. v1 회귀

- `REFINE_V2_ENABLED=0` 전체 smoke 통과
- v1 search-distance gate·arms-only 기본 유지
- 기존 `/refine` request/response 역호환

---

## 17. D0 재실행 판정선

구현 뒤 같은 27개 frozen unit을 회귀용으로 재실행한다.

필수:

- final 구조적 hard violation `0/27`
- exact conservative/base fallback `100%`
- 운영 오류 `0`
- deadline 도달 `0`
- final Joint NME 평균 개선 `>=5%` vs C
- final Endpoint NME 평균 non-regression vs C
- selector 적용 후 unit-level 공통 metric worse `0`
- `2.16.52:p2` final aggressive 직접 반환 금지
- `124702:p0` safe aggressive 이득 유지
- p95 `<=3초`

raw candidate의 구조 violation은 진단에 남을 수 있다. 승격 판단 분모는 final selected artifact다.

D0 재실행은 구현 회귀 증거이며 config 최종 승격 증거는 아니다.

---

## 18. 구현 후 사람 평가

다음 blind pair를 새로 생성한다.

```text
Primary:
  v2.5 final policy ↔ v2 conservative

Promotion:
  v2.5 final policy ↔ v1

Absolute value:
  v2.5 final policy ↔ B0
```

raw aggressive는 진단용 별도 화면에서만 본다. config 승격 판정에는 사용자가 실제 받는 final policy만
사용한다.

라벨 스키마를 다음처럼 바꾼다.

```json
{
  "unit_id": "...",
  "arm": "v25_final",
  "alert_type": "foot_direction_regression",
  "judgment": "confirmed_major|confirmed_minor|false_positive|uncertain"
}
```

pair 전체에 하나의 proxy 판정을 붙여 양쪽 SafeUsable을 동시에 제외하는 현재 문제를 제거한다.

---

## 19. 구현 순서

### Step 1. 회귀 fixture 고정

- D0의 4개 raw 구조 실패
- `2.16.52:p2` 사람 major
- `124702:p0` aggressive positive
- 미생성 4개 reason

### Step 2. `src/refine_selector.py`

- artifact parse/FK
- B0 기준 누적 trust/구조 안전
- C/A′ 공통 metric
- `SelectorDecision`

### Step 3. v2 orchestrator 연결

- conservative temp 보존
- aggressive final temp 생성
- selector 실행
- C/B0 fallback
- diagnostics merge

### Step 4. deadline

- remaining budget 계산
- aggressive early skip
- final reserve 보장

### Step 5. API/config/cache

- effective default mode
- capability/config hash
- capability/diagnostics lineage
- atomic selected artifact 게시

### Step 6. visibility/ownership P1

- shot/crop-aware limb mask
- assignment/ownership fail-closed 검증

### Step 7. 전체 검증

- 단위/스모크/API 테스트
- frozen D0 재실행
- final blind pair 생성
- 승격 판단

---

## 20. 구현 완료 조건

다음을 모두 만족하면 v2.5 코드 구현 완료다.

- `REFINE_V2_CODE_VERSION=v2.5.1`
- raw candidate와 final artifact 경계가 테스트로 고정됨
- final selector가 B0 구조 안전과 C non-regression을 모두 검사
- 모든 fallback이 exact content/geometry를 보존
- default mode가 cache·healthz·diagnostics에 반영
- deadline이 aggressive 미시도/복구를 안전하게 처리
- v1 회귀 0
- D0 final 구조 위반 0, common metric worse 0
- D0 평균 Joint NME 개선 5% 이상 유지
- positive/negative 대표 fixture 모두 통과
- 평가 UI와 label schema가 final 정책 평가를 지원

다음을 만족하면 config 승격 후보가 된다.

- final v2.5 blind 평가에서 conservative/v1 대비 작가 순선호 양수
- final SafeUsable이 conservative/v1 대비 non-regression
- confirmed final major-worse 0
- 신규 구조 violation 0
- timeout/error 0
- cache-off p95 제품 예산 이내

승격 전 최종 config:

```env
REFINE_V2_ENABLED=1
REFINE_V25_SELECTOR_ENABLED=1
REFINE_DEFAULT_MODE=aggressive
REFINE_V2_LOWER_BODY=1
REFINE_V2_TORSO=0
```

이 config의 제품 의미는 끝까지 **safe aggressive attempt**, 즉 aggressive 우선 시도와 안전한
conservative/base 복구다.

---

## 22. 2026-08-18 v2.5.0 기준 구현·D0 회귀 결과

이 절은 `v2.5.1` 최적화 전 `v2.5.0` 기준선이다. 27개 frozen near-gap D0를 기존 사람 라벨과 분리된 임시 실행에서
다시 생성해, 사용자가 실제 받는 final artifact를 conservative와 비교했다.

| 항목 | 결과 |
|---|---:|
| final mode | aggressive 12 / conservative 14 / base 1 |
| final 신규 구조 위반 | 0 / 27 |
| Joint NME 평균 감소 | 4.43% |
| Endpoint NME 평균 감소 | 5.81% |
| Hand-pair 평균 감소 | 6.52% |
| Lower-pair 평균 감소 | 7.80% |
| Lap-contact 평균 감소 | 11.86% |
| selector 활성 metric 회귀 | 0 |
| `124702:p0` | aggressive 유지 |
| `2.16.52:p2` | conservative 복구 |
| cache-off aggressive request p95 | 4.22초 |
| aggressive timeout 후 안전 복구 | 1 / 27 |

`lap_contact` cohort는 두 접촉 부위가 모두 solve block일 때만 여는 방식에서, 한쪽만 움직여도 다른 쪽을
고정 접촉면으로 검사하도록 수정했다. 또한 hand/lower pair selector cohort는 단순 관절 가시성이 아니라
불변 B0와 target 사이에 실제 pair 공백이 있을 때만 고정한다.

구조 안전과 zero-regression/fallback 계약은 통과했다. 사전 완료선인 Joint NME 5%와 cache-off p95
3초는 아직 만족하지 못했지만, 2026-08-18 사용자 결정으로 safe aggressive를 제품 기본값으로 사용한다.
이는 raw aggressive 승격이 아니라 selector와 conservative/base 복구를 항상 포함한 운영 결정이다.
이 문장의 기본값은 코드/config 기본을 뜻한다. v2.5.2 일반 production 출시는 §25의 알려진 사용자 실패가
닫힐 때까지 승인되지 않았다.
품질 컷라인은 폐기하지 않으며, 다음 단계는 selector 완화가 아니라 aggressive solver의 안전한 후보
품질과 지연시간을 개선한 뒤 동일 frozen D0와 final blind 평가를 다시 통과하는 것이다.

---

## 23. 2026-08-18 v2.5.1 속도·NME 구현 결과

`v2.5.1`은 selector 기준을 완화하지 않고 다음을 구현했다.

- C/A 공용 prepared target과 parsed BVH/FK 상태
- B0 기준 잔여 channel trust budget
- final Joint NME 방향의 aggressive surrogate
- full A 탈락 시 동일 selector로 검증하는 C→A global blend `0.75/0.5/0.25`
- 관측 가능한 pair/contact가 없고 C가 전체 예산의 25% 이상을 쓴 경우의 budget-risk skip
- prepare/C/A/selector/elapsed 및 nfev/parameter/objective 진단

동일 frozen D0 27개 cache-off 3회(총 81 요청) 결과는
`out/eval/v25_optimization_v251_x3_20260818/REPORT.md`다.

| 항목 | v2.5.1 결과 |
|---|---:|
| final mode | aggressive 18 / conservative 8 / base 1 |
| Joint NME 평균 감소 | **5.75%** |
| Endpoint NME 평균 감소 | **7.60%** |
| Hand-pair 평균 감소 | **19.73%** |
| Lower-pair 평균 감소 | **4.62%** |
| global blend 채택 | 2 / 27 |
| cache-off p95 / max | **1.75초 / 2.71초** |
| timeout | **0 / 27** |
| 구조 hard violation | **0 / 27** |
| 채택 selector metric 회귀 | **0** |

자동 수치의 3회 engineering 목표는 통과했다. 다만 lap-contact 평가 표본이 0개이고 sealed holdout,
실메시, 작가 blind가 남아 있으므로 이 표만으로 최종 config 승격 완료를 선언하지 않는다.

### 23-1. proxy 경보 사용자 검토 결과

같은 날 자동 proxy 경보 8건을 B0/C/FINAL로 검토한 결과 실제 회귀가 확인됐다.

- `131056:p2`: B0부터 손 모양이 기괴하고 손·손가락 rotation은 B0/C/FINAL에서 동일함. 원본 BVH
  품질 결함이며 FINAL 관통의 refine 귀속은 mesh 비교 대기
- `171734:p0`: FINAL 관통
- `171734:p0`, `2.16.52:p0`: 반신 컷의 불필요한 하체 이동
- `4.56.21:p0`: refine 대상이 아닌 search miss
- `124702:p0`: FINAL이 가장 유사함; foot-direction 단독 경보 false positive
- `131040:p0`: 품질은 좋지 않지만 안전 문제는 없음

구현 상태는 “자동 engineering gate 통과, 사용자 안전 gate 실패”다. 다음 구현은 (1) 불량 B0 asset
quarantine/다음 Top-K 이관, (2) 반신/비관측 하체 eligibility, (3) mode 선택 후 FINAL 관통 post-check,
(4) search miss의 refine 분모 분리를 이 순서로 수행한다. B0 자체가 불량한 경우 exact B0 fallback은
안전 복구로 인정하지 않는다. 사용자 판정 원본은
`out/review/v251_proxy_alerts_20260818/review_labels.jsonl`에 보존한다.

---

## 24. v2.5.2 안전 수정 구현 및 재평가

v2.5.2는 v2.5.1의 selector/NME 최적화를 유지하면서 사용자 검토에서 확인된 두 안전 회귀와 한
평가 오염을 수정한다.

### 24-1. 반신·하체 비관측 fail-closed

- VLM 출력에 `lower_body_visible[]`를 추가한다. `approx_boxes[]`와 정확히 같은 순서·길이여야 한다.
- `/analyze` person 응답의 `lower_body_observed`로 전달한다.
- BFF는 선택한 person의 값을 `/refine.lower_body_observed`로 그대로 돌려보낸다.
- 값이 명시적 `true`가 아니면 `left_leg`, `right_leg`, `lower_pair`, leg-driving `lap_contact`, Foot
  counter-rotation을 모두 닫는다. 누락·길이 불일치·null도 false와 같다.
- 상체 refine은 계속 허용할 수 있지만 모든 하체 BVH channel은 B0와 동일해야 한다.

이 gate는 RTMPose가 화면 밖 관절을 추정해 만든 score로 다시 열지 않는다. crop/가시성은 VLM의 제어
신호, 세부 관절 유효성은 기존 skeleton policy가 맡는다.

### 24-2. mode 선택 뒤 FINAL 충돌 재검사

selector가 aggressive, conservative 또는 partial blend를 선택한 뒤 실제 반환 대상 FINAL BVH를 다시
parse/FK한다. B0와 FINAL에서 arm-torso, arm-leg, leg-leg, leg-torso capsule depth를 비교해 신규 또는
악화 관통을 검출한다.

- 통과: 선택 결과를 그대로 반환한다.
- 신규·악화 관통: 생성한 FINAL artifact를 폐기하고 B0의 `bvh_path`, `loss`, 빈 `limbs`로 exact
  fallback한다.
- 검사 불능·예외: fail-closed로 같은 B0 exact fallback을 적용한다.
- 응답은 `reason=final_collision_gate`, `mode_applied=base`,
  `selector.fallback_stage=final_collision`을 기록한다.

현재 검사는 BVH capsule/COCO proxy다. 실제 CSP 메시 관통과 완전히 같지 않으므로 이 검사를 통과한
결과를 실메시 안전 증명으로 사용하지 않는다. `131056:p2`처럼 B0 자체가 기괴한 asset은 exact fallback도
품질 해결이 아니며 별도 asset quarantine/다음 Top-K 정책 대상이다.

### 24-3. Search miss 분리

`4.56.21:p0`은 near-gap이 아니라 더 유사한 라이브러리 자세를 놓친 search miss다.
`gap_labels.v252_overrides.jsonl`에서 `structural_gap`으로 재분류해 refine D0 분모에서 제외하고,
`tests/fixtures/search_regressions.v1.jsonl`에 현 검색 결과를 고정했다. 사용자가 지목한 정답 BVH의
정확한 `pose_id`를 확인하면 `expected_pose_ids`를 채워 Top-K/Top-1 assertion을 활성화한다.

### 24-4. frozen D0 3회 결과

산출물: `out/eval/v25_optimization_v252_x3_20260818/REPORT.md`

| 항목 | v2.5.2 결과 |
|---|---:|
| 평가 unit | 26 (`4.56.21:p0` 제외) |
| final mode | aggressive 15 / conservative 9 / base 2 |
| Joint NME 평균 감소 | **4.9958946357%** |
| Endpoint NME 평균 감소 | **6.3194578428%** |
| Hand-pair 평균 감소 | **19.8567888536%** |
| Lower-pair 평균 감소 | **4.6165179532%** |
| cache-off p95 / max | **1.548초 / 2.765초** |
| timeout / 구조 hard violation / selector regression | **0 / 0 / 0** |
| proxy alert | **3** |

`171734:p0`과 `2.16.52:p0`의 하체 channel 최대 변화는 `0.0`이다. Joint NME 사전 승격선은 정확히
`>=5.00%`이므로 현재 값은 약 **0.004105%p 미달**로 기록한다. lap-contact 평가 표본도 0개다.
따라서 요청한 v2.5.2 안전 수정은 구현 회귀를 통과했지만 최종 승격선 통과로 선언하지 않는다.

---

## 25. v2.5.3 최종 마감 구현 범위

v2.5.3은 v2를 끝내기 위한 마지막 제한 범위다. 임의 파트 조합이나 구조 공백 복원은 포함하지 않는다.

### 25-1. 최종 fixture 분류

| unit | 분류 | 처리 |
|---|---|---|
| `171734:p0` | 구조 공백/검색 후보 부적합 | refine 분모 제외, 보이는 B0/FINAL 관통이면 후보 폐기 후 다음 Top-K |
| `131056:p2` | 구조 공백 + 원본 hand asset 결함 | asset quarantine, hands-forward/clasped 후보 재검색, refine 분모 제외 |
| `131211:p1` | near-gap | 제한적 proximal-assisted single-leg extension의 필수 통과 fixture |

구조 공백 두 건을 제외한 24개 기존 artifact 재집계는 Joint NME 감소 `5.2675061589%`, Endpoint NME
감소 `6.6529686195%`, Joint NME better/tie/worse `14/10/0`이다. 26개 오염 집계보다 각각
`+0.2716115232%p`, `+0.3335107767%p` 높다.

### 25-2. `single_leg_extension` 활성 조건

다음 조건을 모두 만족할 때만 활성화한다.

- `lower_body_observed is True`
- hip/knee/ankle score와 ownership 유효
- 해당 leg가 `foreshortened_limbs`에 있고 전체 block 탈락 사유가 `low_observability`
- 압축된 뼈가 `hip→knee`이며 `knee→ankle`은 유효
- target knee가 straight class(`>=160°`)이고 B0 대비 extension 필요량이 `15°–45°`
- target foot ground/contact 조건이 독립적으로 판정 가능

각도는 압축된 proximal vector의 노이즈에 민감하므로 exact target angle을 강제하지 않는다. straight
class는 activation 신호로만 쓰고 solver는 유효한 `knee→ankle` 방향과 ankle endpoint를 주목표로 한다.

### 25-3. 파라미터와 손실

- 동결: root translation/rotation, pelvis, torso
- solve: 해당 `Leg` local rotation + 해당 `UpLeg`의 B0 대비 최대 `18°` 보조
- 선택 solve: 해당 `Foot` counter-rotation, B0 대비 최대 `18°`
- `Leg` 누적 trust bound: B0 대비 최대 `45°`; 기존 전역 bound를 완화하지 않는다.

설계 당시 상정한 전용 목적함수는 다음이었다.

```text
L_distal =
  w_dir   * Huber(direction(knee→ankle), target)
  + w_end * Huber(ankle_xy, target_ankle_xy)
  + w_cls * straight_class_penalty(projected_knee_angle)
  + w_move * move_from_B0
```

`straight_class_penalty`는 projected knee angle이 `160°` 이상이면 0이며, 정확히 `171.8°`를 복제하도록
강제하지 않는다. 단축 투시의 깊이 모호성은 최소 이동 해를 선택한다.

#### 25-3-1. 실제 구현(as-built) — 전용 목적함수는 만들지 않았다

**위 `L_distal`과 `straight_class_penalty`는 구현하지 않았다.** 코드에 `L_distal`, `w_cls`,
`straight_class_penalty`는 존재하지 않는다. 새 solver 목적항을 추가하는 대신, 기존 v2 solver를
그대로 쓰고 **진입과 채택만 제어**하는 방식으로 대체했다.

| 설계상 역할 | 실제 구현 | 위치 |
|---|---|---|
| `w_dir` / `w_end` / `w_move` | 기존 v2 direction + Huber endpoint + move 목적을 그대로 사용 | `_metrics` |
| proximal 노이즈 억제 | 압축된 `hip→knee` 뼈의 direction weight를 `×0.25`로 유지(신규 항 아님) | `_foreshortened_direction_weights` |
| `UpLeg` 보조 제한 | 해당 `UpLeg` 축 상한을 `min(기존 한도, 18°)`로 clamp | `_param_limits(extension_limbs=…)` |
| `w_cls` straight-class | **손실이 아니라 채택 게이트로 구현.** `projected angle >= 160°` **또는** target 각도 오차가 B0 대비 50% 이상 감소 | `_single_leg_extension_status` |
| 진입 조건 | `low_observability`로 탈락한 다리를 증거가 있을 때만 되살림 | `_single_leg_extension_evidence` → `extension_observability_rescue` |

이 대체는 의도적이다. 전용 목적함수는 solver 거동 전체를 바꾸지만 게이트 방식은 실패 시 기존 폴백
경로로 그대로 떨어지므로, v2 마감 시점에 지는 리스크가 훨씬 작다. **설계 문구가 아니라 이 표가
구현의 정본이다.**

### 25-4. 채택과 fallback

채택하려면 다음을 모두 만족해야 한다.

- projected knee angle `>=160°` 또는 B0 대비 target angle error 50% 이상 감소
- ankle endpoint NME 양의 개선
- final joint/endpoint NME와 기존 활성 pair/contact metric non-regression
- 반대 다리와 동결 channel byte-equivalent
- anatomy, leg-leg, leg-torso, arm-leg, foot direction/ground 통과
- mode 선택 뒤 FINAL collision post-check 통과

성공 reason은 `ok_foreshortened_extension`이다.

실패 경로는 **두 단계이고 복구 범위가 서로 다르다.** 구현 정본은 다음과 같다.

| 단계 | 위치 | 실패 시 복구 범위 | 진단 reason |
|---|---|---|---|
| solve 루프 내 블록 채택 | `try_single_block` / lower-pair 공동 채택 | **해당 다리만** 미채택. 상체·반대 다리 개선은 유지 | `extension_goal_not_met` |
| 최종 artifact post-check | `_refine_bvh_v2` 말미 | **전체를 exact B0로 되돌린다.** 채택됐던 팔 개선도 함께 폐기 | `final_extension_gate` |

즉 "해당 다리만 복구"는 1단계에만 해당한다. 2단계는 fail-closed를 우선해 전량 되돌린다. 2단계는
1단계와 동일한 판정 함수를 직렬화된 BVH에 재적용하는 이중 검사라 정상 경로에서는 발동하지 않지만,
게이트 경계(`정확히 160°` 부근)에서 직렬화 반올림으로 뒤집히면 품질 절벽이 생긴다. 개선 후보는
§25-8-2/§25-8-5에 남긴다.

`structural_gap_required`는 **구현하지 않았다.** 필요한 변화가 `UpLeg 18°`나 trust bound를 넘는
경우 현재는 `extension_goal_not_met`으로 폴백할 뿐, 다음 Top-K로 이관하지 않는다. 이 경로를 요구하는
실표본은 아직 없다.

### 25-5. 금지 범위

다음은 v2.5.3에서 해결하지 않는다.

- 보이지 않는 하체 생성
- 큰 hip/pelvis 이동
- 다리 꼬기·좌우 교환·큰 보폭 변경
- 의자·바닥 등 외부 접촉면 재구성
- 손목·손가락·손의 전후 깊이 생성
- 서로 다른 후보의 파트 조합

### 25-6. 출시 전 검증

1. `131211:p1` 전용 단위/실제 BVH fixture: 오른쪽 무릎 angle gate, 동결 channel, 관통 0.
2. `171734:p0`, `131056:p2`: 실제 메시 육안 fixture에서 불량 FINAL 반환 0; B0 불량은 next Top-K.
3. 재분류 near-gap D0 전체: Joint NME `>=5%`, endpoint 양수, worse 0, timeout 0, p95 `<=3초`.
4. lap-contact 실표본 최소 1건 추가 또는 해당 기능을 release capability에서 experimental로 명시.

### 25-7. 구현 결과와 engineering closeout

- `REFINE_V2_CODE_VERSION=v2.5.3`
- `131211:p1` 오른쪽 무릎: B0 `139.7°` → FINAL `166.0°`, target `171.8°`
- `right_leg.reason=ok_foreshortened_extension`; angle/ankle gate 모두 통과
- `RightUpLeg` B0 누적 변화 최대 `18°` 이하, root/pelvis/torso 불변
- `131056:p2`, `171734:p0` pose quarantine 및 geometry next Top-K 승격 구현
- stale quarantined BVH/refine 요청은 HTTP 409로 fail-closed
- C→A 누적 발 방향을 B0 기준으로 재검사·counter-rotation

`out/eval/v25_optimization_v253_x3_20260818/REPORT.md`의 24개×3회 결과는 Joint NME
`5.3779318216%`, Endpoint NME `6.8257237633%`, Joint better/tie/worse `14/10/0`, 요청 p95
`1.548초`, hard/proxy/selector 회귀와 timeout 모두 0이다. 이 자동 최소 완료선 통과로 v2.5.3을 v2 최종
engineering release candidate로 마감한다. lap-contact 실표본, 작가 blind, 실제 메시 holdout은 운영 승격
증거로 계속 남긴다.

#### 25-7-1. `>=5.00%` 통과의 기여 분해 — 대부분은 코호트 변경이다

v2.5.2와 v2.5.3 산출물을 유닛 단위로 대조하면, v2.5.3 코드가 값을 바꾼 유닛은 **24개 중
`131211:p1` 하나뿐**이고(`joint_nme 0.19592 → 0.17557`) 나머지 23개는 동일하다. Joint
better/tie/worse도 `14/10/0`으로 변하지 않았다. 따라서 승격선 통과는 다음과 같이 분해해 읽는다.

| 단계 | Joint NME 감소 | 증분 |
|---|---:|---:|
| v2.5.2 코드 / 26 유닛 | `4.9958946357%` | 기준 — 선에 `0.0041%p` 미달 |
| v2.5.2 코드 / 24 유닛 | `5.2675061589%` | **`+0.2716%p` — 코호트 재분류만으로** |
| v2.5.3 코드 / 24 유닛 | `5.3779318216%` | `+0.1104%p` — 코드 기여 |

미달분 `0.0041%p`는 코드가 개입하기 전에 코호트 재분류만으로 이미 초과됐다. 갭 해소의 약 71%가
분모 변경이다. 재분류에는 독립 근거가 있고(구조 공백 != near-gap이며, 두 pose는 quarantine되어
사용자에게 도달하지 않는다) §25-1에 두 수치를 함께 공개했지만, **결정 시점이 실패 유닛을 확인한
이후**라는 점은 명시해 둔다. `n`도 27 → 24로 11% 줄었고, 승격전략 §6이 요구한 image-cluster
bootstrap 95% CI는 아직 산출하지 않았다. 현재 `5.38%`는 유닛 1개가 만든 마진 `0.38%p`의 점추정이다.

덧붙여 이 지표는 RTMPose pseudo-target 적합도이며(승격전략 §6), `131056:p2`에서 2D 손목 metric이
거짓 성공을 만든 사례를 이미 확인했다. `5.38%`와 `5.00%`의 차이는 지표 자체의 유효 마진 안이다.

따라서 v2.5.3의 정확한 성격은 "품질 마일스톤"이 아니라 **"회귀 0 + 구현 누락 1건 수정 + known-bad
자산 차단"**이다. 승격전략 §7의 최종 완료선(작가 blind, SafeUsable non-regression, 실메시)은 그대로
열려 있다.

---

## 26. v2.5.3 시점의 알려진 공백

머지를 막지는 않지만 기록하고 다음 반복에서 닫는다.

### 26-1. 게이트 상수의 1-표본 유래

`160° / 15° / 45° / 18°` 네 상수는 모두 `131211:p1`(target `171.8°`, B0 `139.7°`, delta `32.1°`)
한 유닛에서 나왔고, 플래그는 프로덕션 기본 ON이다. D0 24개 중 foreshortened 다리를 가진 유닛은
5개(`124702:p0`, `131000:p0`, `131211:p1`, `2.16.52:p1`)이며 실제 발화는 1개다. 게이트가 선택적이라는
증거인 동시에, **오발화율에 대한 증거가 사실상 없다**는 뜻이다.

완화 요인은 방어층이 전부 fail-closed라는 점이다. `lower_body_observed is True` → 양다리 동시 충족
시 폐기(`len(evidence) != 1`) → B0 대비 ankle 개선 필수 → angle 게이트 → `UpLeg 18°` clamp →
selector의 C non-regression → FINAL collision post-check. 최악의 결과가 "나쁜 출력"이 아니라
"개선 없음"이다. 운영 로그에 발화율 카운터를 추가하고, 실표본이 5건 이상 쌓이기 전에는 상수를
재조정하지 않는다.

### 26-2. 테스트 공백

v2.5.3을 직접 검증하는 테스트는 `test_v25_product_defaults_are_safe_aggressive`(config 기본값)와
`test_v253_foreshortened_single_leg_extension_closes_131211_fixture`(성공 경로) 2개뿐이다. 다음은
미작성이다.

- 음성 발화: `target_angle < 160°`, `extension_delta`가 `[15,45]` 밖, `proximal > distal`
- 양다리 동시 충족 시 `len(evidence) != 1` 폐기
- `lower_body_observed=False`에서 증거가 완비돼도 미발화 (현재는 `allowed_limbs` 축소를 통한 전이 보장뿐)
- `final_extension_gate` 되돌림 경로 — 가장 파괴적인데 미검증
- 필요한 변화가 `UpLeg 18°`를 초과하는 케이스

`tests/test_pose_quarantine.py`는 로더 / geometry 검색 next-candidate 승격 / `/refine` 409 /
BVH 다운로드 409 4건으로 충분하다.

### 26-3. next Top-K의 사용자 결과가 미검증

§25-6 항목 2의 "실제 메시 육안 fixture"는 `171734:p0`, `131056:p2`를 육안 재검증하는 대신
**quarantine 후 평가에서 제외**하는 방식으로 닫았다. next Top-K 승격 경로는 구현·유닛테스트(합성
인덱스)까지 됐지만, **그 두 러프에서 사용자가 실제로 무엇을 대신 받는지 확인한 eval 유닛은 없다.**
v2.5.3에서 사용자에게 보이는 변화 중 검증이 가장 얇은 지점이다.

### 26-4. §25-6 항목 4는 미해결

lap-contact 실표본은 여전히 `n=0`이고, 대체 조건이던 "release capability에서 experimental로 명시"도
하지 않았다(코드·문서에 해당 표기 없음). 두 갈래 모두 열려 있다.

### 26-5. 미구현 진단 계약

- `structural_gap_required` (§25-4) — 없음
- `visibility_excluded_limbs` (§11-2 P1) — 없음
- `extension_observability_rescue`는 `decisions[limb]["reason"]`에 기록되지만 이후
  `record_choice` 또는 `last_reason`에 항상 덮인다. 구제된 limb 여부는 `limb_decisions`가 아니라
  `diagnostics.single_leg_extension`으로만 판별할 수 있다.
