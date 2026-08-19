# Refine v2.5 사전 승격선 달성 전략

작성일: 2026-08-18  
대상 버전: v2.5 safe aggressive  
목표: 제품 기본값은 유지하면서 **selector를 완화하지 않고** 속도와 최종 NME를 사전 승격선까지 개선한다.

## 1. 현재 기준선과 부족분

Frozen near-gap D0 27개에서 최종 사용자가 받는 artifact를 conservative와 비교한 결과다.

| 항목 | 현재 | 사전 승격선 | 부족분 |
|---|---:|---:|---:|
| Joint NME 평균 감소 | 4.43% | ≥ 5.00% | **0.57%p** |
| Endpoint NME 평균 감소 | 5.81% | 양수 유지 | 통과 |
| Hand-pair error 감소 | 6.52% | 양수 유지 | 통과 |
| Lower-pair error 감소 | 7.80% | 양수 유지 | 통과 |
| Lap-contact error 감소 | 11.86% | 양수 유지 | 통과 |
| 신규 구조 위반 | 0/27 | 0 | 통과 |
| selector 활성 metric 회귀 | 0 | 0 | 통과 |
| cache-off p95 | 4.22초 | ≤ 3.00초 | **1.22초, 약 29% 단축 필요** |
| timeout 후 fallback | 1/27 | timeout 0 | 1건 제거 필요 |

현재 final mode는 aggressive 12, conservative 14, base 1이다. 따라서 다음 단계의 핵심은 안전 기준을
낮추는 것이 아니라, **현재 탈락하는 aggressive 후보를 처음부터 더 안전하고 정확하게 만들고 두 단계의
중복 계산을 제거하는 것**이다.

제품 기본 safe aggressive 전환은 운영 결정이고, 위 사전 승격선이 충족됐다는 뜻은 아니다. 수치 목표는
계속 유지한다.

### 2026-08-18 v2.5.1 구현 후 3회 재측정

위 전략의 P0/P1을 구현하고 같은 frozen D0 27개를 cache-off 3회(총 81 요청) 재측정했다. 결과는
`out/eval/v25_optimization_v251_x3_20260818/REPORT.md`에 고정했다.

| 항목 | v2.5.0 기준선 | v2.5.1 1차 | 자동 목표 |
|---|---:|---:|---:|
| Joint NME 평균 감소 | 4.43% | **5.75%** | ≥ 5.00% |
| Endpoint NME 평균 감소 | 5.81% | **7.60%** | 양수 |
| Hand-pair error 감소 | 6.52% | **19.73%** | 양수 |
| Lower-pair error 감소 | 7.80% | **4.62%** | 양수 |
| 요청 p95 | 4.22초 | **1.75초** | ≤ 3.00초 |
| 요청 max | 약 4.85초 | **2.71초** | < 5.00초 |
| timeout | 1/27 | **0/27** | 0 |
| 구조 hard violation | 0 | **0** | 0 |
| 채택 artifact selector regression | 0 | **0** | 0 |

최종 mode는 aggressive 18, conservative 8, base 1이다. v2.5.1에서 실제 반영한 내용은 다음과 같다.

- target normalize/mask/weight와 B0 parse/FK 준비값을 C/A가 재사용한다.
- `prepare/conservative/aggressive/selector/elapsed` 시간과 solver `nfev`, parameter 수, 활성 objective를 기록한다.
- C에서 관측 가능한 pair/contact가 없고 C가 전체 예산의 25% 이상을 쓴 요청만 A를 생략한다.
- A의 channel bounds와 발목 보정은 C 기준 새 예산이 아니라 B0에서 남은 절대 trust budget을 사용한다.
- aggressive objective에 final Joint NME와 같은 방향의 작은 L2 surrogate를 추가한다.
- full A가 탈락하면 C→A 전역 alpha `0.75/0.5/0.25`를 동일 hard gate와 non-regression으로 재검사한다.

이는 자동 사전 승격선의 **3회 engineering 측정 통과**다. 아직 최종 승격 증거는 아니다. 이후 proxy
경보 8건을 사용자와 진단 검토한 결과 `171734:p0`에서 refine 안전 회귀인 FINAL 관통이 확인됐고,
`171734:p0`, `2.16.52:p0`에서는 반신 컷의 불필요한 하체 이동이 확인됐다. `4.56.21:p0`은
refine으로 풀 근접 공백이 아니라 더 유사한 라이브러리 포즈를 놓친 search miss로 재분류됐다.
`131056:p2`는 B0부터 손 모양이 기괴했고 손·손가락 rotation은 B0/C/FINAL에서 동일해 원본 BVH
품질 결함으로 재분류했다. FINAL 관통이 팔 이동으로 악화됐는지는 mesh 비교가 남았다.
`124702:p0`은 FINAL이 가장 유사해 foot-direction 경보가 false positive였고, `131040:p0`은 품질은
좋지 않지만 안전 문제는 없는 것으로 판정됐다.

따라서 자동 hard violation 0만으로는 안전선을 충족하지 못했다. 제품 기본 safe aggressive는 유지하되
승격 판정은 보류하고, 하체 eligibility와 FINAL 관통 post-check를 수정한 뒤 같은 fixture와 D0를 다시
실행한다. 상세는 `out/review/v251_proxy_alerts_20260818/FINDINGS.md`다.

### 2026-08-18 v2.5.2 안전 수정 후 3회 재측정

반신/하체 비관측 lower-body freeze, mode 선택 후 FINAL 충돌 post-check, `4.56.21:p0` search miss
분리를 구현하고 동일 frozen D0를 3회 재실행했다. 결과는
`out/eval/v25_optimization_v252_x3_20260818/REPORT.md`다.

| 항목 | v2.5.1 | v2.5.2 | 자동 목표 |
|---|---:|---:|---:|
| 평가 unit | 27 | 26 | search miss 제외 |
| Joint NME 평균 감소 | 5.75% | **4.9958946357%** | ≥ 5.00% |
| Endpoint NME 평균 감소 | 7.60% | **6.3194578428%** | 양수 |
| Hand-pair error 감소 | 19.73% | **19.8567888536%** | 양수 |
| Lower-pair error 감소 | 4.62% | **4.6165179532%** | 양수 |
| 요청 p95 | 1.75초 | **1.548초** | ≤ 3.00초 |
| 요청 max | 2.71초 | **2.765초** | < 5.00초 |
| timeout / hard violation / selector regression | 0 / 0 / 0 | **0 / 0 / 0** | 0 / 0 / 0 |

최종 mode는 aggressive 15, conservative 9, base 2다. `171734:p0`, `2.16.52:p0`의 하체 channel
최대 변화는 `0.0`이고 두 unit의 이전 하체 경보가 사라졌다. 남은 proxy 경보는 3건이며 사용자 판정상
정상 개선/안전 문제 없음/B0 asset 결함으로 분류된 unit들이다.

다만 Joint NME는 정확한 5.00% 선에 **0.004105%p 미달**이다. 측정값을 소수 둘째 자리로 반올림해
통과 처리하지 않는다. 또한 lap-contact 표본은 여전히 0이고 FINAL 충돌 검사는 실제 메시가 아닌 capsule
proxy이므로 v2.5.2의 상태는 “요청 안전 수정의 engineering 회귀 통과, 사전 NME 승격선 미달,
실메시/작가 평가 대기”다.

### 2026-08-18 구조 공백 재분류와 v2.5.3 마감 결정

FINAL 6개 육안검사에서 `171734:p0`은 반신 target에 양무릎 꿇기 B0가 선택된 구조 공백,
`131056:p2`는 hands-forward/clasped 의도를 현재 2D 손목 refine으로 만들 수 없는 구조 공백이자 불량
hand asset으로 확정했다. 두 unit은 refine 성능 분모에서 제외하고 검색/asset 품질 트랙으로 넘긴다.

동일 v2.5.2 artifact를 다시 계산한 결과:

| 항목 | 기존 26개 | 유효 near-gap 24개 | 사전 목표 |
|---|---:|---:|---:|
| Joint NME 감소 | 4.9958946357% | **5.2675061589%** | ≥5.00% |
| Endpoint NME 감소 | 6.3194578428% | **6.6529686195%** | 양수 |
| Joint better/tie/worse | 15/11/0 | **14/10/0** | worse 0 |

구조 공백을 포함한 것이 Joint NME를 `0.2716115232%p` 낮췄으며, 더 큰 문제는 2D 수치상 개선과 실제
손 깊이·손가락·메시 안전이 분리되는 거짓 성공을 만들었다는 점이다. 따라서 near-gap solver 성능선은
통과했지만 제품 출시선은 아직 통과하지 않았다.

`131211:p1`은 제외하지 않는다. 같은 sitting 계열에서 목표 오른쪽 무릎 `171.8°`에 대해 B0/FINAL이
모두 약 `139.7°`인 유효 근접 공백이며, `foreshortened → low_observability`로 solve가 누락된 구현
문제다. Leg-only 실험은 무릎 endpoint NME를 악화시켰으므로 v2.5.3은 `UpLeg <=18°`의 제한적 보조와
Leg/제한 Foot을 함께 사용한다.

출시 판단은 다음과 같다.

- v2.5.3 자동 closeout: 24개×3회, Joint NME 5.3779%, Endpoint 6.8257%, worse/hard/proxy/timeout 0,
  p95 1.548초
- ⚠️ 단 이 통과의 `+0.2716%p`는 코호트 재분류, `+0.1104%p`만 v2.5.3 코드다. 유닛 단위로는 24개 중
  `131211:p1` 하나만 값이 바뀌었고 better/tie/worse는 `14/10/0`으로 불변이다. 분해는
  `REFINE_V2_5_IMPLEMENTATION.md` §25-7-1 참조
- known bad 2개 pose는 geometry 검색과 stale BVH/refine 요청에서 차단하고 다음 Top-K로 이관.
  단 그 두 러프에서 사용자가 실제로 받는 대체 후보의 품질은 아직 검증하지 않았다
- v2 최종 engineering release candidate로 채택 — "회귀 0 + 구현 누락 1건 수정 + known-bad 자산 차단"이며
  §7 최종 완료선 달성이 아니다
- 작가 blind·실메시 holdout·lap-contact 실표본·bootstrap 95% CI는 대규모 운영 승격 증거로 계속 추적

## 2. 지켜야 할 불변식

다음 방식으로 숫자를 맞추지 않는다.

- selector epsilon을 키워 작은 회귀를 허용하지 않는다.
- 구조 trust-region, collision, foot direction, ground contact gate를 끄지 않는다.
- timeout을 늘려 p95 문제를 숨기지 않는다.
- D0 27개에만 맞춘 weight를 새 holdout 검증 없이 제품값으로 고정하지 않는다.
- raw aggressive를 selector 전에 publish하지 않는다.
- **평가 코호트(분모)를 사후에 바꿔서 선을 넘기지 않는다.** 재분류가 필요하면 실패 유닛을 보기 전에
  기준을 고정하고, 불가피하게 사후 재분류했다면 통과 수치를 **코호트 기여분과 코드 기여분으로 분리해**
  기록한다. 2026-08-18 v2.5.3이 이 규칙의 첫 사례이며 분해는
  `REFINE_V2_5_IMPLEMENTATION.md` §25-7-1에 있다.

항상 B0 원본 구조 안전, C 대비 공통 metric non-regression, exact fallback을 유지한다.

## 3. 속도 개선 전략

### P0. 단계별 계측부터 세분화

현재 `conservative_ms`, `aggressive_ms`, `selector_ms`만으로는 어디에서 1.22초를 줄일지 부족하다.
다음 값을 unit별 sidecar와 평가 summary에 추가한다.

- BVH parse/FK 준비 시간
- target normalize·mask·coverage 계산 시간
- observability Jacobian/SVD 시간
- conservative residual 평가 횟수(`nfev`)와 parameter 수
- aggressive residual 평가 횟수와 활성 objective(`hand_pair/lower_pair/lap_contact/ankle`) 수
- collision/anatomy/foot safety 검사 시간
- selector 구조 검사와 metric 검사 시간
- 최종 BVH 직렬화·원자적 publish 시간

27개 cache-off를 각 3회 실행해 median을 쓰고, p95 기여도가 큰 상위 unit 5개를 먼저 profile한다.

### P1. C와 A가 공유하는 prepared context 도입

현재 safe aggressive는 conservative를 만든 뒤 aggressive를 실행하면서 parse, FK, target/mask,
observability, 충돌 기준 계산을 상당 부분 다시 수행한다. 다음 immutable context를 요청당 한 번 만든다.

```text
PreparedRefineContext
  - parsed hierarchy/channels + B0 frame
  - target keypoints/scores + normalized target/masks/weights
  - view projection
  - joint/channel mapping + ancestry
  - limb observability + axis Jacobian/SVD
  - B0 collision/contact/foot baselines
  - fixed metric cohorts
  - deadline/budget
```

C와 A는 같은 context 위에서 frame만 변경한다. 임시 BVH를 C 단계에서 쓰고 다시 parse하지 않고, selector가
최종 artifact를 선택한 뒤 **한 번만** BVH를 직렬화한다. 구조 selector는 계산 경로의 독립성을 유지하되
파싱된 hierarchy 같은 불변 데이터는 재사용할 수 있다.

목표 효과: p95 20~30% 단축. 이 변경 하나로 3초에 못 미치면 다음 조건부 solve를 함께 적용한다.

### P1. 활성 objective와 예산 위험을 함께 본 aggressive solve

aggressive 전용 목표가 실제로 하나도 활성화되지 않은 컷이라도 Joint NME surrogate 이득이 있을 수 있다.
따라서 전부 생략하지 않고 timeout 위험이 큰 요청만 두 번째 solver를 생략한다.

- C에서 관측 가능한 `hand_pair/lower_pair/lap_contact`가 모두 inactive이고 C가 전체 예산의 설정 비율 이상을
  소비했으면 C를 즉시 선택한다.
- hand pair만 활성화면 팔 block만 열고 다리 parameter는 동결한다.
- lower/lap만 활성화면 관련 다리와 필요한 접촉 상대 block만 연다.
- aggressive parameter 수와 residual block 수에 따라 `max_nfev`를 가변화한다.
- C를 warm start로 쓰고 gain·step이 연속으로 미소하면 조기 종료한다.

이 최적화는 결과 목표를 줄이는 것이 아니라 **활성되지 않은 변수와 residual만 제거**하므로 안전 정책을
바꾸지 않는다.

### P1. deadline을 단계 예산으로 운영

전체 5초를 한 solver가 다 쓰게 하지 않는다.

1. 준비와 C에 상한을 둔다.
2. final selector/publish reserve를 먼저 확보한다.
3. 남은 시간과 활성 block 크기로 A의 `nfev`를 계산한다.
4. reserve가 부족하면 A를 시작하지 않고 C를 반환한다.

목표는 timeout fallback을 0으로 만드는 것이다. 단순히 `REFINE_TIMEOUT_SECONDS`를 늘리지 않는다.

### P2. FK/Jacobian hot path 최적화

P1 뒤에도 p95가 3초를 넘을 때 진행한다.

- channel ancestry와 joint index lookup을 사전 계산한다.
- 수치 미분 perturbation의 FK를 batch/vectorize한다.
- residual 한 번 안에서 동일 frame의 FK·projection·pair vector를 한 번만 계산한다.
- Jacobian sparsity를 solver에 제공해 관계없는 channel 미분을 건너뛴다.
- soft collision residual은 저비용 proxy로 계산하고, 기존 정밀 hard gate는 최종 후보에 그대로 적용한다.

C와 A는 의존 관계라 동시에 병렬 실행하지 않는다. 작은 문제에서는 thread/process 시작 비용이 더 클 수
있으므로 먼저 계산 재사용과 sparsity를 적용한다.

### 속도 실험 통과 조건

- cache-off D0 27개 × 3회에서 p95 ≤ 3.0초
- max < 5.0초, timeout 0
- cache-on 반복 호출은 기존 content-address hit 유지
- 선택 mode, 최종 BVH hash, selector 판정이 최적화 전과 동일한 parity set 100%

## 4. Joint NME 5% 달성 전략

현재 0.57%p만 부족하므로 큰 동작 범위를 추가할 필요가 없다. 안전하게 회수할 가능성이 높은 순서로
진행한다.

### P0. solver objective와 selector metric의 불일치 줄이기

solver는 direction·Huber endpoint·move·pair/contact의 혼합 목적을 줄이지만, 최종 selector는 Joint NME와
endpoint/pair/contact를 직접 본다. 내부 loss가 좋아져도 Joint NME가 좋아지지 않는 후보가 생길 수 있다.

- 공통 mask와 torso normalization을 solver·selector 단일 소스로 고정한다.
- Joint NME의 differentiable surrogate를 aggressive objective에 작은 primary 항으로 추가한다.
- 변경 사지 endpoint뿐 아니라 공통 visible joint가 나빠지는 방향에도 학습 중 penalty를 준다.
- pair/contact weight는 해당 cohort가 실제 활성일 때만 적용한다.

목표는 selector를 통과시키는 편법이 아니라, solver가 처음부터 selector가 평가하는 기하를 최적화하게 만드는
것이다.

### P1. B0 기준 남은 trust budget 안에서 solve

C에서 이미 관절이 많이 움직였는데 A가 C 기준 local bound를 다시 모두 쓰면 B0 absolute trust-region을
초과해 후보가 폐기될 수 있다.

- 관절별 `remaining_delta = B0_limit - abs(C - B0)`를 A의 실제 bound로 사용한다.
- foot/contact/collision도 B0 baseline 대비 허용 가능한 남은 범위를 residual에 반영한다.
- 최종 gate는 그대로 두되, 위반이 예상되는 탐색 공간을 애초에 제거한다.

폐기될 후보 계산을 줄여 NME와 속도 모두에 도움이 된다.

### P1. selector-guided 안전 부분 채택

현재 A 전체가 한 metric에서 나빠지면 C 전체로 돌아가 A의 유용한 block 개선도 버릴 수 있다. v2.5.1
후보로 다음 deterministic 선택을 검증한다.

1. 1차 구현은 C→A 전체 rotation의 `alpha = 0.75, 0.5, 0.25` 후보를 만든다.
2. 각 후보를 B0 구조 hard gate로 먼저 거른다.
3. C 대비 모든 활성 metric non-regression을 만족하는 후보 중 Joint NME가 가장 낮은 것을 고른다.
4. 동률이면 이동량이 작은 후보, 그다음 C를 선택한다.

이 방식은 selector 기준을 완화하지 않고, 안전한 일부 개선을 회수한다. D0에서는 global blend 2건이
채택됐다. block별 leave-one-out은 global blend로 회수되지 않는 별도 holdout 실패가 확인될 때만 P2로
진행한다.

### P1. timeout unit 회수

현재 1개 timeout은 후보 품질 문제가 아니라 계산 미완료다. §3의 속도 개선으로 안전한 A 또는 C 평가가
완료되면 평균 Joint NME의 일부를 자연스럽게 회수할 수 있다.

### P2. weight 보정은 train/holdout 분리

- D0를 튜닝 집합으로 직접 사용하지 않는다.
- current rough에서 별도 train cluster를 만들고 hand/lower/lap 활성 유형별 소규모 sweep을 한다.
- Joint NME, endpoint, pair/contact와 안전 탈락률을 다목적으로 본다.
- 후보 weight는 frozen D0와 새 작가 holdout에서 한 번만 검증한다.
- 몸통 회전은 이번 NME 목표 달성을 위해 켜지 않는다.

### NME 실험 통과 조건

- 최종 artifact Joint NME 평균 감소 ≥ 5.0%
- unit별 selector 활성 metric worse 0
- 신규 구조 hard violation 0
- aggressive 탈락 시 C/base exact fallback 100%
- Endpoint/Hand/Lower/Lap 평균 개선이 모두 양수 유지
- 작가 blind 결과에서 major-worse 0, SafeUsable non-regression

## 5. 구현·실험 순서

| 순서 | 실험 | 기대 효과 | 실패 시 판단 |
|---:|---|---|---|
| 1 | 세부 phase/nfev 계측 | 병목 확정 | profile 없이 optimizer 변경 금지 |
| 2 | prepared context + 최종 1회 write | p95 20~30% 감소 | hot path profile로 P2 이동 |
| 3 | inactive objective skip + block freeze | 긴 tail 감소 | 결과 parity 확인 후 nfev 조정 |
| 4 | B0 remaining trust budget | 구조 탈락·낭비 감소 | 기존 fixture 회귀 시 즉시 rollback |
| 5 | exact-metric surrogate | Joint NME +0.57%p 이상 | train weight 재보정 |
| 6 | selector-guided block alpha | 안전한 부분 이득 회수 | 후보 수/latency 상한 강화 |
| 7 | frozen D0 + 새 holdout + blind | 사전 승격선 판정 | 실패 유형별 다음 iteration |

속도 변경과 NME 변경을 한 번에 섞지 않는다. 먼저 결과 parity를 유지하는 속도 최적화를 완료한 뒤,
후보 기하를 바꾸는 NME 실험을 하나씩 추가해야 원인 귀속이 가능하다.

## 6. 평가 프로토콜

### 자동 평가

- 동일 27개 frozen D0와 새 이미지-cluster holdout 사용
- cache off, warm process, 각 unit 3회
- p50/p95/max, timeout, phase별 시간, nfev를 함께 보고
- 이미지 cluster bootstrap 95% CI를 NME 개선율에 추가
- `mode_requested/effective/applied`와 selector fallback 사유별 NME를 분해

### 정성 평가

- base/C/final A 이름을 가리고 무작위 순서로 비교
- 첫인상 유사도, 손·다리 공백, 관통, 발 방향/지면, 4-view 붕괴, 수정 없이 사용 여부 기록
- 자동 NME는 RTMPose pseudo target 적합도이므로 작가 의도를 대신하지 않는다.

## 7. 최종 완료선

아래를 동시에 만족하면 사전 승격선 달성으로 기록한다.

- Joint NME 평균 개선 ≥ 5%
- cache-off p95 ≤ 3초
- timeout/error 0
- 신규 구조 violation 0
- selector metric regression 0
- exact fallback 100%
- Endpoint/Hand/Lower/Lap 개선 유지
- blind major-worse 0, SafeUsable non-regression

달성 전에도 제품 기본은 safe aggressive지만, 운영 로그에서 fallback 비율과 p95를 계속 감시한다.
문제가 생기면 `REFINE_DEFAULT_MODE=conservative`로 즉시 mode rollback하고, 필요하면
`REFINE_V2_ENABLED=0`으로 v1을 사용한다.

근거:

- `REFINE_V2_5_IMPLEMENTATION.md` §22
- `out/eval/v25_current_rough_near_gap_d0_20260817/REPORT.md`는 raw aggressive 사전 결과
- `docs/REFINE_V2_DESIGN.md`
