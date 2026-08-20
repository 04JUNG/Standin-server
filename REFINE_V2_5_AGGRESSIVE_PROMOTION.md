# Refine v2.5 — Aggressive 기본 승격 계획

작성일: 2026-08-17  
목표: `aggressive`의 포즈 적합도 이득을 유지하면서 안전 회귀를 제거하고, 기본 config로 승격한다.

## 1. 현재 판단

**개선 신호는 강하지만 승격은 아직 불가하다.**

| 항목 | 하네스 결과 | 판단 |
|---|---:|---|
| Joint NME | v1 대비 **9.54% 감소** | 개선 |
| Endpoint NME | **13.18% 감소** | 개선 |
| Lower-pair error | **20.66% 감소** | 하체 목적이 효과 있음 |
| 단위별 Joint NME | 개선 20 / 동률 1 / 악화 4 | 회귀 제거 필요 |
| v2 신규 hard violation | **9/27건** | 승격 차단 |
| v2 p95 / max | 2.57초 / 5.00초 | deadline 여유 부족 |

v2 최종 결과는 aggressive 21건, conservative fallback 5건, base 1건이었다. 비교 가능한 aggressive 19건 중 18건이 NME 개선이므로 **개선의 주동력은 aggressive**다. 반면 신규 안전 위반 9건 중 8건도 aggressive에 집중됐다.

> 이 수치는 RTMPose pseudo target에 대한 적합도다. 실제 작가 의도 정확도나 검색 정답률을 증명하지 않는다.

## 2. v2.5 필수 개선

### P0. 외부 평가기와 채택 게이트 일치

- aggressive 후보를 반환하기 전에 외부 평가기와 동일한 조건으로 다시 검사한다.
- 현재 발견된 회귀를 모두 hard gate로 처리한다.
  - `foot_direction_regression`: 4건
  - `ground_contact_regression`: 2건
  - `lap_contact_regression`: 6개 violation, 5개 평가 단위
- 하나라도 실패하면 **conservative → base 순서로 정확히 복구**한다.
- solver와 evaluator가 임계값·관절 mask·contact cohort를 공유하도록 단일 소스화하고 parity test를 추가한다.

### P0. Aggressive non-regression 선택

- 내부 objective 감소만으로 aggressive를 채택하지 않는다.
- 최종 투영 결과가 conservative보다 아래 공통 지표에서 악화되면 conservative를 반환한다.
  - 전체 `joint_nme`
  - 변경 부위 `endpoint_nme`
  - 활성된 `hand_pair` / `lower_pair` / `lap_contact`
- 이득이 수치 노이즈 수준이면 aggressive 대신 conservative를 유지한다.
- 현재 NME 악화 4건을 회귀 fixture로 고정한다: `131056:p2`, `131127:p1`, `131211:p0`, `2.16.52:p2`.

### P0. 비교 기준의 운영 오류 제거

- v1에서 한쪽 팔만 유효할 때 발생한 `KeyError: right_arm` 2건을 수정하고 회귀 테스트를 추가한다.
- 동일 27개 단위가 모두 비교돼야 v2.5 전후 수치를 확정한다.

### P1. 5초 deadline 이전의 보수적 종료

- conservative, aggressive, 사후 안전검사에 단계별 시간 예산을 둔다.
- 남은 시간이 부족하면 aggressive를 시작하지 않고 conservative를 반환한다.
- 5초에 도달한 `131211:p0`을 성능 fixture로 고정한다.
- 목표: 운영 오류 0, deadline 도달 0, p95 3초 이하. 제품 예산이 확정되면 이 값을 더 엄격한 합의값으로 교체한다.

## 3. Config 승격 조건

아래를 모두 통과한 뒤에만 기본값을 바꾼다.

- 신규 hard safety violation: **0건**
- aggressive 채택 결과의 Joint NME 악화: **0건**
- 안전·non-regression 탈락 시 conservative/base exact fallback: **100%**
- 운영 오류 및 deadline 도달: **0건**
- 현재 19장 재실행에서 Joint/Endpoint/Lower-pair 평균 개선 유지
- 새 러프 holdout과 아래 직관 테스트 통과

승격 시 목표 config:

```env
REFINE_V2_ENABLED=1
REFINE_DEFAULT_MODE=aggressive  # v2.5에서 신규 추가
REFINE_V2_LOWER_BODY=1
REFINE_V2_TORSO=0
```

현재 API 기본값은 `conservative`이므로 `REFINE_DEFAULT_MODE`를 추가하고, 요청에서 mode가 생략됐을 때 config 값을 적용하도록 변경한다. 비상 복구는 `REFINE_DEFAULT_MODE=conservative` 또는 `REFINE_V2_ENABLED=0`으로 유지한다.

## 4. 내가 직접 할 직관 테스트

### 테스트 방법

각 평가 단위에서 이름을 가리고 다음 세 결과를 무작위 순서로 본다.

1. 검색 베이스
2. v2 conservative
3. v2.5 aggressive

한 단위당 아래 순서로 확인한다.

1. **첫인상 5초:** 러프와 가장 비슷한 결과를 하나 고른다.
2. **안전 확대:** 발 방향·바닥 접촉·손과 허벅지/무릎 관통을 본다.
3. **4-view 회전:** front / three-quarter / side / back에서 해부학적 붕괴를 본다.
4. **실사용 판단:** 직접 고치지 않고 사용할 수 있는지 `예/아니오`로 기록한다.

기록표:

| unit | 가장 유사한 결과 | 발/지면 문제 | 접촉/관통 | 다른 view 붕괴 | 수정 없이 사용 | 메모 |
|---|---|---|---|---|---|---|
|  | base / conservative / aggressive | 없음/있음 | 없음/있음 | 없음/있음 | 예/아니오 |  |

### 반드시 볼 컷

**안전 회귀 9개 단위**

- 발/지면: `124702:p0`, `131040:p0`, `171734:p0`, `2.16.52:p0`
- 접촉: `131000:p0`, `131056:p2`, `2.16.04:p0`, `2.16.52:p1`, `4.56.21:p0`

**NME 악화 4개 단위**

- `131056:p2`, `131127:p1`, `131211:p0`, `2.16.52:p2`

**개선 유지 확인 5개 단위**

- `131112:p0`, `131040:p1`, `131211:p1`, `131112:p3`, `124629:p0`

### 직관 테스트 통과선

- 안전 회귀 단위에서 hard issue **0건**
- NME 악화 4개 단위에서 aggressive가 conservative보다 명백히 나쁜 경우 **0건**
- 개선 확인 5개 중 최소 **4개**에서 aggressive 선호
- 전체에서 “수정 없이 사용” **80% 이상**, major-worse **0건**

## 5. 실행 순서

1. v1 부분 팔 `KeyError` 수정 및 fixture 추가
2. evaluator-solver 안전 parity gate 구현
3. aggressive-vs-conservative 공통 지표 non-regression gate 구현
4. deadline 예산·조기 fallback 구현
5. 동일 19장과 별도 holdout 자동 재실행
6. 위 직관 테스트 블라인드 수행
7. 통과 시 `REFINE_DEFAULT_MODE=aggressive`를 canary부터 단계적으로 적용

근거 산출물: `out/eval/in_refine_auto_final_20260814/REPORT.md`, `summary.json`, `records.json`
