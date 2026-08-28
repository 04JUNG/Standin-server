# Hybrid 검색 다중 러프 파라미터 실험

작성일: 2026-08-28

브랜치: `codex/hybrid-search-lab`

상태: engineering D0 탐색 — 운영 승격 근거 아님

## 1. 결론

15장 러프의 27명 frozen skeleton과 1,434개 파라미터 조합을 비교했다.

- 현재 raw hybrid `w=0.7`은 기본값으로 부적합하다.
- `4.56.21:p0`의 목표 family를 1위로 강제하는 조합도 다른 러프에서 확인 가능한 회귀가 발생했다.
- 보수적인 H2 조합은 목표를 4위에서 2위로 올리면서 기존 compatibility proxy 25개를 모두 Top-5에 유지했다.
- 이 보수 조합은 27명 모두에서 position과 **동일한 Top-5 family 집합**을 반환하고 순서만 바꿨다.
- 다만 H2가 단순 raw H0 `w=0.025`보다 실제로 더 정확한지는 현재 라벨로 판정할 수 없다. 두 후보를 blind 비교해야 한다.
- Scale, angle weight, short-bone threshold는 아직 운영값으로 확정하지 않는다.

따라서 현 단계 권고는 다음과 같다.

```text
운영 기본값 유지: position
blind 비교 후보 A: raw H0, full/reduced w=0.025
blind 비교 후보 B: conservative H2
폐기 후보: raw H0 w=0.7, target-Top1 강제 후보
```

## 2. 평가 범위

### 2.1 데이터

Frozen D0에서 검색 가능한 27명을 사용했다.

| 항목 | 수량 |
|---|---:|
| 러프 이미지 | 15장 |
| 평가 인물 | 27명 |
| full | 18명 |
| reduced | 7명 |
| sparse | 2명 |
| DB projection | 6,152개 |
| pose family | 769개 |

`4.56.21:p0`은 다음 human-approved 보호 사례로 분리했다.

```text
pose_family_id = cmu_124_13_00661
view           = three_quarter
```

나머지 사례 중 DB에서 기존 선택 family를 추적할 수 있는 25명을 compatibility proxy로 사용했다. 이 후보들은 과거 position 검색을 바탕으로 선택되었으므로 독립적인 정확도 정답이 아니다. 다음 용도로만 사용한다.

- 기존에 유용했던 후보가 Top-5 밖으로 사라지는지 탐지
- 파라미터 변화 폭 비교
- 육안 검토 대상을 좁히기

### 2.2 검색 정책

- 모든 projection에 direct vectorized exhaustive search
- position shortlist 없음
- stable sort
- `pose_family_id`별 최선 variant 하나
- Family 단위 Top-5 backfill
- 운영 검색 코드에는 연결하지 않은 독립 실험 모듈

## 3. 탐색한 파라미터

총 1,434개 설정을 비교했다.

### 3.1 Angle weight

- Raw H0 uniform weight: `0.000~0.700`, 간격 `0.025`
- H1/H2 full weight:
  - `0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.200, 0.250, 0.300`
- H1/H2 reduced/sparse weight:
  - `0.000, 0.010, 0.025, 0.050, 0.075, 0.100`

Sparse는 별도 high-confidence 대상이 아니며, 이번 검색 실험에서는 reduced profile을 공유했다.

### 3.2 임시 component scale

Position-derived proxy의 선택 projection component 중앙값을 실험용 scale로 계산했다.

| Profile | Position scale | Angle scale | 표본 수 |
|---|---:|---:|---:|
| full | `0.172674` | `0.077975` | 16 |
| reduced+sparse | `0.139150` | `0.032392` | 9 |

이 값은 최종 calibration이 아니다. Position-derived proxy로 계산했으므로 position 편향이 있다.

또한 H1의 ranking은 다음 상대 계수로 결정된다.

\[
\frac{w / s_{angle}}{(1-w) / s_{position}}
\]

따라서 relevance 라벨 없이 ranking만 보면 scale과 weight를 독립적으로 식별할 수 없다. 이번 sweep은 raw scale과 proxy-median scale을 비교했지만 운영 scale 확정에는 별도의 human-labeled D1이 필요하다.

### 3.3 Short-bone 관측성

Smoothstep 구간:

- `0.02~0.10`
- `0.03~0.12`
- `0.05~0.18`
- `0.08~0.25`

Minimum effective bone mass:

- `3`
- `4`
- `6`

## 4. 주요 후보

### P — Position 기준선

```text
metric = position
```

### R — 기존 raw hybrid

```text
metric = H0
full/reduced weight = 0.700
scale = raw
```

### C — 보수 H2

```text
metric = H2
scale = raw (position=1, angle=1)
full weight = 0.025
reduced/sparse weight = 0.050
short-bone smoothstep = 0.08~0.25
minimum effective bone mass = 4
```

`minimum effective bone mass=3/4/6`은 이번 주요 수치에서 동률이었다. `4`는 중앙 실험값일 뿐 우월성이 입증된 값이 아니다.

### T — 목표 Top-1 H2

```text
metric = H2
scale = proxy median
full weight = 0.125
reduced/sparse weight = 0.010
short-bone smoothstep = 0.05~0.18
minimum effective bone mass = 4
```

## 5. 수치 결과

| 후보 | `4.56.21` 목표 순위/시점 | Proxy Top-5 유지 | Position Top-5 평균 overlap | Top-1 변경 | 치명 호환성 회귀 |
|---|---|---:|---:|---:|---:|
| P position | 4 / three-quarter | 25/25 | 5.00 | 0 | 0 |
| R raw H0 `w=0.7` | 1 / three-quarter | 17/25 | 2.11 | 16 | 8 |
| C conservative H2 | 2 / three-quarter | 25/25 | 5.00 | 2 | 0 |
| T target-Top1 H2 | 1 / three-quarter | 23/25 | 4.15 | 7 | 2 |

Family duplicate는 모든 평가 결과에서 0건이었다.

### 5.1 단순 H0 `w=0.025` 비교

추가 복잡성이 없는 raw H0도 다음 결과를 냈다.

| 항목 | 결과 |
|---|---:|
| `4.56.21` 목표 순위 | 2위 |
| Proxy Top-5 유지 | 25/25 |
| Position Top-5 평균 overlap | 4.93 |
| Top-1 변경 | 0 |

Conservative H2는 position과 Top-5 family 집합이 27명 모두 동일했고, raw H0 `w=0.025`는 일부 Top-5 경계 family를 교체했다. 반면 H2는 Top-1을 두 건 바꿨다. 어느 쪽이 실제로 더 좋은지는 proxy 수치로 결정할 수 없다.

### 5.2 Raw `w=0.7` 회귀

기존 raw H0 `w=0.7`이 선택 proxy를 Top-5 밖으로 밀어낸 사례:

- `124702:p0`
- `131056:p0`
- `131112:p0`
- `131112:p3`
- `131127:p1`
- `131211:p1`
- `2.16.04:p0`
- `2.16.52:p2`

대표적인 순위 변화:

| Unit | 기존 선택 family | Position 순위 | Raw H0 `w=0.7` 순위 |
|---|---|---:|---:|
| `2.16.04:p0` | `rokoko_Lawyer_02_mixamo_00008` | 2 | 100 |
| `2.16.52:p2` | `Talking On A Cell Phone (1)_01` | 1 | 121 |
| `124702:p0` | `rokoko_Interogation_Suspect_mixamo_00199` | 3 | 41 |

### 5.3 Conservative H2의 변경

Top-5 family 집합은 27명 모두 position과 같았으며, 순서만 달라졌다. Top-1 변경은 두 건이다.

| Unit | Position Top-1 | Conservative H2 Top-1 | 기존 선택 순위 변화 |
|---|---|---|---:|
| `131127:p0` | `UAL2__NinjaJump_Start__p01_f0002` | `Shaking Hands 1_01` | 2→1 |
| `2.16.52:p2` | `Talking On A Cell Phone (1)_01` | `Reacting_02` | 1→2 |

첫 사례는 기존 선택 후보가 1위로 올라가지만, 두 번째는 휴대폰 후보가 2위로 내려간다. 두 변화 모두 Top-5 내부 순서 변경이므로 blind slate 비교 대상으로 적합하다.

### 5.4 Target-Top1 H2 회귀

Target-Top1 후보는 `4.56.21` 목표를 1위로 만들지만 다음 proxy를 Top-5 밖으로 밀었다.

| Unit | 기존 선택 family | Position 순위 | Target-Top1 H2 순위 |
|---|---|---:|---:|
| `131127:p1` | `rokoko_Juggling_mixamo_00412` | 4 | 12 |
| `2.16.52:p2` | `Talking On A Cell Phone (1)_01` | 1 | 20 |

`2.16.52:p2` 육안 시트에서는 휴대폰을 든 러프에 대해 `Hands Forward Gesture`가 1위로 올라갔다. 이 변경은 목표 사례 하나를 1위로 만들기 위해 받아들이기 어려운 회귀다.

## 6. 육안 관찰

현재 시트는 알고리즘과 점수를 표시한 diagnostic 자료이며 정식 blind 평가가 아니다.

### 6.1 Conservative H2

- `131127:p0`: `Shaking Hands` family가 2위에서 1위로 상승한다. 팔 방향은 query와 가까워 보이지만 다인 컷의 인물 소유권 영향이 있어 사람 판정이 필요하다.
- `2.16.52:p2`: 휴대폰 family가 1위에서 2위로 내려가고 `Reacting_02`가 1위가 된다. 두 후보가 모두 유사한 상체 형태라 blind 비교가 필요하다.
- `4.56.21:p0`: 목표 CMU family가 4위에서 2위로 상승하며 three-quarter view가 유지된다.

### 6.2 Target-Top1 H2

- `124702:p0`: 손을 모은 형태는 강조되지만 하체 배치가 크게 달라지는 후보가 1위가 된다.
- `131056:p0`: 다인 컷에서 후보 순서 변화가 커지고 person assignment 영향과 metric 효과를 분리하기 어렵다.
- `2.16.52:p2`: 휴대폰 후보가 20위로 하락하여 육안상 명확한 위험 신호다.
- `4.56.21:p0`: 목표 CMU family는 1위가 된다.

### 6.3 Raw H0 `w=0.7`

- `2.16.04:p0`: 팔 자세가 있는 기존 후보들이 밀리고 팔을 아래로 내린 직립 후보가 상위에 집중된다.
- `2.16.52:p2`: 휴대폰 동작 family가 121위까지 하락한다.

## 7. Short-bone 파라미터 해석

이번 D0에서 minimum effective bone mass `3/4/6`은 선택 후보의 주요 지표가 동일했다. 이는 다음 중 하나를 뜻한다.

1. 상위 후보의 effective bone mass가 이미 6 이상이다.
2. mass 감쇠가 순위를 바꿀 정도로 작동하지 않았다.
3. 현재 proxy가 short-bone 문제를 판별하지 못한다.

따라서 `4`를 최종값으로 해석하면 안 된다.

Smoothstep `0.08~0.25`가 보수 후보에서 position Top-5 집합을 가장 잘 보존했지만, 이 또한 compatibility 기준으로 선택된 값이다. 실제 투시 단축 포즈에서 좋은 방향 정보를 과도하게 제거하는지는 별도의 labeled slice로 검증해야 한다.

## 8. 시간

6,152 projection을 사용하는 direct vectorized search는 대표 육안 실행에서 다음 범위였다.

| 실행 | Metric | P50 | P95 |
|---|---|---:|---:|
| Conservative 대표 3명 | H2 | 약 2.08ms | 약 2.22ms |
| Target-Top1 대표 5명 | H2 | 약 2.21ms | 약 2.35ms |
| Raw 대표 5명 | H0 | 약 2.16ms | 약 2.34ms |

초기 engineering gate인 P95 10ms 안에 있으며, position shortlist를 추가할 시간상 이유는 확인되지 않았다.

## 9. 판정

| 후보 | 판정 | 이유 |
|---|---|---|
| Position | 유지 | 현재 운영 기준선 |
| Raw H0 `w=0.7` | REJECT | 8/25 proxy Top-5 회귀, 큰 순위 붕괴 |
| Target-Top1 H2 | REJECT | 휴대폰 등 다른 러프에서 확인 가능한 회귀 |
| Raw H0 `w=0.025` | BLIND CANDIDATE | 단순하며 25/25 유지, 목표 2위 |
| Conservative H2 | BLIND CANDIDATE | 25/25 유지, Top-5 family 집합 동일, 목표 2위 |

현 시점에서는 hybrid를 운영 기본값으로 승격하지 않는다.

## 10. 다음 테스트

정식 다음 단계는 position, raw H0 `w=0.025`, conservative H2의 3-arm blind 비교다.

우선 포함할 러프:

- `4.56.21:p0`: 보호 목표
- `131127:p0`: conservative Top-1 개선 후보
- `2.16.52:p2`: conservative 내부 순서 위험
- `124702:p0`: 손 모으기와 하체 배치 trade-off
- `131056:p0`: 다인/person assignment 영향
- `2.16.04:p0`: raw angle 회귀 보호
- `131127:p1`: target-Top1 회귀 보호

후보의 알고리즘명, rank, distance, pose ID를 숨기고 다음을 라벨한다.

- `direct`: 바로 사용 가능
- `reference`: 참고 후 사용 가능
- `unusable`: 사용 불가
- slate 전체 선호도

Scale과 weight의 최종 보정은 이 D0가 아니라 새로운 scene/artist 기반 D1에서 수행한다.

## 11. 재현 경로

자동 sweep:

```bash
.venv/bin/python scripts/sweep_hybrid_search.py \
  --out out/hybrid_lab/parameter_sweep_20260828
```

산출물:

- `out/hybrid_lab/parameter_sweep_20260828/sweep_results.json`
- `out/hybrid_lab/parameter_sweep_20260828/SWEEP_REPORT.md`
- `out/hybrid_lab/visual_conservative_20260828/`
- `out/hybrid_lab/visual_top1_20260828/`
- `out/hybrid_lab/visual_raw_h0_20260828/`
