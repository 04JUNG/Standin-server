# V3.2 pelvis-boundary discovery-01

## 결론

`CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY` 구현과 수학·회귀·실물 artifact 검증은 통과했다.
V3.1은 수정하지 않았고 production converter도 건드리지 않았다. 현재 상태는
`PENDING_ORIGINAL_FBX_VISUAL_GATE`이며 제품 승격이나 V3.1 대체 판정은 하지 않는다.

- parent V3.1 retarget: `f6d9a352…e9c6b`
- V3.2 retarget: `692e975d…13693`
- V3.1 ankle policy: `79cb19ad…80805f` 그대로
- deterministic math controls: 8/8
- Blender regression: 28/28
- paired real conversions/surface: 20/20
- independent artifact verifier: 10/10
- pre-export scope/Hips exact: 10/10
- forced one-side degeneration -> bilateral exact V3.1 fallback: PASS
- actual mirror + independent verifier: PASS
- render: 생성하지 않음

## 구현 범위

Hips의 output은 계속 legacy C다. 활성 다리의 시작 누적값만 Identity에서 Hips 수송으로
바꾼다.

```text
G_H = R_hips_output @ inverse(R_hips_target_rest)
H0  = MinRotation(G_H @ target_rest_thigh -> source_pose_thigh)
Q0  = H0 @ G_H
```

그 뒤 shin은 순차 최소회전으로 다시 풀고 foot에는 동결 V3.1 soft-cap/hard-guard를 적용한다.
V3.1과 V3.2의 `Q0` 상대회전은 모든 active case에서 source thigh 축 twist였으며 독립
검증기의 최대 방향 오차는 `0.0289°`였다.

## Paired 결과

`boundary L/R`는 V3.1의 Identity seed에서 parent-coherent seed로 바뀌며 제거되는 순수 roll이다.
RMSE와 surface는 진단값이며 자동 합격·선택에 사용하지 않는다.

| case | boundary L/R | RMSE V3.1 -> V3.2 | foot request L/R V3.1 -> V3.2 | hip edge p95 L/R V3.1 -> V3.2 |
|---|---:|---:|---:|---:|
| UAL2 Slide | 2.55° / 1.05° | 0.559458 -> 0.559616 | 109.9/33.3 -> 111.0/33.3 | 105.37/100.20 -> 104.69/100.91 |
| UAL2 SwordHeavy | 33.00° / 45.77° | 0.209977 -> 0.210126 | 51.7/95.1 -> 36.3/58.5 | 87.99/84.87 -> 96.38/91.54 |
| g1 Move1 | 79.53° / 98.76° | **0.184225 -> 0.182824** | 96.4/83.4 -> 41.3/37.1 | 104.95/99.28 -> 104.81/101.22 |
| g1 Move17 | 0.61° / 0.66° | exact | 16.6/18.2 -> 16.5/17.9 | 7.44/7.15 -> 7.30/7.15 |
| g1 Move7 | 12.39° / 24.51° | exact | 29.5/23.6 -> 27.3/23.3 | 78.14/77.00 -> 76.98/76.98 |
| CMU | 16.23° / 21.01° | 0.277728 -> 0.276123 | 29.7/141.7 -> 19.8/134.0 | 110.26/104.95 -> 112.52/106.22 |
| Mixamo control | inactive | **exact** | exact | exact |
| Rokoko | 71.26° / 80.50° | exact | 93.0/31.3 -> 54.8/89.3 | 115.19/109.34 -> 109.50/106.47 |
| MakeHuman | 0.00° / 0.00° | semantic no-op | 34.1/44.8 -> same | same |
| UAL2 Hook | 0.00° / 0.00° | semantic no-op | 28.9/49.7 -> same | same |

CMU 오른발은 요청각이 여전히 120°를 넘으므로 V3.1 `frozen_v3_hard_guard`가 유지됐다.
다만 upstream shin 프레임이 바뀌어 최종 foot world frame도 V3.1과 같지는 않다. CMU 발 방향은
반드시 원본 FBX로 다시 확인해야 한다. UAL2 SwordHeavy는 hip surface 대리지표가 악화됐으므로
수치만으로 골반 개선을 주장하지 않는다.

## 범위 증명

10건 모두 다음을 만족했다.

- Hips pre-export record: V3.1과 raw exact
- spine/neck/head/shoulder/arm/hand: raw exact
- skeleton baseline: exact
- ankle policy SHA: exact
- 변경 가능 canonical: `upleg/leg/foot/toe` 양쪽뿐
- Mixamo control: 전 canonical exact
- MakeHuman/UAL2 Hook: 회전 의미상 no-op

강제 퇴화 control에서는 leg.L parent seed 첫 계산만 실패시켰고, 양쪽 활성 다리의 모든
pre-export canonical record와 solver mode가 V3.1과 raw exact로 복구됐다.

## 육안 확인 우선순위

원본 FBX를 같은 DCC에서 비교한다. 렌더는 만들지 않았다.

1. `g1-move1`: 골반 뒤틀림·사타구니 구멍이 실제로 줄었는가
2. `g1-move7`: 정상 반례가 유지되는가
3. `cmu`: 승인했던 발 방향이 회귀하지 않았는가
4. `ual2-swordheavy`: 골반과 발목이 악화되지 않았는가
5. `rokoko`: 큰 boundary roll 교정이 해부학적으로 맞는가
6. `ual2-slide`: V3.1에서 마무리한 발목 방향이 유지되는가

경계 twist가 제거됐는데도 골반 구멍이 남으면 solver 추가 튜닝을 중단하고 target character의
Hips/UpLeg weight와 pelvis topology 문제로 분류한다.

## FBX 경로

```text
V3.1: /Users/dowon/dev/Standin-server/out/rest-v32-qa/outputs/discovery-01/V31/<case>/artifact.fbx
V3.2: /Users/dowon/dev/Standin-server/out/rest-v32-qa/outputs/discovery-01/V32/<case>/artifact.fbx
```

case 이름: `ual2-slide`, `ual2-swordheavy`, `g1-move1`, `g1-move17`, `g1-move7`,
`cmu`, `mixamo-ctrl`, `rokoko`, `makehuman`, `ual2-hook`.
