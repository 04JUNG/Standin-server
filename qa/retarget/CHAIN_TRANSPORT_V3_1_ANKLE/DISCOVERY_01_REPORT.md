# V3.1 ankle-only discovery-01

## 결론

`CHAIN_TRANSPORT_V3_1_ANKLE` 구현과 수학/회귀/실물 정량 검증은 통과했다. 이후 사용자가
원본 FBX를 직접 비교해 발목 방향 역전이 없고 V3보다 조금 개선됐음을 확인했다. UAL2
Slide의 잔여 굽힘은 원본 BVH에도 같은 방향으로 존재하므로, 상태를
`DISCOVERY_ACCEPTED_SOURCE_CONSTRAINED_RESIDUAL`로 동결한다. 제품 승격은 하지 않는다.

- production converter 수정: 없음
- frozen V3 수정: 없음 (`retarget.py` SHA-256 `2701cc44…8985e`)
- V3.1 retarget SHA-256: `f6d9a352…e9c6b`
- ankle policy SHA-256: `79cb19ad…80805f`
- Blender 회귀: 28/28
- V3/V3.1 실물 변환: 18/18
- 독립 artifact transport verifier: 9/9
- UAL2 Slide actual mirror + 독립 verifier: PASS (`foot.R/toe.R`만 변경)
- render: 생성하지 않음

## 적용량

| case | profile | left raw→applied | right raw→applied | V3 대비 RMSE |
|---|---|---:|---:|---:|
| UAL2 Slide | mixamo_noprefix | 109.91°→87.55° | 33.34°→33.34° | +0.126% |
| UAL2 SwordHeavy | mixamo_noprefix | 51.67°→51.67° | 95.06°→89.39° | −0.071% |
| G1 Move1 | mixamo_noprefix | 96.35°→89.67° | 83.36°→83.08° | +0.766% |
| G1 Move17 | mixamo_noprefix | 16.56°→16.56° | 18.24°→18.24° | exact |
| CMU | cmu_bvh | 29.70°→29.70° | 141.72°→parent-follow | exact |
| Mixamo control | mixamo | compatible legacy | compatible legacy | exact |
| Rokoko | mixamo | 93.02°→93.02° | 31.27°→31.27° | exact |
| MakeHuman | mixamo_noprefix | 34.08°→34.08° | 44.84°→44.84° | exact |
| UAL2 Hook | mixamo_noprefix | 28.94°→28.94° | 49.67°→49.67° | exact |

G1 Move1의 joint-head RMSE는 0.766% 증가했다. 발목 surface 지표는 양쪽 모두 개선됐지만,
이 수치를 감추거나 임계를 다시 맞추지 않는다. 원본 FBX에서 발목과 기존 골반 문제를 분리해
확인해야 한다.

## Active ankle surface

| case/side | edge p95 | >20% edge fraction | min stretch p05 | condition p95 |
|---|---:|---:|---:|---:|
| Slide L V3 | 39.840 | 0.2379 | 0.3284 | 2.7748 |
| Slide L V3.1 | **35.778** | **0.1866** | **0.4155** | **2.3065** |
| SwordHeavy R V3 | 35.394 | 0.1815 | 0.4112 | 2.6469 |
| SwordHeavy R V3.1 | **33.762** | **0.1707** | **0.4409** | **2.4504** |
| G1 Move1 L V3 | 31.355 | 0.1362 | 0.3975 | 2.6119 |
| G1 Move1 L V3.1 | **29.659** | **0.1207** | **0.4404** | **2.3769** |
| G1 Move1 R V3 | 33.133 | 0.1594 | 0.4864 | 2.3057 |
| G1 Move1 R V3.1 | **33.033** | 0.1594 | **0.4880** | **2.2970** |

Slide는 크게 개선됐지만 사전 보수 envelope `edge p95 <=35.4`, `fraction <=0.182`를 각각
0.378, 0.0046만큼 넘었다. `min stretch >=0.411`, `condition <=2.65`는 통과했다.

## Scope proof

pre-export canonical state 비교에서 바뀐 본은 정확히 다음뿐이었다.

- Slide: `foot.L`, `toe.L`
- SwordHeavy: `foot.R`, `toe.R`
- G1 Move1: `foot.L`, `toe.L`, `foot.R`, `toe.R`
- 나머지 6건: 변경 없음

모든 active case에서 그 외 본의 JSON 수치가 frozen V3와 exact였다. 독립 verifier는 report의
`mu`를 신뢰하지 않고 source/target/artifact를 재임포트해 부분 H2와 방향 잔차를 재구성했다.

## 사용자가 확인할 FBX

각 디렉터리의 `V3/artifact.fbx`와 `V31/artifact.fbx`를 원본 상태로 비교한다.

```text
/Users/dowon/dev/Standin-server/out/rest-v31-qa/outputs/discovery-01/V3/
/Users/dowon/dev/Standin-server/out/rest-v31-qa/outputs/discovery-01/V31/
```

우선순위 `ual2-slide`, `ual2-swordheavy`, `g1-move1`을 사용자가 확인했다. 이 판정은 발목
단계의 discovery 동결 근거일 뿐 production 승격 근거는 아니다. 골반/허리와 손목은 범위 밖이며,
향후 새 converter 저장소에서 별도 후보와 holdout으로 다룬다.
