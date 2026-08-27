# CHAIN_TRANSPORT_V3_1_ANKLE QA candidate

> 상태: **DISCOVERY_ACCEPTED_SOURCE_CONSTRAINED_RESIDUAL / FROZEN QA REFERENCE**
> 제품 승격: 없음(새 converter 저장소로 옮기기 위한 재현 기준선)
> 운영 파일 변경: 없음

사용자가 승인한 frozen `CHAIN_TRANSPORT_V3`에서 발목의 incremental 최소회전 `H2`만
SO(3) 최단호에서 부분 적용하는 별도 QA 후보다. frozen V3 snapshot은 수정하지 않았다.

## 수학과 범위

```text
predicted_foot = Q_shin @ target_rest_foot_edge
H2_full        = MinRotation(predicted_foot -> source_pose_foot_edge)
H2(mu)         = Exp(mu Log(H2_full)) = slerp(I, H2_full, mu)
Q_foot         = H2(mu) @ Q_shin
Q_toe          = Q_foot
```

- `mu=1`: frozen V3 full foot solve와 동일
- `mu=0`: frozen V3 parent-follow와 동일
- raw ankle increment `>120°`: 정책보다 먼저 parent-follow; 우회 불가
- pelvis, thigh, shin, arms, hands, translation, hierarchy order: frozen V3 그대로
- 파일명, BVH label, 좌우별 예외: 없음

정책은 `converter/ankle_policy.json` 한 곳에서 읽고 import 시 schema와 범위를 fail-closed로
검증한다. 현재 `mixamo_noprefix`만 soft cap `80°`, hard guard `120°`를 사용한다. 그 외
명시 프로필은 120° 보호 외에 frozen V3를 그대로 사용한다.

80° < raw angle <= 120°일 때:

```text
u       = clamp((raw - 80) / 40, 0, 1)
applied = (1-u) * raw + u * 80
mu      = applied / raw
```

identity까지 회전을 지우는 선형 backoff가 아니라, 적용각을 80° cap 쪽으로 완만히 옮긴다.

## Discovery-01 결과

- Blender converter regression: **28/28 PASS**
- 실물 변환: **9 cases x V3/V3.1 = 18/18 PASS**
- 독립 export/reimport transport verifier: **9/9 PASS**
- UAL2 Slide actual mirror: **PASS**, active side `foot.L -> foot.R`, changed bones `foot.R/toe.R` only
- non-foot/toe pre-export canonical state: frozen V3와 exact
- no-op cases: G1 Move17, CMU, Mixamo control, Rokoko, MakeHuman, UAL2 Hook exact
- UAL2 Slide L: `109.91° -> 87.55°`, ankle edge p95 `39.84 -> 35.78`,
  min-stretch p05 `0.328 -> 0.416`, condition p95 `2.775 -> 2.307`
- UAL2 SwordHeavy R: `95.06° -> 89.39°`, 모든 주요 ankle surface 지표 개선
- G1 Move1 L/R: `96.35° -> 89.67°`, `83.36° -> 83.08°`, 모든 주요 ankle surface 지표 개선

수치는 개선됐지만 UAL2 Slide가 보수적 discovery envelope의 edge p95 `35.4%`와
20% 초과 edge 비율 `0.182`를 각각 `35.78%`, `0.1866`으로 약간 넘는다. 임계를 결과에
맞춰 다시 조정하지 않았다.

## 육안 판정과 동결 결정

사용자가 원본 FBX를 직접 비교한 결과, V3.1은 발목 방향 역전을 재발시키지 않았고 V3보다
조금 개선됐다. UAL2 Slide에는 굽힘이 남지만 원본 BVH 자체에도 같은 방향의 발목 회전이
있음을 확인했다. 따라서 그 잔여량을 소스가 가진 포즈로 보고, 이를 억지로 펴기 위한 추가
튜닝은 하지 않는다.

- 발목 계층: 이 상태로 discovery 기준선 동결
- UAL2 Slide: 잔여 과굴곡은 알려진 source-constrained residual
- 골반/허리 및 손목: 이 solver의 승인 범위가 아니며 별도 후속 단계
- production 적용: 별도의 통합·holdout·제품 승격 결정 전까지 없음

QA 산출물과 manifest:

```text
/Users/dowon/dev/Standin-server/out/rest-v31-qa/outputs/discovery-01/
/Users/dowon/dev/Standin-server/out/rest-v31-qa/manifests/discovery-01.json
/Users/dowon/dev/Standin-server/out/rest-v31-qa/manifests/discovery-01-independent.json
```

렌더는 생성하지 않았다. 각 case의 `V3/artifact.fbx`와 `V31/artifact.fbx`를 직접 비교한다.

## 기준선 무결성

```text
frozen V3 retarget.py:
2701cc44a9ec6584722c194002d505a21bbf4fe199695325bb3b660c2818985e

V3.1 parent retarget.py SHA before edit:
2701cc44a9ec6584722c194002d505a21bbf4fe199695325bb3b660c2818985e
```

현재 후보 파일은 `SHA256SUMS`로 검증한다.
