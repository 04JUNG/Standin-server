# Chain Transport V3.1 Ankle Freeze Record

## 결정

`CHAIN_TRANSPORT_V3_1_ANKLE`을 발목 단계의 재현 가능한 QA 기준선으로 동결한다.

- 상태: `DISCOVERY_ACCEPTED_SOURCE_CONSTRAINED_RESIDUAL`
- 기준 부모: frozen `CHAIN_TRANSPORT_V3`
- 제품 승격: 아직 하지 않음
- 운영 converter 변경: 없음
- Git 목적: 향후 독립 FBX converter 저장소로 옮길 수 있는 원본 기록 보존

## 승인 범위

V3.1은 frozen V3의 발목 incremental 최소회전 `H2`만 SO(3) 최단호에서 부분 적용한다.
골반, 허리, 손목, 손가락, root translation과 나머지 체인은 승인 범위가 아니다.

정책은 파일명·클립명·좌우 예외 없이 source rig profile과 회전량으로만 결정된다. 현재
`mixamo_noprefix`에 `80°` soft cap과 `120°` hard guard를 적용하고, 다른 프로필은 frozen
V3를 유지한다. `120°`를 넘으면 frozen V3와 동일하게 parent-follow로 복구한다.

## 근거

- Blender converter regression: 28/28
- 실물 V3/V3.1 변환: 18/18
- 독립 export/reimport verifier: 9/9
- actual mirror: active side만 반전되고 foot/toe 외 canonical state exact
- 사용자 육안 확인: 발목 방향 역전 없음, V3보다 소폭 개선
- UAL2 Slide 잔여 굽힘: 원본 BVH에도 같은 방향으로 존재하는 source-constrained residual

세부 수치와 범위 증명은
`qa/retarget/CHAIN_TRANSPORT_V3_1_ANKLE/DISCOVERY_01_REPORT.md`에 보존한다.

## 무결성

```text
frozen V3 retarget.py
2701cc44a9ec6584722c194002d505a21bbf4fe199695325bb3b660c2818985e

V3.1 retarget.py
f6d9a35268ff18173d9280baf8e502f5e258dbaeb8de4ffb4dd83637c19e9c6b

V3.1 ankle_policy.json
79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f
```

`qa/retarget/CHAIN_TRANSPORT_V3_1_ANKLE/SHA256SUMS`로 solver 묶음을 검증한다.

## 후속 작업

1. 새 `Standin-fbx-converter` 저장소를 만들면 이 동결 묶음을 이관한다.
2. 골반/허리는 V3.2 별도 후보로 만들고 V3.1 발목 수학은 변경하지 않는다.
3. 손목은 손가락 rest 구조를 사용하는 별도 terminal solver로 미룬다.
4. production 통합 전 holdout과 API/Blender 실행 경로를 별도로 검증한다.
