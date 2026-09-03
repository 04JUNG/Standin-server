# CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY QA candidate

> 상태: **FROZEN / MATH+REGRESSION+ARTIFACT QA PASS / ORIGINAL-FBX VISUAL GATE PASS**
> 사용자 승인: **2026-08-27 — V3.2 골반이 매우 자연스러워졌으며 합격**
> 제품 승격: 알고리즘 동결 승인, converter 서비스 통합은 아직 하지 않음
> 운영 파일 변경: 없음

사용자가 승인한 frozen `CHAIN_TRANSPORT_V3_1_ANKLE`을 부모로 둔다. V3.1 파일은 수정하지
않았고, 이 후보는 활성 다리의 `Hips -> UpLeg` 시작 프레임만 parent-coherent projection으로
바꾼다.

## 중심 수학

```text
G_H = R_hips_output @ inverse(R_hips_target_rest)

predicted_thigh = G_H @ target_rest_thigh_edge
H0              = MinRotation(predicted_thigh -> source_pose_thigh_edge)
Q0              = H0 @ G_H

predicted_shin  = Q0 @ target_rest_shin_edge
H1              = MinRotation(predicted_shin -> source_pose_shin_edge)
Q1              = H1 @ Q0

Q2              = H2(mu_v3.1) @ Q1
```

`Q0`은 source thigh 방향을 만족하는 회전 중 실제 Hips 수송에 가장 가까운 해다. Identity에서
시작한 V3.1 해와 `Q0`의 차이는 수치 오차를 제외하면 source thigh 축 twist다. 이 후보는
그 미정 roll을 부모 Hips에서 이어받는다.

## 동결·변경 범위

- Hips legacy 결과와 root translation: V3.1 exact
- spine, neck, head, shoulder, arms, hands: V3.1 exact
- 이미 rest-compatible인 leg 경로: V3.1 exact
- 활성 leg: `upleg/leg/foot/toe`만 변경 가능
- ankle soft-cap/hard-guard 정책과 SHA: V3.1 그대로
- 한쪽 parent seed라도 퇴화: 양쪽 활성 다리 exact V3.1 복구
- 파일명·클립명·좌우 예외와 메시 자동 selector: 없음

## 현재 검증

- deterministic math controls: **8/8 PASS**
- Blender converter regression: **28/28 PASS**
- 실물 paired 변환+surface 측정: **10 cases x V3.1/V3.2 = 20/20 PASS**
- 독립 export/reimport transport verifier: **10/10 PASS**
- pre-export 변경 범위/Hips exact/ankle policy SHA: **10/10 PASS**
- 강제 한쪽 퇴화 -> 양쪽 exact V3.1 fallback: **PASS**
- UAL2 Slide actual mirror + independent verifier: **PASS**
- 사용자 원본 FBX 골반 육안 gate: **PASS (2026-08-27)**
- production import: 없음
- render: 생성하지 않음

자동 surface 수치는 selector나 합격 gate로 사용하지 않는다. `g1-move1`, `g1-move7`, CMU,
UAL2 Slide/SwordHeavy를 포함한 원본 FBX는 사용자가 직접 확인했고, V3.2 골반 경계를
자연스럽다고 판정했다. 이 승인은 solver 동결 판정이며 HTTP converter 서비스 배포 승격과는
구분한다.

## 기준 해시

```text
parent V3.1 retarget.py
f6d9a35268ff18173d9280baf8e502f5e258dbaeb8de4ffb4dd83637c19e9c6b

V3.1 ankle_policy.json
79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f

V3.2 retarget.py
692e975d32f41e3406144763c7c0b7dbf0a586ff07732f09c4365e3233b13693
```

현재 묶음은 `SHA256SUMS`로 검증한다.

## QA 산출물

```text
/Users/dowon/dev/Standin-server/out/rest-v32-qa/manifests/math-controls.json
/Users/dowon/dev/Standin-server/out/rest-v32-qa/manifests/discovery-01.json
/Users/dowon/dev/Standin-server/out/rest-v32-qa/manifests/discovery-01-independent.json
/Users/dowon/dev/Standin-server/out/rest-v32-qa/manifests/discovery-01-scope.json
/Users/dowon/dev/Standin-server/out/rest-v32-qa/manifests/bilateral-fallback-control.json
/Users/dowon/dev/Standin-server/out/rest-v32-qa/outputs/discovery-01/V31/
/Users/dowon/dev/Standin-server/out/rest-v32-qa/outputs/discovery-01/V32/
```
