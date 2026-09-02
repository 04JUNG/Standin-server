# V3.2.3 Ankle Clearance QA Report

상태: **QA 후보 / 운영 미연결 / 사용자 육안 승인 전**

실행 환경: Blender 5.2.0 LTS, frame 0, `rigged_rest`.

## 결론

V3.2.2의 `mu=1`은 원본 BVH의 종아리–발 swing 각도를 복원했지만, target 메시의
발/종아리 비율과 뒤꿈치 형상 때문에 실제 clearance가 충분하지 않았다. V3.2.3은 길이
비율을 직접 회전 공식에 넣지 않고, 같은 최소회전 축을 더 강하게 적용한 후보를 실제
메시 변형으로 판정한다.

CMU 왼발에서 target/source `foot-to-shin` 비율은 남성 1.3108, 여성 1.2407이었다.
비율 차이가 문제를 증폭한다는 근거는 있으나, gain은 이 비율이 아니라 실제 메시 gate가
선택했다.

## 실제 FBX 결과

| 케이스 | 선택 gain | swing | clearance p01 | area p01 | edge min | strain p99 | 새 sharp fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| CMU 남성 | L=1.50 | 31.2403° | 0.05000 | 0.49707 | 0.38134 | 0.37168 | 0 |
| CMU 여성 | L=1.25 | 25.5496° | 0.05155 | 0.49520 | 0.39067 | 0.37057 | 0 |
| CMU 남성 mirror | R=1.50 | 32.4162° | 0.05071 | 0.48864 | 0.33439 | 0.40930 | 0 |
| Talking On Phone 정상 대조군 | L=0, R=0 | 0° | exact parent | exact parent | exact parent | exact parent | 0 |

V3.2.2 `mu=1` 대비:

- 남성 clearance p01 `+0.01172`, area p01 `+0.08905`, strain p99 `-0.06422`.
- 여성 clearance p01 `+0.00581`, area p01 `+0.03710`, strain p99 `-0.02337`.
- 남성 `pose_fidelity_rmse`: 0.185063 → 0.183768.
- 여성 `pose_fidelity_rmse`: 0.172235 → 0.171795.

## 불변식

- correction twist 최대 절댓값은 0.000003° 미만이다.
- toe 상대회전 오차는 0°이다.
- proximal 본 행렬 오차는 0이다.
- post-bake 정점 오차는 0이다.
- mirror에서는 활성 보정이 L에서 R로 정확히 이동한다.
- 정상 대조군은 승인 parent와 52개 본 행렬 최대 오차 0, 전체 메시 정점 최대 거리 0이다.
- 파일명, 성별명, BVH family별 분기는 없다.

## 판정

수학/메시 정량 gate와 정상 대조군 보존은 통과했다. 다만 이 후보는 사용자가 원본 FBX를
직접 확인하기 전에는 운영 승격하지 않는다. 육안 결과가 부족하면 임계를 낮추어 억지로
통과시키지 않고 exact V3.2.1로 되돌린다.
