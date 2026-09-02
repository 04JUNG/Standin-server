# V3.2.4 Foot Plant QA Report

상태: **QA 후보 / 운영 미연결 / 사용자 육안 승인 전**

실행 환경: Blender 5.2.0 LTS, frame 0, `rigged_rest`.

## 문제 재정의

V3.2.2/V3.2.3은 종아리–발 bone bend와 heel/calf clearance를 개선했지만, 사용자가
확인한 원본 포즈의 발바닥 전체 접지를 재현하지 못했다. CMU 왼쪽 source toe pitch는
`-1.5808°`인데 exact parent의 실제 target plantar pitch는 남성 `+65.1839°`, 여성
`+67.2670°`였다. 따라서 남은 문제는 작은 gain 부족이 아니라 source toe frame과 target
메시 발바닥 frame 사이의 큰 불일치였다.

## 선택 결과

| 케이스 | 선택 swing | source toe | parent plantar | final plantar | pitch 오차 |
|---|---:|---:|---:|---:|---:|
| CMU 남성 | L `-78°` | -1.5808° | +65.1839° | -1.8898° | 0.3089° |
| CMU 여성 | L `-78°` | -1.5808° | +67.2670° | -1.8584° | 0.2775° |
| CMU 남성 mirror | R `-78°` | -1.5808° | +65°대 | -1.3775° | 0.2033° |
| Talking On Phone 정상 대조군 | 0° | side별 측정 | parent와 호환 | parent 그대로 | exact parent |

## 실제 메시 안전성

| 케이스 | clearance p01 | area p01 | edge min | strain p99 | dihedral p99 | 새 sharp fold |
|---|---:|---:|---:|---:|---:|---:|
| CMU 남성 | 0.09639 | 0.90574 | 0.83007 | 0.09430 | 2.9573° | 0 |
| CMU 여성 | 0.10936 | 0.91747 | 0.84700 | 0.08321 | 3.1920° | 0 |
| CMU 남성 mirror | 0.09938 | 0.88635 | 0.80975 | 0.11670 | 3.3802° | 0 |

추가 불변식:

- correction twist 최대 절댓값 `0.000004°` 미만.
- toe 상대회전 오차 `0°`.
- proximal 본 행렬 오차 `0`.
- post-bake 정점 오차 `0`.
- mirror에서 활성 보정이 L에서 R로 이동.
- 정상 대조군은 승인 parent와 52개 본 행렬 최대 오차 `0`, 전체 메시 정점 최대 거리 `0`.
- 남성 `pose_fidelity_rmse`: V3.2.3 `0.183768` → V3.2.4 `0.180821`.
- 여성 `pose_fidelity_rmse`: V3.2.3 `0.171795` → V3.2.4 `0.170500`.

## 안전 경계

- source toe pitch가 수평에서 8° 이내이고 parent plantar pitch 오차가 12° 이상일 때만 탐색한다.
- coarse/fine 후보가 pitch 오차 3° 이내와 실제 메시 절대 gate를 동시에 통과해야 한다.
- 최종 양측 결합 상태를 다시 측정하고 실패하면 bilateral exact V3.2.1로 복구한다.
- 파일명, 프로파일명, 성별, 다리 길이 비율은 회전량 선택에 사용하지 않는다.
- 단일 프레임에는 force·velocity·floor collision이 없으므로 물리적 접지 여부의 완전한
  증명은 아니다. source toe 방향과 target plantar 방향을 맞춘 결과이다.

## 판정

정량 contact/mesh gate와 정상 대조군 보존은 통과했다. 운영 승격은 하지 않으며, 남성·여성
원본 FBX에 대한 사용자 육안 확인을 다음 gate로 둔다.
