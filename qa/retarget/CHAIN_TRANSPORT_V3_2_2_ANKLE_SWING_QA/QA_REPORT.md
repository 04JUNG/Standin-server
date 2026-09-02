# V3.2.2 Ankle Flexion Swing QA Report

작성일: 2026-08-31  
상태: **구현 완료 / QA 후보 유지 / 운영 승격 안 함**

## 구현 결론

승인된 `V3.2.1 Palm Roll(mu=0.5)`을 부모로 고정하고 발목 굽힘 swing만 추가했다.
부모 solver 파일은 수정하지 않았다. 좌우 후보는 `mu=0/.25/.5/.75/1`이며, 모든 후보는
부모 포즈에서 새로 시작한다. 선택 실패·측정 불가·다중 메시 입력은 FBX를 거부하지 않고
exact 부모(`mu.L=mu.R=0`)를 내보낸다.

```text
source_motion = beta_source_pose - beta_source_rest
beta_desired  = beta_target_rest + source_motion
Q(mu)         = slerp(I, MinRotation(foot_parent -> foot_desired), mu)
```

`Q`는 foot 방향을 목표 굽힘으로 보내는 최소회전이다. 같은 `Q`를 toe에도 적용하므로
foot-toe 상대회전은 유지된다. unsigned 굽힘 목표가 `[0.5°, 179.5°]` 밖이면 축의 부호를
지어내지 않고 해당 발을 exact 부모로 복구한다.

## 자동 선택 안전 경계

비영 후보는 다음을 모두 통과해야 한다.

- 부모 굽힘 오차와 target-rest-relative 운동이 활성 임계 이상
- 실제 발목 메시 ROI 또는 극단 target motion에서 이상 징후 확인
- 굽힘 오차 최소 1° 개선
- 추가 twist `<=0.05°`
- foot 상위 본 행렬 변화 `<=1e-6`
- foot-toe 상대회전 변화 `<=0.001°`
- 메시 면적·edge·dihedral·foot-calf clearance 비퇴행
- clearance/면적/edge strain 중 하나가 실질 개선

정책 숫자는 `ankle_swing_policy.json` 한 곳에 있고, clip명·파일명·성별명 분기는 없다.
남녀 차이는 실제 rest 기하와 vertex weight로만 측정된다.

## 검증 결과

### 1. 순수 수학 및 negative controls

Blender 5.2.0 LTS:

- 목표 굽힘 119° 복원 오차: `0.0000109°`
- 추가 twist: `0.0°`
- `mu=0` identity 행렬 오차: `0.0`
- mirror 각도 오차: `0.0000109°`
- 동일 메시(실질 개선 없음) 후보: 거부
- 면적 퇴행 후보: 거부
- 정상 발목: 비활성
- 과잉 굽힘 + 메시 이상: 활성

### 2. 문제 사례 — 남성 `cmu_05_10_00400`

선택: `L=1.0`, `R=0.0`.

| 항목 | 부모 | 선택 결과 |
|---|---:|---:|
| 왼발 굽힘 목표 오차 | 20.82687° | 0.000067° |
| 추가 twist | - | 0.00000185° |
| toe 상대회전 오차 | - | 0.0° |
| 상위 본 행렬 오차 | - | 0.0 |
| foot-calf clearance p01 | 기준 | +0.02110 leg ratio |
| 발목 면적 ratio p01 | 기준 | +0.11473 |
| log edge strain p99 | 기준 | −0.15047 |
| bake 정점 오차 | - | 0.0 |

오른발은 unsigned 목표가 범위 밖이라 `UNMEASURABLE`로 보호되어 부모를 유지했다.

### 3. 정상 대조군 — `Talking On Phone_02`

양쪽 모두 비활성, 선택 `L=0`, `R=0`. 승인 부모 FBX와 독립 재임포트 비교:

- 52개 본 집합 동일
- 본 rest matrix 최대 절대차 `0.0`
- 모든 메시 정점 최대거리 `0.0`

즉 정상 대조군 artifact는 exact 부모다.

### 4. actual mirror

같은 CMU 입력을 mirror하면 문제 발목이 반대로 이동해 `L=0`, `R=1`이 선택됐다.
오른발 목표 오차는 21.61084°에서 0.000063°로 감소했고 twist·toe 상대회전·상위 본
변화는 0 수준이다. 메시 clearance, 면적, edge strain도 모두 개선됐다.

### 5. 여성 정규화 메시

동일 CMU 입력에서 `L=1`, `R=0` 선택:

- 굽힘 목표 오차 `20.43975° -> 0.000024°`
- twist `0.00000101°`
- clearance p01 `+0.02281 leg ratio`
- 면적 p01 `+0.13608`
- log edge strain p99 `−0.11656`
- bake 정점 오차 `0.0`

여성 전용 분기 없이 실제 여성 메시의 기하·웨이트가 같은 공통 정책을 통과했다.

## 산출물

QA FBX와 JSON은 저장소 밖 임시 디렉터리에 있다.

```text
/tmp/v322-ankle-swing-qa/
```

주요 파일:

- `cmu_05_10_00400.fbx/json`
- `cmu_05_10_00400_mirror.fbx/json`
- `cmu_05_10_00400_female.fbx/json`
- `talking_on_phone.fbx/json`
- `talking_on_phone_parent.fbx/json`
- `talking_on_phone_exact_compare.json`

## 미완료와 승격 제한

- 실제 사용자가 FBX를 열어 발 모양을 육안 승인하는 gate는 아직 안 했다.
- 현재 검증은 문제 1건, 정상 대조 1건, mirror, 남녀 두 메시다. reviewed ankle cohort
  전체 비회귀는 아직 안 돌렸다.
- 저장소 스모크는 시스템 `python3`에 `pydantic`이 없어 시작 전에 중단됐다. 새 모듈은
  운영 import 경로 밖이지만, 정식 환경에서 기존 스모크를 다시 확인해야 한다.
- 따라서 이 구현을 운영 converter에 연결하거나 `V3.2.2`로 승격하지 않는다.
