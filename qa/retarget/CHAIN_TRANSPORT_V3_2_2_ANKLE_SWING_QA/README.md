# V3.2.2 Ankle Flexion Swing QA

상태: **QA 후보 / 운영 미연결 / 자동 승격 금지**

부모는 사용자 승인 `V3.2.1 Palm Roll(mu=0.5)`이며 부모 파일을 복사하거나 수정하지
않는다. 새 자유도는 `foot.L/R`, `toe.L/R`에 같은 월드 최소회전을 적용하는 굽힘
`swing`뿐이다. Hips, 허벅지, 정강이, 손목, root translation은 바꾸지 않는다.

## 목표식

```text
beta_source_motion = beta_source_pose - beta_source_rest
beta_desired       = beta_target_rest + beta_source_motion
Q_full             = MinRotation(foot_parent -> foot_desired)
Q(mu)              = slerp(I, Q_full, mu)
```

`foot_desired`는 현재 V3.2.1 정강이 축 주위의 발 방향(azimuth)을 그대로 두고 굽힘각만
`beta_desired`로 바꾼다. 그러므로 새 회전에는 발 길이축 twist가 없다. toe에는 같은
월드 회전을 적용해 foot-toe 상대 회전을 보존한다.

## 정상 발목 보호

`mu=0`은 exact 부모 기준선이다. 비영 후보는 다음을 모두 통과해야 한다.

1. 부모 출력의 rest-relative 굽힘과 목표식 오차가 활성 임계보다 크다.
2. 출력 굽힘이 실제로 과도하거나 발목 메시 ROI에서 낮은 foot-calf clearance,
   면적/edge 압축, 큰 edge strain, 새 sharp fold가 관측된다.
3. 후보 굽힘 오차가 감소한다.
4. twist, foot-toe 상대회전, foot 상위 모든 본 행렬이 불변이다.
5. 실제 메시의 면적·edge·dihedral·clearance가 부모보다 나빠지지 않고 적어도 한 지표가
   실질적으로 개선된다.

좌우는 같은 공통 ladder `0/.25/.5/.75/1`에서 독립 선택한다. 파일명, BVH family,
성별 이름으로 분기하지 않는다. 정책 숫자는 `ankle_swing_policy.json` 한 곳에만 있으며
실제 리그 기하와 vertex weight로 측정한다.

## 가용성 계약

측정 불가, 퇴화, 후보 전부 탈락, 다중 메시 입력은 FBX 제공을 막지 않는다. 항상
`mu=0`인 exact `V3.2.1 Palm Roll(mu=0.5)`을 다시 한 번 실행해 내보낸다. 이 QA 후보는
라이브러리 포즈를 격리하거나 제거하지 않는다.

## 실행

```bash
BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
"$BLENDER_BIN" --background --python tools/test_math.py

"$BLENDER_BIN" --background --python tools/run_case.py -- \
  --bvh /absolute/input.bvh \
  --character /absolute/character.fbx \
  --out /tmp/v322-ankle.fbx \
  --report /tmp/v322-ankle.json
```

현재는 QA 후보다. actual FBX 육안 대조와 정상 대조군 cohort 비회귀가 끝나기 전에는
운영 converter에 연결하지 않는다.
