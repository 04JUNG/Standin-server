# V3.2.5 Airborne Plantar QA

상태: **QA 후보 / 운영 미연결 / V3.2.4 exact rollback 필수**

V3.2.4는 source toe가 수평에 가까운 접지 방향 사례만 소유한다. V3.2.5는 그 경계를
바꾸지 않고, V3.2.4가 소유하지 않는 공중 발에 한해서 source rest 대비 종아리–발 상대
굽힘을 target 종아리 프레임의 가상 plantar plane으로 운반한다. 목표면과 실제 target
메시 발바닥의 차이는 V3.2.4와 같은 foot/toe swing 및 실제 메시 탐색으로 줄인다.

## 절대 롤백 계약

1. 변환 시작 시 V3.2.4 FBX를 먼저 별도 임시 파일로 완성한다.
2. source pitch `|θ| <= 8°`, V3.2.4 contact 보정 선택, 측정 불가, 안전 후보 없음,
   결합 상태 실패, 예외, 미지원 output mode 중 하나라도 성립하면 그 임시 FBX를 요청
   출력으로 byte-for-byte 복사한다.
3. fallback report에 parent/final SHA-256과 `exact_parent_artifact_restored=true`를 남긴다.
4. V3.2.4 코드·정책 SHA-256이 동결값과 다르면 시작하지 않는다.

수동 kill switch도 제공한다. `tools/run_case.py`에 `--force-exact-v324`를 주면 V3.2.5
policy 파싱·활성·후보 계산을 모두 우회하고 V3.2.4를 직접 실행한다.
`fallback_reason=FORCED_EXACT_V324`와 parent/final SHA 동일성을 기록한다.

## 공중 발 목표

```text
source_motion = source_pose_bend - source_rest_bend
target_desired = target_rest_mesh_plantar_bend + source_motion
```

월드 수평면을 공중 발에 강요하지 않는다. 실제 target 메시의 rear/front plantar bottom
band로 현재 방향을 재고, target shin과 이루는 상대 굽힘이 목표에 가까워지는 후보만
고른다. correction은 foot/toe에 같은 최소회전을 적용하므로 toe 상대회전과 proximal
chain은 보존한다.

## 안전 경계

- 접지 경계는 V3.2.4가 계속 소유한다.
- correction 절댓값은 90°를 넘지 않는다.
- 목표 오차 3° 이내인 후보만 선택한다.
- 부모 artifact는 이미 rest bake되어 변형 지표가 1/1/0이므로 그 값과 직접 비교하지
  않는다. 대신 사용자가 육안 승인한 V3.2.4 접지 결과보다 엄격한 contact deformation
  envelope(clearance 0.05, area 0.90, edge 0.82, strain 0.10, dihedral 3°, fold 0)와
  기존 절대 메시 gate를 모두 통과해야 한다.
- 최종 양측 결합 상태를 다시 검사하고 한 항목이라도 실패하면 전체 artifact를 exact
  V3.2.4로 복구한다.
- 파일명, 성별, BVH family, profile명으로 회전량을 선택하지 않는다.

실제 검증 결과는 `QA_REPORT.md`에 기록하며, 사용자 육안 승인 전에는 운영 승격하지 않는다.

## 후속 라이브러리 검수 도구

전체 활성 라이브러리 스캔과 육안 검수에 사용한 다음 도구는 V3.2.5 core 커밋 범위에서
제외한다.

- `tools/scan_library.py`
- `tools/build_review_assets.py`
- `tools/build_review_viewer.py`

검수 결과 자체는 `QA_REPORT.md`에 보존한다. core 런타임은 `tools/run_case.py`,
`airborne_plantar_safe.py`, 정책 파일과 exact V3.2.4 rollback 경계만으로 완결된다.
