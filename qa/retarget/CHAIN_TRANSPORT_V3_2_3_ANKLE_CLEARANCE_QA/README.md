# V3.2.3 Ankle Clearance QA

상태: **QA 후보 / 운영 미연결**

V3.2.2의 `mu=1` 결과가 각도상 정확해도 실제 target 메시의 뒤꿈치–종아리 간격이
부족할 수 있어 만든 후속 후보이다.

- source/target의 thigh, shin, foot rest 길이와 `foot/shin` 비율을 기록한다.
- 길이 비율은 회전 공식에 직접 넣지 않는다. 길이만으로 mesh heel 형상과 weight를 알 수
  없기 때문이다.
- 기존 최소회전 축은 유지한 채 `mu=1.25`, `mu=1.5` 후보를 추가한다.
- 강한 후보는 `mu=1`보다 clearance/면적/edge strain 중 하나가 실질 개선되고 다른 지표가
  비퇴행이어야 한다.
- source-rest 목표 overshoot는 최대 12°이며, 전체 swing은 60° hard guard 안에 있어야 한다.
- 실제 메시 rescue target을 처음 만족하는 후보군에서 가장 안전한 결과를 선택한다.
- 모든 후보가 실패하면 exact V3.2.1로 복구한다.

파일명·BVH family·성별명별 보정은 없다. 실제 target 기하와 vertex weight가 선택한다.

## 선택 원리

1. exact V3.2.1을 기준 상태 `mu=0`으로 만든다.
2. V3.2.2와 같은 최소회전 축으로 `mu=0.5, 1, 1.25, 1.5` 후보를 만든다.
3. 각 후보에서 실제 변형 메시의 발–종아리 clearance, 삼각형 면적, edge 압축/신장,
   dihedral 변화를 측정한다.
4. `mu>1` 후보는 `mu=1`보다 실제 메시가 개선되고 다른 지표가 비퇴행일 때만 허용한다.
5. rescue gate를 만족하는 가장 작은 gain을 선택한다. 없다면 안전한 후보 중 최소 위험을
   선택하며, 측정 불가/전 후보 불안전이면 exact V3.2.1로 복구한다.

`target/source foot-to-shin` 비율은 원인 진단용으로만 기록한다. 비율 차이는 같은 관절
각도가 서로 다른 메시 간격을 만드는 증폭 요인이지만, 메시 형상과 웨이트를 모르는 길이
비율만으로 회전량을 강제하지 않는다.

검증 결과는 [QA_REPORT.md](QA_REPORT.md)에 기록한다.
