# V3.2.4 Foot Plant QA

상태: **QA 후보 / 운영 미연결 / 사용자 육안 승인 전**

V3.2.3은 target의 종아리–발 각도 오차와 heel/calf clearance를 개선했지만 원본의
발바닥 접지를 복원하지 못했다. V3.2.4는 다음 두 방향을 분리한다.

- 원본 BVH: toe bone의 world pitch를 방향 증거로 사용한다.
- target FBX: 실제 변형 메시의 rear/front plantar bottom band pitch를 측정한다.

원본 toe가 수평에 가까우면서 target plantar pitch가 크게 다를 때만 foot/toe에 동일한
최소 swing을 적용한다. `-90°..+90°` coarse/fine 탐색에서 실제 메시 안전 gate와 pitch
일치를 동시에 통과한 후보만 선택한다.

한 프레임 BVH에는 force·velocity·floor collision이 없으므로 이를 물리적 접지의 완전한
증명이라고 부르지 않는다. source toe 방향과 target 발바닥 방향의 일치 문제만 해결한다.

파일명, BVH family, 성별, source/target 다리 길이는 회전 공식이나 활성 조건에 사용하지
않는다. 측정 불가 또는 안전 후보 없음은 exact V3.2.1을 내보낸다.

실제 검증값은 [QA_REPORT.md](QA_REPORT.md)에 기록한다.
