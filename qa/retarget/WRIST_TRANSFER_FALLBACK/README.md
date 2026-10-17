# palm-roll fallback 손목 전달 (2026-10-16)

## 문제
CMU(353)·MH(22) 리그는 손바닥 기준점(index+pinky / index+thumb 쌍)이 없어 arm chain의 palm-roll이
fallback으로 빠진다. 이 경로는 `frozen_hand = maps[1] @ rest_dst_hand`를 그대로 써서 원본 BVH 손목 채널
회전(예: cmu_135_01_04368 L 67.7° / R 12.0°)을 버리고 손을 target rest 관계(0°)에 고정했다.
사용자가 지적한 "머리 위 손목이 꺾여 보인다"(cmu_135_01_04368_mirror)의 원인이다.

## 수정
- fallback일 때 원본의 **전완 대비 손 상대회전**(굽힘+비틀림)을 source 전완 posed 프레임에서 구해 frozen 손에 곱한다.
- 전완축 비틀림의 50%는 전완 본에 나눠 싣는다(`STANDIN_WRIST_TWIST_FOREARM_SHARE`, 기본 0.5).
  CMU 손 채널에는 회내·회외가 함께 실려 있어, 비틀림 본이 없는 캐릭터 손목에 다 몰면 스킨이 꼬인다.
- 손가락 위치로 방향만 맞추는 swing 방식은 비틀림을 가짜 굽힘으로 바꿔 폐기했다.
- 끄기: `STANDIN_WRIST_TRANSFER=0` → 수정 전과 같은 결과.
- 동결 계보 해시(protocol.py, ankle_swing_policy.json, SHA256SUMS.v325, 테스트 고정값)만 새 retarget 해시로 교체.

## 검증 (Blender 5.2.0, standin-master-v2 / standin-female-v2-lbs)
- CMU 353개 전체: 손 전달량 = 원본 채널 (오차 ≤ 0.01°)
- MH 22개: 손 채널 회전이 없어 전달량 0° (변화 없음)
- fallback 비대상 무작위 200개: 수정 전후 손 회전·관통 값 완전히 동일
- `test_wrist_transfer.py` 5 passed (재현, 전달, 미러 대칭, 끄기 스위치, 대조군) — `pytest_result.log`
- 실행: `PY=<bpy 5.2 python> MESH_ROOT=... W_ROOT=... OLD_CONV=<수정 전 server> NEW_CONV=<수정 후 server> python -m pytest -q test_wrist_transfer.py`
  (`jtool.py`가 변환·측정을 담당. CI에는 Blender가 없어 이 폴더는 수동 QA용이다.)
