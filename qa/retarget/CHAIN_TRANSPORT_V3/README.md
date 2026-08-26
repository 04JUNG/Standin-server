# CHAIN_TRANSPORT_V3 frozen QA snapshot

> 상태: **FROZEN-BASELINE**
> 동결일: 2026-08-27
> 제품 승격: 없음

사용자가 원본 FBX로 육안 확인해 현재 기준선으로 선택한 외부 FBX retarget 후보의 물리
snapshot이다. 런타임에서 이 디렉터리를 import하지 않는다. 목적은 실패한 후속 후보가 기준선을
덮어쓰지 못하게 하고, byte-identical 복원을 가능하게 하는 것이다.

## 동결 파일

~~~text
converter/__init__.py
converter/bone_map.py
converter/convert.py
converter/retarget.py
SHA256SUMS
~~~

`retarget.py` SHA-256:

~~~text
2701cc44a9ec6584722c194002d505a21bbf4fe199695325bb3b660c2818985e
~~~

## 상태

- 중심 수학: 순차 최소회전 체인 수송
- hips: legacy 경로 유지
- foot incremental rotation `>120°`: foot solve를 적용하지 않고 frozen terminal-follow 유지
- terminal hand roll: `UNRESOLVED`
- `CHAIN_TRANSPORT_V3_SAFETY_V1`: 원본 FBX 육안 gate 실패로 `REJECTED`

자세한 수학·동결 범위와 실패 분석:

- `docs/CHAIN_TRANSPORT_V3_SAFETY_POLICY.md`
- `docs/CHAIN_TRANSPORT_V3_SAFETY_V1_QA_REPORT.md`

## 복원 검증

snapshot 복원 시 `SHA256SUMS`를 검증하고 원본 snapshot과 전체 트리 `diff -qr` 차이가 없어야
한다. 재생성한 FBX의 binary hash를 코드 동치성 근거로 사용하지 않는다.
