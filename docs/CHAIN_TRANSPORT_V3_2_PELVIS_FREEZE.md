# Chain Transport V3.2 골반 경계 동결

## 결정

`CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY`를 승인된 solver 동결본으로 기록한다.

- 승인일: 2026-08-27
- 사용자 육안 판정: 골반이 매우 자연스러워졌으며 합격
- 제품 상태: 알고리즘 동결 완료, converter HTTP 서비스 통합·배포는 아직 하지 않음
- V3.1 부모는 변경하지 않음

## 계보와 해시

```text
V3.1 parent retarget.py
f6d9a35268ff18173d9280baf8e502f5e258dbaeb8de4ffb4dd83637c19e9c6b

V3.1 ankle_policy.json
79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f

V3.2 retarget.py
692e975d32f41e3406144763c7c0b7dbf0a586ff07732f09c4365e3233b13693
```

동결 파일은 `qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY/`에 있으며
`SHA256SUMS`로 복원 동치성을 확인한다.

## 동결 범위

- Hips 회전과 root translation은 V3.1과 동일하다.
- spine, neck, head, shoulder, arm, hand는 V3.1과 동일하다.
- 활성 다리의 `upleg/leg/foot/toe`만 parent-coherent pelvis boundary를 적용할 수 있다.
- 발목 적용량은 동결된 V3.1 soft-cap/hard-guard 정책을 그대로 사용한다.
- 한쪽 parent seed라도 퇴화하면 양쪽 활성 다리를 exact V3.1로 복구한다.
- 클립명·파일명·좌우 본 이름에 따른 예외는 없다.

## 완료된 검증

- deterministic math controls: 8/8
- Blender converter regression: 28/28
- 실물 paired 변환과 surface 측정: 20/20
- 독립 export/reimport verifier: 10/10
- 변경 범위, Hips exact, ankle policy SHA: 10/10
- 강제 한쪽 퇴화의 bilateral exact V3.1 fallback: PASS
- actual mirror 경로: PASS
- 사용자 원본 FBX 골반 육안 gate: PASS

수치·진단의 세부 내용은 동결 디렉터리의 `DISCOVERY_01_REPORT.md`를 따른다.

## 이후 통합 규칙

운영 `converter/`로 옮길 때는 이 스냅샷을 직접 편집하지 않는다. 별도 구현 디렉터리로
복사한 뒤 먼저 `SHA256SUMS` 동치를 확인하고, 28건 회귀와 대표 실물·mirror 검증을 다시
통과시킨다. 추론 API와 Blender converter는 같은 저장소를 사용하더라도 별도 컨테이너와
별도 서비스로 배포한다.
