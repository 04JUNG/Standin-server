# Refine v5의 앞뒤 관계 보정

검수 그림의 첫 네 컷을 001, 006, 029, 039로 묶었다. 사용자 검수에서 확인한 앞뒤 관계를 별도 입력으로 받고, 같은 B0와 에셋에서 회전만 계산한다. 관측 워크트리는 입력의 출처와 축 부호를 고정한다. 보정 워크트리는 회전 계산, Blender 메시 검사, 저장 후 재검증을 맡는다.

깊이는 축을 지정해 기록한다. 몸통 기준 뒤는 몸통의 전방축 반대 방향이고, 카메라 기준 멀어짐은 고정 투영의 깊이 방향이다. Blender 월드 Z는 발 높이에 쓴다. 각 간격의 하한은 몸통 길이로 정규화한 보정 기준이다. 러프에서 측정된 절대 3D 깊이는 입력에 없다.

| 컷 | 깊이 목표 | 검사 |
|---|---|---|
| 001 | 오른 손바닥을 허리 옆면으로 이동 | 옆면 앞쪽 허용량, 손바닥과 표면 거리, 허벅지·몸통 관통 |
| 006 | 왼발을 오른발보다 뒤로 보내고 들어 올림 | 몸통·카메라 깊이, 오른발 기준 발바닥 높이, 실제 오른발 메시 이동 |
| 029 | 오른 팔꿈치와 손목을 오른 어깨보다 뒤로 보냄 | 몸통 깊이 차, 관절 화면 위치, 몸통과의 교차 |
| 039 | 오른 손바닥을 머리 뒤쪽으로 보냄 | 머리 뒤쪽 깊이 기준, 손·전완과 머리 삼각형 교차 |

같은 카메라 아래 보이는 관절 위치와 파츠 방향을 맞추면서 깊이 하한을 푼다. 손과 전완의 대표 피부점에 표면 간격을 주고, B0 기준 회전 크기를 제한한다. 작은 범위와 구조 수정 범위를 분리했다. 구조 수정 범위는 이 오프라인 실험에 한정하며 기존 보정 프로필의 인증 상태와 별개다.

Blender에서 정확한 변형 메시를 평가해 새 관통, 지지발 이동, 머리 교차를 제거한다. 2D 위치와 실제 깊이를 함께 보고 후보를 고정한 뒤 정답 FBX를 읽는다. 저장 후에도 깊이와 메시를 독립적으로 재검사한다. 전신 피부 인증과 표면 접촉의 품질은 별도 판정으로 기록한다.

재현에는 기존 원근 A/B run과 P0 원본 폴더가 필요하다. 보정 워크트리에서 아래 명령을 순서대로 실행한다. 실행 디렉터리는 비어 있는 새 경로를 사용한다.

```bash
python -m experiments.refine_v5.depth_evidence --run "$depth_parent" --previous "$depth_previous" --out "$depth_run"
"$depth_blender" -b --python-exit-code 1 --python experiments/refine_v5/ab_blender.py -- prepare "$depth_run"
python -m experiments.refine_v5.depth_fit --root "$depth_run" --parent "$depth_parent"
"$depth_blender" -b --python-exit-code 1 --python experiments/refine_v5/depth_native.py -- select "$depth_run" "$depth_parent"
"$depth_blender" -b --python-exit-code 1 --python experiments/refine_v5/depth_native.py -- audit "$depth_run" "$depth_previous/data/정답_3d인형"
"$depth_blender" -b --python-exit-code 1 --python experiments/refine_v5/depth_native.py -- readback "$depth_run"
python -m experiments.refine_v5.depth_report --root "$depth_run"
python -m experiments.refine_v5.check
```

이번 실행은 네 컷 모두에서 깊이 조건과 B0 대비 지역 피부 비악화를 통과했다. 006은 오른발을 고정하고 카메라 깊이 순서를 바꿨지만, 정답 대비 왼쪽 대퇴·정강이 평균 방향 오차는 20.1°에서 25.2°로 커졌다. 네 컷의 전신 피부 인증 결과는 모두 미통과다. 결과의 적용 범위는 검수자가 앞뒤 관계를 제공한 개발용 후보까지다.
