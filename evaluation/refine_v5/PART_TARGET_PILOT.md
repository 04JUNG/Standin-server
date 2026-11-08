# Refine v5 파츠 목표 파일럿

10컷에 같은 원본 B0와 시점 보정 정사영을 사용한다. 러프 검수 기록에서 대상의 해부학적 좌우를 읽는다. 들린 다리를 움직이려면 반대쪽 지지발 확인이 필요하다. 043의 다리 대응은 재검수 대상으로 남겼으며 어느 실험군도 그 다리를 움직이지 않는다. 깊이 순서는 이전에 봉인한 축·조건을 그대로 쓴다. 새 관측은 화면 굽힘과 단축을 사용하고 006에는 B0 굽힘의 약한 선호와 최대 25° 변화 한도를 더한다. 이 한도는 오프라인 가설 검증용이다.

세 실험군은 순서와 기존 피팅, 굽힘·단축·표면 접촉 추가, 그리고 동일 목표에 기존 지도학습 모델의 시작값을 넣은 방식이다. 각 실험군에서 회전 범위 두 개와 범위당 시작값 두 개를 사용한다. 학습 방식은 두 번째 시작값을 기존 체크포인트의 제안으로 바꾼다. 이전 학습에서 노출된 사례가 포함되므로 일반화 실험으로 집계하지 않는다. 접촉은 001에만 명시했고, 나머지 손–머리 표면은 접촉이 확정되지 않은 검사용이다.

001 고개 보정은 화면에서 읽은 기울기의 독립 비교다. 다른 축은 합성 관측으로 검증한다. 척추·쇄골은 별도의 독립 기준 관측이 필요하며 이 10컷에서는 후보를 만들지 않는다. 생성 가능한 두 파츠는 각각 및 함께 여는 합성 테스트로 확인했다. 실제 적용 전 native 검증이 필요하다.

아래 명령은 보정 워크트리에서 실행한다. `previous_depth`는 봉인된 10컷 실험 폴더, `part_run`과 `head_out`은 아직 없는 경로다. `seed_json`은 P0의 all-seeds/seeds.json이며 `ground_truth`는 P0에 고정한 정답 FBX 폴더다. 체크포인트를 생략하면 학습 시작값 실험을 건너뛴다.

```bash
python -m experiments.refine_v5.part_observations --prior "$previous_depth" --out "$part_run"
python -m experiments.refine_v5.part_pilot --root "$part_run" --checkpoint "$checkpoint" --seeds "$seed_json"
"$blender" -b --python-exit-code 1 --python experiments/refine_v5/depth_ten_native.py -- select "$part_run"
"$blender" -b --python-exit-code 1 --python experiments/refine_v5/depth_ten_native.py -- audit "$part_run" "$ground_truth"
"$blender" -b --python-exit-code 1 --python experiments/refine_v5/depth_ten_native.py -- readback "$part_run"
python -m experiments.refine_v5.part_pilot --root "$part_run" --head-only
"$blender" -b --python-exit-code 1 --python experiments/refine_v5/part_head_native.py -- "$part_run" "$head_out"
python -m experiments.refine_v5.part_report --root "$part_run" --head "$head_out" --out "$review_out"
```
