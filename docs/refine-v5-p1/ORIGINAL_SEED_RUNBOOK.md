# 원본 BVH 인계본 P0/P1 재현

P0의 조인 키는 검색 잡의 `rough_id + person_index`와 `final_files[].sample_id`다. 컷 접미사로 인물을 추정하지 않는다. 원본은 제작 시작 기록과 당시 선택이 일치하고 BVH SHA-256도 일치하는 항목이다. 나중에 저장된 선택은 별도 상태로 남긴다. 확정된 92쌍 중 봉인 평가 17쌍 및 개발·검증의 귀속 모호 2쌍을 빼고 73쌍에서 솔버를 진단한다. P0의 해시 대조와 P1의 계산 여부를 별도로 기록한다.

```sh
python -m experiments.refine_v5.original_seed_intake \
  --archive "$HANDOFF_ZIP" --old-data "$P0_DATA" --out "$RUN"
blender -b -t 4 --python-exit-code 1 \
  --python experiments/refine_v5/original_seed_native.py -- prepare "$RUN" "$BODY_FBX"
python -m experiments.refine_v5.original_seed_oracle \
  --root "$RUN" --gt-audit "$P0_DATA/native_audit.json"
blender -b -t 4 --python-exit-code 1 \
  --python experiments/refine_v5/original_seed_native.py -- audit "$RUN"
python -m experiments.refine_v5.original_seed_report \
  --root "$RUN" --gt-audit "$P0_DATA/native_audit.json"
```

`P0_DATA`는 기존 핀된 정답 커밋의 파일과 curated_inventory_v2.json이 있는 폴더다. 검색 원본과 검수용 러프를 별도 보존한다. `BODY_FBX`는 기존 실험의 Standin 남자 기본형이다. 원본 BVH의 0번 프레임을 현재 컨버터로 변환하므로, 당시 편집 앱이 생성한 시작 FBX와 바이트 일치한다고 간주하지 않는다.

정답 몸통 프레임을 원본에 맞춘 후 관절 위치와 방향, 머리의 위·앞 축을 목표로 준다. 루트·골반 이동과 뼈 길이는 고정한다. 세 회전 범위의 수치 후보를 비교하고 `limbs_head`를 전량 FBX로 저장해 지역 피부 비악화 여부를 확인한다. `*-regional.fbx`는 지역 검사 결과이며, 거절되면 정확한 B0로 돌아간다. 전신 피부 인증은 별도 상태다. GT를 입력한 진단이라 러프 자동 보정 성능이나 학습 성능으로 집계할 수 없다.

029에는 별도 전체 파츠 진단을 만들었다. 새 폴더에 029만 담은 manifest와 동일한 시작 FBX를 두고, 가려진 부위까지 GT 목표로 사용한다고 표시한 뒤 별도로 계산했다. `audit-review`는 실제 GT 메시의 파츠 가중치를 확인하고 정면·측면 렌더에 필요한 메시를 저장한다. 이 결과는 73쌍의 러프 가시성 기준 집계에 합산하지 않는다.

## P2–P4: 같은 원본에서 자동 관측과 학습 비교

P1의 `RUN`을 `P1_RUN`으로 넘기고 별도 `P2_RUN`을 쓴다. 개발 쌍에서만 파츠별 3D 방향 잔차를 지도 학습한다. 러프의 SHA와 일치하는 기존 관절 검출만 입력으로 쓴다. 검증 자동 추론에서 정답 FBX를 읽지 않는다. VLM에는 검증 러프만 보내고 3D 파일은 로컬에 둔다.

```sh
python -m experiments.refine_v5.original_seed_observe \
  --p1 "$P1_RUN" --out "$P2_RUN" --env-file "$VLM_ENV"
python -m experiments.refine_v5.original_seed_prior \
  --p1 "$P1_RUN" --out "$P2_RUN" --seeds "$SEEDS" --gt-audit "$P0_DATA/native_audit.json"
blender -b -t 4 --python-exit-code 1 \
  --python experiments/refine_v5/original_seed_p4.py -- prepare "$P1_RUN" "$P2_RUN"
python -m experiments.refine_v5.original_seed_p3 \
  --p1 "$P1_RUN" --out "$P2_RUN" --seeds "$SEEDS"
for method in oracle automatic learned; do
  blender -b -t 4 --python-exit-code 1 \
    --python experiments/refine_v5/original_seed_p4.py -- audit "$P1_RUN" "$P2_RUN" "$method"
done
blender -b -t 4 --python-exit-code 1 \
  --python experiments/refine_v5/original_seed_p4.py -- capture "$P1_RUN" "$P2_RUN"
python -m experiments.refine_v5.original_seed_p2_p4_report \
  --p1 "$P1_RUN" --run "$P2_RUN" --gt-audit "$P0_DATA/native_audit.json"
```

`SEEDS`는 P0 러프의 고정 관절 검출 기록이다. `oracle`은 P1의 3D 정답 목표 계획을 사용한다. 두 자동 조건은 같은 관측·초기 포즈·관절 한도로 풀며, `learned`만 관측과 충돌하지 않는 학습 방향을 약한 제약으로 더한다. 머리 목표는 두 자동 조건에서 같다. 세 조건 모두 원본 B0에서 100%, 50%, 25% 순으로 보정량을 시험한다. 최초 국소 비악화 및 재로드 통과를 저장하고, 실패하면 바이트가 같은 원본을 보관한다. 전신 피부 인증 결과는 별도이며 이미지와 최종 점수에는 원본 복귀가 포함된다.

### 부위별 복귀 비교

기존 P4 실행은 전체 보정량 100%, 50%, 25%가 모두 탈락하면 시작 포즈로 복귀했다. 동결된 P3 계획을 별도 실행 폴더에 복사하고 `audit`를 `audit-partial`로 바꾸면 부위별 복귀를 비교한다. 이전에 전체 보정을 채택한 경우 그 결과를 유지한다.

부분 채택은 머리·목과 각 팔다리를 연결된 체인 단위로 다룬다. 실패한 시도의 관통에 연루된 편집 체인은 원본에 둔다. 나머지 체인을 기존 채택 체인과 합쳐 100%, 50%, 25%로 시험한다. 합친 전체 포즈가 원본 대비 국소 피부 비악화와 FBX 재로드를 통과해야 그 결과를 남긴다. 정답은 부분 채택에 쓰지 않는다. `group_fractions`는 남긴 체인과 비율, `blocked_groups`는 관통 때문에 제외된 체인, `partial_attempts`는 실제 합친 검사의 순서와 판정이다.

### 중립 관절 표식으로 좌우 검증

기존 검증과 같은 별도 `P2_RUN`에 `model.json`과 `base-mesh/`를 가져온다. `original_seed_observe` 대신 아래 모듈을 실행하고, 기존 P3·P4 명령을 이어서 실행한다. 두 자동 조건에 `audit-partial`을 쓰면 이전 부분 보정 결과와 비교할 수 있다.

```sh
python -m experiments.refine_v5.original_seed_laterality \
  --p1 "$P1_RUN" --out "$P2_RUN" --seeds "$SEEDS" --env-file "$VLM_ENV"
```

러프 원본과 A/B 팔, C/D 다리 표식을 함께 보낸다. 기본값인 2단계는 1단계의 깊이·굽힘·머리 판정 기준을 포함한다. VLM은 표식으로 실제 사지를 추적하고, 앱은 몸통 방향과 어깨·골반 위치를 확인해 표식을 리그 좌우에 연결한다. VLM은 각 발의 지면 접촉과 체중 지지를 별도로 읽는다. 양쪽 다리가 확인되고 한쪽만 체중을 받으면 앱이 그 발을 고정한다. P3는 검출점과 점수의 해시를 다시 확인하고, 머리 각도가 미확정인 경우 검출점과 몸통 방향을 근거로 기본 목표를 보완한다. 3단계는 `--stage 3`으로 실행한다. 손의 앞·옆·뒤와 별개로 간격을 `palm_touching`, `near`, `separated`, `unknown`으로 관측한다. 손바닥 접촉과 대상 면이 확인될 때만 접촉용 손목 회전을 열며, 근접과 분리는 손 전체 표면의 표본 거리로 제한한다. 실험 임계값은 몸통 길이의 9%다. 근접은 상한만, 분리는 하한만 적용하고 미확정이면 거리 목표를 두지 않는다.
