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
