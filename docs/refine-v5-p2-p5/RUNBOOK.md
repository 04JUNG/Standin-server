# Refine v5 P2–P5 재현 절차

오프라인 실험 전용이다. P0 데이터를 입력으로 사용하며 GT는 학습과 별도 평가 과정에서만 읽는다. 실서비스 API 연결은 포함하지 않는다.

한국어 결과와 제한 사항은 [REPORT_2026-11-07.md](REPORT_2026-11-07.md)를 먼저 확인한다.

## 현재 로컬 결과

- 학습 worktree: `/private/tmp/standin-refine-v5-p2-p5-20261107/learning`, branch `codex/refine-v5-learning-p2-p3`.
- 통합 worktree: `/private/tmp/standin-refine-v5-p2-p5-20261107/native`, branch `codex/refine-v5-native-p4-p5`.
- 최종 run: `/private/tmp/standin-refine-v5-p2-p5-supported-20261107`.
- P0 원본: `/private/tmp/standin-refine-v5-20261106`.
- 실험 Python: `/private/tmp/standin-refine-v5-p2-p5-20261107/venv/bin/python`.

최종 run의 `native-data`, `dataset`, `models`는 이전 단계의 불변 산출물을 참조하는 로컬 symlink다. `summary.json`의 SHA와 `evaluation/refine_v5/p2-p5-summary-v1.json`으로 checkpoint/에셋을 확인한다. 단일 실행 스크립트는 새로운 out 디렉터리를 요구한다.

## 전체 재현

통합 worktree에서 별도 Python 환경에 저장소 실행 의존성과 `experiments/refine_v5/requirements-learning.txt`를 설치한다. RTMPose 재검색은 P0에서 끝났으므로 이 단계의 새 모델 다운로드·provider 호출은 없다. Blender와 P0의 `all-seeds/seeds.json`, `data/native_audit.json`, 러프/GT 파일이 필요하다.

```bash
python -m experiments.refine_v5.run_p2_p5 \
  --previous /private/tmp/standin-refine-v5-20261106 \
  --out /private/tmp/standin-refine-v5-p2-p5-reproduction-20261107 \
  --body '/Users/dowon/dev/Standin-server/data/3D model/output/standin-master-v2.fbx' \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

순서는 native source/GT mesh 추출 → 분할 검사·합성 → 5개 학습 ablation → 기하/관측/균형 1·3회 비교 → 독립 GT 평가 → 실제 target mesh backtracking·FBX export/readback → 별도 프로세스 재읽기 → 검수 그림이다. `--epochs 140`, `--variants 32`가 이번 값이다. 조건을 바꾸면 별도 run으로 비교한다.

이번 실험은 각 단계를 실제로 실행했다. 새 orchestration wrapper 자체를 두 번째 전체 학습으로 다시 검증하지는 않았고, 각 단계의 실행과 wrapper의 import/help/compile을 확인했다. 모든 Blender 호출에 `--python-exit-code 1`을 사용해 Python 예외를 실행 실패로 처리한다. 본문에 없는 자동 성공/skip fallback은 없다.

## GT 없는 단일 추론

```bash
python -m experiments.refine_v5.predict_v3 \
  --checkpoint /Users/dowon/dev/Standin-server/out/eval/refine_v5/p2-p5-20261107/mixed_randomized.pt \
  --input /Users/dowon/dev/Standin-server/out/eval/refine_v5/p2-p5-20261107/strict-input-001.json \
  --out /private/tmp/standin-v5-predict-001.json
```

입력은 이미지·source FBX SHA·source bone state·기존 2D 관절/score·검수 유형/가시성·고정 카메라·소유권 상태만 받는다. GT 필드나 임의 추가 필드는 거부한다. 제공된 fixture의 `ownership_verified=False`를 유지했으므로 제안은 계산해도 적용 권한 마스크는 0이다. 실험용 benchmark의 `offline_ownership_assumption`은 서비스 승인으로 쓰지 않는다.

`proposals`는 뼈 프레임에서 도출한 길이 보존 관절 목표와 독립 local rotation 제안이다. 서로 영향을 주는 관절은 이후 FK solve에서 함께 푼다. 좌표는 몸통 길이, 회전은 radian이다. 예측된 깊이·표면 방향을 관측 사실로 바꾸지 않는다.

## 검증과 산출물 구분

```bash
python -m experiments.refine_v5.check
python tests/test_smoke.py
```

`check`는 unittest 33개와 `test_learning.py`의 독립 함수 6개를 함께 실행한다. 총 39개다. 기존 smoke는 54개다.

- `models/metrics.json`: 학습 조건별 검증과 사례별 오차.
- `frame-evaluation.json`: 학습 재현/검증, 합성/실제 분리. 완전 관측 8뼈 분모 포함.
- `iterative*/manifest.json`: GT 없는 추론 입력·체크포인트 출처.
- `evaluation*.json`: 추론이 끝난 다음의 독립 정답 비교. FK 후보 단계.
- `native-audit/case-ledger.json`: 12슬롯 전부와 후보 미생성 사유.
- `native-audit/records.json`: 실제 피부 triage, backtracking, export/readback, final-byte fallback.
- `native-audit/independent-readback.json`: 별도 Blender 프로세스 결과.
- `native-review.png`: 저장된 **diagnostic** 메시와 GT 비교. body-frame 표시이며 실제 러프 카메라는 아님.
- `*.diagnostic.fbx`: 검수용 후보, 서비스 전달 대상 아님.
- `*.safe.fbx`: 현재는 전부 B0와 바이트까지 동일. 후보 개선치를 안전 출력 성능으로 인용하지 않는다.

보류된 finger/contact/support/surface-normal/profile 인증과 P6는 보고서의 다음 개선 항목으로 남아 있다.

### VLM 팔다리 목표 10컷

`part_vlm_observe`는 이전 20컷 재생에서 다리가 관측된 9컷과 다리가 잘린 대조 1컷을 고정한다. OpenAI에는 러프만 보낸다. VLM은 해부학적 좌우, 굽힘 범위, 앞뒤, 허벅지 높이, 지지와 손 접촉 단서를 제공한다. 솔버는 기존 RTMPose와 독립 가시성 검수에서 확인한 화면 위치를 결합하고, 같은 B0에서 네 팔다리를 한 번에 조정한다.

아래에서 `REPLAY20`은 기존 20컷의 원본 실행 폴더, `RUN`과 `OUT`은 새 폴더, `P0`는 정답 FBX 및 동결된 검출 결과가 있는 P0 폴더다. `VLM_ENV`는 OpenAI 설정을 가진 로컬 env 파일이다.

```sh
python -m experiments.refine_v5.part_vlm_observe \
  --root "$RUN" --prepare-control "$REPLAY20" --env-file "$VLM_ENV"
python -m experiments.refine_v5.part_vlm_fit \
  --root "$RUN" --seeds "$P0/all-seeds/seeds.json"
blender -b --python-exit-code 1 --python experiments/refine_v5/part_vlm_native.py -- \
  "$RUN" "$REPLAY20" "$P0/data/정답_3d인형"
python -m experiments.refine_v5.part_vlm_report \
  --root "$RUN" --control "$REPLAY20" --out "$OUT"
```

같은 러프와 스키마에 이미 받은 응답을 재사용할 때는 `--prepare-control`을 빼고 같은 `RUN`을 지정한다. 비교 시트는 기존 검출 방식, 피부 검사 전 VLM 보정안, 피부 검사 후 저장 결과를 따로 표시한다. 지역 피부 검사에서 회귀하면 원본 B0를 그대로 저장한다. 전체 피부 인증 상태와 컷별 손 접촉은 비교 JSON에 기록한다.

### 029 기준 후보와 자동 재생

`evaluation/refine_v5/leading_reference_029.json`이 029의 수동 결과를 오프라인 기준으로 지정한다. 백업 ZIP에는 시작 FBX, 수동 결과 FBX, 계획, 검증 기록, 비교 이미지와 해시를 보관했다.

자동 경로는 동일한 `Part goals + head` FBX에서 시작한다. 새 VLM 응답은 러프의 격자 영역과 다리의 앞뒤를 관측한다. 별도 팔다리 VLM 응답의 손 분리 단서는 RTMPose와 대조한다. 각도 범위는 격자와 리그 길이로 추정한 3D 방향에서 보완한다. 이 실행은 정면에서 앞으로 들린 오른 무릎과 골반에서 분리된 왼팔을 다룬다. 다른 포즈 유형의 평가와 자동 머리 목표 생성은 별도로 검증해야 한다.

```sh
python -m experiments.refine_v5.part_grounded_observe \
  --input "$RUN/input.json" --out "$RUN/grounded-response.json" --env-file "$VLM_ENV"
python -m experiments.refine_v5.part_grounded_fit --root "$RUN" \
  --seeds "$P0/all-seeds/seeds.json" --limb-response "$LIMB_VLM_RESPONSE"
blender -b --python-exit-code 1 --python experiments/refine_v5/part_single_native.py -- \
  export "$RUN" grounded
python -m experiments.refine_v5.part_single_report \
  --root "$RUN" --prior "$PART_GOALS_RUN" --out "$OUT" --comparison-mode grounded
```

`RUN`에는 같은 시작 포즈의 `input.json`, `base.npz`, 비교용 `reviewed/`가 필요하다. `LIMB_VLM_RESPONSE`는 같은 러프에서 `part_vlm_observe`가 받은 JSON이다. 정답과 수동 결과는 자동 목표 생성에 사용하지 않고 보고서에서만 읽는다.
