# Set 01 자동 관절 인식 평가

2026-10-28. 사용자가 `vlm-pose-30x5-v1`을 평가셋으로 사용하고 **set-01부터 시작**하도록 지정했다. 자료 사용 허용도 확인했으므로 323장 접수·권한 작업표 작성은 이번 자동 평가의 선행 조건이 아니다. 실제 외부 provider 호출·운영 배포는 수행하지 않았다.

## 고정 입력

- 저장소: [Standin-master-docs의 고정 소스](https://github.com/04JUNG/Standin-master-docs/tree/5680035be8354916d1b11a980e7817a6cece2015/%EB%9F%AC%ED%94%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0/vlm-pose-30x5-v1).
- dataset: `vlm-pose-set01-r3-exclude119-230-252`, **27컷·48명**. 폴더 전체가 아니라 `set-01/dataset.json`의 `cuts`가 분모다.
- 기존 제외 `p_rough_119`, `p_rough_230`, `p_rough_252`는 재포함하지 않는다. 다른 set은 자동 실행하지 않는다.
- 보이는 GT 관절(visibility=2) **162개**, 가림 위치 추정(visibility=1) **210개**, 미라벨(visibility=0) **444개**. 보이는 신체 관절(얼굴 5개 제외)은 61개다.
- 48명 중 5명은 보이는 유효 관절이 없다. 이들은 primary 인물 대응에서 unknown을 유지한다. 가림 추정 좌표만으로 직접 관측된 인물 대응을 인증하지 않는다.
- GT의 `human_verified=false`, `evaluation_ready=false`, `draft_unverified`는 그대로 보존한다. 자동 점수는 계산할 수 있지만 `PROVISIONAL_DRAFT_GT`로 표시한다. 이 상태를 이유로 실행 자체를 막지는 않는다.

로컬 고정 입력은 `out/eval/vlm-pose-suite-source-5680035/`, 정규화·봉인된 인식 dataset은 `out/eval/datasets/vlm-pose-set01-r3-recognition-v1/`에 준비했다. 입력 PNG·GT 원문과 모델 출력은 Git에 추가하지 않는다. [소스 lock](../../evaluation/pose_set01_source.lock.json)은 커밋·분모·제외·매니페스트 해시만 기록한다.

## 실행

처음부터 재현하려면 고정 소스 저장소를 로컬에 checkout하고 다음 명령을 사용한다. 상대 경로는 source suite 폴더와 각 set manifest 기준으로 해석하며, `metadata_path`, `source_paths`, `prediction_source` 같은 역사적 경로는 열지 않는다.

```bash
python -m standin_eval dataset import-pose-suite \
  --source /path/to/Standin-master-docs/러프\ 데이터/vlm-pose-30x5-v1 \
  --set 1 --source-commit 5680035be8354916d1b11a980e7817a6cece2015 \
  --usage-authorized --output out/eval/datasets/set01-new-version

# 현재 준비된 dataset을 이용해 미실행 상태/분모부터 확인 (네트워크 호출 없음)
python -m standin_eval run pose-benchmark \
  --dataset out/eval/datasets/vlm-pose-set01-r3-recognition-v1 \
  --policy evaluation/pose_set01_policy.example.json \
  --output out/eval/set01-new-unmeasured-run

# 실제 수집한 예측을 오프라인으로 자동 채점
python -m standin_eval run pose-benchmark \
  --dataset out/eval/datasets/vlm-pose-set01-r3-recognition-v1 \
  --predictions /path/to/recorded-predictions.jsonl \
  --policy evaluation/pose_set01_policy.example.json \
  --output out/eval/set01-new-recorded-run
```

예측 JSONL은 컷별 한 행이다. `cut_id`, 입력의 `image_sha256`, `status`, `execution_kind`, `requested_backend`, `actual_backend`, `people`를 사용한다. 각 person의 `keypoints`는 COCO17 순서의 픽셀 좌표 17쌍이며 미출력은 `[null,null]`이다. 기존 API의 `people[].skeleton.keypoints` 형식도 읽는다. `scores`는 선택 필드이고, 모델별 점수 스케일이 다르므로 기본값에서는 임의 신뢰도 임계치를 적용하지 않는다. `vlm_count`, `elapsed_ms`, `inference_metadata`를 함께 기록할 수 있다.

`execution_kind=recorded_model/http`는 실제 수집 결과에만 사용한다. mock/replay/GT 복사 fixture는 명시적인 fixture 종류·실제 backend로 기록한다. 예측에 GT를 복사한 계약 검사는 모델 성능 결과로 세지 않는다. 이미지 해시 누락은 unknown, 충돌은 FAIL이며, 제외/중복 컷 입력은 거부한다.

이미 운영 중인 호환 `/analyze` 서버를 **사용자가 명시적으로 실행 대상으로 정할 때만** 기존 multipart/HTTP transport로 요청할 수 있다. 이 작업에서는 해당 경로를 mock transport로만 검증했다.

```bash
python -m standin_eval run pose-benchmark \
  --dataset out/eval/datasets/vlm-pose-set01-r3-recognition-v1 \
  --target '<명시한 서버의 base URL>' --allow-http \
  --timeout '<사전에 정한 요청 timeout 초>' \
  --requested-vlm '<요청 provider>' --requested-pose '<요청 pose backend>' \
  --policy /path/to/preregistered-policy.json \
  --output out/eval/set01-new-http-run
```

정책과 전체 컷/인물 ID는 요청 전에 `plan.json`에 고정한다. requested/actual backend, 응답의 실제 모델·prompt·pose·DB·배포 버전을 기록한다. 정책의 `expected_inference_identity`에 기대하는 버전/해시를 고정하면 충돌은 FAIL이다. 미고정·누락 정체성은 unknown이다. mock fallback을 실제 모델 성공으로 바꾸지 않는다. timeout/error도 해당 컷·인물·관절 행을 유지한다. 코드 실행 중 원본/예측이 바뀌면 실패를 기록한다.

## 지표와 보고서

- 인물수: GT 인물수와 예측 인물 행수, VLM 보고 인물수를 별도로 비교한다.
- 대응: 기존 하네스의 Hungarian solver로 1:1 대응한다. GT bbox가 없으므로 보이는 관절의 이미지 대각선 정규화 거리로 대응하며, 전체 인물 bbox를 새 정답으로 만들지 않는다. 순서/`personIndex`가 곧 GT 인물 ID라고 가정하지 않는다.
- 관절: 보이는 관절과 가림 추정 좌표를 분리한 PCK, comparable 관절의 평균 pixel error, 얼굴 제외 신체 관절과 관절명별 결과를 출력한다.
- PCK는 **이미지 대각선의 0.05 이내**라는 명시적 engineering 진단이다. bbox/head 정규화 PCK와 다르며 제품 품질 합격선이 아니다. 매칭 거리 0.15도 engineering 규칙이다. 기준은 정책에 봉인한다.
- 미실행/불가·무효 GT는 unknown, 실제 실행한 인물 누락/관절 미출력/timeout/error는 incorrect로 분모에 남는다. unknown이 있으면 aggregate PCK는 null로 표시하고 고정 분모의 하한·correct/incorrect/unknown 수를 함께 제공한다. 미입력을 0 좌표나 모델 정확도 0%로 채우지 않는다.
- 실제 backend가 확인된 모델 관측만 지연 집계에 사용한다. fixture/GT 복사/identity의 짧은 실행 시간을 제품 지연으로 사용하지 않는다. 별도 제품 SLO는 설정하지 않았다.

각 run은 `report.json`, `report.html`, `cuts.jsonl`, `persons.jsonl`, `joints.jsonl`, `observations.jsonl`, 봉인 manifest를 만든다. 현재 [미측정 예시](../../evaluation/pose_set01_unmeasured.example.json)는 27컷 not_run, 48명 unknown, 실제 모델 관측 0건이며 INCONCLUSIVE다. 실제 결과를 넣으면 자동으로 수치가 계산된다. 품질·메시·FBX 활용성·출시 승격 판정은 이 관절 인식 보고서와 별개다.

## 검증

합성 회귀에는 고정 제외, 해시/좌표계/COCO 순서/인원 무결성, 순서가 바뀐 다인 대응, 중복 인물, null/0 구분, 가림 분리, 실패 분모, source mutation, 모델 버전 충돌, 네트워크 default-off와 HTTP mock 실행을 포함했다. 실제 set-01의 27컷·48명도 GT-derived 계약 fixture로 스키마·분모·좌표 계산을 확인했다. 이 검사는 **모델 평가가 아니며 실제 모델 관측 수는 0**으로 유지된다.

```bash
python -m pytest -q tests/test_pose_benchmark.py
```

운영 미배포. 실제 provider 호출·원본 외부 전송·Blender·설치 앱·CSP 실행 없음.
