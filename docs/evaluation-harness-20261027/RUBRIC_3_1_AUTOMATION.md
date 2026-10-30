# Rubric 3.1 자동 판정·집계 — 2026-10-28

## ZIP 전체 확인 결과

사용자가 제공한 `standin-harness-handoff-20261028.zip`의 7개 파일을 확인했다. package-manifest의 6개 SHA-256이 실제 파일과 모두 일치했다. HTML 내부 실제 예시 이미지 15개와 설명용 SVG 2개, 수정 방향 오버레이, 인쇄/확대 스크립트까지 확인했다. 외부 이미지 다운로드는 필요하지 않았다.

| 파일 | 확인한 정보 / 구현 반영 |
|---|---|
| `standin-evaluation-harness-handoff.html` | 10개 절: 배점, 관찰 가능성, 입력·실행·출력 고정, 모드, 상태, 지표 분모, 계약, 12개 수용 예시, 미확정 사항, 구자료 주의 |
| `standin-scoring-guide.html` | 6개 절, 실제 사례 8개와 설명용 전신 그림. 실제 사례도 AI 시각 1차 판정·팀 검토용이며 독립 검증 정답 아님 |
| `rubric.json` | 배점 v3.1 확정. handoff spec은 draft-1.0. 부분 평가 제외 및 미확정 실행 옵션 |
| `review-record.schema.json` | 인물별 7항목 상태+이유, 원래 후보 순위, 최종 이미지 해시, refine 상태, 평가자. 임의 점수 필드 금지 |
| `contract-examples.json` | 12개 가상 계산 사례. placeholder 이미지와 0 해시는 실제 출력 아님. 실제 결과와 섞지 않는 전용 검증 명령 |
| `README.txt` | 50장 전체 재채점 미실행, 예전 PDF/v2.3 점수 제외, 크롭·부분 점수·실행 제한 미정 |
| `package-manifest.json` | 패키지 무결성, `new_evaluation_run=false`, spec/rubric 버전 |

### 확정된 판단 규칙

100점에서 오른팔/왼팔/오른다리/왼다리 각 10, 몸통 20, 머리·목 5, 전체 동작 35를 전액 또는 0점 차감한다. 인물 자신의 좌우이며 다인 컷의 같은 항목은 컷당 한 번만 차감한다.

팔은 어깨·팔꿈치·손의 배치/펴짐/도달 위치, 다리는 골반·무릎·발/굽힘/벌어짐/지지 방향, 몸통은 실제 숙임·젖힘·옆 기울기, 머리는 몸통 대비 숙임·들기·돌림이다. 팔짱과 배 앞 느슨한 겹침을 같게 취급하지 않는다. 전체 동작은 큰 동작군 변화의 별도 근거가 필요하며 팔다리 오류에서 자동 전파하지 않는다. 서로 다른 항목의 오류는 함께 차감할 수 있다.

체형·복장·손가락·작은 접촉·작은 각도 차이를 제외한다. 시점 차이만으로 감점하지 않는다. 시점과 자세를 분리하기 어렵거나 바닥 근거 없이 접지를 판단해야 하면 보류한다. 눈동자 시선만으로 머리를 감점하지 않는다. 가림/크롭 부위를 추정해 match로 만들지 않는다.

### 사례 해석의 차이

| 가이드 사례 | 표시 점수 | 하네스에서의 취급 |
|---|---:|---|
| 42 / CB027 | 65 | 오른다리 지지발 보류: diagnostic 65, case_score null, 전신 집계 제외 |
| 44 / CB041 | 90 | 머리 보류: diagnostic 90, case_score null |
| 47 / CB064 | 85 | 왼다리+머리 감점 설명. 7항목 전체 판정 레코드 확보 전에는 확정 점수로 수입하지 않음 |
| 10 / CB024 | 60 | 양팔+몸통 감점, 깊은 쪼그림 유지, 머리 보류: case_score null |
| 11 / CB051 | 25 | 사지+동작 감점. refine timeout 후 기존 후보 평가 가능. 전체 판정·출력 증거 확보 필요 |
| 12 / CB053 | 5 | 머리 보류: diagnostic 5, case_score null |
| 5 / CB009 | 0 | 정상 실행 후 후보 전무라는 증거 필요. 부위 오답을 만들지 않음 |
| 7 / CB013 | 50 | 2명 중 1명 반환 + 4/7 부분 평가. 새 계약에서는 case_score null; 전부 판정됐을 때만 50 상한 적용 |
| 설명용 전신 그림 | 100 | 평가 표본 아님. 실제 100점 수에 절대 포함하지 않음 |

50장 전체의 최신 점수/평균/비율, 독립 정답 FBX, 확정 관절 임계값은 ZIP에 없다. v2.3의 동작 40점 결과와 혼합하면 안 된다. 기존 01번 PCK는 관절 인식 진단이며 이 점수로 환산하지 않는다. 01번 27컷을 50장 세트와 동일하다고 가정하지 않는다.

## 구현

- `standin_eval/rubric_score.py`: 엄격한 판정 스키마 및 인원/후보/파일 검증, 컷 점수, 고정 분모, 부위별 인물·컷 분모, HTML/JSON 보고서.
- `standin_eval/rubric_judge.py`: 원본/동결된 최종 결과의 Gemini 시각 판정 → 7항목 JSON → 동일 계산기. 판정은 항상 `ai_assisted_unreviewed`. 사람 확정 GT라고 표시하지 않는다.
- `evaluation/rubrics/standin-v3.1/`: 제공된 JSON 계약을 원문 그대로 보존.
- 제품 경로와 분리된 오프라인 CLI. 검색/리파인/수동 포즈 수정/후보 재선택을 실행하지 않는다. 선택은 이미 동결된 `selections`를 사용한다. 이 구현만으로 best-of-Top-5 선택의 최적성을 검증하거나 전체 live E2E를 실행한 것은 아니다.
- 실패한 AI 판정은 7항목 `not_run`. 제품 `execution_error`나 부위 `mismatch`로 바꾸지 않는다. 직렬 실행, 인물당 원본 해석+출력 비교 최대 2회, 명시적 모델/최대 호출/요청 timeout, SDK 재시도 1회 시도(재시도 0), 호출 전 전체 입력 검증.
- 각 호출 후 `.pending/`에 판정과 trace를 원자적으로 저장한다. 중단 후 같은 출력 경로로 재호출하지 않으므로 묵시적 중복 비용이 없다. 필요하면 저장된 판정으로 `rubric-score`를 새 경로에 실행한다.
- 이미지/manifest/설정/판정 해시를 보존하고 결과 디렉터리 재사용을 거부한다. 재검수는 새 결과 버전으로 저장한다. 사진 쌍은 보고서 안으로 복사한다. HTTP/절대경로/상위 경로/루트 밖 symlink는 아티팩트 입력으로 거부한다.

### 지표

N=고정 입력 수, E=기대 인물 전부 반환 및 전원 7항목 판정 완료 컷 수, H=전신100점, P=설정 기준 이상 전신컷.

- 전신 평균: 점수 합/E. E=0이면 null.
- 전신100점: H/N, 평가 가능 범위: E/N.
- 기준 도달률: P/E와 P/N을 별도 표시. 합격선은 config로 명시하며 공식 목표로 간주하지 않음.
- 하나 이상 결과 반환율, 모든 인물 반환율, 정상 무결과율, 실행 오류율.
- 실행 완료 수, 제외 분류/이유 수, 부위 오류율의 인물 및 컷 분모.
- 정상 무결과: 점수0, 전신 평균 제외. 실행 오류: 점수null. 보류/비가시/미실행: 점수null. 일부 인물 누락: 반환된 인물 전원 판정 완료 때만 min(진단점수,50), 전신 지표 제외.

## 실행

```bash
# 네트워크 없이 ZIP의 12개 가상 수용 사례를 계산한다.
python -m standin_eval run rubric-contract-check \
  --output out/eval/rubric31-contract-v1

# 동결된 사람이 검수한 판정 또는 기존 AI 초안을 재계산한다.
python -m standin_eval run rubric-score \
  --manifest /path/to/dataset-manifest.json \
  --reviews /path/to/reviews.jsonl --config /path/to/run-config.json \
  --artifacts-root /path/to/shared-artifacts \
  --output out/eval/rubric31-reviewed-v1

# 원본과 최종 이미지 쌍을 Gemini로 자동 판정한다. 모델/예산은 실행자가 명시한다.
# GEMINI_API_KEY는 환경변수로만 공급한다. 이 작업에서는 실제 호출하지 않았다.
python -m standin_eval run rubric-judge \
  --manifest /path/to/dataset-manifest.json --config /path/to/run-config.json \
  --artifacts-root /path/to/shared-artifacts --output out/eval/rubric31-ai-v1 \
  --model MODEL_ID --max-calls 96 --judge-timeout 60 --allow-provider
```

### manifest 확장 계약

기본 `review-record.schema.json`은 변경하지 않는다. 이미지·후보 실재 확인을 위한 sidecar manifest는 다음 형식이다. paths는 artifacts-root 기준 상대 경로이며 이미지 해시는 실제 바이트 해시다. rank는 배열 위치가 아니라 실제 반환 순위다. 인물당 후보는 최대 5개, 최종 선택은 하나다.

```json
{
  "dataset_version": "fixed-set-v1",
  "cases": [{
    "case_id": "cut-001",
    "input_image": "inputs/cut-001.png",
    "input_sha256": "<64 lowercase hex>",
    "reference_person_ids": ["P1"],
    "result_status": "returned_all",
    "response_completed": true,
    "execution_error": null,
    "candidates": [{
      "reference_person_id": "P1", "candidate_id": "actual-id", "rank": 3,
      "final_image": "outputs/cut-001-final.png",
      "final_image_sha256": "<64 lowercase hex>",
      "refine_status": "returned"
    }],
    "selections": [{
      "reference_person_id": "P1", "candidate_id": "actual-id", "rank": 3,
      "target_description": "원본 중앙의 인물; 최종 이미지의 유일한 인물"
    }]
  }]
}
```

AI 판정에는 `result_status`와 `selections`가 필요하다. 사람이 작성한 리뷰만 재집계할 때는 리뷰가 해당 정보를 제공한다. 선택 후보의 최종 이미지/해시/refine 상태를 candidate sidecar와 대조한다. `response_completed=true`와 빈 후보 목록이 있어야 `returned_none`을 인정한다. 시스템 오류를 정상 무결과로 기록하지 않는다.

```json
{
  "run_id": "frozen-output-review-v1",
  "mode": "replay_review",
  "review_origin": "ai_assisted_unreviewed",
  "threshold": 90,
  "official_comparison": false,
  "manual_pose_edit_allowed": false,
  "refine_attempt_limit": null,
  "refine_time_limit_seconds": null,
  "selection_before_or_after_refine": null,
  "retry_limit": null,
  "timeout_seconds": null
}
```

null은 미확정 상태 그대로다. 공식 비교는 미확정 옵션이 남으면 거부한다. 정식 비교 설정은 `human_reviewed`와 pipeline/model/library/prompt/refine 버전, camera_policy, fallback_policy, selection_method도 요구한다. selection timing 값은 `before_refine` 또는 `after_refine`. 판정 출처와 실행 설정은 사용자가 기록한 provenance이며 사람 신원이나 실제 실행을 인증하는 시스템은 아니다.

아직 정해야 하는 사항: 리파인 횟수/시간, 재시도/timeout, 선택 전후 시점, fallback, 공식 합격선과 분모, 시각 판정 검수 방식. 부분 점수 재분배·상반신 출력 크롭은 미구현 제안이다. 독립 정답 FBX·단위·좌표계·카메라 정렬이 없는 상태에서 3D 관절 정확도라고 부르지 않는다. 실패 원인 태그는 점수와 분리하며 이미지에서 단정하지 않는다.

## 이번 검증과 준비 결과

- 신규 계약/집계/해시/후보/AI 오류·보류/호출 예산 검증과 기존 pose benchmark, 평가 하네스, 관절 보존 회귀를 합쳐 **105 passed**.
- ZIP의 가상 예시 **12/12** 기대 결과 일치. 로컬 보고서: `out/eval/rubric31-contract-20261028-v1/report.html`. 실제 모델 성능으로 해석하지 않는다.
- 기존 01번 봉인을 검증해 `out/eval/set01-rubric31-preparation-20261028-v1/`에 **27컷·48명·336개 부위 판정 작업표**를 작성했다. `dataset prepare-rubric-review --dataset <sealed pose dataset> --output <new path>`로 재생성한다.
- 기존 01번 실측은 검색 라이브러리를 비운 인식 평가이므로 Top-5/최종 선택/리파인 출력에 대한 100점 채점 자료가 아니다. 이번에 새 시각 점수나 50장 평균을 만들어내지 않았다.
- 실제 Gemini 추가 호출 0회. 시각 판정 어댑터→계산→보고서 경로는 주입한 고정 응답으로 검증했다. 제공된 시각 예시를 정답으로 삼은 AI 성능 검증은 하지 않았다.
- 제품/API 경로, 모델 설정, 배포는 변경하지 않았다. 구현은 현재 평가 하네스 worktree의 로컬 변경이다.


## 결과 ZIP 수입과 사람의 마지막 검수 (추가 구현)

새 컷마다 사람이 먼저 목표 자세나 7항목 정답을 작성할 필요는 없다. 기존 제품 실행이 남긴 입력·선택 후보·최종 출력과 인물 대응 정보를 입력받아 아래 단계로 처리한다.

1. **입력만 해석**: AI가 원본만 보고 인물별 7항목의 목표 배치와 관찰 가능성을 추정한다. 이 호출에는 최종 결과 이미지를 전달하지 않는다. `targets.jsonl`을 먼저 저장하고 해시로 동결한다.
2. **출력 자동 평가**: 원본과 동결된 최종 결과를 비교한다. 원본 단계의 비가시/불확실 항목은 출력이 그럴듯해도 일치로 승격하지 않는다. AI 목표 추정은 `ai_estimated_target_not_gold`이다.
3. **자동 집계와 보고서**: 점수, 분모, 제외, 판정 근거를 계산한다. AI 실패 시 `not_run`으로 남긴다. 인물당 최대 2호출, 재시도 없음, 순차 실행이다.
4. **사람의 마지막 검수**: 모든 컷이 `final-review.jsonl`에 pending으로 생성된다. 사람은 보고서를 보고 approve 또는 replace를 기록한다. replace는 판정 상태·이유만 바꾸며 후보·이미지·제품 결과 상태는 바꿀 수 없다.
5. **확정과 피드백**: 별도 결과 버전에 사람 확정 점수와 변경 이력을 저장한다. `review-feedback.json`은 일치율, 비교 가능한 판정 비율, 과잉 감점, 오류 누락, 수정 항목을 기록한다. 기존 AI 결과를 덮어쓰지 않는다. 원본 목표 추정 자체를 인간 정답으로 자동 승격하거나 모델을 자동 재학습하지 않는다.

인물 대응에는 기존 기계 대응의 `reference_bbox_xyxy`(원본 픽셀 좌표)를 사용할 수 있다. `target_description`은 선택 사항이다. 단일 인물은 별도 설명 없이 진행하며, 다인 컷에서 대응 근거가 없으면 불확실로 처리한다. 새 컷의 인물 수/대응도 자동 파이프라인의 추정이지 인간 정답이 아니므로 마지막 검수 시 확인해야 한다. 인원·후보 대응 자체를 정정해야 하면 원래 실행을 보존한 새 manifest 버전을 만든다.

```bash
# 받은 ZIP: 최상위 폴더 유무 모두 허용, 기본 분모는 50컷.
python -m standin_eval run rubric-import-zip \
  --zip /path/to/standin-rescore-v3.1.zip \
  --output out/eval/received50-v1

# 자동 결과 또는 수입 결과에 최종 검수 양식을 생성한다.
# rubric-judge 실행 시에는 이 파일이 자동 생성된다.
python -m standin_eval run rubric-review-queue \
  --scored out/eval/received50-v1/scored \
  --output out/eval/received50-v1/final-review.jsonl

# 모든 행의 decision을 approve/replace로, reviewer/reviewed_at을 실제 검수 정보로 작성한 후:
python -m standin_eval run rubric-finalize \
  --scored out/eval/received50-v1/scored \
  --decisions out/eval/received50-v1/final-review.jsonl \
  --artifacts-root out/eval/received50-v1/package \
  --output out/eval/received50-human-v1

# 사람 검수한 50장과 동일 원본·동일 최종 출력에 대한 AI 판정을 대조한다.
python -m standin_eval run rubric-audit-judge \
  --reference out/eval/received50-human-v1 \
  --automatic out/eval/ai50-v1 \
  --output out/eval/ai50-audit-v1
```

ZIP은 schema/rubric/50개 ID/인원/후보 순위/실제 이미지 해시를 검사한다. ZIP 내부 파일을 실행하지 않으며 경로 탈출·심볼릭 링크·중복 파일·과대 압축 해제는 거부한다. 필수 파일 누락이나 `missing-artifacts.json`에 누락 목록이 있으면 `validation.json`에 오류를 남기고 전체 점수는 발행하지 않는다. 누락 목록 형식은 `[]` 또는 `{"items": []}`가 빈 상태다. 실패 컷을 빼서 49장으로 집계하지 않는다. 성공은 종료코드0, 검증 실패는2다.

동봉 summary를 신뢰하지 않고 직접 재계산한다. 같은 이름의 지표가 다르면 `summary_differences`에 남긴다. 원본 ZIP hash와 받은 summary도 보존한다. 성공한 패키지의 필요한 로컬 이미지와 계약은 `package/`, 재계산 결과는 `scored/`에 보존한다.

검수 JSONL 한 행은 `case_id`, `review_sha256`, `decision`, `replacement`, `reviewer`, `reviewed_at`이다. 승인 시 replacement는 null, 수정 시 전체 review-record를 넣는다. 원래 판정 hash가 달라지면 오래된 승인으로 간주하여 거부한다. reviewed_at은 실제 검수 시각을 기록하며 사람 신원을 인증하는 시스템은 아니다.

`rubric-audit-judge`는 인간 검수 reference만 받으며 같은 컷·입력 해시·최종 이미지 해시·후보·제품 상태를 요구한다. 새로운 컷과 기존 50장 점수를 억지로 일대일 비교하지 않는다. 알려진 50장 내 판정 일치도를 새 입력 일반화 정확도로 주장하지 않는다. 기준 보완용/검증용 컷을 구분해야 실제 성능 검증이 가능하다.

검증: 가상 50컷 ZIP을 실제 CLI로 수입하여 잘못 첨부된 평균100을 직접95로 재계산하고 차이를 탐지했다. 이 숫자는 제품 성능이 아닌 계약 검증값이다. 로컬 증거는 `out/eval/rubric31-zip-smoke-20261028-v1/imported/validation.json`. 50컷 수입 → 마지막 승인/수정 → 인간 확정 결과 → AI 대조 흐름과 원본 해석의 가림 보존, ZIP 공격 경로, 검수 hash 불일치까지 관련 **105개 테스트 통과**. 실제 추가 모델 호출과 운영 배포는 없다.
