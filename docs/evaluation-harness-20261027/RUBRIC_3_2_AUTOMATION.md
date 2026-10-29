# v3.2 평가 자동화와 50컷 재집계

2026-10-28 기준. 현재 계약은 `evaluation/rubrics/standin-v3.2/`의 rubric과 schema다. v3.1 경로는 이전 결과 재현을 위해 유지한다.

## 받은 자료와 검증

`standin-rescore-v3.2.zip` SHA-256:
`be85f93b2e3f4bc276d221ce8dd5c716b3360f546802c5dcf65a703d7bcf00b3`

전체 367파일을 조사했다: JSON 212, Python 6, CJS 2, JSONL 2, PNG 99, Markdown 1, HTML 3, FBX 30, BVH 12. 문서·계약·판정·실행 증거·계산/검증/보고서 소스를 읽고, 원본 50장과 최종 이미지 48장을 접촉 시트로 확인했다. BVH 구조와 FBX 바이너리 헤더를 확인했으며, 이는 3D 형상 품질 인증을 뜻하지 않는다. ZIP의 프로그램을 실행하지 않고 검토한 순수 점수 계산 규칙을 하네스에 적용했다.

아카이브 해시 목록 자체를 제외한 **366개 파일의 바이트 수와 SHA-256**, 50컷의 고정 집합, 인물 수, 후보·선택·리파인·FBX 영수증의 대응을 검증했다. root의 v3.2 계약이 현재 기준이며 `source-contract`는 역사 자료다. 원본 파일은 `package/`에 그대로 보존한다.

실제 import 결과: `out/eval/received50-v32-20261028-v2/validation.json`.
컷별 재계산은 제출된 `computed-scores.json`과 완전히 일치했고, summary 차이도 없다. 선택적 FBX 미확인 18건은 시각 평가 필수 자료 누락과 분리했다. 정확한 평가 AI 모델 등 미확정 메타데이터는 추정해 채우지 않는다.

## 적용한 계산 규칙

| 구분 | 전신 | 상반신 |
|---|---|---|
| 만점 | 100 | 45 |
| 활성 항목 | 양팔·양다리·몸통·머리·전체 동작 | 양팔·몸통·머리 |
| 비활성 항목 | 없음 | 양다리·전체 동작은 `not_applicable` |
| 90점 기준 | 적용 | 적용하지 않음 |

원본 manifest의 `framing == "01_상반신"`으로 범위를 고정한다. 결과 이미지의 잘림을 보고 범위를 바꾸거나 상반신 점수를 100점으로 환산하지 않는다.

확정 불일치 감점을 D, 불확실 항목의 가능한 감점을 U라 하면 점수는 `[만점-D-U, 만점-D]`다. 확률이나 통계적 신뢰구간이 아니다. 여러 인물의 같은 항목은 한 번만 차감하고, 불일치가 있으면 같은 항목의 uncertain을 중복 차감하지 않는다. 이때 범위가 한 점으로 좁아져도 모든 인물의 판정이 확정됐다는 뜻은 아니다.

활성 항목에 `unobservable` 또는 `not_run`이 하나라도 있으면 범위 평균에서 제외한다. 인물 누락도 제외하며, 반환 인물의 활성 판정이 전부 확정됐을 때만 최대 50점(상반신은 45점)의 부분 결과 진단값을 남긴다. 정상 완료 무결과와 실행 오류도 별도 집계한다. 확정 만점은 모든 활성 판정이 match일 때만 인정한다.

## 실제 50컷 결과

| 범위 | 분류된 컷 | 집계 가능 | 모두 확정된 컷 | 평균 점수 범위 |
|---|---:|---:|---:|---:|
| 전신 | 40 | 27 | 10 | 57.78–67.59 / 100 |
| 상반신 | 10 | 10 | 7 | 28.50–33.50 / 45 |

전체 50컷 중 13컷은 평균에서 제외된다: 인물 누락 7, 정상 무결과 4, 활성 항목 관찰 불가 2. 결과 반환은 46/50, 전원 반환은 39/50, 인물 반환은 48/60이다. 전신 확정 만점은 0컷, 상반신 확정 만점은 1컷이다. 전신 90점 이상 확정은 1/27, 가능 범위 포함은 4/27이다.

v3.1에서 v3.2로 활성 항목 판정이 바뀐 것은 0건이며 비활성 30항목이 평가 대상 아님으로 표시됐다. 이번 차이를 모델 개선으로 해석하지 않는다. 50컷 전부 `ai_assisted_unreviewed`이며 사람 정답 데이터로 자동 승격하지 않았다.

## 실행 방법

저장소 루트에서 의존성이 설치된 Python을 사용한다. 이 작업 환경의 실행 파일은 `/Users/dowon/dev/Standin-server/.venv/bin/python`이다. 출력 경로는 매번 새 경로여야 한다.

```bash
python -m standin_eval run rubric-import-zip \
  --zip /Users/dowon/Downloads/standin-rescore-v3.2.zip \
  --output out/eval/received50-v32-new
```

이 한 명령이 검증, 재집계, 원본/최종 이미지와 항목별 근거가 있는 `scored/report.html`, 검수용 `final-review.jsonl`을 만든다. 제출 점수를 그대로 믿지 않으며 차이는 `validation.json`에 기록한다.

새 컷에는 원본 이미지, 제품의 최종 출력, 후보와 인물 대응, 원본 framing, 파일 해시를 담은 manifest와 `rubric_version: "3.2"` 실행 설정을 제공한다. 받은 native manifest와 하네스 normalized manifest 둘 다 지원한다. 테스트 fixture `tests/test_rubric_v32.py::bundle`이 최소 normalized 형식 예시다. 사람에게 7항목 정답을 먼저 작성하게 하지 않는다.

```bash
# 제공된 50컷과 같은 실행 산출물을 AI로 다시 판정하는 예시.
# MODEL_ID는 사용할 평가 모델로 지정한다. 이 명령만 외부 모델을 호출한다.
python -m standin_eval run rubric-judge \
  --manifest out/eval/received50-v32-20261028-v2/scored/dataset-manifest.json \
  --config out/eval/received50-v32-20261028-v2/scored/run-config.json \
  --artifacts-root out/eval/received50-v32-20261028-v2/package \
  --output out/eval/ai50-v32-new \
  --model MODEL_ID --max-calls 96 --judge-timeout 60 --allow-provider
```

AI는 원본만으로 목표 자세를 추정하고 저장·동결한 다음, 최종 출력과 비교한다. 원본 단계의 불확실/비가시 정보는 비교 단계에서도 보존한다. 상반신 비활성 항목은 계산 정책으로 제외한다. 이후 점수 범위·제외 사유·근거·최종 검수 큐를 자동 생성한다. 목표 추정은 `ai_estimated_target_not_gold`로 구분한다.

사람은 마지막에 report를 보고 검수 JSONL의 각 행에 approve 또는 replace와 실제 reviewer/reviewed_at을 적는다. 수정은 판정과 이유에 한정되며 제품 출력·후보를 바꾸지 않는다. 원본 승인 대상 hash가 달라지면 확정을 거부한다.

```bash
python -m standin_eval run rubric-finalize \
  --scored out/eval/received50-v32-20261028-v2/scored \
  --decisions out/eval/received50-v32-20261028-v2/final-review.jsonl \
  --artifacts-root out/eval/received50-v32-20261028-v2/package \
  --output out/eval/received50-human-v32-new

python -m standin_eval run rubric-audit-judge \
  --reference out/eval/received50-human-v32-new \
  --automatic out/eval/ai50-v32-new \
  --output out/eval/ai50-audit-v32-new
```

source schema의 AI-only 상태는 원본 그대로 보존한다. 내부 검수 완료 결과는 동일 계약에 `human_reviewed` provenance를 허용하는 확장을 사용한다. 확정 결과와 수정 이력은 별도 버전으로 저장하고, audit는 같은 입력·출력의 사람 검수 결과와 AI 결과만 대조한다. 상반신 비활성 항목을 판정 일치율에 넣지 않는다. 사람이 수정한 정보를 피드백으로 남기지만 자동 재학습이나 추정 정답의 자동 확정은 하지 않는다.

## 검증 범위

v3.2 경계값·다인 중복 감점·비가시 제외·상반신 범위 고정·자동 목표 추정부터 사람 확정/audit까지 포함해 관련 **134개 테스트 통과**. 실제 제공 ZIP import도 최종 코드로 재실행해 50컷 계산과 제출 집계 일치를 확인했다. 이번 검증에서 추가 외부 모델 호출과 운영 배포는 하지 않았다.

## 2026-10-29 PR 전 회귀 검증·코드 리뷰

`tests/test_rubric_v32_package.py`에 외부 파일·네트워크 없이 실행하는 native ZIP 회귀 테스트 17개를 추가했다. 최상위 폴더 유무, 선택적 누락 허용, 파일 누락·해시·바이트 크기 불일치, choice/refine/FBX/실행 영수증 불일치, 인물 대응, 계약 해시, 고정 컷 수, schema 누락, 경로 탈출, 제출 집계 차이 감지를 검증한다. ZIP 안의 프로그램은 실행되면 실패하도록 작성한 fixture로 비실행을 확인한다.

코드 리뷰에서는 CLI → ZIP 추출 → native 변환 → schema·점수 → AI 판정 → 사람 확정 → audit 경로를 확인했다. 발견한 두 문제를 수정했다: 반환 후보 풀에 있는 다른 후보로 재채점 대상을 바꿀 수 없도록 frozen selections와 판정 대상을 대조하며, native 실행 실패 사유를 변환 후에도 보존한다. refine 영수증의 person_index도 선택 인물과 대조한다. 새 회귀 테스트에서 수정된 동작을 확인했다.

최종 코드로 실제 제공 ZIP을 재수입한 결과는 `out/eval/received50-v32-pr-20261029-v1/validation.json`이다. 366개 해시 검증, 50컷 계산 일치, summary 차이 없음. 실제 이미지·FBX·수신 ZIP과 생성 보고서는 Git에 추가하지 않는다. 외부 평가 모델의 판정 정확도는 이 코드 리뷰/계약 테스트의 검증 대상이 아니다.

최종 전체 테스트(`tests/`, converter 제외): **2763 passed, 8 skipped, 31 subtests passed**. 스모크 **54/54**, `compileall src api scripts standin_eval`, `git diff --check` 통과. 처음에는 로컬 소켓 바인딩 제한으로 SDK 모의 HTTP 테스트 10개가 실패했으나, 로컬 테스트 서버를 허용한 전체 재실행에서 모두 통과했다. 코드 리뷰에서 발견한 문제는 수정 및 회귀 검증을 마쳤다. GitHub CI 결과는 별도로 확인한다.
