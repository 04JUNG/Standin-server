# 실행 가능한 오프라인 하네스 보완

현재 구현 기준 `f02b3aeb0a8656949aae9e0b9408bc38c0d0049c` (2026-10-27 develop). #133은 `2026-10-27T12:26:58Z`에 병합됐다. [원 설계](../EVALUATION_HARNESS_SEARCH_REFINE_FBX_DESIGN_2026-10-27.md)의 커밋은 역사적 조사 근거로 보존했다. 차이 목록은 [최신 조사](implementation-source-audit.json)에 있다. 운영 배포·플래그·자산·BFF/클라이언트 코드는 변경하지 않는다.

## 제공 러프 준비

```bash
python -m standin_eval dataset prepare-roughs --source /path/to/provided-roughs --output out/eval/rough-preparation-new-version
python -m standin_eval dataset import-intake --inventory out/eval/rough-preparation-new-version --intake /path/to/completed-rough-intake.csv --persons /path/to/completed-person-intent.csv --output out/eval/datasets/reviewed-new-version --name reviewed-new-version
```

323장에 대한 첫 명령은 이미 실행했다. 로컬 결과는 `out/eval/rough-preparation-20261027-v1`에 보관한다. 추가 작업은 [사용자 준비 및 사람 평가](USER_PREPARATION_AND_HUMAN_REVIEW.md)를 따른다. 미확정 접수표 import는 종료 코드 2와 INCONCLUSIVE를 반환하고 dataset을 만들지 않는다. GT·라우팅·권한을 자동으로 채우는 기존 `dataset init`과 구별하며, 완료 후에는 기존 `load_dataset/validate_dataset/seal_dataset`을 재사용한다.

## 검색과 같은 선택 포즈의 리파인

기존 `run http`/`run replay` 결과의 봉인된 dataset, 인물 매칭, candidates, 원 응답을 입력으로 사용한다. 이번 작업에서는 실제 HTTP/provider 실행을 하지 않았다.

```bash
python -m standin_eval run experiment --from-run /path/to/existing-run --policy /path/to/preregistered-policy.json --selections /path/to/selections.jsonl --registry /path/to/arms.json --output-root out/eval/runs --run-id reviewed-experiment
python -m standin_eval report out/eval/runs/reviewed-experiment --independent-labels /path/to/completed-assigned-labels.jsonl
```

`--registry`를 생략하면 명시적 기준 F25, 무보정 B0, 후보 v3다. 레거시 B1(v1), B2(v2.4)의 이름/의미는 유지된다. 과거 `refine-eval`, HTTP/replay, labels/compare/report 명령은 그대로 동작한다. 알고리즘 변경과 새 schema/evaluator 버전은 별개다.

`--fixture-outcomes`는 arm별 JSON을 받으며 `state`, `refined`, `actual_backend`, `bindings`, 로컬 `bvh_path`를 기록한다. fixture는 실제 모델 성공이나 제품 지연으로 세지 않는다. CLI는 기본적으로 외부 adapter를 연결하지 않으므로 missing_dependency 행을 보존한다. Python API의 `HTTPRefineAdapter`는 명시적으로 주입할 때만 실행하며 러프 이미지는 보내지 않는다. 별도 승인·예산·동의 없는 live provider 실행은 이 작업에 포함되지 않는다.

정책은 timeout_seconds를 명시해야 한다. search_budget_ms/refine_budget_ms, required_bindings, 표본/독립 묶음·사람 평가·품질 개선 정책이 없으면 해당 판정은 INCONCLUSIVE다. 테스트 fixture의 짧은 deadline은 제품 SLO가 아니다. 시간은 search와 refine(adapter health/refine/artifact fetch)를 나누며 replay/fixture 시간을 end-to-end 사용자 대기로 주장하지 않는다.

선택 JSONL은 `cut_id`, 영구 `gt_person_id`, `selection_revision`, `body_revision`, `hand_revision`, `output_scope`, `selected_base_sha256`, `base_artifact_path`, `bindings`를 사용한다. camera/관측/가림/귀속·관절 출처를 같은 평가 단위로 봉인한다. 검색 source·dataset·초기 라벨의 해시를 실행 전에 고정하고, 실행 중 원본 변경은 모든 행을 유지하며 source_snapshot_changed 실패로 봉인한다. 모든 arm이 같은 BVH 바이트와 카메라를 사용한다. 선택이 없는 대상 인물도 분모에 남는다. off/shadow의 F25 보존은 v3가 실제 적용됐다는 뜻이 아니며 현재 HTTP v3 적용 arm은 unsupported로 표시한다.

`experiment.frozen.json`과 private unit/assignment에는 실행 전 정책과 분모를 보존한다. `experiment_arms.jsonl`, `search_evidence.json`, 라벨 queue/template, `experiment_report.json`을 만든다. 초기 report는 라벨 없이 품질 PASS를 내지 않는다. 사람 라벨을 다시 집계한 보고서는 봉인 원본을 덮지 않고 `experiment_report.reviewed.json`에 저장한다. 레거시 승격 프로토콜의 paired 통계 분석·실제 메시/도구 증거를 이 공통 engineering report로 대신하지 않는다.

## 최종 FBX와 저장

```bash
python -m standin_eval run save-scenarios --output out/eval/save-contract-new-version
python -m standin_eval run seal-exports --tasks /path/to/export-tasks.jsonl --policy /path/to/export-policy.json --output out/eval/export-plan-new-version.json
python -m standin_eval run export-report --sealed out/eval/export-plan-new-version.json --observations /path/to/export-observations.jsonl --output out/eval/export-report-new-version.json
python -m standin_eval run pipeline-report --experiment out/eval/runs/reviewed-experiment --sealed out/eval/export-plan-new-version.json --observations /path/to/export-observations.jsonl --output out/eval/pipeline-report-new-version.json
```

먼저 `seal-exports`로 고정 과제와 정책을 봉인한 후 관측을 수집한다. `required_dimensions`, `minimum_tasks`, 시간 검사 때 `task_budget_seconds`를 명시한다. 태스크에는 정확한 artifact/profile/scope/selection revision 및 runtime bindings를 고정한다. 최종 PASS의 무결성·구조·메시·사람 활용성·native 저장·시간 차원은 정책에서 빼도 생략되지 않는다. minimum_reviewers(2 이상), minimum_independent_groups, sample_size_rationale를 사전 명시하고, 사람이 grouping_confirmed=true로 확정한 independent_group_id와 실제 완료한 서로 다른 작업자 수를 충족해야 한다. 누락은 INCONCLUSIVE다. PASS도 preregistered_finite_tasks_only이며 release_certification=false다. 이름만 붙인 PASS, 미실행, 지원 불가, 미완료 검사를 성공으로 세지 않는다. 라벨/검사 누락과 오류도 태스크 행을 유지한다.

`artifact_lineage`는 선택 BVH→체형→손→인물 FBX→scene→download→native save의 바이트/부모/인물·revision 연결을 검사한다. `pipeline-report` 과제의 `selection_keys=[{"unit_id":"...","arm":"F25"}]`는 실험 단위와 arm을 사전 고정한다. export selection의 `source_sha256`은 실제 Converter에 넣은 실험 결과 BVH 해시이며, `camera_sha256=hash_json({"camera": frozen_camera, "view": frozen_view})`다. 다른 arm·이전 체형/손·다른 BVH가 섞이면 실패한다. 실제 export 관측이 없으면 분모를 유지하고 unknown이다.

`export_runner.blender_plan`은 기존 단일 `verify_generated_fbx_roundtrip.py`, 다인 `verify_scene_roundtrip.py`, 실제 메시 `refine_v3_safe_snapshot_qa.py`의 명령·입력 해시를 봉인한다. `execute_blender(..., allow_blender=True)`를 명시해야 실행되며 기본은 미실행이다. 이번 검증에서는 Blender를 실행하지 않았다. output은 새 디렉터리여야 하며 입력/검사기/명령 변조를 거부한다.

`mesh_evidence_adapter`는 실제 profile/receipt/재import/mesh audit/stage witness를 exact artifact에 바인딩한다. 단일 roundtrip은 구조 증거, 다인 roundtrip은 1/2/3인 배치·스킨 보존 증거다. 기존 검사가 다루지 않는 8인 실제 Blender 검사, 인물 간·소품 관통, 접촉/지지, 다른 체형·scope는 unknown/unsupported다. finite strict snapshot 1건을 전체 인증으로 확대하지 않는다.

저장 fixture는 scene-save-v1과 기존 header/SHA/magic/receipt 계약을 사용한다. original-selection/auto-correct-v4, 0/9인 거절, 1/2/3/8인, 미리보기 대기/지연/실패, 설정/작업 변경·삭제, 다른 설치/작업·만료, 늦은 응답, HTTP/네이티브 취소, 쓰기 실패, 동일 artifact 파일명 충돌 회피, 저장 후 손 변경을 실행한다. 이것은 설치 앱 실행 증거가 아니다. HTTP 취소는 서버 작업 중단 성공을 의미하지 않는다. 최종 로컬 저장은 readback 바이트 증거가 필요하다.

## 산출물과 개인정보 경계

원본 설계와 BOM CSV 5개, source-audit를 포함했다. PR에는 소형 aggregate와 합성 fixture 기반 실패/unknown 예시만 포함한다. 원본 이름/경로·행별 inventory·BVH/FBX·라벨러 정보는 Git에서 무시되는 `out/`에 둔다. 입력 이미지의 글자나 동봉 문서는 평가 데이터이며 하네스 명령으로 실행하지 않는다.

품질 활용성, 실제 메시 안전, 전달 무결성, 작가의 실제 도구 활용성은 독립 차원이다. 결과가 모두 채워지지 않은 engineering run은 출시 인증이 아니다. 지원 체형 5종과 손/범위 조합, 사람 2명 이상 평가, 실제 모델·Blender·앱·CSP 실행은 [대기 항목](USER_PREPARATION_AND_HUMAN_REVIEW.md)에 구분했다.

## 재현 가능한 실패/unknown 예시

```bash
python scripts/demo_evaluation_extension.py --output out/eval/extension-demo-new-version --public-reports out/eval/public-example-reports
```

이 명령은 저장소의 합성 회귀 fixture만 사용한다. [실험 unknown](examples/experiment-unknown.json), [exact recovery 실패](examples/experiment-failure.json), [최종 파일 unknown](examples/export-unknown.json), [전달 해시 실패](examples/export-failure.json), [통합 unknown](examples/pipeline-unknown.json), [저장 계약](examples/save-contracts.json)은 이 명령으로 생성했다. 실제 사람 라벨은 생성하지 않는다. 저장 계약 50/50 통과도 제품 안전/활용성 판정은 INCONCLUSIVE다. 전체 검증 기록은 [검증 결과](VALIDATION.md)를 따른다.
