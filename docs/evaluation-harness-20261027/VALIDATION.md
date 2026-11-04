# 오프라인 구현 검증 — 2026-10-27 KST

운영 미배포. origin/develop `f02b3aeb0a8656949aae9e0b9408bc38c0d0049c` 기준. A/B를 별도 관리 Git 워크트리의 브랜치에서 병렬 구현한 뒤 통합했다. 제품 API/플래그/자산·BFF/클라이언트와 analytics PR #134/#156을 수정하지 않았다.

## 수행한 검사

- 신규 intake/experiment/export/pipeline + 기존 evaluation/evaluator/three-arm/mesh + v3 계약/evidence/runtime/solver/target profile + Converter mesh/scene/final artifact 회귀: **253 passed, 0 skipped**, 추가 subtest 3개 통과.
- 첫 실행에서 빠진 legacy fixture capture/replay 2개는 현재 워크트리에 로컬 합성 DB(5개 포즈/20개 투영)를 생성해 재검증했다. 최종 suite에 skip은 없다. 이 DB는 Git에서 무시되며 실제 라이브러리/운영 자산을 사용하거나 변경하지 않았다. 실 BVH/비공개 DB로의 모델 품질 평가는 별도다.
- `python tests/test_smoke.py`: **54/54 passed**.
- `python -m compileall -q src api scripts standin_eval`, `git diff --check`: 통과.
- CLI 8개 경로 실행: prepare-roughs, 미완성 import-intake(예상 exit 2), fixture experiment, report(예상 exit 2), save-scenarios, seal-exports, export-report(예상 exit 2), pipeline-report(예상 exit 2). **8/8 예상 종료 코드**.
- 저장 계약 original-selection/auto-correct-v4 50개 시나리오: **50/50**. 제품 판정 INCONCLUSIVE.
- 실행 중 source candidates 변경 감지/실패 봉인, 잘못된 다인 count에 따른 scene 검사 우회 차단 회귀를 포함했다.
- 실패/unknown 예시 생성: exact fallback 해시 위반과 저장 byte lineage 위반은 FAIL, 라벨·실제 export 미실행은 INCONCLUSIVE. 예시에 원본 경로가 없는지도 확인했다.
- 제공 원본 읽기 전용 inventory: **PNG 323개, 고유 SHA 323개, exact duplicate 0, 크기 미확인 0**. 원본 84,092,609 bytes를 변경하지 않았다. 기계 작업을 완료했지만 인물/출처/동의 판단은 하지 않았다.

실행 환경 Python 3.12.13. 기존 개발 venv의 numpy/Pillow와 임시 pytest 환경을 사용했다. 외부 네트워크는 git/GitHub metadata 및 임시 pytest 설치에만 사용했으며 원본 러프를 전송하지 않았다. 기존 fastapi TestClient의 httpx deprecation warning 1건은 테스트 실패가 아니다.

## 재실행

```bash
python -m pytest -q tests/test_eval_experiment.py tests/test_eval_export_contracts.py tests/test_eval_pipeline.py tests/test_eval_intake.py tests/test_eval_harness.py tests/test_refine_evaluator.py tests/test_refine_three_arm.py tests/test_refine_mesh.py tests/test_refine_v3_contracts.py tests/test_refine_v3_evidence.py tests/test_refine_v3_runtime.py tests/test_refine_v3_solver.py tests/test_refine_v3_target_profile.py tests/converter/test_refine_v3_mesh_validation.py tests/converter/test_scene_contract.py tests/converter/test_final_artifact.py
python tests/test_smoke.py
python -m compileall -q src api scripts standin_eval
```

## 수행하지 않은 실제 품질 평가

실제 provider/원본 외부 전송/대규모 생성 평가, 실제 Blender 실행, 설치 앱 native 저장·취소·경쟁 상태, 실제 CSP/Modeler 활용성, 작가 2명 이상 라벨·수정 시간, 지원 체형 5종/손/범위 조합, 독립 holdout과 통계적 승격 분석은 미실행이다. off/shadow v3 출력은 v3 적용 성공으로 세지 않는다. 현재 일반 인증 profile 0개 상태를 전체 출시 승인으로 바꾸지 않는다.

현재 저장 fixture는 제품 경계를 모델링한 injected driver다. 설치 앱을 실제로 조작한 증거가 아니다. Blender adapter는 기존 검사 도구를 호출·봉인할 수 있으나 이번에 해당 검사를 실제로 실행하지 않았다. 메시 adapter의 수학·계약 fixture는 실제 변형 메시 안전 승인을 대신하지 않는다. 사람 준비 대기 항목은 [별도 목록](USER_PREPARATION_AND_HUMAN_REVIEW.md)에 있으며 구현 미완료와 구분한다.
