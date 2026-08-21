# Standin evaluation harness

실행 진입점은 `python -m standin_eval`이다. 저장소의 `.venv`를 쓰는 예시는 다음과 같다.

```bash
# 현재 inventory 확인. selected12-v1은 GT가 아직 없어 성능 수치를 내지 않는다.
.venv/bin/python -m standin_eval dataset validate selected12-v1
.venv/bin/python -m standin_eval dataset stats selected12-v1

# persons.jsonl과 cuts.jsonl을 사람이 검토·수정한 뒤 봉인한다.
.venv/bin/python -m standin_eval dataset seal selected12-v1

# 실제 서버의 E2E/latency run. 요청 backend와 실제 backend가 다르면 실패한다.
.venv/bin/python -m standin_eval run http \
  --target http://127.0.0.1:8000 \
  --dataset selected12-v1 \
  --requested-vlm gemini \
  --requested-pose rtmlib \
  --note "baseline"

# VLM/pose 원시 입력을 고정해 코드 변경만 replay한다.
.venv/bin/python -m standin_eval fixture capture-vlm \
  --dataset selected12-v1 --provider gemini --cache-miss capture
.venv/bin/python -m standin_eval fixture capture-pose \
  --dataset selected12-v1 --vlm-fixture <fixture-id> --backend rtmlib \
  --cache-miss capture
.venv/bin/python -m standin_eval run replay \
  --dataset selected12-v1 --fixture <fixture-id> --note "candidate"

# A/B 후보를 합치고 출처·rank를 숨긴 라벨 풀을 만든다.
.venv/bin/python -m standin_eval labels pool <run-a> <run-b>
.venv/bin/python -m standin_eval labels validate \
  --pool <pool-dir> --labels <completed-labels.jsonl>
.venv/bin/python -m standin_eval compare <run-a> <run-b> \
  --labels <completed-labels.jsonl>

# 검색과 분리한 legacy refine pair probe(승격 판단용 아님).
.venv/bin/python -m standin_eval run refine-pairs \
  --target http://127.0.0.1:8000 --from-run <http-run-id>

# 의사결정용 B0(no-refine)/B1(v1)/B2(v2.4 aggressive) 3-arm 평가.
# 두 서버는 각각 REFINE_V2_ENABLED=0/1이며 새 REFINE_DIR로 cache-off 기동한다.
# 승격 기준은 서버/holdout 결과를 보기 전에 반드시 이 명령에서 봉인한다.
.venv/bin/python -m standin_eval run refine-eval \
  --v1-target http://127.0.0.1:8001 \
  --v2-target http://127.0.0.1:8002 \
  --from-run <http-or-replay-run-id> \
  --promotion-criteria evaluation/refine_promotion_criteria.example.json

# CSP/avatar mesh 검사를 template 계약으로 완료하고, 라벨을 받기 전
# 불변 result manifest에 연결해 봉인한다.
.venv/bin/python -m standin_eval evidence seal-mesh \
  --run <refine-eval-run-id> \
  --mesh-evidence <completed-mesh-safety-evidence.jsonl>

# `refine_label_queue.jsonl` 순서로 두 template의 unknown 라벨을 채운다.
# private assignment 파일은 UI/라벨러에게 공개하지 않는다.
.venv/bin/python -m standin_eval report <refine-eval-run-id> \
  --independent-labels <completed-independent-labels.jsonl> \
  --pair-labels <completed-pair-labels.jsonl>
```

`report`는 run 안의 `promotion_criteria.frozen.json`만 사용한다. 다른
`--promotion-criteria`를 나중에 넣어 기준을 바꿀 수 없고, 같은 파일을 명시하더라도
봉인 hash가 정확히 같아야 한다. 예시 기준은
[`refine_promotion_criteria.example.json`](refine_promotion_criteria.example.json)에 있다.
실제 D2를 열기 전 D1 결과로 MCID·표본 수·latency 예산을 확정해 별도 파일로 복사하고,
그 파일을 `run refine-eval`에 넘긴다.

각 run은 기본적으로 `out/eval/runs/<run-id>/`에 생성된다. HTTP run의
`timings.jsonl`은 live 왕복시간과 `Server-Timing` 단계별 시간을, replay run은
제품 latency로 해석하면 안 되는 고정-input 실행시간을 기록한다.
fixture capture는 기본적으로 `.eval-cache/model-cache`의 checksum 검증된 원시
VLM/pose 결과를 재사용한다. 외부 모델 호출을 절대 허용하지 않는 재현 run은
`--cache-miss error`, 강제 재수집은 `--refresh`를 사용한다.

`refine-eval`은 실행 전에 동일 query/skeleton/selected base를 봉인하고 서버의
v1/v2 capability·config hash 및 양쪽 base BVH hash를 검사한다. 모든 평가 단위에
B0/B1/B2 행을 남겨 gate·fallback·timeout도 ITT 분모에서 제외하지 않는다. 세 최종
BVH에는 같은 외부 평가기(2D NME, limb/endpoint/pair/contact, 선택적 synthetic MPJPE,
BVH/FK·충돌·관절·발·ground 검사)를 적용한다. 동일 geometry는 blind pair에서 자동
tie가 되며, 그 외에는 버전 정보를 숨긴 SVG와 사람 라벨이 필요하다. 라벨이나 사전
승격 기준이 빠졌거나 ownership/mesh 안전 검증이 부족하면 결과는 `INCONCLUSIVE`다.

사람 평가 작업은 기본적으로 20% 이중 라벨과 5% 숨은 반복을 사전 생성한다.
`refine_label_assignments.private.jsonl`에는 같은 artifact인지 검증할 assignment lineage가,
공개 queue에는 무작위 `assignment_id`만 들어간다. reporter는 최소 2명, 15% 이중 라벨,
5% 유효 숨은 반복과 실제 labeler 관계를 확인한다. `result_manifest.json`은 라벨 전에
arm 결과·BVH·SVG·blind provenance·assignment를 봉인하므로, 결과를 본 뒤 artifact나
좌우 순서를 바꾸면 승격 판정은 `INCONCLUSIVE`가 된다.

SVG 스틱피겨의 capsule 검사는 진단용 proxy다. 승격에는 별도 CSP/avatar mesh evaluator가
각 `unit_id × arm`의 artifact/geometry hash, evaluator/body version, 완결된 검사와
absolute/new violation ID를 기록한 evidence가 필요하다. 이 증거가 없으면 자동 2D 지표가
좋아도 `PASS`가 되지 않는다.

Mesh evidence는 `mesh-checks-v2` 계약을 따른다. 필수 검사는 `parse_fk`, `ownership`,
`anatomy`, `collision`, `contact`, `ground`, `foot_direction` 7개이며, 실패한 검사는
`<check>:<stable_detail>` 형식의 hard violation ID를 반드시 가진다. 같은 unit에서 B0가
통과한 검사를 B1/B2가 실패하면 같은 prefix의 `new_hard_violations`도 반드시 기록해야 한다.
템플릿·봉인 명령·reporter가 이 동일 계약을 검증하므로 필드 누락이나 신규 위반 누락은
`INCONCLUSIVE`로 처리된다.

## 현재 데이터 상태

`selected12-v1`은 14개 파일을 SHA-256으로 중복 제거한 12개 고유 이미지 inventory다.
원본 provenance, `artist_id`, `project_id`, `scene_group_id`, person bbox, eligible/scale을
아직 사람이 확정하지 않았다. 따라서 `target_persons=0`이며 생성되는 report는 의도적으로
`INCOMPLETE`다. 이 상태의 0% 또는 100%는 성능 지표가 아니다.

다음 수작업은 이미지 결과를 보지 않은 상태에서 `cuts.jsonl`과 `persons.jsonl`을 채우고
검토 후 `dataset seal`로 hash를 갱신하는 것이다. D1/D2에는 라이선스·외부 VLM 전송 동의와
작가/프로젝트/scene group 분리도 반드시 기록한다.
