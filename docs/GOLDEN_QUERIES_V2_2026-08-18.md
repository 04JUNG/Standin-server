# Semantic search golden queries v2 보고서

> 생성일: 2026-08-18  
> 상태: query·split 동결 유지 · member PoseCode DB schema v2에 재고정 · development 평가 PASS

## 결과

| 항목 | 결과 |
|---|---:|
| 전체 query | 45 |
| development / holdout | 30 / 15 |
| 측정 기반 exact query | 31/31 완성 |
| source-context query | 4/4 완성 |
| no-exact·강건성 query | 10/10 정책 고정 |
| active pose / semantic unit | 1,232 / 616 |
| 제외 CMU pose 누수 | 0 |
| 미완료 판정 규칙 | 0 |
| 회귀 테스트 | 10/10 pass |

## v1에서 해결한 항목

- 이전 1,307-pose library hash를 현재 semantic build ID로 교체했다.
- 검색 제외한 CMU 76 pose를 모든 정답 집합에서 제거했다.
- C03·C04·C05·D04·E03·E05 정답 집합을 실제 posecode 측정값으로 계산했다.
- `torso_lateral_lean_deg`는 body-local left→right 축에서 양수=오른쪽, 음수=왼쪽으로 확정했다.
- G01–G04는 `semantic_vocab.v2` source mapping의 contextual unit 정답을 사용한다.
- F07은 과거 orphan 검사가 아니라 완전한 Typing UsingMouse 미러쌍의 contextual 회수 검사로 바꿨다.
- 이미 진단에 사용한 B01·F01은 development로 두고 15개 holdout을 별도 고정했다.

## 판정 경계

- A–E의 정답은 포즈 의미를 사람이 확정한 것이 아니라 **결정적 측정 규칙으로 계산한 정답 집합**이다.
- E 자세분류는 웅크림·착지·눕기 후보가 섞일 수 있어 사람 precision 검토를 병행한다.
- C01–C05는 unit embedding만으로 끝낼 수 없고 concrete member 측정값으로 원본/미러를 선택해야 한다.
- F는 context 후보를 보여줄 수 있지만 exact/observed 사실로 승격하면 실패다.
- G는 contextual recall을 평가하며 pose truth로 표시하면 실패다.
- H는 되묻기 또는 다양한 결과가 정답이다.

## 고정 대상

- semantic build ID: `sha256:217d56e31c42634ec920db8704dd151b088b2f12291d5e3726f5b34c9be50196`
- dataset fingerprint: `sha256:798734eb9761ba1c5bc73c08cd2af780fc37182178ac7801c1e661cb34f50aed`
- pose library version: `sha256:22eb5e9c24a954c11b68f684f327a71e42b694a9aed7e721589d30d84f724c76`

## 산출물

- `data/semantic/golden_queries/golden_queries.v2.json`: 기계 평가 단일 소스
- `data/semantic/golden_queries/golden_queries.v2.csv`: 사람 검토용 파생 파일
- `data/semantic/golden_queries/README.v2.md`: 운영 규칙과 분할 설명
- `scripts/build_golden_queries_v2.py`: 재현 가능한 builder
- `tests/test_golden_queries_v2.py`: build·분할·제외·미러·문맥 회귀 테스트

## 현재 결과와 다음 단계

내부 runtime은 development 30개에서 PASS했다. 결과는 `docs/SEMANTIC_RUNTIME_POC_2026-08-18.md`에
저장했고 내부 `POST /semantic-search`와 health 계약도 구현했다. 다음은 설정·release bundle을
동결한 뒤 holdout 15개를 최종 gate에서 한 번 실행하는 것이다. holdout 결과를 보고 설정을 바꾸면 해당 쿼리를 development로 강등하고 새
holdout을 만들어야 한다.
