# Semantic search runtime PoC 평가

> 평가일: 2026-08-18  
> split: `development`  
> holdout 사용: `아니오`  
> development gate: `PASS`

## 요약

| 지표 | 결과 |
|---|---:|
| query | 30 |
| parser↔GT 집합 일치율 | 1.000 |
| exact pose P@10 | 1.000 |
| exact unit macro R@50 | 0.693 |
| source-context macro R@50 | 1.000 |
| no-exact 안전율 | 1.000 |
| clarification 정확도 | 1.000 |
| latency p50 / p95 | 6.5 / 8.6 ms |

## 구현된 범위

- `src/semantic_search.py`: 한국어 concept parser, typed measurement 제약식, 3값 matcher, E5+FTS 회수, source-context 경계, mirror member 선택
- `scripts/semantic_search.py`: 내부 CLI 검색
- `scripts/eval_semantic_search.py`: development/holdout 분리 평가와 holdout 이중 잠금
- semantic DB schema v2: 1,232 member마다 PoseCode v2 연속 측정값 27개와 observed atom 저장
- 기존 `src/search.py` geometry 검색, `Pipeline.process_cut`, refine 입력은 변경하지 않음

검색 결과는 `observed exact`, `contextual candidate`, `library_gap`, `clarification_required`를
분리한다. 출처명에 dance/typing이 있어도 포즈 자체에서 전통성·소품·의도를 관찰했다고 주장하지 않는다.

## 쿼리별

| ID | mode | status | 핵심 지표 | ms |
|---|---|---|---:|---:|
| A01 | exact_pose_set | success | P@10=1.00, R@50=0.13 | 9.2 |
| A03 | exact_pose_set | success | P@10=1.00, R@50=0.46 | 6.0 |
| A04 | exact_pose_set | success | P@10=1.00, R@50=0.32 | 5.7 |
| A06 | exact_pose_set | success | P@10=1.00, R@50=0.45 | 5.8 |
| A07 | exact_pose_set | success | P@10=1.00, R@50=0.98 | 6.8 |
| A08 | exact_pose_set | success | P@10=1.00, R@50=0.42 | 6.8 |
| B01 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 8.0 |
| B02 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 7.9 |
| B04 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 6.2 |
| B05 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 8.5 |
| B07 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 7.2 |
| B08 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 8.6 |
| C03 | exact_pose_set | success | P@10=1.00, R@50=0.44 | 7.1 |
| C04 | exact_pose_set | success | P@10=1.00, R@50=1.00 | 6.8 |
| C05 | exact_pose_set | success | P@10=1.00, R@50=0.89 | 5.7 |
| C06 | exact_pose_set | success | P@10=1.00, R@50=0.42 | 7.0 |
| D01 | exact_pose_set | success | P@10=1.00, R@50=0.09 | 7.8 |
| D03 | exact_pose_set | success | P@10=1.00, R@50=0.50 | 7.2 |
| E01 | exact_pose_set | success | P@10=1.00, R@50=0.63 | 7.4 |
| E03 | exact_pose_set | success | P@10=1.00, R@50=0.91 | 5.9 |
| E04 | exact_pose_set | success | P@10=1.00, R@50=0.91 | 7.2 |
| F01 | no_exact_evidence | contextual_candidates | safe=1 | 4.2 |
| F03 | no_exact_evidence | contextual_candidates | safe=1 | 3.9 |
| F04 | no_exact_evidence | library_gap | safe=1 | 3.5 |
| F06 | no_exact_evidence | library_gap | safe=1 | 3.5 |
| F07 | no_exact_evidence | contextual_candidates | safe=1 | 3.6 |
| G01 | source_context_recall | contextual_candidates | context R@50=1.00 | 3.8 |
| G03 | source_context_recall | contextual_candidates | context R@50=1.00 | 3.4 |
| H01 | clarification_or_diversity | clarification_required | clarify=1 | 0.0 |
| H02 | clarification_or_diversity | clarification_required | clarify=1 | 0.0 |

## 해석

- exact P@10은 측정 제약을 통과한 concrete member만 반환하므로 허위 정답을 차단한다.
- exact R@50은 정답 집합이 50 unit보다 큰 광범위 쿼리에서는 구조적으로 1이 될 수 없다.
- source action은 `contextual` 후보로만 반환되며 `exact_pose_claim=false`다.
- `unknown` 측정값은 `violation`으로 취급하지 않지만 현재 active 1,232 member에는 누락이 없다.
- semantic 후보에는 `refine_allowed=false`를 고정해 기존 geometry/refine 경로와 섞지 않는다.
- holdout은 설정 동결 후 최종 승격 gate에서만 별도 명시 플래그로 실행한다.
- 후속 단계에서 내부 `POST /semantic-search`, bounded concurrency/cache, semantic health를 연결했다.
  이 서버는 무인증 내부 추론 API이므로 인터넷에 직접 공개하지 않는다.

## 실행

```bash
.venv/bin/python scripts/semantic_search.py "왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세" --top-k 5
.venv/bin/python scripts/eval_semantic_search.py --split development
```
