# 동작 의미를 보존하는 A/B1 결합 실험

> 후속 방향: [검색·조합 전략 v2](SEARCH_EDITABLE_POSE_STRATEGY_2026-09-22.md). 이 문서와 구현은 초기 shadow 비교 기준이다. 고정 순위 구간, 기존 A 제외 재유입 금지, 부족할 때만 조합하는 방식을 새 목표 파이프라인의 고정 정책으로 간주하지 않는다.

상태: **오프라인 shadow 결합 모듈·재생기 구현, 운영 승격 전**. 2026-09-21.

## 핵심 답변

A/B1은 유지한다. 이미 `Pipeline`에 기본 off 훅이 있다. 다만 현재 A는 **확실한 non_stand 입력 → 다중 시점에서 stand로 분류된 후보 제거**만 한다. 서기 입력을 서기 후보로 제한하는 기능은 아니다. `stand/non_stand`는 2D 모양을 이용한 보수적 규칙 이름이며 지면 접촉/앉기/꿇기의 실제 의미 라벨이 아니다. 두 무릎이 굽었다고 항상 서 있지 않은 것도 아니다.

B1은 현재 Top-K의 팔꿈치·손목 높이·무릎·몸통 기울기를 비교한다. 팔짱/주머니/안기 같은 동작 의미를 이해하거나, 후보군 밖에서 적합한 포즈를 가져오지는 않는다.

## 역할 분담과 구현

```text
동일 DB의 기하 후보 ── 기존 A 게이트 ─────┐
                                       ├─ 후보 합집합
동일 DB의 의미 검색 후보 ─ 관측 관절 투영 ┘
       ↓ A가 제외한 pose_id는 재유입 금지
자세 형태의 일치 / 불확실 / 모순 구간 (별도 soft 정렬)
       ↓
기하 순위 + 의미 순위로 결합 (RRF, 원점수 합산 없음)
       ↓
같은 자세 구간·검색 순위 구간 안에서 B1 재정렬
       ↓
실험 제안 + 원본 기하 결과 보존
```

- 구현: `src/experimental/intent_fusion.py`. 기존 A/B1 추출기와 인덱스를 재사용하고 새 동작 라벨이나 DB를 만들지 않는다. A/B1 모듈 안에 semantic 자산 접근을 넣지 않았다.
- `SemanticBatch`는 실제 검색된 concrete pose ID의 순서를 받는다. `semantic_user`/`semantic_vlm`, 문장, build ID, geometry DB SHA-256을 보존한다. DB 불일치나 미등록 member가 하나라도 있으면 의미 batch 전체를 사용하지 않는다.
- 의미 후보의 썸네일 정면은 매칭 카메라가 아니다. 동일 기하 인덱스와 관측 마스크로 해당 member의 최선 투영을 다시 선택한다. 검역·family 중복 제거를 재사용한다.
- A가 실제 채택한 제외 목록을 의미 후보에도 적용한다. A distance guard가 복구한 경우 제외 목록을 활성화하지 않는다.
- 추가 자세 정렬은 full/valid/full_body이고 하체가 전부 관측된 때만 작동한다. 기존 A를 양방향 hard gate로 바꾼 것이 아니다. 일치 후보를 우선하되 unknown을 삭제하지 않고 `support_match_verified=false`를 반환한다. 아는 일치 후보가 없으면 `no_known_compatible_proposal`을 기록한다.
- RRF 상수 60, 기본 검색 순위 구간 크기 3은 **미보정 실험값**이다. 의미 후보가 있을 때 B1은 그 구간을 넘지 못한다. 이를 동작 의미의 정답 보장으로 해석하지 않는다. 의미 입력이 없으면 같은 자세 구간 안의 전체 후보에 기존 B1을 적용한다.
- `published_candidates`는 입력 기하 결과 객체 그대로다. 새 제안은 기존 confidence·refine 결과를 상속하지 않고 전부 `refine_allowed=false`다. 운영 연결 전 기존 소유권/coverage/stability/geometry 안전 재판정과 출처별 계약이 필요하다.

## 의미와 구조가 충돌하면

이 구현은 두 검색기의 점수를 함께 다룰 **실험 경계**다. VLM 동작 인식이나 의미 검증기를 새로 구현한 것이 아니다.

- “서 있는 자세”가 명시되어 있어도 현재 `SemanticBatch` 문장을 구조 규칙으로 자동 변환하지 않는다. v3.3의 명시 조건 구조화·검증을 이후 어댑터로 연결한다.
- 사용자가 명시한 조건, VLM 추정, 2D 관절에서 얻은 사실을 구분해야 한다. VLM의 자기 확신만으로 hard gate나 exact claim을 만들지 않는다.
- “서기”라는 명시 조건과 “양 무릎 굽힘”이 동시에 관측될 수 있다. 이때 2D A를 실제 자세 의미의 확정 판정으로 쓰면 올바른 후보를 제외할 수 있다. 이런 충돌 사례는 별도 라벨/검증 대상이며 현재 실험을 그대로 운영 승격하지 않는다.
- 높은 의미 유사도는 손 위치·좌우·하체·접촉의 정확성을 보장하지 않는다. B1도 2D 검증이므로 3D 무릎과 실제 모델 검토를 함께 한다.

## 이번에 실행한 검증

새 통합 테스트 **10/10**, 기존 A/B1 테스트 **14/14** 통과. 검증 항목은 A 제외 재유입 방지, DB/member 불일치 차단, 의미 후보의 기하 재투영, B1의 순위 구간 보존, 하체 결측, 입력 객체/거리 불변, refine 자격 미승계다. 의미 검색 품질을 입증하는 테스트는 아니다.

동일한 frozen 실제 입력 28명을 재생했다. 결과: `artifacts/intent_fusion_20260921_01/`.

| 조건 | 기존 대비 Top-1 변경 |
|---|---:|
| A | 0 / 28 |
| B1 | 13 / 28 |
| A + B1 | 13 / 28 |
| 추가 자세 구간 + A + B1 | 13 / 28 |

A의 2D 입력 분류는 stand 4명, non_stand 2명, unknown 22명이다. A gate는 2명에서 평가·채택됐지만 Top-1은 바뀌지 않았다. 추가 자세 구간은 6명에서 활성화됐다. **22명이 unknown이라는 사실 때문에 A만으로 서기 문제를 해결한다고 약속할 수 없다.** 과거 27개 쿼리/14개 변경 결과와 이번 28개 쿼리/13개 변경은 다른 입력 프로토콜이며 섞지 않는다.

이번 재생에는 실제 의미 후보 캐시를 넣지 않았다(`semantic_accepted_cases=0`). 의미 결합은 fixture로 계약만 검증했으며 E5/VLM 검색의 정확도 향상은 아직 측정하지 않았다. 기존 사용자 라벨도 다른 후보·시점에 자동 승계하지 않았다.

## 다음 실제 의미 실험

1. 동일 DB와 일치하는 semantic build를 명시적으로 준비한다. 문장별 retrieval을 실행하고 아래 schema로 캐시한다.
2. 사람이 쓴 원본 러프 설명과 VLM 설명을 분리한다. current 28명은 개발용이며 새 holdout을 별도로 준비한다.
3. baseline / A / B1 / A+B1 / 의미 단독 / 의미 결합을 비교한다. 이 재생기의 선택적 semantic 경로는 의미 결합을 담당한다. 의미 단독 품질은 독립 검색 평가와 함께 보고한다.
4. 앞뒤/원근 렌더 규약을 먼저 검증한 실제 모델로 상체 의미, 하체 상태, 손발 관계, 활용 가능, 기존 성공→실패를 판정한다.
5. 조건이 맞는 전체 포즈가 부족한 사례에서만 상·하체 조합 실험을 추가한다.

```json
{"cases":[{"case_id":"124629:p0","image_sha256":"<frozen image hash>","query":"<사용자 문장 또는 VLM 분석 문장>","source":"semantic_user","geometry_db_sha256":"<semantic build가 검증한 geometry DB hash>","build_id":"<실제 semantic build ID>","pose_ids":["<실제로 검색된 member ID, 순위순>"]}]}
```

캐시는 실험 입력이며 이 값들을 임의로 현재 DB 값으로 고쳐 lineage 검사를 우회하면 안 된다. API/service에서 실제 build를 검증한 결과로 생성한다.

```sh
.venv/bin/python tests/experimental/test_intent_fusion.py
.venv/bin/python tests/experimental/test_a_b1_search.py
.venv/bin/python experiments/intent_fusion/evaluate.py \
  --protocol artifacts/camera_real_rough_20260921_v2/protocol.json \
  --db data/poses.db \
  --semantic-cache <검증된_실제_의미검색_캐시.json> \
  --out artifacts/<새_실험_폴더>
```
