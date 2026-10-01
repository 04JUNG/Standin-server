# Dense-first Semantic Search v3.3 최종 설계

> 확정일: 2026-08-20
>
> 상태: **구현 기준 설계 확정 · 구현 시작 전** — 미결 항목 없음
>
> 적용 범위: `POST /semantic-search`와 그 전용 색인·평가·운영 계약
>
> 기준 자산: **`4ebc489a…0441f`** (mirror 수정 후 재생성, 769 unit / 1,538 member)
>
> 검토 입력: `archive/semantic/SEMANTIC_V3_DESIGN_REVIEW_2026-08-20.md`,
> `archive/semantic/SEMANTIC_V3_2_DESIGN_REVIEW_2026-08-20.md`,
> `archive/evaluations/MIRROR_DEFECT_ANALYSIS_2026-08-20.md`, `archive/evaluations/MIRROR_FIX_VALIDATION_2026-08-20.md`,
> DCR-001 ~ DCR-007, 현재 runtime 실측

이 문서는 Semantic Search v3 구현의 **단일 기준** 이다. DCR 001~007의 결정은 모두 이 문서에
병합되었으므로, 구현 시 DCR을 함께 읽을 필요가 없다. 앞으로 아래 결정을 바꾸려면 코드에서
임의 변경하지 않고 **새 DCR과 재평가를 먼저 만든다.**

### v3.2에서 바뀐 것

| 항목 | v3.2 | **v3.3** | 출처 |
|---|---|---|---|
| 기준 자산 | 1,552 / 776, build `ae3b…e895` | **1,538 / 769, build `4ebc48…0441f`** | DCR-003 |
| mirror retention | 96.96% (버그 포함) | **100.00% 전 family** | DCR-003 |
| structurer 모델 | `gemini-2.5-flash` 고정 | 후보 4개 bake-off · provider 어댑터 | DCR-001·005 |
| predicate | 단일 measurement만 | **compound 허용** (corpus 근거 강제) | DCR-005 |
| `exact_partial` | 있음 | **삭제** | DCR-002 |
| 결과 개수 | 고정 Top-K | **적응적 대표 집합** | DCR-007 |
| `top_k` | 반환 개수 | **상한** | DCR-007 |
| 되묻기 | 선택적 사용 | **semantic 경로에서 사용 안 함** | DCR-007 |
| ID 체계 | 3개 | **4개** (`structurer_model_id` 분리) | DCR-001 |
| registry | 57개 (`near_thigh` 포함) | **53개** · lateral 부호 수정 | DCR-008 |

---

## 0. 최종 결정

1. 공백이 아닌 모든 유효 자연어 query는 **원문 그대로 raw E5 검색 channel을 반드시 적용**한다.
   cache miss에서는 실제 E5를 실행하고, 동일 index의 정상 dense cache hit만 그 결과를 재사용한다.
   parser, structurer, 규칙이 dense channel 적용 전에 후보 생성을 차단할 수 없다.
2. v3 목표 자산은 **1,538 pose member / 769 mirror unit** (`4ebc48…0441f`)이다.
   1,232-member snapshot은 v2 비회귀 비교에만 쓴다. 두 snapshot 모두 production 승격 전이다.
3. 1,538 member의 27개 연속 measurement와 좌우 `observed_atoms`는 이미 보존돼 있다.
   BVH를 다시 추출하지 않고 이 원천에서 member atom과 document를 **materialize**한다.
4. 첫 구현 우선순위는 structurer가 아니라 member-scope atom과 search document v3다.
5. LLM structurer는 사용한다. 단 검색·rerank·포즈 판정이 아니라 원문의 명시 조건을
   제한된 measurement DSL로 구조화하는 데만 쓴다.
6. uncached query당 structurer 호출은 **최대 1회**다. 후보별 호출, rerank 호출, 자동 retry,
   같은 모델의 A/B self-consistency 호출은 하지 않는다.
7. structurer 자기신고만으로 exact를 허용하지 않는다. 서버가 원문 전체 clause coverage,
   literal span, 허용 predicate, 좌우·부정·수량 anchor를 결정적으로 검사한다.
8. structurer는 predicate ID만 고른다. measurement, comparator, threshold, 그리고
   compound predicate의 내부 식은 versioned server registry가 고정하며 모델이 만들 수 없다.
9. exact 검증은 E5 Top-N에 한정하지 않고 **1,538 member 전체를 스캔**한다.
10. API는 MVP에서 단일 동기 응답이다. structurer가 짧은 deadline 안에 끝나지 않으면 같은
    HTTP 200 응답에서 raw E5 후보를 `exact_pose_claim=false`로 반환한다.
11. 비어 있지 않은 정상 query를 `clarification_required`와 빈 후보로 종료하지 않는다.
    **그리고 작가에게 추가 질문을 하지 않는다** — `clarification_question`은 semantic
    경로에서 사용하지 않는다.
12. 최초 staging은 `shadow`와 `SEMANTIC_EXACT_CLAIM_ENABLED=0`으로 시작한다. 승격 gate를
    통과한 frozen release에서만 exact claim을 켠다.
13. golden v2는 품질 증거에서는 퇴역하지만 기존 기능의 비회귀 tripwire로 보존한다.
    v3 품질은 실제 사용자 표현으로 만든 별도 development/밀봉 holdout으로 평가한다.
14. 기존 semantic build를 즉시 삭제하지 않는다. `유지 / 논리적 퇴역 / 재작성 / 재생성`을
    구분한다.
15. **반환 개수는 데이터가 정한다.** 고정 Top-K를 쓰지 않고, 대표 집합을 구성해 전부 반환한다
    (상한 있음). `top_k`는 목표가 아니라 상한이다.
16. **결정성을 계약으로 걸지 않는다.** LLM은 랜덤하다. 대신 D4에서 노이즈 바닥을
    1회 측정해 기록하고, 모델 비교와 holdout 해석의 기준으로 쓴다.
17. **provider는 어댑터다.** 활성 structurer는 항상 하나지만 Gemini·OpenAI·Anthropic 중
    설정으로 고른다. 모델 은퇴가 위기가 아니라 설정 변경이 되게 한다.

---

## 1. 기준 자산과 확인된 병목

### 1.1 snapshot 이름과 해시

| 이름 | pose member / unit | 고정 식별자 | 용도 |
|---|---:|---|---|
| `v2_reference_snapshot` | 1,232 / 616 | semantic build `217d…0196` | 현재 v2 runtime 비회귀 기준 |
| **`v3_target_snapshot`** | **1,538 / 769** | **semantic build `4ebc489a…0441f`** | v3 재색인 목표 입력 |

두 manifest 모두 `production_ready=false`다. `v2_reference_snapshot`을 production이라고 부르지
않는다. `4ebc48…0441f` 역시 그대로 v3로 승격하지 않는다 — 해당 build의 BVH, inventory,
provenance, mirror lineage, source mapping, member PoseCode를 **입력 자산으로 재사용**한다.

> **`ae3b…e895`는 폐기됐다.** mirror 결함이 있던 자산이며 어떤 경로에서도 참조하지 않는다.

평가 artifact에는 항상 다음을 함께 기록하며 두 snapshot 결과를 한 지표로 섞지 않는다.

- `pose_library_version`
- geometry DB SHA-256
- `semantic_index_build_id`
- member/unit 수

### 1.2 원천에는 좌우 데이터가 있고 검색 DB에는 없다

1,538 snapshot 실측:

| 계층 | 실측 |
|---|---:|
| `member_posecodes[*].measurements` | member당 27개 연속값 |
| `member_posecodes[*].observed_atoms` | side-specific atom 존재 (`left_elbow`, `right_wrist` 등) |
| 현재 `semantic_atoms` | **unit scope 100%.** member scope 0 |
| 축소 projection `(predicate, subject, bucket)` | 19종 |

**"좌우 데이터가 없다"는 현재 semantic index 관점에서만 맞다.** 원천 `member_posecodes`와
`pose_semantic_members.observed_atoms_json`에는 side-specific 사실이 이미 있다.

> **정확한 원인**: `scripts/build_semantic_tagging.py`는 member scope atom과 연속 measure를
> 정상 생성한다. 그런데 document/index 단계(`src/semantic_documents.py::render_search_unit`,
> `direction_neutral_source_text`)가 unit scope로만 렌더하고 `_DIRECTION_EN` 정규식으로
> left/right를 **의도적으로 제거** 한다.
>
> 즉 v3의 member fact materialization은 **없는 것을 만드는 작업이 아니라
> tagging→index 사이에서 버리는 것을 보존하는 작업** 이다.
> `data/semantic/tagging_review.v1.db`가 이미 `scope`·`pose_id`·`measure`·`measure_unit`
> 컬럼을 갖고 있으므로, schema v3는 그 스키마를 index DB로 **승계** 하는 것이 본질이고
> 신설 필드는 §4.2의 4개뿐이다.

### 1.3 mirror 무결성 — 해결됨

`scripts/mirror_bvh.py`가 MOTION만 미러링하고 HIERARCHY `OFFSET`을 복사하던 결함이
수정됐다(`MIRROR_DEFECT_ANALYSIS` · `MIRROR_FIX_VALIDATION`). 769쌍 전수 재검증 결과:

| 지표 | 수정 전 | **현재 (`4ebc48…`)** |
|---|---:|---:|
| side-specific atom retention | 96.96% | **100.00%** |
| 10개 predicate family 전부 | 86.32 ~ 98.77% | **각각 100.00%** |
| bucket 불일치 unit | 98 / 769 | **0 / 769** |
| 원시값 위반율 (2°/0.02) | 18.43% | **0.00%** |
| 부호 규칙 self-check 불일치 | 0 | **0** |

`kp_mirror == swapLR(reflect_x(kp_original))` 항등식 정규화 최대 오차가 전 소스에서 `0.0`이다.

### 1.4 현재 document 병목

| 항목 | 1,538 staging |
|---|---:|
| 전체 text document | 3,449 |
| posecode_render 총 문서 | 1,538 |
| **posecode_render 고유 문서** | **288** |
| 고유 한국어 posecode 문서 | 144 |
| indexed atom scope | unit 100% |
| distinct `(predicate, subject, bucket)` | 19 |

`상체를 세움` 같은 짧은 문서가 111개 member에 반복된다. E5가 서로를 구분할 근거가 부족하고,
색인 문서에 concrete left/right 사실이 없어 `왼손`과 `오른손` 최소쌍 회수가 구조적으로 약하다.

**그래서 document/atom v3가 structurer 연결보다 먼저다.**

---

## 2. 제품 목표와 비목표

### 2.1 제품 계약

```text
임의 자연어
→ 원문 E5 channel로 의미 후보를 항상 생성
→ structurer가 검증 가능한 명시 조건만 제한된 DSL로 제안
→ 서버가 원문 coverage와 DSL을 결정적으로 검증
→ 전체 member의 관절 measurement로 exact 여부 검증
→ 해석하지 못한 조건은 후보를 유지하고 exact=false
→ 결과는 대표 집합으로 구성해 전부 반환
```

지원은 두 층이다.

- **항상 지원**: 자연어와 의미가 가까운 pose/action context 후보 반환
- **조건부 지원**: 현재 registry와 관절 measurement로 증명 가능한 조건의 exact member 반환

자연어의 모든 문화·감정·의도·소품 조건을 한 프레임 관절값으로 증명한다고 약속하지 않는다.
그런 표현도 E5 후보를 받을 수 있지만 exact claim 대상은 아니다.

### 2.2 비목표

- structurer가 BVH, 관절 좌표, pose truth, retrieval score 생성
- structurer rewrite를 E5 query로 사용
- structurer를 후보별로 호출하거나 LLM reranker로 사용
- E5 cosine을 확률이나 exact 근거로 사용
- query parser가 dense 후보를 hard filter
- geometry `/analyze`, image→pose kNN, refine 경로 변경
- **스켈레톤 추출 실패 시 의미 검색 폴백** — v3.1 승격 이후 별도 DCR (§10.5)
- v3에서 text-to-pose 공동 임베딩 모델 신규 학습
- polling, SSE, background exact 갱신
- **작가에게 추가 질문을 던지는 상호작용**

### 2.3 검토 후 기각한 대안

| 대안 | 결정 | 이유 |
|---|---|---|
| 정규식 parser 확장 | 폐기 | 표현을 미리 나열해야 하며 E5 전 조기 종료 문제를 반복함 |
| **같은 모델** A/B 2회 합의 | 폐기 | 같은 모델의 일치는 사실성 보장이 아니고 비용·지연이 2배가 됨 |
| structurer 자유문장 rewrite 검색 | 폐기 | 원문 의미가 모델 해석으로 변형되고 책임 경계가 흐려짐 |
| E5 Top-N만 exact 검증 | 폐기 | exact recall이 dense recall에 종속됨 |
| 첫 버전 비동기 2단계 API | 보류 | 현재 규모는 단일 deadline fallback으로 계약이 충분함 |
| LLM 없이 모든 구조화 | 폐기 | 임의 자연어의 좌우·부정·복합 조건을 정규식 열거 없이 해석하기 어려움 |
| **결과 집합 기반 되묻기** | 폐기 | 작가 워크플로우에 마찰을 더한다. 결과 구성으로 해결한다 (DCR-007) |
| **중립 자세 선호 가중치** | 폐기 | §7.2 대표 집합이 그 자리를 채웠다. 동적 포즈 역효과도 함께 사라진다 (DCR-007) |
| farthest-first 다양화 | 폐기 | 극단값은 대표가 아니다. 실측 커버율 25.8%로 무작위 수준 (DCR-006) |

> **다중 provider는 "A/B 합의"와 다르다.** 합의는 같은 질문을 두 번 물어 일치를 보는 것이고,
> provider 어댑터는 활성 모델을 하나만 두되 교체 가능하게 하는 것이다. 충돌하지 않는다.

---

## 3. 확정 아키텍처

```text
                         ┌─ raw query → E5 → 769 unit 전역 랭킹 ────────┐
사용자 query → normalize┤                                               │
                         └─ structurer 1회 → clause/DSL 제안 ─┐         │
                                                              ▼         ▼
                                        schema/span/coverage/anchor 검증
                                                          │
                  ┌───────────────────────────────────────┴────────────────┐
                  │                                                        │
          완전하고 검증 가능                                      실패/부분/timeout/
                  │                                              context·unsupported
       1,538 member 전역 measurement scan                                  │
                  │                                                        │
         ┌────────┴────────┐                                      raw E5 후보
         │                 │                                           │
      exact 있음       exact 없음                               exact_pose_claim=false
         │                 │
   대표 집합 구성      library gap + dense fallback
```

raw E5와 structurer는 별도 executor/semaphore에서 병렬로 시작한다. structurer capacity가
부족하거나 deadline을 넘겨도 E5 실행과 응답을 점유하지 않는다.

### 3.1 구성 요소별 책임

| 구성 요소 | 하는 일 | 하지 않는 일 |
|---|---|---|
| raw E5 | 원문과 모든 v3 passage의 의미 순위 | exact 판정, 좌우·부정 논리 판정 |
| FTS5 | lexical rank 보조 | dense unit 탈락, hard filter |
| structurer (LLM) | clause 분할과 허용 predicate 제안 | threshold 생성, pose 판정, 검색 rewrite |
| deterministic validator | 원문 coverage/span/anchor/schema 검증 | 의미 후보 생성 |
| measurement verifier | 1,538 member의 3값 사실 판정 | 자연어 해석 |
| **selector** | **exact 집합에서 대표 집합 구성** | 미검증 후보를 exact로 승격 |
| ranker | dense fallback의 순서 결정 | exact 판정 |

### 3.2 structurer 모델 계약

**provider는 어댑터다.** `src/vlm/client.py`의 기존 패턴(`BaseVLMClient` / 구현체 /
`build_vlm_client()` 팩토리)을 따른다. `CLAUDE.md` 규칙 2에 부합한다.

- `SEMANTIC_STRUCTURER_PROVIDER`: `gemini | openai | anthropic`
- **활성 structurer는 항상 하나.** uncached query당 1회 불변
- 모델 후보 (D4 bake-off로 선택, D8에서 동결):

  | 순위 | 후보 | 선택 근거 |
  |---|---|---|
  | 1 | `gemini-3.5-flash-lite` | thinking 기본 `minimal`, 은퇴 모델과 동가 |
  | 2 | `gemini-3.7-flash` | 최신 세대(수명 최장), 3.6보다 저렴. thinking 최저 `low` |
  | 3 | `gemini-3.6-flash` | 공식 대체 경로. 1·2가 eligibility floor 미달 시 |
  | 4 | **비-Gemini 헤지 후보 1개** | provider 경계를 넘는 교체 가능성 확보 |

- `SEMANTIC_STRUCTURER_MODEL`에 **정확한 버전 문자열** 을 명시한다.
  **`-latest` alias와 preview 모델은 금지** — alias는 hot-swap되어
  `structurer_model_id`의 재현성 주장을 거짓으로 만들고, 평가된 적 없는 모델이
  exact를 서빙하는 경로를 연다
- **`thinking_level`(또는 provider 등가 설정)을 명시 고정** 한다. 후보가 지원하는 최저값을
  기본으로 하고, eligibility floor 미달 시에만 한 단계 올린다. **미지정 금지** —
  미지정 시 기본 thinking이 §8.1의 deadline을 상시 초과해 exact 경로가 조용히 붕괴한다
- temperature: `0`
- 출력: JSON structured output (enum 제약)
- 자동 provider retry: 없음
- 검색 grounding, tool calling, candidate/BVH 전달: 없음

> `gemini-2.5-flash`는 **2026-10-16 은퇴** 예정이므로 사용하지 않는다.

structured output은 문법적으로 올바른 JSON만 보장하며 값의 의미 정확성은 보장하지 않는다.
서버의 결정적 validator와 measurement verifier가 필수인 이유다.

**schema 제약**: provider가 매우 크거나 깊게 중첩된 schema를 거부할 수 있다. §5.1 schema는
비재귀이고 `predicate_id` enum은 현재 57개(상한 150)이므로 여유가 있다. 다만 D4 착수 시
**각 후보 provider의 schema 수락 여부를 먼저 확인** 한다.

---

## 4. atom·document schema v3

### 4.1 사실 계층

| 계층 | key | 내용 | exact 근거 |
|---|---|---|---|
| unit observed | `semantic_unit_id` | mirror pair 공통 방향 중립 사실 | 보조 진단만 |
| member observed | `semantic_unit_id + pose_id` | concrete original/mirror의 좌우 보존 사실 | passage는 힌트, 연속 measurement가 최종 근거 |
| contextual | `source_clip_id` | action/style/prop/source 문맥 | 불가 |

context document가 exact verifier에 들어가는 경로는 없다.

### 4.2 DB schema 변경

`tagging_review` 스키마를 index DB로 승계한다. **신설은 아래 표시된 4개뿐이다.**

`semantic_atoms`:

- `pose_id` (nullable) — 승계
- `measure`, `measure_unit` (nullable) — 승계
- **`quantizer_version`** — 신설
- **`render_eligible`** — 신설
- **`mirror_validation_status`** — 신설
- `scope CHECK(scope IN ('unit','member'))` — 승계 + **CHECK 신설**
- `scope='unit'`이면 `pose_id IS NULL` / `scope='member'`이면 `pose_id IS NOT NULL`
- composite FK `(semantic_unit_id, pose_id) -> pose_semantic_members`

`semantic_text_documents`:

- `document_scope CHECK(document_scope IN ('unit','member'))`
- `pose_id` (nullable), 위와 동일한 CHECK
- composite FK `(semantic_unit_id, pose_id) -> pose_semantic_members`

source context도 검색할 unit에 붙이고 실제 `source_clip_id`는 provenance에 보존한다.
atom/document ID hash에는 `scope`, `pose_id`, renderer/quantizer version을 포함한다.
schema v2와 v3는 cache 및 runtime을 공유하지 않는다.

### 4.3 member atom materialization

1. 기존 `member_posecodes[*].observed_atoms`와 27개 measurement를 읽는다
2. original/mirror의 continuous measurement를 measurement별 mirror transform
   (left↔right 교환과 lateral 축 부호 반전)으로 비교한다
3. 두 member가 같은 안전 bucket이면 각 concrete side atom을 materialize한다
4. bucket 불일치 또는 transform 불일치면 raw atom은 진단용으로 보존하되
   `render_eligible=false`로 두고 양쪽 search document에서 해당 주장을 생략한다
5. 생략해도 continuous measurement는 보존하며 exact verifier는 각 member의 실제 값을 사용한다
6. build마다 `mirror_atom_report`와 round-trip invariant 결과를 남긴다

unit atom도 기존 것을 그대로 복사하지 않는다. 양 member의 `render_eligible=true` atom을
measurement별 mirror transform한 뒤 **교집합만** side-neutralize해 v3 unit atom으로 재생성한다.
한쪽 member에서 omitted된 사실은 unit claim에서도 omit한다.
original 한쪽만 canonical source로 사용하는 v2 방식을 폐기한다. 이 단계는 BVH를 다시 읽지 않는다.

#### 4.3.1 retention의 정의 — 확정

> **retention은 `(predicate, subject, bucket, polarity)` 집합의 mirror round-trip 일치율이다.**
> 원시 연속값 일치는 요구하지 않는다. 130°와 137°는 둘 다 `bent`이므로 같은 사실이다.
> 연속값은 verifier가 member별 실제 값을 그대로 쓰므로 document 주장의 안전성과 별개다.

#### 4.3.2 core family와 floor

> **core = registry predicate가 exact 근거로 사용하는 atom family.**
> 어떤 predicate도 근거로 쓰지 않는 family는 search document(dense 회수)에만 기여하므로
> exact gate 대상이 아니다.

목록과 판정은 `config/semantic_core_families.v3.json`이 단일 소스다.

| floor | 값 | 현재 실측 |
|---|---:|---:|
| side-specific atom retention 전체 | ≥ 0.95 | **1.00** |
| core family별 retention | ≥ 0.90 | **각각 1.00** |
| non-core family retention | 없음 | 1.00 |

**mirror 수정 후 전 family가 100%이므로 floor는 자명하게 충족된다.**
그러나 규칙은 원칙으로 유지한다 — 비대칭 rest skeleton을 가진 새 소스가 들어오면 다시 필요하다.

`render_eligible=false` 생략 경로는 **현재 데이터에서 0건이 기대값** 이다.
D2b build report의 omission 수가 0이 아니면 새 소스에 같은 종류의 문제가 있다는 신호다.

coverage floor를 못 맞추면 모든 atom을 생략해 invariant를 통과시키지 못하도록 D2에서 중단한다.

### 4.4 기본 document set과 cap

| document type | 개수 상한 | scope | evidence |
|---|---:|---|---|
| `observed_unit_summary` ko/en | unit당 2 | unit | observed |
| `observed_member_summary` ko/en | member당 2 | member | observed |
| `canonical_context` | unit당 최대 2 | unit | contextual |
| `source_context` | unit당 최대 1 | unit | contextual |

**769 unit 전부가 정확히 2 member를 갖는다.** 따라서 unit당 문서는
`unit 2 + member 4 + canonical_context ≤2 + source_context ≤1 = 최대 9`이고,
index 규모는 **약 6,921 문서** (현재 3,449의 약 2배)다. D5에 embedding 재생성 소요를 실측해 기록한다.

모든 unit에 같은 cap을 적용하고, document 수가 많은 unit이 유리하지 않도록 rank 집계는
unit/document-type별 max score를 사용한다. member passage의 `pose_id`는 dense fallback
힌트일 뿐이며 exact concrete member는 verifier만 선택한다.

renderer 원칙:

- LLM caption과 자동 paraphrase를 저장하지 않음
- versioned atom→문장 template으로 결정적 렌더
- 부위 순서: torso → left arm → right arm → left leg → right leg → global
- `torso_length_bvh_units`는 semantic fact에서 제외
- dead-zone/boundary 사실은 주장하지 않고 coverage/omission report에 기록

### 4.5 renderer ablation

전체 v3 index를 확정하기 전에 같은 corpus에서 다음을 비교한다.

| variant | passage 구성 |
|---|---|
| T0 | 현재 짧은 방향 중립 문서 |
| T1 | 전체 방향 중립 composite |
| T2 | T1 + member 좌우 summary |
| T3 | T2 + 신체 부위별 chunk |

context 문서와 unit cap은 네 variant에서 동일하게 유지한다. raw-E5-only unit P@10/nDCG@20,
member directional P@20, contextual R@50, text collision을 비교해 T2/T3 중 더 단순하면서 실제
사용자 corpus 성능이 좋은 안을 고정한다. 개선이 없으면 structurer 연결로 넘어가지 않고
renderer를 다시 설계한다.

**여기서 `왼쪽 손을 든 포즈` 실험을 재측정한다.** v3.2 §1이 인용한 bi-encoder 관측은
좌우 정보가 없는 passage에서 잰 것이라 결론을 지지하지 못했다. document v3 이후 값으로 교체한다.

---

## 5. query-understanding schema

### 5.1 비재귀 clause schema

재귀 AST 대신 원문을 순서대로 완전히 덮는 clause 배열을 사용한다.

```json
{
  "schema_version": 3,
  "clauses": [
    {
      "clause_id": "c1",
      "start": 0,
      "end": 5,
      "text": "왼손 들기",
      "kind": "verifiable",
      "operator": "all",
      "terms": [
        {
          "predicate_id": "left_wrist.above_shoulder",
          "evidence_start": 0,
          "evidence_end": 5,
          "evidence_text": "왼손 들기"
        }
      ]
    }
  ],
  "model_confidence": "high"
}
```

허용값과 제한:

- `kind`: `verifiable | context | unsupported | boilerplate`
- verifiable `operator`: `all | any | exactly_one | none`
- top-level clause 결합: 모두 `all`
- clause 최대 8개, 전체 term 최대 8개
- `predicate_id`: server registry enum만 허용 (atomic 또는 compound)
- offset: 공백 정규화된 원문의 Unicode code point 기준, `end`는 exclusive
- context/unsupported/boilerplate의 `operator=null`, `terms=[]`
- `should`, `preferred`, 자유 `semantic_text`, 숫자 threshold 필드 없음

예시 의미: `all` 모든 term이 참 · `any` 하나 이상 참 · `exactly_one` 정확히 하나만 참 ·
`none` 모든 term이 거짓.

임의 중첩 논리를 지원하지 않는다. 비중첩 schema로 표현하지 못한 문장은 `unsupported`를 포함해
검색 후보는 반환하되 exact를 금지한다.

### 5.2 deterministic validation

서버는 structurer 응답에 대해 다음을 순서대로 검사한다.

1. clause가 normalized query의 첫 code point부터 끝까지 순서대로, gap/overlap 없이 덮는다
2. `start:end` substring과 `text`, term evidence substring과 `evidence_text`가 정확히 일치한다
3. term evidence는 반드시 parent clause 안에 있어야 하며 cross-clause evidence를 금지한다
4. `boilerplate`는 서버 allowlist의 검색 요청 표현만 허용한다
5. `verifiable` clause는 허용 operator와 term을 하나 이상 가진다
6. 모든 predicate ID가 frozen registry에 존재한다
7. verifiable clause의 evidence **내부와 외부 모두**를 versioned lexer(`kiwipiepy`, 버전 핀)로
   분해한다. 공백·조사·활용 어미 같은 function word 외의 모든 content span은 허용된
   side/body/relation/polarity/cardinality/operator/idiom anchor여야 하며 미등록 content가
   하나라도 있으면 coverage 실패다
8. content span은 원칙적으로 term 하나에만 귀속한다. evidence overlap은 금지하며, 완전히 같은
   span이 server-owned side/cardinality expansion rule로 좌우 predicate 전체를 생성할 때만
   허용한다. 예를 들어 `팔`은 `any(left,right)`, `한쪽 팔`은 `exactly_one`, `양팔`은
   `all(left,right)`로 확장할 수 있다. 좌우 일부만 생성하거나 operator가 다르면 실패다
9. predicate가 가진 side, body part, relation, polarity, cardinality를 해당 term 자체의 anchor가
   **양방향으로** 허가하는지 검사한다. 원문에 없는 relation을 predicate가 추가하거나 원문의
   anchor를 predicate가 소비하지 못하면 실패다
10. 논리 접속 표현은 function word로 버리지 않고 operator anchor로 소비한다.
    `과/와/그리고/및 → all`, `또는/혹은/이나/거나 → any`, `중 하나/한쪽만 → exactly_one`,
    부정 → `none`을 양방향 검증한다. 단일 term의 default `all`과 server-owned side expansion을
    제외하면 원문 operator anchor 없이 operator를 만들 수 없다
11. 한 clause에 복수 logical scope나 중첩 connective가 있거나 단일 operator로 표현할 수 없는
    혼합 부정·수량 scope는 `unsupported`로 내린다
12. constraint를 결정적 한국어 문장으로 다시 렌더링해 trace에 저장한다

#### 5.2.1 compound predicate의 anchor 검증

관용 표현은 어휘적으로 원자적인데 기하학적으로 복합이다. 예: `허리에 손을 짚고`는
손 위치와 팔꿈치 굽힘을 함께 뜻하지만 "팔꿈치"라는 anchor가 원문에 없다.
rule 9를 그대로 적용하면 조합이 불가능하다.

> **compound predicate는 관용구 anchor 하나가 predicate 하나를 허가한다.**
> 그 predicate의 **내부 조건은 anchor 검증 대상이 아니다** — 서버가 소유하는 식이기 때문이다.
> registry의 `anchors_ko`에 등재된 표현만 이 경로를 탄다.

이는 rule 8의 server-owned side expansion과 같은 안전 패턴이며 축만 다르다.

#### 5.2.2 safety scanner의 역할

safety scanner는 검색 parser가 아니다. 실패해도 raw E5 후보를 필터링하거나 빈 결과를 만들지
않고 **exact eligibility만 거부**한다. `_OBSERVABLE_PATTERNS`를 다른 이름으로 되살리는 용도가
아니다.

다음 중 하나라도 있으면 query 전체가 exact-ineligible이다.

- term 0개
- 원문 coverage 불완전
- `context` 또는 `unsupported` clause 존재
- boilerplate allowlist 위반
- schema/span/operator/registry 검증 실패
- evidence 안팎에 미등록 content span이 남음
- content anchor가 term별로 단일 귀속되지 않음
- 승인되지 않은 evidence overlap 또는 불완전 bilateral expansion
- predicate의 side/body-part/relation/polarity/cardinality가 원문 anchor에 의해 양방향 허가되지 않음
- anchor와 clause operator scope가 불일치
- `model_confidence != high`

`model_confidence`는 exact를 낮출 수만 있고 high 자체가 exact 근거가 되지 않는다.

---

## 6. registry와 measurement verifier

### 6.1 registry

조합형 ID를 무한히 늘리지 않고 measurement와 relation profile을 묶은 predicate를 쓴다.

**atomic predicate** (단일 비교):

```json
{
  "predicate_id": "left_wrist.above_shoulder",
  "expression": {
    "measurement_id": "left_wrist_height_from_shoulder_torso_units",
    "comparator": "gt",
    "threshold": 0.10
  },
  "dead_zone": 0.0071,
  "scope": "member",
  "render_ko": "왼손이 어깨보다 위에 있음",
  "render_en": "the left hand is above the shoulder"
}
```

**compound predicate** (소규모 `all` conjunction):

```json
{
  "predicate_id": "left_hand.on_hip_akimbo",
  "expression": {"op": "all", "children": [
    {"measurement_id": "left_hand_to_hip_torso_units", "comparator": "lt", "threshold": 0.38},
    {"measurement_id": "left_elbow_flexion_deg",       "comparator": "lt", "threshold": 135}
  ]},
  "anchors_ko": ["짚", "얹"],
  "render_ko": "왼손을 허리에 짚음 (팔꿈치를 굽힘)",
  "corpus_evidence": {
    "query_ids": ["q_0031", "q_0088", "q_0142"],
    "provenance": ["artist_session", "artist_session", "team_user"]
  }
}
```

structurer는 `predicate_id`만 선택한다. registry의 measurement/comparator/threshold/dead-zone과
compound의 내부 식은 사람이 검토하고 평가로 동결한다.

registry 변경은 `query_understanding_version`과 `semantic_release_id`를 바꾸며
index/golden 영향 분석을 요구한다.

#### 6.1.1 compound predicate 생성 규칙

> **한국어 표현이 어휘적으로 원자적인데(한 단어·한 관용구) 기하학적으로 복합이면**
> compound predicate를 만든다.
> 표현을 anchor로 쪼개 각 predicate에 배분할 수 있으면 만들지 않는다.

| 표현 | 어휘 | 기하 | 판정 |
|---|---|---|---|
| 허리에 손을 **짚고** | 원자 | 복합 | **compound** |
| 왼손을 들고 팔꿈치를 굽혀 | 분해됨 | 복합 | atomic 조합 |
| 손이 골반 근처에 | 원자 | 원자 | atomic |

**투기 금지를 기계로 강제한다.** compound 항목은 `corpus_evidence`가 필수이며
빌드 게이트(§12.2)가 다음을 검사한다.

| 검사 | 기준 |
|---|---|
| `corpus_evidence.query_ids` | **≥ 3건** |
| 그 query들의 provenance | `artist_session` 또는 `team_user`만. `team_synthetic` 불가 |
| 인용된 query가 corpus에 실재하고 anchor를 포함하는지 | 전수 대조 |
| atomic predicate | corpus 근거 불요 (measurement 축에서 파생) |

근거 없는 compound가 하나라도 있으면 registry 빌드를 중단한다.
**D1 corpus 없이는 compound를 만들 수 없다 — 상상으로 넣는 경로가 물리적으로 막힌다.**

관용구가 재수집 corpus에서 등장 0이 되면 `deprecated`로 표시하고 다음 release에서 제거한다.

#### 6.1.2 enum 크기

현재 atomic registry는 **53개** 다(DCR-008로 `near_thigh`/`far_from_thigh` 좌우 4개 제거).
`torso_length_bvh_units`와 `hand_to_thigh`를 제외한 measurement를 모두 사용하며,
**match=0 predicate 0개 · 희소(<25) predicate 0개 · 좌우 비대칭 0건** 이다.

> **단일 `predicate_id` enum을 쓴다. 상한 150.**

분해 구조(`body_part` 12 + `relation` 17 = 29)는 enum이 작지만 서버가 조합 유효성을 따로
검증해야 하고 structurer가 존재하지 않는 조합을 만들 여지가 생긴다. 단일 enum은 그 경로를 없앤다.
좌우 확장은 §5.2 rule 8의 server-owned rule이 처리하므로 registry는 side-atomic으로 둔다.
상한 150을 넘기면 분해 구조를 재검토한다.

#### 6.1.3 dead-zone 기본값

> **`dead_zone` 기본값은 해당 measurement **IQR의 2%** 다.**

predicate별 override를 허용하되 override에는 사유를 적는다.

| ε 정책 | 전체 unknown 비율 |
|---|---:|
| 고정 (2° / 0.02) | 3.07% |
| **IQR의 2%** | **1.66%** |
| IQR의 5% | 4.25% |
| IQR의 10% | 8.18% |

measurement마다 IQR이 크게 다르므로 고정값은 어떤 축에서는 넓은 dead-zone이고 어떤 축에서는
사실상 0이다. 분산 스케일이 근거를 데이터에 둔다.

#### 6.1.4 lateral 축 부호 규칙

lateral measurement는 body-local **left→right** 축이다. **왼쪽이 음수, 오른쪽이 양수** 다
(왼손목 중앙 −0.45, 오른손목 +0.45).

> 따라서 "바깥쪽(outward)"은 **왼쪽에서 `lt`(−), 오른쪽에서 `gt`(+)** 이며 부호가 side에 종속된다.
> height·forward 축은 side와 무관하게 부호가 같다.

발목처럼 양·음 방향 threshold가 비대칭이면(`threshold`, `threshold − 0.30`) 그 값도 함께
반전해야 한다(`-(threshold − 0.30)`).

이 규칙을 어기면 좌우 predicate의 match 수가 어긋난다. mirror retention이 100%인 라이브러리에서
좌우가 다르면 **데이터가 아니라 registry 정의가 틀린 것** 이므로 §12.2의 대칭성 게이트가 잡는다.

#### 6.1.5 threshold 배치

> threshold는 분포가 **성긴 곳** 에 놓여야 한다. 봉우리 한가운데 있으면 측정 오차로
> match/violation이 뒤집히고 dead-zone에도 많이 걸린다.
>
> **단, 데이터가 말해 주는 것은 "경계가 싼 곳"이지 "경계가 옳은 곳"이 아니다.**

`scripts/eval_threshold_candidates_v3.py`가 각 predicate의 현재 T 주변
±(탐색폭 × IQR)를 훑어 dead-zone 최소 지점을 후보로 낸다. 동률이면 원래 값에 가장 가까운 쪽.
`between`은 경계 둘을 각각 옮긴다. match 수가 25% 넘게 변하면 `semantic_review_required`.

**2단계로 확정한다.**

- **Tier 1 (자동)**: 탐색폭 `0.05 × IQR`. `semantic_review_required=false`인 55개는 후보를
  그대로 채택한다. **dead-zone 29.4% 감소, 사람 검토 2건.** 이동량이 대부분 1~2° 또는
  0.03 torso-unit이라 의미가 바뀌지 않는다
- **Tier 2 (사람)**: 검수 시트에서 개별 판단한다. 실제 포즈를 값 순서로 늘어놓은
  `scripts/build_threshold_ladder_v3.py`의 사다리를 근거로 쓴다

**Tier 2 확정분**

| predicate | T | 근거 |
|---|---|---|
| `left_hand.near_hip` / `right_hand.near_hip` | **0.38** | 사다리 관찰. 0.31~0.37은 손이 골반 옆, 0.41부터 팔이 다른 일을 한다. 분포에 골짜기가 없으므로 근거는 "여기부터 자세가 달라 보인다" |
| `left_hand.near_head` / `right_hand.near_head` | **0.50** | 이전 0.45가 `Talking On A Cell Phone`(0.469)을 잘라냈다. 0.46~0.53에 전화·권투 가드·트럼펫이 몰려 있고 대부분 정당하다 |
| `left_hand.near_thigh` 계열 | **제거** | 손목↔**무릎** 거리라 "손을 허벅지에 댄"을 표현하지 못한다. `<0.30`의 88.9%가 앉은 자세이고, 무릎이 펴진 592개 중 `<0.60`은 4개뿐이다. 확충으로도 해결되지 않는다 (DCR-008 §1) |

**Tier 2 미결분은 없다.**

### 6.2 3값 진리표

atomic 결과는 `match | violation | unknown`이다.

| operator | 입력 | 결과 |
|---|---|---|
| `all` | 전부 match | match |
| `all` | 하나라도 violation | violation |
| `all` | violation 없음 + unknown 존재 | unknown |
| `any` | 하나라도 match | match |
| `any` | match 없음 + unknown 존재 | unknown |
| `any` | 전부 violation | violation |
| `exactly_one` | match 2개 이상 | violation |
| `exactly_one` | match 1개 + unknown 없음 | match |
| `exactly_one` | match 0~1개 + unknown 존재 | unknown |
| `exactly_one` | match 0개 + unknown 없음 | violation |
| `none` | 전부 violation | match |
| `none` | 하나라도 match | violation |
| `none` | match 없음 + unknown 존재 | unknown |

부정(`none`)에서도 unknown은 "거짓으로 확인됨"이 아니다.
compound predicate의 내부 식도 같은 진리표로 합성한다.

`unknown_member_count`는 atomic unknown 개수가 아니라 모든 clause를 진리표로 합성한
**member query state가 unknown인 member 수**다. 예를 들어 `any=[match, unknown]`은
aggregate match이므로 unknown member로 세지 않는다.

현재 실데이터 measurement null은 0이지만 dead-zone unknown은 발생한다.
missing-measurement와 threshold-boundary synthetic fixture를 operator별로 테스트한다.

### 6.3 전역 scan과 exact 판정

schema/coverage 검증이 완료된 query는 **1,538 member 전체**를 검사한다.
E5 점수는 검증 대상을 자르지 않는다.

`exact_pose_claim=true`의 필요조건은 다음 전부다.

1. `SEMANTIC_STRUCTURER_MODE=on`
2. `SEMANTIC_EXACT_CLAIM_ENABLED=1`
3. raw E5 dense channel이 적용됨(실행 또는 동일 index의 정상 cache hit)
4. structurer structured output 성공
5. clause/schema/span/coverage/anchor 검증 성공
6. verifiable term이 하나 이상 존재
7. context/unsupported clause 없음
8. **해당 member의** 모든 verifiable clause aggregate가 match
9. loaded release approval manifest의 `exact_claim_approved=true`
10. approval manifest의 네 ID, holdout run hash, loaded asset/policy hash가 모두 일치

#### 6.3.1 `unknown=0`을 요구하지 않는다

`exact_pose_claim`은 **candidate 단위 주장** 이다. 다른 member가 dead-zone unknown이어도
known-match candidate 자체의 exact는 성립한다.

실측 근거: registry 초안으로 1,538 member를 진리표 합성했을 때
**1-term query에서 `unknown_member_count=0`인 비율이 0.0%** 다(2,000회 시뮬레이션).
2-term 7.6%, 3-term 21.9%. 연속 분포에 임계값을 그으면 경계 근처에 항상 누군가 있으므로
**`unknown=0`은 구조적으로 도달 불가능하다.**

따라서:

- `unknown_member_count`는 계산·노출하되 **상태 이름을 좌우하지 않는다**
- `matching_set_known_complete`는 `unknown_member_count == 0`이며 대개 `false`다.
  정직한 값이고 UI 주 표시의 기준으로 쓰지 않는다
- `library_gap`은 `matching_member_count == 0`만으로 주장한다
- `exact_denial_reasons`의 `measurement_unknown`은 **candidate 단위 사유**로만 남기며
  query 전체를 exact-ineligible로 만들지 않는다

**안전선은 그대로다.** 반환되는 candidate는 여전히 자기 measurement가 match인 member뿐이다.
aggregate가 unknown인 member는 결과에 들어가지 않는다(§12.2 gate 유지).

structurer, E5 cosine, source filename, document atom만으로 exact를 부여하는 경로는 없다.
`false exact=0`은 수학적 보장이 아니라 development와 새 holdout에서 요구하는 승격 gate다.

---

## 7. retrieval과 결과 구성

### 7.1 raw dense는 항상 전역 실행

모든 비어 있지 않은 유효 cache-miss query는 공백만 정규화한 원문으로 `query:` prefix E5
embedding을 만든다. schema v3의 모든 dense document를 점수화하고 **769 unit 전부**에
unit score를 만든다. 동일 `semantic_index_build_id`의 정상 dense cache hit는 이 계산 결과만
재사용하며 parser/structurer 결과 cache로 dense channel을 대신할 수 없다.

폐기할 gate:

- `dense_candidate_depth`
- `contextual_only`에 따른 dense 제외
- parser intent 기반 document gate
- `_context_match` hard filter
- E5 전 `clarification_required` 조기 종료

FTS5는 raw lexical rank를 RRF에 추가할 수 있지만 dense unit을 탈락시키지 않는다.
structurer가 만든 rewrite나 back-render 문장은 dense input으로 사용하지 않는다.

### 7.2 exact lane — 적응적 대표 집합

#### 7.2.1 문제 정의

exact 후보는 **전부 똑같이 정답** 이다. 예: `왼손이 어깨 위` 조건을 만족하는 member가 264개다.
그러므로 "관련도 순 정렬"이 성립하지 않는다. 정렬할 관련도 차이가 없다.

> **바른 목적함수: 보여준 결과 중에 작가가 원하던 게 하나라도 있을 확률을 최대화한다.**
>
> 이건 랭킹 문제가 아니라 **대표 표본(cover) 문제** 다.

실측이 이를 뒷받침한다(264개 기준, 지정 축 제외 24축, IQR 표준화, 후보 간 거리 중앙 0.89):

| 전략 | 커버율 (r=0.50) |
|---|---:|
| 임의(앞에서 5개) | **6.8%** — 무작위보다 4배 나쁘다 |
| 무작위 5개 | 26.8% |
| farthest-first | 25.8% — **극단값은 대표가 아니다** |
| action domain 층화 | 42.8% |
| **greedy coverage** | **53.4%** |

#### 7.2.2 개수는 데이터가 정한다

완전 커버(r-net)를 요구하면 대표 수가 폭발한다(r=0.5에서 중앙 89개). 24축 공간에 1,538개가
흩어져 있어 전부 덮으려면 중심이 아주 많이 필요하다. greedy의 한계이득 급감을 이용한다.

93개 쿼리(단일 term 57 + 2-term 36) 실측:

| 정지 규칙 | 중앙 | p75 | p90 | 최대 | 30개 초과 |
|---|---:|---:|---:|---:|---:|
| **커버 50% 도달** | **5** | **8** | **12** | 37 | **1.1%** |
| 커버 70% 도달 | 20 | 30 | 40 | 89 | 23.7% |
| 한계이득 <3% 정지 | 4 | 5 | 8 | 15 | 0.0% |
| 도메인당 1개 | 14 | 14 | 14 | 14 | 0.0% |

70%는 화면에 못 담고, 한계이득 정지는 고정 5개와 다를 게 없고, 도메인당 1개는
후보가 9개인 쿼리에도 14를 시도해 적응하지 않는다.

#### 7.2.3 확정 규칙

```
1. 배제      quarantine · mirror unit 중복 · 근접 중복(정규화 거리 <0.15)
2. 거리      쿼리가 지정한 measurement 축을 제외한 나머지 축, IQR 표준화, 평균 |차이|
3. 선택      greedy coverage (반경 r = 0.50)
4. 도메인 보정 이득이 최대치의 85% 이상인 후보들 중 아직 안 나온 action_domain 우선
5. 정지      누적 커버 50% 도달 시
6. 상한      24개 (초과 시 greedy 순서대로 잘라내고 results_truncated=true)
7. 동률      stable semantic_unit_id
```

**도메인 보정은 거의 공짜다.**

| | 대표 중앙 | 도메인 다양성 | 커버율 | 고유 clip 비율 |
|---|---:|---:|---:|---:|
| greedy 단독 | 5 | 3.0 | 54% | 0.98 |
| **greedy + 도메인 우선** | **6** | **5.0** | 54% | 0.98 |

대표 1개 늘고 커버율은 그대로인데 도메인 다양성이 67% 늘어난다.
고유 clip 비율 0.98이므로 **"같은 clip ≤2" 같은 별도 제약은 이 규칙에 흡수되어 불필요하다.**

unit당 concrete member 하나만 반환한다. side-specific query는 실제로 조건을 만족한
original/mirror를 선택하고, 대칭 동률이면 `side_resolution=ambiguous_symmetric`를 표시한다.

#### 7.2.4 중립 자세 가중치 — 채택하지 않는다

한 관절만 명시한 쿼리에서 나머지 관절이 중립(기본 자세)에 가까운 것을 선호하자는 안은
**채택하지 않는다.** 이유는 "나쁘다"가 아니라 **"그 자리가 이미 채워졌다"** 이다.
§7.2.3의 대표 집합이 "무엇을 보여줄지"의 근거를 제공하므로 중립도가 메우려던 공백이 사라졌고,
동적 포즈를 원하는 작가에게 역효과라는 문제도 함께 사라진다.

계산 코드는 만들지 않는다. 선택 로그(§12.4)가 쌓이면 재검토할 수 있다.

### 7.3 semantic fallback lane

구조화 실패, 부분 coverage, context/unsupported, provider 장애에서는 hard constraint filter를
적용하지 않는다. raw E5/FTS RRF Top-K를 반환하고 모든 후보를 exact false로 둔다.

> **fallback lane에는 대표 집합 규칙을 적용하지 않는다.**
> 대표 집합은 *후보가 전부 동등하게 정답일 때* 를 위한 장치다. fallback에서는 dense 점수가
> 실제로 의미 있는 순서를 만들므로 대표화가 오히려 정보를 버린다.
> **근접 중복 배제와 mirror unit 중복 배제는 양쪽 lane 모두 적용한다.**

- best member document가 이긴 unit: 해당 `pose_id`를 preview 힌트로 사용
- unit document가 이긴 unit: canonical original을 preview로 사용
- 어느 경우든 `candidate_basis`와 `verification_state`를 노출

일부 인식된 term을 진단 정보나 soft annotation으로 표시할 수는 있지만, partial structure를 hard
filter 또는 exact 근거로 사용하지 않는다.

---

## 8. API 계약

### 8.1 단일 동기 latency 계약

```text
raw E5 ──────────────────────┐
structurer 최대 1,200ms ─────┤ 병렬
                             ▼
                    단일 JSON 응답
```

- `SEMANTIC_STRUCTURER_TIMEOUT_MS`: 기본 1,200ms
- inference endpoint 목표 budget: 1,500ms
- BFF 권장 timeout: 2,000ms
- deadline 후 structurer를 기다리거나 자동 재시도하지 않음
- timeout/off/busy/error: HTTP 200 + raw dense 후보 + exact false
- background 갱신, polling, SSE: v3 범위 밖

수치는 development 실측으로 holdout 전에 한 번 동결한다. 단일 응답과 timeout fallback 구조 자체는
v3에서 바꾸지 않는다.

### 8.2 입력 계약

- 공백 정규화 후 1~500자
- 공백-only, 제어 문자, 500자 초과는 기존 validation error
- 201~500자는 E5 검색은 실행하지만 structurer는 건너뛰고 `understanding.status=too_long`
  (컷오프는 D1 corpus의 길이 분포로 재확정한다)
- **`top_k`는 반환 개수의 상한(cap)이다.** 실제 개수는 §7.2.3이 정한다.
  생략 시 서버 기본 상한 24. BFF가 `top_k=5`를 계속 보내도 깨지지 않으며 대표 5개가 나온다
- `view_hint`는 preview decoration 전용이며 retrieval/cache core key에 넣지 않음
- `refine=false` 불변

### 8.3 clean v3 응답

동일 `POST /semantic-search`를 사용하되 `contract_version=3`으로 clean break한다.

```json
{
  "contract_version": 3,
  "status": "exact_matches",
  "exact_match_status": "exact_verified",
  "semantic_index_build_id": "sha256:...",
  "query_understanding_version": "sha256:...",
  "structurer_model_id": "sha256:...",
  "semantic_release_id": "sha256:...",
  "understanding": {
    "status": "complete",
    "coverage_status": "complete",
    "exact_eligible": true,
    "cache_hit": false,
    "latency_ms": 312,
    "clauses": [],
    "verified_interpretation_ko": "왼손이 어깨보다 위에 있음"
  },
  "retrieval": {
    "raw_dense_applied": true,
    "raw_dense_executed": true,
    "units_scored": 769,
    "cache_hit": false,
    "latency_ms": 11
  },
  "verification": {
    "scope": "all_members",
    "evaluated_member_count": 1538,
    "matching_member_count": 264,
    "unknown_member_count": 23,
    "matching_set_known_complete": false,
    "latency_ms": 4
  },
  "selection": {
    "strategy": "coverage_greedy_domain_aware",
    "radius": 0.50,
    "coverage_target": 0.50,
    "coverage_achieved": 0.54,
    "candidate_count": 264,
    "representative_count": 12,
    "cap": 24
  },
  "exact_denial_reasons": [],
  "returned_count": 12,
  "results_truncated": false,
  "results": [
    {"pose_id": "example_pose", "exact_pose_claim": true,
     "represents": 31, "action_domain": "dance"}
  ]
}
```

`understanding.status`: `complete | partial | disabled | timeout | invalid | provider_error |
busy | circuit_open | too_long`

`coverage_status`: `complete | partial | not_evaluated`

`understanding.exact_eligible`은 구조·coverage만 통과했다는 뜻이며, operational switch나 실제
matching member 존재 여부를 뜻하지 않는다. UI의 "조건 검증됨" 표시는 원문 전체가 맞다는 포괄적
주장 대신 `verified_interpretation_ko`에 적힌 기하 조건이 확인됐다는 의미로 사용한다.

`exact_denial_reasons`의 stable code:

```
structurer_off · exact_claim_disabled · structurer_timeout · structurer_busy
provider_error · query_too_long_for_structurer · schema_invalid · coverage_incomplete
residual_content · anchor_unmapped · anchor_predicate_mismatch · evidence_overlap_invalid
operator_scope_mismatch · context_present · unsupported_present · no_verifiable_terms
measurement_unknown · no_matching_member · release_not_approved · release_identity_mismatch
model_retired · model_unavailable
```

candidate 필수 진단:

- `candidate_basis=verified_geometry|dense_pose_document|source_context`
- `verification_state=exact|not_evaluated|violation|unknown`
- `side_resolution=not_requested|unique|ambiguous_symmetric`
- `evidence_state`
- `semantic_unit_id`, `pose_id`, `variant_kind`
- `exact_pose_claim`
- `constraint_results`
- **`represents`** — 이 대표가 덮는 후보 수
- **`action_domain`** — UI가 그룹으로 묶을 수 있게 (강제 아님)
- `refine_allowed=false`

### 8.4 status와 lane

| 조건 | `status` | `exact_match_status` | 결과 lane |
|---|---|---|---|
| 완전 구조화 + matching ≥ 1 + 승인된 exact 활성 | `exact_matches` | `exact_verified` | 대표 집합 |
| 완전 구조화 + matching = 0 + 승인된 exact 활성 | `library_gap` | `no_exact_match` | dense fallback만 |
| 구조화 불완전/장애/context | `semantic_candidates` | `not_verifiable` | dense fallback만 |
| structurer off/shadow 또는 exact claim 비활성 | `semantic_candidates` | `not_verifiable` | dense fallback만 |

**`exact_partial`은 삭제됐다** (§6.3.1). `unknown_member_count`는 `verification` 블록에
그대로 노출되며 상태 이름을 좌우하지 않는다.

`returned_count`와 `results_truncated`는 실제 반환량과 상한 도달 여부를 나타낸다.
`candidate_count`와 `representative_count`를 분리해 노출하므로 "264개 중 12종류"가
정직하게 표현된다.

exact와 fallback을 한 결과 배열에 섞지 않는다. 유효 query에 `clarification_required` 빈 결과를
반환하지 않으며, **`clarification_question`을 semantic 경로에서 사용하지 않는다.**

HTTP 정책:

- structurer timeout/429/5xx/schema invalid/circuit open: HTTP 200 fallback
- local E5/index not-ready 또는 E5 admission busy: 503
- query validation 실패: 기존 4xx

### 8.5 같은 URL의 무중단 contract 전환

request shape은 유지되지만 response v3는 현재 BFF의 v2 enum/field와 비호환이다. 같은 URL을
유지하므로 배포 순서를 다음으로 고정한다.

1. BFF에 dual parser와 v3 status mapping을 먼저 배포한다. 현재 legacy 응답에는
   `contract_version`이 없으므로 `contract_version==3`이면 v3, field absent 또는
   `service_version==1`이면 legacy v2로 판별한다
2. BFF가 아직 v2 inference를 호출한 상태에서 v2/v3 contract fixture를 모두 통과시킨다
3. `bff_contract_compatibility_approved=true`인 뒤에만 inference current pointer를 v3로 바꾼다
4. 최초 v3 production의 operational rollback은 같은 승인 v3 bundle에서
   `SEMANTIC_STRUCTURER_MODE=off`, `SEMANTIC_EXACT_CLAIM_ENABLED=0`으로 dense-only 전환한다.
   v3 dense/index 자체가 불능이면 BFF feature flag로 semantic endpoint를 비활성화한다.
   `production_ready=false`인 v2 reference build를 rollback으로 승격하지 않는다
5. v2 response parser 제거는 별도 rollback window 종료 후에만 한다

구 BFF가 남아 있는 상태에서 inference v3를 먼저 전환하지 않는다. 이 compatibility approval은
atomic release preflight의 필수 입력이다.

---

## 9. 장애·비용·보안

### 9.1 운용 스위치

```text
SEMANTIC_STRUCTURER_MODE=off|shadow|on
SEMANTIC_EXACT_CLAIM_ENABLED=0|1
SEMANTIC_STRUCTURER_PROVIDER=gemini|openai|anthropic
SEMANTIC_STRUCTURER_MODEL=<정확한 버전 문자열>          # latest alias·preview 금지
SEMANTIC_STRUCTURER_THINKING_LEVEL=minimal|low|medium|high   # 미지정 금지
SEMANTIC_STRUCTURER_SHADOW_PROVIDER=<헤지 provider 또는 빈 값>
SEMANTIC_STRUCTURER_SHADOW_MODEL=<후보 모델 또는 빈 값>
SEMANTIC_STRUCTURER_SHADOW_SAMPLE_RATE=0.0~1.0
SEMANTIC_STRUCTURER_TIMEOUT_MS=1200
SEMANTIC_STRUCTURER_MAX_CONCURRENCY=4
SEMANTIC_STRUCTURER_SHADOW_MAX_CONCURRENCY=2
SEMANTIC_DENSE_MAX_CONCURRENCY=2
SEMANTIC_DENSE_ACQUIRE_TIMEOUT_MS=250
```

- `off`: structurer 호출 없음, dense-only
- `shadow`: 결과를 검증·기록하지만 응답 exact에는 사용하지 않음
- `on`: frozen release와 gate 통과 시에만 verifier 결과를 응답에 반영
- exact switch가 0이면 mode가 on이어도 exact false
- mode와 exact switch가 모두 켜져도 loaded approval manifest의 `exact_claim_approved=true`,
  `semantic_release_id`, holdout run hash가 일치하지 않으면 exact false로 fail-closed

**shadow 이중 모델**: 후보 provider/모델을 운영 트래픽에서 미리 검증한다.
shadow 호출은 응답 latency에 포함하지 않고, **§0.6의 "uncached query당 1회" 예산에도
포함하지 않는다.** 별도 예산·별도 concurrency·별도 circuit breaker를 쓰며 shadow 실패가
활성 경로에 영향을 주지 않는다.

### 9.2 동시성·circuit breaker

validation 후 dense admission을 먼저 얻고, 성공한 요청만 raw E5와 structurer를 병렬 시작한다.
E5와 structurer는 semaphore/executor를 완전히 분리하고 permit을 서로 점유하지 않는다.

`/semantic-search`는 sync 핸들러이므로 async 전환 없이 `ThreadPoolExecutor`로 병렬화한다.

- dense admission: `SEMANTIC_DENSE_MAX_CONCURRENCY`, acquire timeout 기본 250ms
- dense acquire 실패: `DenseBusy`로 HTTP 503, structurer 호출 시작 금지
- structurer admission 실패: `understanding.status=busy`, HTTP 200 dense fallback
- "raw dense 100%" gate의 분모: validation 통과 후 dense admission에 성공한 요청

structurer capacity가 차면 queue를 늘리지 않고 즉시 fallback으로 간다.

circuit open 조건:

- 연속 실패 5회, 또는
- 최근 20회 중 최소 10회 관측 후 timeout/429/5xx/schema-invalid 비율 50% 이상
- 인증·권한 오류, **모델 은퇴(404/model_not_found)** 는 즉시 open하고 운영 경고

open 30초 후 half-open probe 1회를 허용한다. 재실패 시 60/120/240/300초로 늘리고 최대 5분으로
제한한다. breaker open은 semantic 503 사유가 아니다.

### 9.3 cache와 single-flight

| cache | key | 저장 정책 |
|---|---|---|
| dense | `semantic_index_build_id + normalized query + top_k` | 정상 결과만 versioned LRU/TTL |
| understanding | `query hash + query_understanding_version + structurer_model_id` | schema-valid 정상 구조화만 LRU/TTL |
| structurer prompt prefix | `query_understanding_version` | provider context cache, TTL은 운영값 |
| undecorated core result | `semantic_release_id + approval_manifest_sha256 + query + top_k + mode + exact_claim_enabled` | transient fallback은 장기 저장 금지 |

- timeout, 429, 5xx, invalid JSON/schema invalid: **cache하지 않음**
- "지원 가능한 predicate 없음"이라는 정상 구조화 결과: 성공 cache 가능
- 같은 query 동시 요청: single-flight로 합쳐 structurer 1회
- `view_hint`는 core cache key에서 제외하고 cache read 뒤 decoration 단계에서 붙임
- fallback 최종 응답이 provider 복구 후에도 고착되지 않게 함

고정 프롬프트 접두부(시스템 + registry + anchor lexicon)는 매 호출 동일하므로 provider의
context caching을 적용한다. 캐시 키를 `query_understanding_version`으로 두면
프롬프트·schema·registry가 바뀔 때 무효화가 자동으로 맞는다.

### 9.4 rate limit과 로그

- BFF 권장: 인증 사용자당 20 req/min, burst 5, submit debounce
- inference: bounded structurer concurrency로 비용 상한 설정
- API key는 inference 서버 환경변수에만 보관
- raw query를 access/error log에 남기지 않음
- release-versioned salted query hash, 길이, status, latency, call count만 운영 로그에 기록

---

## 10. 유지·퇴역·재작성·재생성

### 10.1 유지

- 1,538 BVH와 769 mirror lineage
- inventory, provenance, source/action mapping 원장
- member별 27개 연속 measurement와 side-specific `observed_atoms`
- `evaluate_expression`의 3값 원칙과 mirror selector 원리
- pinned multilingual E5 ONNX, prefix/pooling/normalization 계약
- unit/member/context evidence 분리 원칙
- `POST /semantic-search` 경계와 semantic refine 금지
- `/healthz` 내부 semantic 상태
- geometry `/analyze`, kNN, refine 전체

### 10.2 production 경로에서 논리적 퇴역

구현 승인 후 아래를 v3 production 경로에서 제외한다. 즉시 물리 삭제하지 않는다.

- `_OBSERVABLE_PATTERNS`
- 정규식 `parse_semantic_query`
- inline `contextual_rules`
- `_context_match` action-key hard filter
- E5 전 `clarification_required` 종료
- `contextual_only`/parser intent 기반 dense gate
- 조합형 constraint ID를 계속 늘리는 방식
- parser/retrieval profile v1
- build `217d…0196`의 production 승격
- **build `ae3b…e895`** — mirror 결함 자산. 참조 금지
- golden v2 PASS를 자연어 일반화 품질 증거로 사용하는 설명

### 10.3 재작성할 runtime 경계

**repo `CLAUDE.md` 규칙 2에 따라 새 파일 + 팩토리 선택으로 공존시킨다.**
`src/semantic_search.py`(v1)는 §11.3 tripwire가 frozen v2 predicate profile로 판정하는 데
필요하므로 **수정하지 않는다.**

```
src/semantic_search.py            v1 — 유지. tripwire가 import
src/semantic_search_v3.py         v3 런타임 (동일 진입점 시그니처)
src/semantic_query_understanding.py  structurer 호출 + 결정적 validator
src/semantic_constraints.py       registry 로더 + 3값 verifier
src/semantic_selector.py          대표 집합 구성 (§7.2.3)
src/semantic_lexer_ko.py          한국어 lexer 래퍼 (kiwipiepy, 버전 핀)
src/semantic_documents_v3.py      renderer v3
```

선택은 `SEMANTIC_IMPL=v1|v3` env 하나. **`api/app.py`와 `semantic_service.py`에 v1/v3 if 분기를
넣지 않는다.** 공유 로직(`semantic_embedding`, `semantic_index` 읽기, 3값 판정 원리)은
복제하지 않고 import한다(규칙 3). v3 승격 후 v1 삭제는 별도 PR이며, 그때까지 어느 쪽이 실행
기준인지 `docs/DECISIONS.md`에 명시한다(규칙 5).

### 10.4 재생성

- `config/semantic_observable_predicates.v3.json`
- structurer prompt + structured-output schema v3
- anchor / boilerplate / idiom lexicon (D1 corpus에서 유도)
- query-understanding/retrieval/document/verification profile v3
- member atom/document schema-v3 materialization과 mirror report
- `search_documents.v3.jsonl`
- semantic DB/index build v3
- golden v3 intent/expected artifact
- evaluation JSON/CSV/dashboard
- API/OpenAPI와 API/BFF handoff 문서
- atomic release bundle과 current pointer

golden v2와 build v1은 삭제하지 않고 immutable archive/tripwire로 둔다.
BVH/geometry는 재생성하지 않는다.

### 10.5 다음 버전으로 미룬 것

**스켈레톤 추출 실패 시 의미 검색 폴백** — v3.1 승격 이후 별도 DCR.

`src/pipeline.py`에 `soft_fallback`(prior 유지, confidence low)과
`hard_fallback`(`candidates=[]`, `refine_allowed=False`, `skeleton=None`)이 이미 있고,
`hard_fallback`에서도 VLM 태그는 살아 있어 텍스트 검색으로 넘길 재료가 있다.

전제 조건:

1. `hard_fallback` 발생률 실측 — **지금 미리 해둔다**(§12.4)
2. `match_source=semantic_vlm_fallback` source 신설, `exact_pose_claim=false` 항상,
   `refine_allowed=false`
3. `route=skip` / `route=bust` 제외 게이트 — 얼굴·흉상 컷에 전신 포즈 후보는 방해다
4. `CutResult`에 lane 필드 추가 + BFF/UI 합의

가장 큰 위험은 작가가 결과를 "포즈를 맞춘 것"이 아니라 "VLM이 짐작한 텍스트로 찾은 것"인 줄
모르는 것이다. false-exact와 같은 종류의 신뢰 붕괴이므로 lane 표시가 필수다.

---

## 11. query corpus, golden, holdout

### 11.1 실제 query corpus

vocabulary와 renderer를 고정하기 전에 작가 인터뷰·실사용·팀 테스트에서 실제 표현을 모은다.

**출처 등급을 분리한다.**

| provenance | 허용 용도 | 금지 용도 |
|---|---|---|
| `artist_session` · `team_user` | 전부 (renderer 선택, golden, holdout, 승격 gate, compound 근거) | ─ |
| `team_synthetic` | **D3 renderer ablation 한정** | golden 정답, holdout, 승격 gate, compound 근거 |
| LLM 생성 | 없음 | 전부 |

- vocabulary 초안/renderer 선택 전 최소 100개 (`team_synthetic` 포함 가능)
- **golden·holdout·승격 gate의 100개/200개 하한은 `artist_session|team_user`만 세어 충족**
- 개인정보·프로젝트 식별 정보 제거
- provenance, 수집일, 언어를 필수 필드로
- `query_family_id` 단위 split로 near-paraphrase 누수 방지
- 좌우/부정 최소쌍은 같은 split에 함께 배치
- eval runner가 등급별 개수를 report에 찍는다 — 등급을 섞어 gate를 통과시키면 즉시 드러난다

**수집 시 함께 받을 것**: "무엇을 검색했나"만이 아니라 **"그래서 어느 걸 골랐나"**.
이것이 §7.2의 대표 집합 파라미터를 실증으로 검증할 유일한 근거다.

현재 대화에서 나온 `왼쪽 손을 든 포즈`, 조합 자세, 전통춤은 이미 공개됐으므로 development에만 넣는다.

### 11.2 golden v3 artifact 분리

```text
golden_intents.v3.jsonl
  원문, query_family_id, 사람이 승인한 clause/span/조건,
  verifiable/context/unsupported 판정, provenance

golden_expected.<golden_dataset_id>.json
  독립 predicate/SQL로 materialize한 snapshot-specific member/unit 정답 집합
```

ground truth 생성기는 runtime parser/structurer/constraint evaluator를 import하지 않는다.
intent는 build-independent, expected set은 snapshot-specific이다.

```text
golden_dataset_id = sha256(
  canonical intent + qrels bytes
  + independent GT rule/profile bytes
  + pose_library_version
)
```

`golden_dataset_id`를 manifest와 모든 eval run에 기록해 query, 사람 판정 규칙, threshold 중 하나가
바뀌면 서로 다른 dataset으로 취급한다.

필수 development 묶음:

- 자유 paraphrase와 구어체
- 왼쪽/오른쪽 최소쌍
- 언급하지 않은 반대쪽을 추측하지 않는 query
- 긍정/부정 최소쌍
- `all/any/exactly_one/none`
- **compound predicate 관용구** (anchor 인식과 오탐)
- 행동·소품·문화·감정 context
- vocabulary 밖 open-set
- structurer disabled/timeout/invalid/partial/busy
- measurement가 빠진 synthetic unknown
- mirror round-trip metamorphic fixture
- **한 관절만 명시한 쿼리** — 대표 집합의 `representative_count` 분포와 커버율을 잰다.
  정답을 특정 정렬로 고정하지 않는다

### 11.3 golden v2 tripwire

기존 45개, builder, expected label은 immutable legacy artifact로 보존하고 v3 holdout으로 재사용하지
않는다. frozen label을 재산정하거나 수정하지 않는다.
동일 자산 비교를 위해 1,232 snapshot을 v3 renderer/runtime로 만든 `v3_shadow_1232`에서 검사한다.

> **실현 가능함을 확인했다**: 1,232 member ⊂ 1,538 member (strict subset, 1,232-only 0개),
> 616 unit ⊂ 769 unit.

- 31 exact legacy projection: frozen v2 predicate profile로 frozen GT 전체 matching set 완전 일치
- 31 exact v3 profile: `v3 match ⊆ legacy match`; 차이는 선언된 dead-zone 안의
  `legacy match|violation → v3 unknown`만 허용하고 반대 방향 match 생성은 금지
- 4 context: recall/evidence purity 비회귀
- 7 no-exact: false exact 0
- 3 legacy clarification: `non-empty dense candidates + exact=false`라는 계약 전환 assertion

즉 42개는 frozen legacy compatibility와 v3의 보수적 dead-zone migration을 함께 검사하고,
3개는 의도한 response migration을 검사한다. 기존 evaluator는 legacy로 동결하고 v3 evaluator는
`pose_library_version`과 독립 GT를 사용한다.

### 11.4 밀봉 holdout

- 전부 신규 실제 사용자 문장 (`artist_session|team_user`)
- plaintext는 개발 브랜치에 두지 않고 fingerprint/count/class 분포 manifest만 저장
- 한 frozen `semantic_release_id`에 공식 실행 1회
- 동일 bits의 인프라 실패만 1회 재시도 허용
- 결과를 본 뒤 model/prompt/schema/registry/threshold/renderer/ranking/selection/deadline을
  바꾸면 전량 폐기
- 실패한 holdout을 일부 development로 옮기고 나머지를 재사용하지 않음
- 새 query family로 새 holdout 작성

> **해석 규칙**: holdout 결과의 차이가 `structurer_noise_floor`(§12.3) 이내면
> "차이 없음"으로 읽는다. LLM은 랜덤하므로 노이즈 바닥 없이는 1회 실행이 무엇을
> 증명하는지 말할 수 없다.

---

## 12. 평가 자동화와 승격 gate

### 12.1 실행 도구와 산출물

- `scripts/ablate_semantic_documents_v3.py`: T0~T3 raw dense 비교
- `scripts/eval_semantic_v3.py --suite v2-tripwire|v3-development|v3-holdout|failure`
- `scripts/eval_mirror_residual_v3.py`: mirror round-trip 잔차 report
- `scripts/eval_predicate_deadzone_v3.py`: registry enum·dead-zone 실측
- `scripts/eval_threshold_candidates_v3.py`: threshold 후보 탐색
- `scripts/build_review_sheet_v3.py`: 사람 검수 시트
- `scripts/build_threshold_ladder_v3.py`: 경계 판단 사다리
- immutable run JSON · `summary.csv` · query별 `triage.csv` · v2/v3 comparison
- Markdown/HTML dashboard · 좌우 mirror thumbnail/contact sheet
- latency/cache/structurer call 수/추정 비용 요약

query별로 raw top document, clause/span/term, 전체 verifier set count, 반환 member,
대표 집합 구성(`representative_count`·`coverage_achieved`), exact denial reason,
latency와 fallback 원인을 기록한다. dense, understanding, verifier, selection, end-to-end
지표를 분리한다.

### 12.2 필수 safety gate

- validation + dense admission 성공 query의 raw dense channel 적용률 `100%`
- 위 분모의 dense cache-miss raw E5 실제 실행률 `100%`
- validation과 dense admission을 통과한 non-empty query의 dense fallback non-empty율 `100%`
- structurer disabled/timeout/error/invalid/busy fallback 성공률 `100%`
- global verifier GT의 member별 `match|violation|unknown` 상태 equality `100%`
- shadow/active false exact `0건`
- constraint violation을 포함한 exact `0건`
- aggregate clause/query state가 unknown인 candidate의 exact `0건`
- 좌우 최소쌍 concrete member 정확도 `100%`
- mirror round-trip 위반 `0건`
- semantic unit 중복 반환 `0건`
- v2 tripwire regression `0건`
- schema-v3 data coverage `1,538 member / 769 unit`
- **compound predicate corpus 근거**: 전 항목 `≥3건`, provenance가 실사용 등급, 인용 query 실재
- **registry 좌우 대칭성**: 모든 `left_X` predicate에 대해 `match(left_X) == match(right_X)`.
  불일치 시 registry 빌드 중단 (§6.1.4)
- **희소 predicate 0개**: match < 25인 predicate가 있으면 threshold·의미를 재검토한다
- **고정 모델의 structured-output schema 수락 `100%`**
- production 승인 corpus `≥200` actual queries와 provenance/dataset fingerprint 일치

`SEMANTIC_EXACT_CLAIM_ENABLED=0`인 shadow에서도 evaluator는 "켜졌다면 claim했을 결과"를 계산해
false-exact gate를 검증한다. 단순히 응답 flag가 꺼져 있어 0건이 되는 것을 통과로 세지 않는다.

### 12.3 retrieval·운영 gate

- actual corpus raw dense relevant HitRate@20 `≥ 0.90`
- query당 최대 20개로 사람이 고정한 must-retrieve exemplar Recall@20 `≥ 0.90`
- verifiable query end-to-end Success@5 `≥ 0.90`
- 선택 renderer의 P@10/nDCG@20이 T0 대비 비회귀
- member directional P@20이 T0 대비 개선
- document text collision 감소
- **exact eligibility rate** — 사람이 `verifiable`로 라벨한 query 중
  `understanding.exact_eligible=true` 비율. **절대 하한 0.70.**
  D4~D7은 measure-only, D7 종료 시 동결. **데이터를 본 뒤 올릴 수는 있어도 내릴 수 없다**
- `understanding.status=complete` 비율 하한 · structurer timeout 비율 상한 (D4 실측 후 동결)
- **대표 집합**: `representative_count` 분포(중앙·p90·상한 도달률)와 `coverage_achieved`
- cache miss structurer 호출 query당 `≤ 1`, cache hit `0` (shadow 제외)
- p95 endpoint latency가 frozen budget 이내
- timeout fallback이 설정 deadline + local budget 이내

**`structurer_noise_floor` — D4에서 1회 측정**

```
noise_floor = development corpus 를 같은 설정으로 5회 반복했을 때
              validated clause/predicate 집합의 불일치율   (캐시 off)
```

상시 게이트가 아니다. release manifest에 기록하고 §11.4와 DCR-001 equivalence gate의
해석 기준으로만 쓴다. **자기 노이즈를 모르면 모델 회귀를 탐지할 수 없다.**

수백 unit이 정답인 넓은 query의 전체 matching set을 Recall@20 분모로 쓰지 않는다. 전체 set
정확성은 전역 verifier의 3값 equality로 검증하고, dense Top-K는 HitRate와 제한된 exemplar qrels로
평가한다.

절대 threshold를 변경하면 holdout 전 release config와 ID를 새로 동결한다. holdout 결과를 본 뒤
threshold를 낮춰 통과시키지 않는다.

### 12.4 지금부터 수집하는 것

측정은 나중에 쓰더라도 수집은 지금 시작해야 한다.

- **선택 로그**: `semantic_release_id`, 쿼리 해시(salted), 반환 후보 ID와 순서, 선택된 후보,
  선택까지의 순위. §9.4 로그 규칙을 따른다.
  대표 집합 파라미터(§7.2.3)와 랭킹 선호를 실증으로 해결할 유일한 근거다
- **`hard_fallback` 발생률**: `data/webtoon_rough_verified`로
  `hard_fallback` / `soft_fallback` / `route=skip|bust` 비율, 그리고
  **`hard_fallback`이면서 VLM 태그가 유효한 비율**(= §10.5 폴백의 효용 상한)

---

## 13. health, version, release

### 13.1 health

`GET /healthz`의 `semantic` 아래에서 dense readiness와 exact capability를 분리한다.

```json
{
  "semantic": {
    "ready": true,
    "dense": { "ready": true, "semantic_index_build_id": "sha256:..." },
    "semantic_release_id": "sha256:...",
    "exact_capable": false,
    "exact_claim_approved": false,
    "approval_manifest_sha256": "sha256:...",
    "structurer": {
      "mode": "shadow",
      "configured": true,
      "provider": "gemini",
      "model": "gemini-3.5-flash-lite",
      "thinking_level": "minimal",
      "structurer_model_id": "sha256:...",
      "query_understanding_version": "sha256:...",
      "model_retirement_date": "2027-xx-xx",
      "days_until_retirement": 000,
      "timeout_ms": 1200,
      "max_concurrency": 4,
      "circuit_state": "closed"
    },
    "fallback_counters": {}
  }
}
```

structurer 장애는 `semantic.ready=false`로 만들지 않는다. dense E5/index가 준비되면 endpoint는
ready이고 `exact_capable=false` 또는 degraded counter로 보인다. `exact_capable=true`는
mode/flag만 보지 않고 approval manifest, 네 ID, holdout run hash, loaded policy hash가 모두
일치할 때만 가능하다.

`days_until_retirement`가 임계일(기본 30) 이하면 degraded counter를 올린다.

### 13.2 네 가지 ID

- `semantic_index_build_id`: geometry/inventory/document/embedding/profile hash
- `query_understanding_version`: prompt bytes + JSON schema + predicate registry +
  anchor/boilerplate/idiom lexicon + lexer 버전 + span/anchor policy hash **(모델 제외)**
- **`structurer_model_id`**: provider + 정확한 모델 버전 문자열 + thinking_level +
  temperature + generation config
- `semantic_release_id`: 위 세 개 + retrieval/verification/selection/API/cache/latency policy hash

응답, cache, health, 평가 artifact에 네 ID를 저장한다. `SEMANTIC_SERVICE_VERSION=3`을 사용하고
v1/v2 cache와 공유하지 않는다.

**모델을 ID로 분리하는 이유**: 프롬프트·schema·registry가 byte-identical인 상태에서
모델만 바뀐 변경을 기계적으로 증명할 수 있어야 한다. 그래야 §13.4의 교체 프로토콜이 성립한다.

### 13.3 startup preflight와 atomic release

production startup은 manifest의 geometry DB hash와 실제 `DB_PATH`를 대조하고, 1,538 pose ID와
BVH 파일 존재를 전수 확인한다. **고정 `structurer_model_id`의 가용성을 최소 호출로 확인** 하고,
실패 시 dense-only로 기동한다.

exact 경로는 별도로 `release-approval.json`의 `exact_claim_approved`, 네 ID, holdout run hash,
`structurer_noise_floor`, `model_equivalence_run_hash`(모델 교체를 거친 release인 경우),
gate summary, actual query corpus count/fingerprint, approver, approval signature와
artifact hash를 확인한다. env flag만으로 이 검사를 우회할 수 없다.
자산 불일치는 service를 fail-closed하고, approval 불일치는 dense-only로 기동한다.

아래를 하나의 atomic release로 배포한다.

1. geometry DB와 1,538 BVH
2. semantic DB와 E5 artifact/profile
3. member atom/document artifact와 mirror report
4. predicate registry와 anchor/idiom lexicon
5. structurer prompt/schema/query-understanding profile
6. retrieval/verification/selection/API 정책
7. golden/eval 승인 manifest와 content-addressed `release-approval.json`
8. BFF contract compatibility 승인

preflight와 holdout을 통과한 bundle만 current pointer로 바꾼다.

### 13.4 모델 교체 프로토콜

모델 교체를 **제한된 변경 등급** 으로 정의한다. 전면 재승인보다 싸지만 느슨하지 않다.

**모델 교체로 인정되는 조건 — 전부 충족**

1. `query_understanding_version`이 **byte 단위로 동일**
2. `semantic_index_build_id` 동일
3. retrieval/verification/selection/API/cache/latency 정책 동일
4. 새 모델의 thinking level이 기존과 같거나 더 낮음

하나라도 어긋나면 설계 변경이다 → D4부터 전면 재실행.

**`model_equivalence_suite`** — 동결된 development corpus 전체에 대해 구/신 모델을 실행하고
**validator를 통과한 뒤의 clause/predicate 집합** 을 비교한다.

| 항목 | 기준 |
|---|---|
| 신규 false-exact | `0건` (shadow `would_claim_exact` 기준) |
| exact eligibility rate | 구모델 대비 선언된 밴드 이내 비회귀 |
| 검증 통과 후 predicate 집합 불일치 query | 전량 목록화 · 사람 검토 |
| v2 tripwire regression | `0건` |
| schema-valid rate | 하한 이상 |
| p95 latency | frozen budget 이내 |

> **차이가 `structurer_noise_floor` 이내면 "차이 없음"으로 읽는다.**

**봉인 holdout 소비 여부**: 위 4개 조건 + equivalence gate를 모두 통과하면
**holdout을 다시 소진하지 않는다.** holdout은 *설계* 를, equivalence suite는 *치환* 을
검증한다. 실패하면 설계 변경으로 강등되어 새 holdout 작성 후 D4~D8 전면 재실행.

교체 후 `structurer_model_id`와 `semantic_release_id`를 갱신하고 approval manifest에
`model_equivalence_run_hash`를 기록한다(기존 holdout run hash는 유지·참조).

`provider`가 바뀌어도 위 조건을 만족하면 같은 등급의 변경이다.

---

## 14. 구현 순서

상세 단계·수용 기준·중단 조건은 **`docs/SEMANTIC_V3_IMPLEMENTATION_PLAN.md`** 가 단일 소스다.
요약만 둔다.

| 단계 | 내용 | 상태 |
|---|---|---|
| D0 | legacy 동결 · 팩토리 골격 · kiwipiepy 핀 + 래퍼 + 결정성 테스트 | 착수 가능 |
| D1 | 실제 query corpus (병렬 진행) | 착수 가능 |
| **D2a** | mirror 잔차 report · core family 확정 | **완료** (전 family 100%) |
| D2b | registry 동결 · schema v3 materialization · document 구축 | **착수 가능** |
| D3 | renderer ablation (T0~T3) | D1 100개 후 |
| **D5** | schema-v3 index · dense-first runtime · **v3.0 dense-only 릴리스** | D3 후 |
| D4 | structurer schema · bake-off · golden v3 | v3.1 시작 |
| D6 | structurer shadow · 전역 verifier · 대표 집합 selector | D4 후 |
| D7 | API/BFF · 평가 자동화 · corpus 200개 | D6 후 |
| D8 | 설정 동결 · 봉인 holdout · 승격 | D7 후 |
| D9 | 모델 교체 리허설 | D8 후 |

**릴리스는 2단으로 나눈다.**

| release | 범위 | 승격 조건 |
|---|---|---|
| **v3.0 dense-only** | D1~D3 + D5. structurer off, exact 없음 | dense 관련 safety gate + v2 tripwire. **봉인 holdout 소비 없음** |
| **v3.1 exact** | D4 + D6~D9 | §12 전체 gate + 봉인 holdout 1회 |

v3.0에서 이미 사용자 가치의 대부분이 나온다 — `clarification_required` 막다른 길이 사라지고
임의 자연어가 항상 후보를 받는다. atomic release 원칙은 각 릴리스 안에서 유지한다.
v3.0 bundle은 §13.3의 1~3, 6~8이며 4~5는 v3.1에서 추가한다.

---

## 15. 승인 기준

이 문서로 다음이 확정됐다.

- "모든 자연어를 E5로 의미 검색"은 예외 없이 raw-query dense channel always-on이다
- LLM은 필요하지만 query structuring 한 곳에만 제한하며, provider는 어댑터다
- exact의 자연어 해석 안전선은 자기신고가 아니라 full-span coverage와 fail-closed
  deterministic validation이다
- exact의 pose 사실 근거는 전체 member의 연속 measurement뿐이다
- 좌우 member 원천은 이미 있고 mirror 무결성도 확보됐으므로 BVH 재태깅부터 다시 하지 않는다
- 결과는 고정 Top-K가 아니라 대표 집합이며 개수는 데이터가 정한다
- 작가에게 추가 질문을 하지 않는다
- 결정성은 계약이 아니며, 노이즈 바닥을 재서 다른 게이트를 해석한다
- 첫 구현은 D0 · D1 · D2b다

문서 승인 시에도 아직 하지 않은 것:

- runtime 코드 변경
- v1 regex parser 물리 삭제
- structurer API 실제 호출
- document/index/golden 재생성
- BVH/geometry 변경 또는 삭제
- holdout 실행
- production manifest 승격

---

## 16. 참고

- [Gemini API models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini API thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [Gemini API structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- `docs/SEMANTIC_V3_IMPLEMENTATION_PLAN.md` — 구현 순서·수용 기준
- `docs/archive/evaluations/MIRROR_DEFECT_ANALYSIS_2026-08-20.md` · `docs/archive/evaluations/MIRROR_FIX_VALIDATION_2026-08-20.md`
- `docs/archive/semantic/SEMANTIC_V3_DCR_001` ~ `008` — 이 문서에 병합된 결정 기록 (이력 참조용)
- `config/semantic_core_families.v3.json`
- `docs/NEXT_SPRINT/VLM_SEMANTIC_SEARCH.md`

---
