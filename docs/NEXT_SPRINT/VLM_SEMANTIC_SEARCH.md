# VLM 의미 검색 (VLM Semantic Search)

> **2026-08-18 어휘 기준:** 오프라인 라이브러리 의미 표준화의 단일 소스는
> `config/semantic_vocab.v2.json`과 검증기 `src/semantic_vocab.py`다. 런타임 `src.schema.Action`은
> 기존 라우팅 계약으로 유지한다. 행동명이 없던 CMU 35개 source clip은
> `config/library_exclusions.v1.json`에 따라 검색 제외·삭제 대기 상태다. 이름이 있는 나머지 328개는
> `data/semantic/action_mapping.v2.jsonl`에 source context로 매핑했으며 결과는
> [`SEMANTIC_ACTION_MAPPING_REPORT_2026-08-18.md`](../SEMANTIC_ACTION_MAPPING_REPORT_2026-08-18.md)에 있다.
> 활성 616개 semantic unit의 최종 방향 중립 검색 문서는
> `data/semantic/search_documents.v2.jsonl`로 생성했으며 빌드 결과는
> [`SEMANTIC_SEARCH_DOCUMENT_BUILD_2026-08-18.md`](../SEMANTIC_SEARCH_DOCUMENT_BUILD_2026-08-18.md)에 있다.

> 상태: vocabulary·source mapping·pinned E5/member PoseCode DB·내부 runtime/API 완료 · development PASS · holdout/release 전
>
> 갱신일: 2026-08-18
>
> 기준 문서: `PIPELINE_OVERVIEW.md`, `SKELETON_EXTRACTION_IMPROVEMENT.md`,
> `STRONG_ROUGH_GUIDED_GEOMETRY_RECOVERY.md`, `NEXT_STEPS_AFTER_HANCHI_INTERVIEW.md`,
> [`POSE_LIBRARY_EXPANSION_GUIDE.md`](../POSE_LIBRARY_EXPANSION_GUIDE.md)

## 0. 결론

RTMPose가 러프의 인물을 안정적으로 복원하지 못해도 바로 빈 폴백으로 끝내지 않고,
VLM이 읽은 **인물별 포즈 의미**로 라이브러리의 참고 후보 Top-5를 찾는다.

이 기능은 스켈레톤 추출을 대체하지 않는다. 결과를 다음 두 종류로 명확히 구분한다.

- **기하 검색**: 유효한 2D 스켈레톤과 라이브러리 투영을 비교한 후보
- **VLM 의미 검색**: 행동·제스처·문맥을 바탕으로 찾은 참고 후보

먼저 사람이 입력하는 텍스트 검색을 검증하고, 같은 검색 엔진에 VLM 자동 검색어를 연결한다.
단, 런타임 검색 구현보다 **라이브러리 inventory·태깅·사람 승인·버전 고정이 선행**한다. 현재 기하
projection은 존재하지만 의미 색인의 근거가 되는 provenance·caption·posecode가 없기 때문이다.
inventory·provenance·사람 검수 원장과 실행 순서는 `POSE_LIBRARY_EXPANSION_GUIDE.md` §4를 기준으로
삼고, 검색용 atom·text document·embedding 계약은 이 문서 §5를 기준으로 삼는다.

초기 제품 검색기는 **다국어 dense embedding 후보 회수 + typed semantic atom/posecode 제약
검증**으로 만든다. 이는 태그 문자열이나 미리 붙여 둔 여러 문장과 검색어가 겹칠 때만 반환하는
방식이 아니다. `활짝 편`, `옆으로 뻗은`, `날개처럼 벌린`처럼 표면형이 달라도 의미가 가까우면
같은 후보군을 회수하고, 좌우·부정·정도·소품·상호작용은 embedding 유사도에 맡기지 않고
구조화 조건으로 다시 판정한다.

BVH에서 규칙으로 뽑은 관찰 가능한 자세 속성은 **typed semantic atom**으로 저장하고, 파일명·공식
catalog·검수된 caption은 행동·스타일·소품의 문맥 근거로 분리한다. SQLite FTS5/BM25는 춤 이름,
소품명, 고유 동작명 같은 정확 어휘를 보완하는 lexical 채널이다. 생성형 LLM은 필수 검색기가
아니며 복잡한 쿼리를 `must/should/must_not`로 구조화할 때만 선택적으로 사용한다.

---

## 1. 해결하려는 문제

사람은 러프를 보고 “인사한다”, “칼을 들이민다”처럼 동작을 이해할 수 있지만 다음 경우에는
RTMPose 관절이 누락되거나 여러 인물로 합쳐질 수 있다.

- 선이 적고 비례가 과장된 러프
- 가림·접촉이 심한 다인 장면
- 극단적인 단축과 원근
- 신체 일부만 암시된 포즈

현재 파이프라인은 crop 재추론 후에도 `missing`, `invalid`, `insufficient`이면 hard fallback한다.
후속 흐름에서는 먼저 `STRONG_ROUGH_GUIDED_GEOMETRY_RECOVERY.md`의 20초 포즈 지정을 제안하고,
작가가 건너뛰거나 수동 기하 복구가 성립하지 않을 때 VLM 의미 검색으로 전환한다. VLM 의미 검색은
이때 **정확한 이미지 매칭이 아니라 작가가 참고할 수 있는 포즈 후보**를 제공한다.

이 기능은 한치 작가 인터뷰에서 나온 “포즈를 문장으로 검색하고 싶다”는 요구와 같은 기반을 쓴다.

---

## 2. 제품 원칙

1. 의미 검색 결과를 스켈레톤 기반 결과처럼 표시하지 않는다.
2. VLM 의미만으로 높은 일치 신뢰도를 주장하지 않는다.
3. 자동 검색어는 사용자가 확인하고 수정할 수 있다.
4. 모든 의미·하이브리드 후보에는 refine을 적용하지 않는다.
5. 기존 Gemini 분석을 확장하되 정상 컷에 별도 VLM 호출을 추가하지 않는다.
6. 의미 검색 실패를 스켈레톤 추출 실패와 별도로 계측한다.
7. 사용자 원문을 보존하고 deterministic parser·선택적 LLM 구조화 결과와 함께 추적한다.
8. BM25·cosine 같은 검색 점수를 일치 확률처럼 표시하지 않는다.
9. 얽힘 포즈는 개별 인물이 아니라 기존 `set_id` 단위로 검색한다.
10. dense similarity는 후보 회수 신호이지 자세·행동 사실의 증거가 아니다.
11. 좌우·부정·정도·소품·set role은 typed constraint로 검사하고 `unknown`을 `false`로 취급하지 않는다.
12. BVH에서 관찰한 `observed`, 이름·catalog가 의도한 `intended`, 문화·스타일의 `contextual` 근거를
    서로 합치지 않는다.
13. 의미 atom·embedding은 기존 `PersonDescriptor.tag_dict()`·`knn_geometric`·refine 입력에
    전달하지 않는다.
14. 제품 의미 후보는 `product_bvh_export=yes && delivery_mode=original`인 member만 선택·전달한다.
    `refined_only`는 의미 후보를 refine할 수 없으므로 초기 제품 의미 Top-K에서 제외한다.
15. 행동 ID가 비어 있거나 `unknown`이어도 검색 대상에서 제거하지 않는다. 검증된 observed posecode로
    계속 검색하고, 원본 이름은 contextual candidate 신호로만 사용한다.

권장 UI 문구:

```text
포즈를 정확히 읽지 못했습니다.
장면의 의미로 찾은 참고 포즈입니다.

검색어: 한 손을 들고 가볍게 인사하는 자세  [수정]
```

---

## 3. 목표 워크플로우

```text
컷 이미지
→ Gemini 1회
   ├─ 기존: shot, 사람 수, 대략 박스
   └─ 추가: 슬롯별 pose_query, query_atoms, description_confidence
→ RTMPose 전체 이미지 + 기존 조건부 crop 재추론
→ 슬롯 품질 판정
   ├─ valid / 안정적인 reduced
   │  └─ 기존 기하 Top-5
   ├─ partial / sparse
   │  ├─ 기존 기하 후보
   │  └─ 사용자 요청 시 의미 참고 후보
   └─ missing / invalid / insufficient
      └─ 강한 러프 20초 포즈 지정
         ├─ 성공: manual geometry Top-5 → 조건부 refine
         └─ 건너뛰기·불충분·실패: VLM 의미 검색 Top-5
            → 검색어를 사용자가 수정해 재검색 가능
```

**계산은 병렬화할 수 있지만 결과 채택은 조건부다.** Gemini 응답에서 `pose_query`를 받은 뒤
로컬 의미 검색을 RTMPose와 함께 준비할 수 있다. 정상 기하 결과가 나오면 의미 후보는 노출하지 않는다.

### 상태별 정책

| 스켈레톤 상태 | 기본 결과 | 의미 검색 | refine |
|---|---|---|---|
| `valid/full` | 기하 Top-5 | 숨김 | 기존 정책 |
| 안정적인 `reduced` | 기하 Top-5 | 선택 제공 | 기하 후보만 유효 사지, 의미 후보는 금지 |
| `partial/sparse` | 저신뢰 기하 후보 | 별도 참고 후보 | 기하 후보만 기존 엄격 정책, 의미 후보는 금지 |
| `missing/invalid/insufficient` | 먼저 수동 기하 복구 제안 | 수동 복구 실패·건너뛰기 시 참고 Top-5 | manual geometry만 별도 정책으로 허용, 의미 후보는 금지 |

의미 점수와 기하 거리를 가중평균하지 않는다. 두 점수의 척도와 신뢰도 의미가 다르다.
`partial/reduced`에서 두 신호를 함께 쓸 때는 **의미 Top-N 후보 생성 → 남아 있는 관절로 기하 재정렬**
순서의 cascade만 실험한다.

의미 색인은 `knn_geometric`의 tag prefilter나 `valid/full`·안정적인 `reduced` 기본 경로에 개입하지
않는다. partial cascade는 `hybrid_partial`이라는 별도 실험 결과이며 기존 geometry confidence나
fallback distance를 승계하지 않는다. 의미가 후보 생성에 들어간 결과는 항상 `refine_allowed=false`다.
여기서 dense-first는 의미 검색 엔진 내부 순서를 뜻하며 이미지→포즈의 기존 순수 기하 검색을 dense
text retrieval로 바꾼다는 뜻이 아니다.

---

## 4. 공용 의미 검색 엔진

사람이 입력한 문장과 VLM이 만든 문장은 같은 인터페이스를 사용한다.

```text
semantic_search(raw_query, query_constraints=None, top_k=5)
→ 원문 보존 + query embedding + 구조화 제약 추출
→ multilingual dense + lexical exact 채널에서 semantic unit 후보 회수
→ must/must_not/좌우/정도/evidence compatibility 검사
→ concrete mirror variant 선택 또는 pose set 멤버 원자 결합
→ Top-5 + matched/unmatched/unknown 속성 + coverage 상태
```

### 4-1. 기본 검색기: dense retrieval first

첫 제품 기준선은 `intfloat/multilingual-e5-small`을 사용한 다국어 dense retrieval이다. 한국어
검색어와 한국어·영어 색인 문서를 같은 벡터 공간에 넣고 cosine으로 가까운 semantic unit Top-N을
찾는다. 이때 unit당 하나의 긴 합성문만 만들지 않고 다음처럼 근거가 다른 짧은 document를 각각
embedding한다.

- 사람이 승인한 `caption_ko`, 필요하면 `caption_en`
- 승인된 alias·공식 action/style 이름
- typed atom을 결정적으로 자연어로 렌더링한 posecode document
- 공식 catalog의 문맥 설명. 선택 프레임에서 확인되지 않은 사실은 별도 context document

unit 점수는 문서별 유사도의 `max`로 후보를 회수하되 공개 확률로 사용하지 않는다. 좌우 표현은
방향 중립 family 문서에 복제하지 않고 member atom에서만 처리한다. E5의 `query:`/`passage:` prefix,
checkpoint revision, pooling/normalization, 차원과 문서 렌더러 버전을 `embedding_version`으로 함께
고정한다. document가 많은 unit이 `max`에서 유리해지는 편향을 막기 위해 type별 document 수와
alias 병합 규칙을 template에 고정하고 모든 unit에 같은 cap을 적용한다.

검색 순서는 다음과 같다.

1. 사용자 원문을 보존하고 query embedding을 만든다.
2. 원문에서 body part, 좌우, 관계, 부정, 정도를 deterministic parser로 먼저 추출한다.
3. 복잡하거나 모호한 문장만 선택적으로 LLM이
   `must/should/must_not/unspecified/unparsed_explicit`를 제안한다.
4. dense Top-N과 lexical exact 후보를 합치고 semantic unit 단위로 중복 제거한다.
5. 각 후보의 typed atom과 evidence를 쿼리 제약에 대조한다.
6. 방향 중립 unit을 찾은 뒤 좌우 조건을 만족하는 concrete original/mirror member를 선택한다.
7. `pose_set`은 모든 역할을 원자적으로 반환하고, `view`는 표시용 투영 hint로만 쓴다.

LLM 정규화가 실패하거나 API 키가 없어도 dense 검색과 deterministic constraint parser는 동작해야
한다. pinned embedding 모델이 없거나 index와 버전이 다르면 production 의미 endpoint는
`semantic_not_ready`로 실패한다. 명시적인 개발/진단 모드에서만 lexical+atom fallback을 허용하고
`search_components=["lexical_fallback", "atom_constraints"]`라고 표시한다. 이 경량 경로를 dense 의미 검색과 같은 것으로
보고하거나 제품 품질 기준선으로 삼지 않는다.

### 4-2. 구조화 쿼리와 3값 제약 판정

embedding은 후보 **회수**만 담당한다. 쿼리 파서의 최소 출력은 다음과 같다.
parser는 원문을 지우지 않고 조사·어미와 표기 변형을 canonical concept로 정규화한다. 예를 들어
`다리를 → 다리`, `양팔은 → 양팔`, `춤을 → 춤`, `왼쪽 다리 → left_leg`로 해석한다. 이 사전은
문장을 1:1 매칭하기 위한 alias 목록이 아니라 typed constraint를 만들기 위한 제한된 vocabulary다.

```json
{
  "semantic_text": "왼쪽 다리를 뒤로 들고 양팔을 넓게 벌린 자세",
  "must": [
    {"predicate": "limb_state", "subject": "left_leg", "value": "raised"},
    {"predicate": "relative_direction", "subject": "left_ankle", "relation": "behind", "object": "pelvis"},
    {"predicate": "limb_configuration", "subject": "both_arms", "value": "widely_spread"},
    {"predicate": "joint_flexion", "subject": "both_elbows", "value": "extended"}
  ],
  "should": [
    {"predicate": "body_support", "value": "right_leg_support"}
  ],
  "must_not": [],
  "unspecified": [],
  "unparsed_explicit": []
}
```

`unspecified`는 사용자가 말하지 않은 선택적 slot이고 exact를 막지 않는다. `unparsed_explicit`는 원문에
분명히 있지만 parser가 canonical constraint로 옮기지 못한 표현이며, 해당 조건을 해결할 때까지
exact를 금지하고 parse warning을 반환한다. 두 상태를 하나의 `unresolved`로 합치지 않는다.

각 조건은 `match | contradiction | unknown`으로 판정한다. `unknown`을 `false`로 바꾸지 않는다.

- `must`가 모두 match이면 해당 조건 범위에서 exact가 될 수 있다.
- 일부 `must`가 unknown이어도 base intent를 허용된 evidence로 match하면 partial이며 누락 근거를
  공개한다. 전부 unknown인 후보는 dense 유사도만으로 partial에 올리지 않는다.
- `must`와 반대 atom이 확인되면 후보에서 제외한다.
- `must_not`은 명시적 positive evidence가 있는 후보를 제외한다. 미태깅을 “없음”으로 간주하지 않는다.
- `살짝`, `깊게`, `완전히`는 embedding에 맡기지 않고 연속 측정값과 버전된 bin으로 판정한다.
- 좌우·신체 관계는 concrete BVH member, 소품·행동·스타일은 해당 필드에 허용된 provenance로 판정한다.

일반 bi-encoder는 부정에 취약하므로 부정과 최소 대조쌍을 dense 순위만으로 판정하지 않는다. LLM이나
cross-encoder도 사실 근거를 새로 만들 수 없으며, 구조화 결과가 원문과 충돌하면 원문을 우선하고
`query_parse_conflict`를 남긴다.

### 4-3. lexical 보조, 선택 모델과 검색 비용

SQLite FTS5/BM25는 dense의 대체물이 아니라 exact-name 보조 채널이다. 춤 이름, 소품명, 고유 action,
짧은 body part는 lexical 결과를 함께 회수한다. 한국어 caption·alias에는 `trigram`, 영문에는
`porter unicode61`을 사용하되 `칼`, `창`, `팔`처럼 3자 미만인 말은 정규화된 exact 컬럼으로 보완한다.
두 순위를 합칠 때 BM25와 cosine 원점수를 직접 더하지 않고 RRF를 사용한다.

| 구성 | 기본 역할 | 도입·유지 조건 |
|---|---|---|
| `intfloat/multilingual-e5-small` | 첫 dense 제품 기준선 | 한국어 paraphrase와 좌우 제약 회귀 관문 통과 |
| `BAAI/bge-m3` dense | 품질·다국어 비교 후보 | 같은 holdout에서 E5보다 유의미한 개선 |
| FTS5/BM25 + exact facet | 이름·소품·짧은 핵심어 보완 | dense 단독의 명시 조건 누락 감소 |
| dense + lexical RRF | 서로 다른 후보군 결합 | dense 단독보다 `Success@5`/`nDCG@5` 개선 |
| multilingual cross-encoder | 합집합 Top-20 재정렬 | Recall@20은 충분하고 Top-5 순서만 나쁠 때 |
| query LLM | 복잡한 조건의 구조화 제안 | deterministic parser 대비 제약 위반률 감소 |

현재 654 semantic unit에 대표 document가 하나씩 있을 때 float32 384차원 행렬은 약 0.96 MiB이고,
NumPy exact cosine 전수 검색은 로컬 실측 중앙값 약 9 μs다. unit마다 짧은 document를 네 개씩 두어도
행렬은 약 3.8 MiB에 불과하다. 1,307 member를 하나씩 비교하는 경우도 약 15 μs 수준이므로
ANN·vector DB는 필요 없다. 런타임 병목은 벡터 비교가 아니라 query embedding 모델 추론과 선택적
LLM 네트워크 호출이다. document embedding은 오프라인에서 미리 계산하고 query 결과를 cache한다.

Cross-encoder는 누락 후보, 잘못된 provenance, library gap을 복구하지 못한다. VLM이 매 요청마다
전체 접촉시트를 직접 고르게 하는 방식도 비용·지연·재현성 때문에 사용하지 않는다.

---

## 5. 라이브러리 의미 색인

기존 `Action` enum은 라우팅용 거친 어휘로 유지한다. 세부 의미 검색 데이터는 별도 테이블이나
별도 색인으로 관리한다.

원본 제공처·공식 이름·제공처 번호는 의미 테이블에 복사하지 않는다. `pose_id → pose_lineage →
source_clip`으로 join하며, `source_clip_id`(내부 join key), `native_clip_id`(제공처 번호),
`selected_frame_index`, `sample_ordinal`을 서로 다른 필드로 유지한다. 기준 스키마는
`POSE_LIBRARY_EXPANSION_GUIDE.md` §4.3이다. `source_context` document는 승인된 source 내용을 검색용으로
렌더링한 파생 passage와 provenance ref만 가지며, 공식 title의 authoritative record는 계속
`source_clips.v1.jsonl`이다.

```text
pose_semantics
├─ semantic_unit_id        # pose:<mirror_group_base> | set:<set_id>
├─ semantic_unit_type      # pose_variant_group | pose_set
├─ review_status           # generated | needs_review | accepted | rejected | blocked
└─ semantic_version

semantic_text_documents
├─ document_id
├─ semantic_unit_id
├─ language                # ko | en | multilingual
├─ document_type           # caption | alias | posecode_render | source_context
├─ text
├─ text_sha256
├─ provenance_json
└─ review_revision

semantic_embeddings        # 승인 text document에서 재생성하는 build artifact
├─ document_id
├─ embedding_blob
└─ embedding_version

semantic_atoms
├─ atom_id
├─ semantic_unit_id
├─ pose_id                 # unit atom이면 NULL, 좌우 member atom이면 concrete pose_id
├─ scope                   # unit | member
├─ predicate
├─ subject
├─ relation
├─ object
├─ axis
├─ value_json
├─ measure
├─ measure_unit
├─ bucket
├─ polarity
├─ evidence_state          # observed | intended | contextual
├─ provenance_json
└─ atom_version

pose_semantic_members
├─ semantic_unit_id
├─ set_variant_id          # pose_variant_group이면 NULL, pose_set이면 atomic instance ID
├─ pose_id
├─ set_role
├─ handedness              # original | mirrored
├─ posecode_measurements_json
└─ posecode_version

pose_set_variants
├─ set_variant_id
├─ semantic_unit_id
├─ mirror_of_set_variant_id
├─ expected_roles_json     # 이 atomic set을 완성하는 역할 목록
└─ is_canonical

semantic_index_meta
├─ semantic_schema_version
├─ content_version
├─ semantic_build_id
├─ pose_library_version
├─ geometry_manifest_sha256
├─ geometry_db_sha256
├─ atom_schema_version
├─ embedding_model_id
├─ embedding_revision
├─ encoder_artifact_sha256
├─ tokenizer_revision
├─ embedding_dimension
├─ dtype / l2_normalized
├─ query_prefix / passage_prefix
├─ pooling / max_length / truncation
├─ passage_template_version
├─ query_parser_version
├─ resolution_policy_version
├─ retrieval_policy_version
├─ retrieval_config_sha256
├─ lexical_config          # 보조 채널을 켠 build만
└─ built_at
```

`axis`, `measure`, `measure_unit`, `bucket`은 해당 predicate에 없으면 `null`이다. `polarity`는 모든 atom에
필수이며 `positive | negative`만 허용한다. `provenance_json`은 문자열이 아니라 항상
`{kind, ref, version, review_status}` object다. predicate별 필수/nullable 필드와 단위는
`atom_schema_version`에 고정하고, 임의 key를 `value_json`에 숨겨 matcher가 해석하게 만들지 않는다.

예시:

```json
{
  "semantic_unit_id": "pose:Waving_02",
  "semantic_unit_type": "pose_variant_group",
  "text_documents": [
    {
      "document_id": "pose:Waving_02:caption:ko:1",
      "document_type": "caption",
      "language": "ko",
      "text": "한 손을 들어 인사하는 선 자세",
      "provenance": {
        "kind": "vlm_contact_sheet",
        "ref": "proposal:Waving_02:1",
        "version": 1,
        "review_status": "accepted"
      }
    },
    {
      "document_id": "pose:Waving_02:alias:en:1",
      "document_type": "alias",
      "language": "en",
      "text": "greeting with one hand raised",
      "provenance": {
        "kind": "filename_hint",
        "ref": "Waving_02",
        "version": 1,
        "review_status": "accepted"
      }
    }
  ],
  "unit_atoms": [
    {
      "predicate": "action_intent",
      "value": "greeting",
      "polarity": "positive",
      "evidence_state": "intended",
      "provenance": {
        "kind": "source_label",
        "ref": "source_clip:Waving",
        "version": 1,
        "review_status": "accepted"
      }
    }
  ],
  "review_status": "accepted",
  "semantic_version": 1
}
```

embedding은 원본 태깅 레코드가 아니라 승인 text document와 고정된 모델 설정에서 언제든 다시 만들 수
있는 파생 색인이다. 사람이 embedding용 합성문을 따로 작성하지 않는다. builder가 필드 순서·구분자·
빈 값 처리까지 고정한 `passage_template_version`으로 document를 생성하고 `text_sha256`을 검증한다.

`pose_variant_group`은 원본·미러가 공유하는 **방향 중립 caption 검수 단위**일 뿐 export 단위나
기하 dedup 단위가 아니다. 좌우가 명시된 쿼리는 멤버별 posecode로 맞는 concrete `pose_id`를 고른다.
좌우가 없는 의미 검색은 같은 설명을 가진 두 멤버가 Top-5를 중복 점유하지 않도록 canonical variant
하나를 반환한다. 접미사 `_mirror`는 초기 import 힌트로만 쓰고 실제 연결은 `mirror_group_id`와 멤버
테이블에 저장한다. 최종 후보는 항상 BVH를 조회할 수 있는 concrete `pose_id`를 가져야 한다.
방향 중립 text document와 embedding도 group당 한 벌만 만든다. `왼손`과 `오른손` 문장을 공통
passage에 함께 넣지 않으며, 좌우는 dense 후보 회수 뒤 member atom으로만 선택한다. original과
mirror의 posecode는 각각 계산해 좌우 swap+반사를 검증하고 orphan 또는 검증 실패는 의미 build에서
fail-closed한다. posecode passage는 검증된 두 member atom에서 `왼쪽/오른쪽`을 `한쪽`으로 치환해
결정적으로 만들고 원본과 mirror가 공유하지 않는 관찰은 공통 passage에 넣지 않는다. 이 규칙은 기하
DB에서 두 pose를 제거한다는 뜻이 아니다.

이 키는 기존 다양성용 `pose_family_id`와 다르다. `pose_family_id`는 검증된 source clip provenance로
부여하고, 같은 source clip의 다른 프레임은 각자 다른 semantic unit과 caption·posecode를 유지한다.
미러 멤버도 기존 기하 검색에서는 각각 정당한 후보로 남는다.

`pose_set`은 `set_id`로 한 번 검색하고 모든 역할 멤버를 원자적으로 반환한다. A/B를 person별 Top-K에서
임의 조합하거나 일부 멤버만 반환하지 않는다. 현재 로컬 DB에는 `set_id` 데이터가 없으므로 interaction
검색은 set 데이터와 완전성 검사가 구축되기 전까지 `unsupported`로 응답한다.
set text document와 embedding도 set당 한 벌만 만들며 interaction과 role atom은 set/member에 각각
저장한다. mirrored set은 `pose_set_variants`에서 original/mirrored atomic instance를 나누고
`mirror_of_set_variant_id`로 연결한다. 각 `set_variant_id` 안에서 expected role은 정확히 한 번씩
존재해야 하며, 검색은 variant 하나의 완전한 역할만 반환한다. 개별 member embedding을 런타임에
임의 조합하거나 original A와 mirrored B를 섞어 상호작용을 만들어내지 않는다.

제품 semantic selectable pool에는 모든 member가 `product_bvh_export=yes`이고
`delivery_mode=original`인 unit만 넣는다. `delivery_mode=refined_only` 자산은 의미 후보에 refine을
금지한 정책과 양립하지 않으므로 초기 제품 Top-K에서 제외한다. 개발 진단에서 필요하면
`selectable=false`, `bvh_url=null`, `exclusion_reason=semantic_refine_forbidden`인 별도 결과로만 보여주고
`Success@5`에 포함하지 않는다. 서버의 `/pose`·export도 manifest를 다시 검사해 허용되지 않은 원본
BVH 전달을 fail-closed해야 한다.

같은 semantic unit의 좌우 variant 속성은 멤버 행에 둔다.

```json
{
  "semantic_unit_id": "pose:Waving_02",
  "pose_id": "Waving_02",
  "handedness": "original",
  "posecode_measurements_json": {
    "right_wrist_height_torso_units": 1.42,
    "torso_lean_degrees": 2.1
  },
  "observed_atoms": [
    {
      "predicate": "relative_position",
      "subject": "right_wrist",
      "relation": "above",
      "object": "right_shoulder",
      "axis": "vertical",
      "measure": 0.31,
      "measure_unit": "torso_length",
      "bucket": "clear",
      "polarity": "positive",
      "evidence_state": "observed",
      "provenance": {
        "kind": "bvh_rule",
        "ref": "Waving_02.bvh",
        "version": 1,
        "review_status": "generated"
      }
    }
  ],
  "posecode_version": 1
}
```

### BVH 기반 posecode

PoseScript 연구의 핵심만 차용해 기존 BVH 3D 관절에서 설명 가능한 자세 속성을 규칙으로 계산한다.
새 학습 모델은 넣지 않는다.

- 손/팔꿈치가 어깨·머리보다 위인지
- 팔꿈치·무릎의 굽힘 구간
- 상체의 전후·좌우 기울기
- 양발 간격과 서기·앉기·무릎 꿇기 구분
- 손과 엉덩이·허벅지 등 신체 부위의 상대 위치

계산한 posecode는 typed atom과 연속 측정값으로 저장하고, `손이 허벅지 가까이에 있음`처럼 결정적
한·영 passage로 렌더링해 dense document와 구조화 matcher가 같은 관찰 사실을 보게 한다. lexical
보조 색인을 켠 build에서는 같은 renderer가 exact 검색 토큰도 생성한다.

자세 기하는 `bvh_rule`, 행동명은 파일명·caption, 소품·상황은 파일명 또는 오프라인 VLM처럼
출처를 분리한다. BVH에 없는 칼·창을 자세만 보고 있다고 기록하면 안 된다. 임계값이나 규칙을
바꾸면 `posecode_version`과 `semantic_version`을 올리고 전체 의미 색인을 재빌드한다.
PoseScript의 관절 구성·절대 임계값을 복사하지 않고, BVH 소스별 크기와 전방축을 먼저 정규화한 뒤
이 프로젝트 관절 매핑과 평가 데이터로 임계값을 보정한다. 같은 BVH에는 항상 같은 속성이 나와야
하므로 캡션 다양화용 random noise·random skip은 검색 색인에 사용하지 않는다.

### atom의 근거와 3값 논리

- 자세·관절 관계·연속 측정값처럼 BVH에서 보이는 정보는 `observed`다.
- `waving`, `throwing`, `attacking` 같은 행동 의도와 보이지 않는 소품은 catalog·파일명·사람 검수의
  `intended`다.
- 시대·문화·전통·작품 같은 스타일 문맥은 공식 source 또는 사람 검수의 `contextual`이다.

모든 atom에는 provenance가 있어야 하며, 한 종류의 근거를 다른 종류의 사실로 승격하지 않는다.
예를 들어 한 손이 어깨 위에 있다는 observed atom은 `waving`의 후보 근거일 수 있지만 인사 행동을
확정하지 않는다. 자산에 negative atom이 없다는 사실도 “그 행동·소품이 없다”는 뜻이 아니다.
검색기는 `match | contradiction | unknown`을 유지하고 응답에 unknown 조건을 공개한다.

### 문화·스타일 태그의 한계

현재 라이브러리는 모두 1-frame BVH이며 검색용 관절도 COCO17 수준이다. 따라서 리듬·궤적·손모양·
복식·소품·문화 맥락을 관찰할 수 없고, 정적 실루엣만 보고 `전통춤`, 특정 국가의 춤, 시대를 확정하면
안 된다. `style_context`는 공식 catalog 또는 사람 승인 근거가 있을 때만 hard facet으로 사용한다.
4-view VLM 추측은 proposal일 뿐이며 승인 전 기본값은 `unknown`이다.

문화 개념은 `{concept_id, labels_ko/en, region, tradition_or_work, provenance, review_status}`로 관리한다.
검증된 스타일 후보가 없으면 비슷한 정적 포즈를 `partial` 참고로 분리하거나 `library_gap`을 반환한다.
**쿼리 언어는 문화권이 아니다.** 한국어 입력이라고 한국 전통춤을 자동 선택하지 않는다.

### 초기 데이터 구축 순서

inventory·provenance·검수 rubric·템플릿은 `POSE_LIBRARY_EXPANSION_GUIDE.md` §4가 기준이다.
검색용 atom·text document·embedding 형식은 이 문서 §5가 기준이며 여기서는 연결 순서도 고정한다.

1. `init_bvh_tag_inventory.py`로 concrete BVH별 hash·파싱·mapping·미러 관계를 전수 기록한다.
2. 원본 catalog/manifest로 `source_clip_id`·원본 frame·license를 복구한다. 정규식 결과는 hint로만 둔다.
3. 각 concrete BVH에서 결정적 posecode의 연속 측정값과 typed member atom을 만들고 원본·mirror
   좌우 swap을 검증한다.
4. 파일명/source label을 행동·alias·intended prop **후보**로 저장한다.
5. CMU처럼 이름이 불투명하거나 근거끼리 충돌하는 unit부터 4-view VLM 제안을 만든다.
6. P0/P1은 전수 검수하고, 그 밖의 반복군은 묶음 승인+표본 감사를 한다.
7. 결정적 BVH observed posecode는 구조·미러 검증 통과 시 자동 검색 문서로 렌더링한다. 사람 승인이
   필요한 intended/contextual 필드는 `accepted` 또는 `accepted_with_edits` revision의
   **명시적으로 검수된 필드만** hard facet으로 렌더링한다. `reviewed_fields` 밖의 proposal 값은
   candidate-only 문맥으로 격리하거나 `unknown`으로 남기며,
   nested proposal의 `generated | needs_review` 상태는 해당 필드의 decision overlay가 승인 상태를
   승계한다. unit 전체를 묵시적으로 승인하지 않는다.
8. 고정된 embedding model/revision으로 전량 embedding하고 별도 semantic DB를 검증한다.
9. development 질의로 query parser·atom matcher·passage renderer를 조정하고 holdout은 설정 동결
   전까지 열지 않는다.

2026-08-13 로컬 DB 기준으로 1,307개 포즈 중 1,275개가 `action=other`이고 상세 `meta_json`은
비어 있다. `source=synthetic`, `license=n/a`도 실제 데이터 특성이 아니라 import 기본값이다. 따라서
기존 action 필터나 파일명만 재사용해서는 의미 검색이 성립하지 않는다.

기하 `FEATURE_VERSION`과 의미 색인 버전은 분리한다. 포즈 DB가 재빌드되었는데
`pose_library_version`이 맞지 않거나 orphan member·불완전 set이 있으면 오래된 의미 색인을 조용히
사용하지 말고 기동/빌드를 실패시킨다.

### 저장·빌드·런타임 계약

의미 색인은 기하용 `poses.db`와 분리한 `pose_semantics.db`에 둔다. 개발 기본 경로는
`SEMANTIC_DB_PATH=data/pose_semantics.db`이며 release에서는 활성 version bundle 안의 절대 경로를
주입한다. 현재
`repo.build_db()`는 기하 table만 전체 재빌드하므로 같은 DB에 embedding을 넣으면 stale semantic
row가 남을 수 있다. JSONL proposal/decision은 원장이고 semantic DB와 embedding matrix는 언제든
재생성 가능한 build artifact다.

`semantic_build_id`는 다음을 고정한다.

```text
accepted proposal/decision revision set
+ pose_library_version
+ geometry_manifest_sha256 + geometry_db_sha256
+ semantic/atom/posecode schema version
+ passage_template_version
+ embedding_version(model + revision + tokenizer + pooling + prefix + dimension + normalization)
+ encoder_artifact_sha256
+ query_parser_version + resolution_policy_version + retrieval_policy_version
+ retrieval config(document aggregation/type cap + dense/lexical depth + FTS config + RRF k + constraint order)
```

startup validator는 accepted unit마다 승인 document와 embedding이 하나 이상 있고, 모든 벡터가 finite,
지정 dimension, L2 norm, `text_sha256`를 만족하는지 검사한다. 모든 member `pose_id`가 기하 DB의
**고유 pose 1,308개**에 존재해야 하며 projection row 5,232개를 pose 수로 사용하면 안 된다. 실제
`pose_id → BVH sha256` manifest, geometry DB hash, encoder artifact hash도 build metadata와 비교한다.
`pose_library_version` 문자열만 같다고 통과시키지 않는다. 이 값, orphan mirror, set 완전성 중
하나라도 어긋나면 semantic endpoint만 fail-closed한다. set validator는 `set_variant_id`별 expected
role의 유일성·완전성과 mirrored variant 연결까지 검사한다.

모델은 런타임 `latest`로 다운로드하거나 다른 encoder로 조용히 대체하지 않는다. model artifact의
revision/hash/license를 deployment에 고정하고 startup 때 한 번 load·warmup한다. immutable embedding
matrix도 한 번만 메모리에 올리고 query cache key에는 `(embedding_version, raw_query,
query_constraints)`를 포함한다. `query_encode_ms`, `exact_cosine_ms`, `constraint_match_ms`, 전체
p50/p95를 분리 계측한다. vector DB/ANN은 unit이 수만~수십만으로 늘고 exact search p95가 실제
병목으로 확인될 때만 검토한다.

release provisioner는 전체 bundle을 임시 version directory에 풀고 `content-sha256.txt`, geometry와
semantic DB, BVH manifest, encoder artifact 호환성을 모두 검증한 뒤 `current` pointer를 원자적으로
전환한다. live `data/`에 직접 덮어쓰거나 `poses.db` 존재만 보고 download를 건너뛰는 현재 방식은
semantic production 전에 교체해야 한다. 이전 version directory를 보존해 DB·BVH·encoder를 한 묶음으로
롤백한다.

---

## 6. VLM 슬롯 확장

현재 컷 단위 `action` 하나로는 다인 장면의 각 인물을 설명할 수 없다. `approx_box`와 함께
인물별 의미를 받아 `PersonSlot`에 연결한다.

```json
{
  "people": [
    {
      "vlm_person_id": "person_0",
      "approx_box": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.9},
      "pose_query": "오른손을 들어 인사하며 몸을 약간 숙인 자세",
      "query_atoms": [
        {"predicate": "action_intent", "value": "greeting", "strength": "should"},
        {"predicate": "limb_state", "subject": "right_arm", "value": "raised", "strength": "should"},
        {"predicate": "torso_lean", "value": "forward", "degree": "slight", "strength": "should"}
      ],
      "description_confidence": "medium"
    }
  ]
}
```

최소 런타임 필드:

```python
VLMPersonHint.vlm_person_id: str
VLMPersonHint.approx_box: BBox | None
VLMPersonHint.pose_query: str | None
VLMPersonHint.semantic_query_atoms: list[dict] | None
VLMPersonHint.description_confidence: str | None

PersonSlot.vlm_person_id: str | None
PersonSlot.semantic_query: str | None
PersonSlot.semantic_query_atoms: list[dict] | None
PersonSlot.description_confidence: str | None
```

`pose_query`는 관절 좌표가 아니며 기존 “VLM은 관절 좌표를 생성하지 않는다”는 불변식을 유지한다.
프롬프트 확장으로 사람 수·박스 정확도가 떨어지지 않는지 기존 슬롯 평가셋으로 회귀 검사한다.
`description_confidence`는 VLM 설명 자체의 자신감일 뿐 검색 결과의 일치 확률이 아니다. 검색 순위나
UI의 높은 신뢰도 판정에 직접 사용하지 않는다.
VLM이 만든 `query_atoms`는 기본적으로 `should`인 검색 제안이며 정확한 관절 measure를 생성하지
않는다. 사용자 원문에 명시된 좌우·부정만 deterministic parser가 `must`/`must_not`으로 승격한다.
atom 허용값은 posecode 생성기·색인·VLM 프롬프트가 공유하는 별도 semantic vocabulary의 단일 소스로
관리한다. 기존 `Action` enum을 세부 의미 검색어로 확장하지 않는다. 이 atom은 semantic engine
전용이며 기존 `tag_dict()`나 geometry `search.search()`에 전달하지 않는다.
따라서 기존 이미지→기하 검색에서 VLM 태그는 계속 `shot + 사람 수`만 제어 신호로 사용한다.
`pose_query`와 `query_atoms`는 별도 의미 검색/fallback 계약이며 action/view 태그 필터를 기하 검색에
되살리는 변경이 아니다.

`vlm_person_id`와 힌트 객체는 `VLMAnalysis → PersonSlot → PersonDescriptor → API PersonOut`까지
같이 이동한다. 최종 배열 index로 query를 다시 결합하지 않는다. invalid box로 슬롯 하나가 빠지거나
좌→우 정렬이 다시 일어나도 뒤 인물의 query가 당겨지면 안 된다. VLM 인물 힌트가 없는
`rtm_provisional` 슬롯에는 자동 의미 query를 만들지 않는다.

---

## 7. 결과 계약 초안

응답에는 실제 실행된 검색 구성과 표시 정책을 포함한다.

```json
{
  "status": "ok",
  "match_source": "semantic_user",
  "search_components": ["dense_e5_exact", "lexical_fts5", "rrf", "atom_constraints"],
  "query_text": "한 손을 들어 인사하는 자세",
  "unparsed_explicit": [],
  "parse_warnings": [],
  "candidates": [
    {
      "semantic_unit_id": "pose:Waving_02",
      "semantic_unit_type": "pose_variant_group",
      "pose_id": "Waving_02",
      "members": [],
      "recommended_view": "front",
      "match_level": "reference",
      "coverage_status": "exact",
      "rank": 1,
      "matched_atoms": ["action_intent:greeting", "right_arm:raised", "body_support:standing"],
      "unknown_constraints": [],
      "selected_variant_reason": "right_arm:raised matched member atom",
      "selectable": true,
      "delivery_mode": "original",
      "refine_allowed": false
    }
  ],
  "gap_constraints": [],
  "unsupported_reason": null,
  "semantic_build_id": "sha256:...",
  "embedding_version": "m-e5-small:<revision>:passage-v1",
  "pose_library_version": "sha256:..."
}
```

`pose_variant_group` 후보는 선택된 concrete `pose_id`를, `pose_set` 후보는 `pose_id=null`과
`set_variant_id`, `members=[{"pose_id": "...", "set_role": "A"}, ...]`를 반환한다. 한 응답의
members는 반드시 같은 set variant에 속한다.

`match_source`는 `geometry | semantic_user | semantic_auto`를 유지하되 의미 검색 후보의
`match_level`은 항상 `reference`다. response `status`는 `ok | library_gap | unsupported`, candidate
`coverage_status`는 `exact | partial`로 분리한다. `library_gap`은 빈 `candidates`와
`gap_constraints`, `unsupported`는 빈 후보와 `unsupported_reason`을 반환한다. 두 상태는 정상적으로
처리된 검색 결과이므로 HTTP 200이고, 잘못된 request는 422, 준비되지 않은 의미 서비스는 503이다.
제약 충족 범위는 확률이 아니며 명시적 `must`와 모순되는 후보는 반환 전에 제외한다.
cosine·FTS5 BM25·RRF는 모두 내부 순위값이므로 공개 계약에서 `semantic_score: 0.82` 같은 0~1
점수는 제거한다. 진단에
원점수가 필요하면 `retrieval_score`와 `score_type`을 내부 로그에만 함께 남긴다.

기존 `CandidateOut.distance`와 `people[].candidates`는 기하 전용으로 유지한다. 의미 후보는
`SemanticCandidateOut`과 `people[].semantic_candidates` 또는 독립 endpoint 응답으로 분리해 기존
refine 흐름이 의미 후보를 기하 후보로 오인하지 않게 한다. `dense`, `rrf`, `hybrid_partial`도 모두
의미 계열이며 `refine_allowed=false`다. `hybrid_partial` 진단에는 `semantic_rank`와
`geometry_distance`를 따로 기록하고 공통 confidence로 합치지 않는다.

`refine_allowed=false` 응답 필드만으로는 서버 안전선이 되지 않는다. `/refine`은 `/analyze`의 geometry
후보에만 발급하는 서버 서명 selection token을 필수로 검증하고 semantic 후보에는 token을 발급하지
않는다. token 도입 전 최소 안전선은 `match_source`를 필수로 받고 `semantic_*`를 거절하며,
`refine_allowed is True`일 때만 실행하는 것이다. 현재 optional boolean을 생략하면 통과하는 동작은
semantic 출시 전에 제거한다.

초기 API는 독립 `POST /semantic-search`로 구현했고 `API_CONTRACT.md`에 반영했다.
request는 `{query, top_k=5, view_hint?}`이며 public caller가 `match_source`를 정하지 않고 서버가
`semantic_user`를 부여한다. 별도 `SEMANTIC_DB_PATH`의 DB나 pinned encoder가 없거나 버전이 맞지
않으면 이 endpoint만 `503 semantic_not_ready`를 반환하고 `/analyze` 기하 경로는 계속 동작한다.
`/healthz`의 `semantic` 객체에는 `ready`, `semantic_unit_count`, `semantic_build_id`,
`embedding_version`, cache/concurrency stats를 기하 상태와 분리해 추가했다.
`SEMANTIC_REQUIRED=0`이면 geometry health는 200을 유지하되 semantic
ready=false를 별도 alert/SLO로 감시한다. `SEMANTIC_REQUIRED=1`인 배포는 semantic mismatch에서
startup/readiness를 실패시켜 조용한 기능 장애를 허용하지 않는다.

---

## 8. 구현 순서

### 0단계 — 라이브러리 inventory·태깅 승인

- ✅ `POSE_LIBRARY_EXPANSION_GUIDE.md` §4의 JSONL 원장과 템플릿을 기준으로 1,308개 inventory를 만들었다.
- ✅ 1,308 concrete BVH의 posecode와 363 source clip·1,308 lineage 원장을
  생성했다. CMU 공식 title 218개를 복구했고 unresolved provenance/license는 review queue에 남겼다.
- ✅ 654개 자동 proposal과 비제품 `tagging_review.v1.db`를 생성하고 structural validator를 통과했다.
- ✅ 미러 임계값 차이 61 unit을 원본 기준으로 정규화하고, P2 observed tag 616 unit을 자동 검증했다.
- ✅ 행동명이 없는 CMU 35 source clip(38 unit)은 검색 제외했고 orphan mirror를 복구했다.
- ✅ 이름 있는 활성 source 328개를 v2로 매핑했다. 행동 ID unknown 46 source/94 unit도 observed
  posecode와 contextual 원본 이름으로 검색 가능하며 fallback 검색 공백은 0이다.
- ✅ 활성 616 unit/1,232 pose member를 2,892개 최종 text document와 5,044개 observed atom으로
  묶었다. 원본/미러는 한 방향 중립 문서 세트를 공유하며 제외 38 unit의 출력 혼입은 0이다.
- ✅ `intfloat/multilingual-e5-small`의 revision·ONNX/tokenizer hash·runtime·pooling·prefix를 고정하고
  2,892개 embedding과 별도 staging `pose_semantics.db`를 만들었다. build validator도 통과했다.
- ✅ DB schema v2에 1,232 concrete member별 PoseCode v2 측정값 27개와 observed atom을 고정했다.

### 1단계 — 현재 기준선 고정

- 스켈레톤 추출 보완을 먼저 병합한다.
- 고정 평가셋에서 hard/soft fallback 인물 수를 기록한다.
- ✅ 현재 semantic build에 고정된 `golden_queries.v2` 45개를 만들었다. 측정 정답 31개,
  source-context 정답 4개, 안전·강건성 10개이며 development 30/holdout 15로 분리했다.
- VLM 의미 검색은 별도 브랜치와 PR로 진행한다.

### 2단계 — 사람 입력 텍스트 검색 PoC

- ✅ `pose_variant_group` 단위의 최종 text document와 typed atom 입력을 만들었다. 실제 다인
  `pose_set`이 추가되면 같은 원자적 member 계약으로 확장한다.
- ✅ pinned embedding과 별도 `pose_semantics.db`를 생성했다. holdout/release promotion 전이므로
  `production_ready=false`를 유지한다.
- ✅ pinned multilingual E5 exact cosine, 보조 FTS, deterministic 한국어 query parser, 3값
  PoseCode matcher, 원본/미러 concrete member selector가 있는 내부 CLI를 만들었다.
- ✅ golden v2 development 30개에서 parser↔GT 1.000, exact P@10 1.000, source-context R@50 1.000,
  no-exact 안전율 1.000, p95 8.4ms를 기록했다. holdout 15개는 열지 않았다.
- 실제 `pose_set` assembler는 다인 semantic unit 데이터가 들어올 때 구현·평가한다.
- 같은 E5 model/revision을 고정하고 `파일명만 → caption·alias 추가 → posecode passage 추가 →
  typed constraint 적용` 순서로 색인 내용의 ablation을 평가한다.
- 사람이 쓴 문장으로 유효성이 확인되기 전에는 이미지 VLM 자동 검색을 붙이지 않는다.

현재 내부 실행:

```bash
.venv/bin/python scripts/semantic_search.py \
  "왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세" --top-k 5
.venv/bin/python scripts/eval_semantic_search.py --split development
```

holdout은 `--allow-holdout --config-frozen`을 동시에 주지 않으면 실행 자체가 차단된다.

### 3단계 — 제품 검색 기능

- ✅ 독립 `POST /semantic-search`, semantic health/version, `semantic_not_ready`, bounded concurrency,
  build-aware LRU query cache를 연결했다. staging manifest는 production에서 fail-closed한다.
- 검색창, Top-5 썸네일, BVH 선택·저장을 연결한다.
- 사용자가 검색어를 수정하고 재검색할 수 있게 한다.
- 실제 작가가 검색 기능을 사용하는지 관찰한다.

### 4단계 — VLM 자동 의미 폴백

- 기존 Gemini 응답에 `vlm_person_id`와 슬롯별 `pose_query`·`query_atoms`·`description_confidence`를 추가한다.
- crop 재추론 후 hard fallback인 슬롯에만 자동 후보를 노출한다.
- `semantic_auto`와 `reference`를 UI와 로그에 표시한다.

### 5단계 — 선택적 reranker·partial cascade 실험

- BGE-M3는 고정 E5+atom 기준선보다 holdout이 개선될 때만 교체 후보로 삼는다.
- lexical+dense RRF는 dense 단독이 소품·고유명 recall을 잃을 때, cross-encoder는 Recall@20은 충분하지만
  Top-5 순서가 나쁠 때만 검토한다.
- `partial/reduced`에서는 의미 Top-N을 만든 뒤 기하로 재정렬한다.
- 후속 구성이 고정 E5+atom보다 개선될 때만 기본 경로에 넣고 모델·색인 버전을 고정한다.

---

## 9. 최소 PoC 평가

### 평가셋 분리

- **development**: 인터뷰 필수 4문장과 진단 의도 15~20개. passage renderer·query parser·atom
  matcher·검색 규칙을 조정하는 데 사용한다. model revision은 이 안에서 먼저 고정한다.
- **holdout**: 개발 중 사용하지 않은 포즈 의도 8~12개. 같은 의도의 동의어·바꿔 말하기는
  development와 holdout에 나눠 넣지 않는다.
- holdout을 열기 전에 `semantic_version`, 검색 설정, `top_k=5`를 고정한다. 결과를 보고 수정한
  질의는 다음 평가부터 development로 이동한다.

인터뷰 필수 4문장은 색인 보정에도 사용하므로 holdout 성능이 아니라 필수 회귀 관문이다.

1. 양손을 허벅지에 대고 상체를 앞으로 숙인 자세
2. 인사하는 자세
3. 창을 던지려고 하는 자세
4. 칼을 들이미는 자세

각 의도는 필요한 범위에서 한국어 표현 2~3개를 둔다.

- 짧은 표현: `인사 자세`
- 자연어 표현: `한 손을 들고 인사하는 자세`
- 동의어·조사 변형: `손을 흔들며 인사하는 포즈`
- 좌우·소품·상호작용을 구분해야 하는 의도에는 해당 조건과 혼동 후보를 넣는다.

추가 골든 쿼리는 아래 두 개를 고정한다.

#### A. 조합형 자세

입력: `왼쪽 다리를 뒤로 들고 양팔은 활짝 편 자세`

```json
{
  "must": [
    "left_leg:raised",
    "left_leg:behind_pelvis",
    "both_arms:widely_spread",
    "both_elbows:extended"
  ],
  "should": ["right_leg:support"],
  "unspecified": ["action"],
  "unparsed_explicit": [],
  "forbidden_inferences": ["dance", "arabesque"]
}
```

dense는 `한발 균형`, `날개처럼 양팔을 벌림`, `arabesque-like` 같은 우회 표현까지 후보로 회수할 수
있지만 action을 사실로 승격하지 않는다. exact 판정은 concrete BVH의 observed atom이 왼다리 상대
높이, body-local 뒤 방향, 양팔의 넓은 벌림과 양 팔꿈치 신전을 모두 만족할 때만 한다. 방향 중립
unit을 찾은 뒤 original과 mirror 중 조건에 맞는 member 하나를 고른다. 반대쪽 다리만 맞고 대응
mirror가 없으면 dense 순위가
높아도 제외한다. `뒤`는 카메라 depth가 아니라 body-local forward axis의 반대다.

#### B. 문화·문맥형 자세

입력: `옛 전통 춤을 추는 자세`

```json
{
  "must": ["activity:dance", "style:traditional", "period_context:historical"],
  "unspecified": ["culture", "dance_name", "specific_period"],
  "unparsed_explicit": [],
  "forbidden_inferences": ["Korean_dance", "탈춤", "부채춤", "elderly_pose"]
}
```

dense와 lexical 채널은 `전통춤`, `민속춤`, `고전 무용`, `traditional/folk/ceremonial dance`를
회수할 수 있지만 전통성의 증거가 되지는 않는다. `traditional`과 `historical`은 각각 공식 catalog
또는 사람 승인 context atom이 있는 후보만 match다. 둘을 하나의 OR atom으로 합치지 않는다. 현재
`cmu_05_*` 샘플의 공식 catalog에는
`dance/arabesque` 근거만 있고 `traditional` 근거는 없으므로 dance만 맞는 partial이며 exact가 아니다.
exact가 없으면 누락 조건을 공개한 partial 후보 또는
`library_gap`을 반환하고 “어느 나라나 특정 춤 종류를 원하나요?”라고 좁힐 수 있다.

판정은 다음처럼 고정한다.

- `exact`: 모든 required constraint가 허용된 evidence로 match한다.
- `partial`: 기본 의도는 맞지만 하나 이상이 unknown이다. 누락 조건을 그대로 표시한다.
- `library_gap`: accepted index에서 기본 의도조차 근거 있게 만족하는 후보가 없다.
- `rejected`: required constraint와 명시적으로 모순된다. retrieval 순위와 무관하게 반환하지 않는다.

### 비교 실험

동일한 질의·라이브러리·Top-5 조건에서 순서대로 비교한다.

1. 파일명만 넣은 E5 dense
2. 승인 caption·alias document를 추가한 E5 dense
3. 2번에 방향 중립 posecode-render document를 추가한 E5 dense
4. 3번에 typed constraint·mirror selector·set assembler를 적용한 제품 후보
5. 같은 문서의 FTS5/BM25 lexical 진단 기준선
6. 4번과 lexical RRF. dense 단독의 명시적 이름·소품 recall을 실제로 보완할 때만 유지
7. BGE-M3 또는 cross-encoder. 고정 E5+atom보다 holdout이 개선될 때만 채택

LLM query parser도 deterministic parser 대비 별도 ablation으로 평가하며, LLM이 없을 때도 4번은
동작해야 한다.

### 판정과 지표

후보는 `바로 사용=2`, `참고 가능=1`, `사용 불가=0`으로 사람이 판정한다.

- `Success@5`: Top-5에 1점 이상 후보가 하나라도 있는 질의 비율
- `nDCG@5`: `바로 사용` 후보가 앞 순위에 오는 정도
- `MRR@5`: 첫 1점 이상 후보의 순위
- `constraint_violation_rate`: 반환 후보의 명시적 `must`/`must_not` 위반률
- 좌우·mirror member 선택, 부정 최소쌍, 정도 bin, 소품·상호작용 조건 정확도
- `unknown_as_false_rate`: 미확인을 부재로 잘못 처리한 비율
- `set_completeness_rate`: 반환 set의 역할 완전성
- `culture_hallucination_rate`: 근거 없이 문화·전통·국가를 exact로 승격한 비율
- `semantic_unit_duplicate_rate = 1 - (Top-5의 고유 semantic unit 수 / 반환 후보 수)`
- `query_encode_ms`, `exact_cosine_ms`, `constraint_match_ms`, 전체 응답의 cold/warm p50/p95와
  Gemini 추가 호출 수
- 실패 원인: `query_description`, `semantic_index`, `library_gap`
- 사용자의 검색어 수정률과 후보 선택률

핵심 회귀 케이스:

- 왼손/오른손 최소쌍과 방향 없는 쿼리의 semantic unit 중복 제거
- 긍정/부정, `살짝/깊게`, 한국어/영어 paraphrase 최소쌍
- 검증된 전통 스타일과 `unknown`의 대조. 한국어 쿼리를 한국 문화로 해석하지 않음
- invalid·순서 변경 VLM box에서도 `vlm_person_id`와 query 결합 유지
- `pose_set` 전 멤버 반환, 일부 멤버 단독 반환 금지, 전 후보 refine 금지
- pose DB와 semantic index 버전 불일치·orphan member에서 fail-closed

표본이 작으므로 평균만 보고하지 않고 질의별 Top-5와 실패 사례를 함께 저장한다.
`semantic_user`와 `semantic_auto`도 합산하지 않는다.

- `semantic_user`: 사용자 원문부터 결과까지 검색 엔진 자체를 평가한다.
- `semantic_auto`: 검색 설정 동결 후 hard fallback 이미지에서 VLM 설명 정확도와 Top-5 유용성을
  각각 평가한다.
- 자동 검색 실패는 `VLM 설명 오류`와 `설명은 맞지만 색인·라이브러리가 실패`한 경우로 나눈다.

초기 통과 기준:

- 필수 4문장 모두 `Success@5`를 통과한다.
- 조합형 골든 쿼리는 올바른 mirror member를 선택하고, 문화형 골든 쿼리는 `dance` 근거만 있는
  후보를 전통춤 exact로 승격하지 않는다.
- E5+typed constraint 제품 후보가 filename-only dense와 lexical 기준선보다 holdout에서 개선된다.
- 후속 모델은 고정 E5+atom 기준선보다 개선될 때만 채택한다.
- 명시적 `must` 위반, incomplete set 반환, culture hallucination은 0건이다.
- 자동 의미 검색이 정상 기하 후보를 덮어쓰지 않는다.
- 의미 검색만 사용한 결과를 높은 기하 일치로 표시하는 경우가 없다.
- semantic unit 중복 제거와 좌우 concrete variant 선택이 동작한다.
- 정상 컷의 Gemini 추가 호출은 0회이고 전체 후보 표시 p95가 기존 5초 예산 안에 든다.

---

## 10. 이번 기능에서 하지 않는 것

- VLM이 2D/3D 관절 좌표 생성
- 의미 검색 후보의 무조건 refine
- 의미 점수와 기하 거리의 임의 가중합
- 모든 컷에서 두 결과를 동시에 노출
- 런타임마다 전체 라이브러리를 VLM에 전달
- 처음부터 전 포즈를 수동 태깅
- 텍스트 검색 검증 전 자동 VLM 캡셔닝 배포
- embedding 유사도만으로 좌우·부정·정도·소품·set 제약을 판정
- 단일 프레임 BVH나 VLM 추측만으로 문화·시대·전통 스타일을 확정
- 고정 E5 기준선보다 holdout 개선 없이 BGE·cross-encoder·vector DB 추가
- semantic atom을 geometry tag filter·`FEATURE_VERSION`·refine 입력에 주입
- PoseScript 모델 재학습이나 자체 text-to-pose 공동 임베딩 학습

---

## 11. 완료 정의

- 사람 입력 텍스트 검색과 VLM 자동 검색이 하나의 의미 색인을 공유한다.
- accepted unit마다 승인 text document, typed atom과 필드별 provenance가 있다.
- pinned encoder 설정, `embedding_version`, `semantic_build_id`, document hash가 재현 가능하게 저장된다.
- `source_clip_id`·`mirror_group_id`·`semantic_unit_id`와 다양성용 `pose_family_id`가 구분된다.
- `pose_variant_group`과 상호작용 `pose_set`, 원본과 mirror concrete variant가 구분된다.
- 좌우 member atom/mirror 변환과 set 완전성이 자동 검증된다.
- `must`/`must_not` 위반과 unknown-as-false가 0건이고 `exact/partial/library_gap`이 구분된다.
- 결과가 `geometry`, `semantic_user`, `semantic_auto`로 구분된다.
- hard fallback 슬롯에서 참고 Top-5와 수정 가능한 검색어를 제공한다.
- 의미 검색 후보에는 refine 금지 정책이 적용된다.
- 평가 스크립트가 필수 문장과 holdout 문장의 Top-5 결과를 재생성한다.
- 기존 geometry Top-K, `knn_geometric`, `FEATURE_VERSION` 회귀 결과가 바뀌지 않는다.
- API·BFF·클라이언트의 표시 계약과 실패 상태가 문서화된다.

---

## 12. 차용 연구·문서

| 연구·문서 | 이 설계에서 차용하는 것 | 지금 하지 않는 것 |
|---|---|---|
| [PoseScript: Linking 3D Human Poses and Natural Language](https://arxiv.org/abs/2210.11795) | 3D 관절에서 low-level posecode/atom을 결정적으로 만들고 자연어 document와 연결 | 학습 모델·데이터셋을 그대로 런타임에 도입 |
| [Multilingual E5](https://arxiv.org/abs/2402.05672) · [공식 model card](https://huggingface.co/intfloat/multilingual-e5-small) | 초기 한국어/영어 dense 후보 회수와 `query:`/`passage:` 규약 | 공개 checkpoint 성능을 우리 포즈 검색 성능으로 간주 |
| [SQLite FTS5](https://www.sqlite.org/fts5.html) | 고유 동작·문화명·짧은 소품명을 보존하는 보조 lexical 채널과 진단 기준선 | lexical 일치를 semantic retrieval 본체로 표시 |
| [BEIR](https://arxiv.org/abs/2104.08663) | lexical·dense·rerank를 같은 holdout에서 비교하는 평가 원칙 | 공개 벤치마크 순위를 우리 도메인 성능으로 간주 |
| [BGE-M3](https://arxiv.org/abs/2402.03216) | E5 대비 dense 품질과 dense+sparse 통합이 필요한지 비교 | 개선 확인 전 sparse·multi-vector까지 도입 |
| [Reciprocal Rank Fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) | BM25와 dense처럼 척도가 다른 순위를 결합 | 원점수 임의 정규화·가중합 |
| [NevIR: Negation in Neural Information Retrieval](https://arxiv.org/abs/2305.07614) | bi-encoder 순위에 부정 판정을 맡기지 않고 explicit `must_not`과 최소 대조쌍 평가 | cross-encoder도 부정의 사실 판정기로 신뢰 |
| [PoseEmbroider](https://arxiv.org/abs/2409.06535) | 향후 이미지·텍스트·3D pose를 함께 쓸 때 uncommon pose의 세밀한 multimodal 정렬을 평가할 근거 | 현재 geometry 검색과 text semantic 검색을 하나의 점수로 합침 |
| [TMR: Text-to-Motion Retrieval](https://arxiv.org/abs/2305.00976) | multi-frame motion과 설명이 확보된 뒤 춤·리듬·motion phase 검색을 검토할 후보 | 1-frame BVH에 sequence 모델을 적용하거나 전통성을 추론 |

PoseScript는 설계 아이디어만 독립 재구현한다. 공식 코드·데이터를 직접 반입하려면 프로젝트 배포
조건과 라이선스를 별도로 검토한다.
