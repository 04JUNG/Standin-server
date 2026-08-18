# Semantic Vocabulary v2

> 상태: 2026-08-18 표준 어휘·328개 source mapping·616개 최종 문서·pinned E5 staging DB 완료
> 단일 소스: `config/semantic_vocab.v2.json`
> 검증기: `src/semantic_vocab.py`

## 목적

런타임 라우팅용 `src.schema.Action`과 라이브러리 의미 검색 어휘를 분리한다. 원본 이름은 변경하지
않고 `raw_action_label` 또는 source context로 보존한다. hard facet에는 승인된 canonical ID만 쓰지만,
행동 ID가 비어도 observed posecode와 contextual 원본 이름으로 후보 검색은 유지한다.

## 필드

| 필드 | 수 | 의미 |
|---|---:|---|
| `action_domain` | 15 | 이동·춤·스포츠·전투 등 큰 행동 영역 |
| `action_ids` | 97 | `walk`, `step_over`, `swordplay`, `traditional_dance` 등 구체 행동 |
| `posture` | 10 | 서기·앉기·쪼그리기·한발 균형·역자세 등 현재 지지 상태 |
| `motion_phase` | 7 | 준비·수행·접촉·회수·유지·전환·unknown |
| `style_context` | 12 | 전통·인도 전통·발레·브레이크댄스 등 근거가 필요한 문맥 |
| `intended_props` | 28 | 검·골프채·휴대전화 등 source/사람 근거가 필요한 소품 |
| `interaction_kind` | 12 | solo·대화·포옹·키스·싸움·위로 등 다인 관계 |

## 불변식

1. source clip의 복합 행동명은 선택된 단일 프레임의 행동 정답이 아니다.
2. `unknown`은 구체 ID와 함께 저장하지 않는다.
3. `action_ids`의 domain은 `action_domain`에 포함돼야 한다.
4. 단일 프레임으로 보이지 않는 소품·동작 단계·문화 문맥은 source catalog 또는 사람 근거가 없으면
   비워 두거나 `unknown`으로 둔다.
5. 부분 문자열이나 LLM 자유 생성값을 canonical ID로 자동 채택하지 않는다. 정확 alias 후보를 만든
   뒤 vocabulary validator와 승인 결정을 통과시킨다.
6. 기존 기하 검색과 런타임 `Action` enum은 변경하지 않는다.
7. `source_action_ids=[]` 또는 `unknown`은 semantic 검색 제외나 해당 행동의 부재를 뜻하지 않는다.
8. 원본 이름은 lexical/dense 후보 회수용 contextual 문서로 사용할 수 있지만 pose truth나 hard filter로
   사용하지 않는다.

## 예시

```json
{
  "raw_action_label": "indian dance",
  "source_action_ids": ["traditional_dance"],
  "pose_action_ids": ["dance_step"],
  "action_domain": ["dance"],
  "posture": ["one_leg_balance"],
  "motion_phase": "unknown",
  "style_context": ["traditional_indian"],
  "intended_props": [],
  "interaction": {"kind": "solo"}
}
```

## 현재 라이브러리 정책

- 행동명이 없고 포즈 품질이 낮다고 결정된 CMU 35 source clip / 38 semantic unit은
  `config/library_exclusions.v1.json`에서 semantic·geometry·release 모두 제외하고 실제 삭제를
  대기한다.
- `_00882` Typing UsingMouse orphan mirror는 역미러 원본을 생성해 완전한 미러쌍이 됐다.
- 기존 semantic vocab v1 proposal은 덮어쓰지 않았다. 이름이 있는 328 source clip은
  `data/semantic/action_mapping.v2.jsonl`에 vocabulary v2 source-context mapping으로 별도 저장했다.
- 328개 모두 observed posecode와 source context fallback으로 검색 가능하다. 그중 행동 ID가 비어 있는
  46 source clip/94 semantic unit도 검색 공백 없이 유지되며, 사람 검수는 행동 정확도 개선용이다.
- 활성 616 semantic unit은 `data/semantic/search_documents.v2.jsonl`의 최종 문서 세트로 렌더링했다.
  `posecode_render`는 observed, canonical/source context는 candidate-only로 분리하며 어느 문맥 문서도
  hard filter로 사용하지 않는다.

## 다음 단계

1. 고신뢰 단일 행동 source mapping을 승인 후보로 확정한다.
2. 중간신뢰·facet-only·unknown 항목은 사용·검색 실패 빈도를 기준으로 우선순위 검수한다.
3. ✅ v2 source mapping과 observed posecode를 결정적 text document renderer에 전달했다.
4. ✅ `multilingual-e5-small` model/revision·renderer·runtime을 고정하고 2,892개 embedding과 별도
   staging `pose_semantics.db`를 생성했다.
5. ✅ 현재 semantic build에 맞춘 `golden_queries.v2` 45개를 development 30/holdout 15로 고정했다.
6. development로 dense+atom runtime을 구현·조정한 뒤 holdout으로 production 승격 여부를 결정한다.
