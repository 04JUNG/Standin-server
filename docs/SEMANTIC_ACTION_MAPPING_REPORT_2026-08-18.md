# Semantic action mapping v2 보고서

> 생성일: 2026-08-18  
> 대상: 검색 제외되지 않은 source clip 328개 (원본 이름 있음 328개)  
> 상태: source context 자동 매핑 완료, pose 단위 행동 정답으로는 미승인

## 결과

| 항목 | 수 |
|---|---:|
| 전체 source clip | 328 |
| 행동 ID 매핑 | 242 |
| facet만 매핑 | 40 |
| canonical 행동 미매핑 | 46 |
| 고신뢰 | 125 |
| 중간신뢰 | 157 |
| 저신뢰 | 46 |
| 복합 source 이름 | 31 |
| canonical 정확도 검수 권장 | 203 |
| 규칙/어휘 검증 오류 | 0 |
| 안전한 fallback 검색 가능 | 328 |
| 행동 unknown이지만 검색 가능 | 46 |
| fallback 검색 공백 | 0 |

## 판정 기준

- 원본 행동명은 그대로 보존하고 camel case·밑줄·문장부호만 검색용으로 정규화했다.
- 버전 관리되는 결정적 규칙과 vocabulary exact alias만 사용했다. LLM 자유 생성값은 없다.
- `mapped`는 canonical 행동 ID가 있는 항목, `facets_only`는 domain/posture 등만 안전하게
  판정한 항목, `unknown`은 이름만으로 행동을 확정하지 않은 항목이다.
- `source_action_ids`는 원본 clip 전체의 문맥이다. 선택된 한 프레임의 행동 정답이 아니다.
- 단일 프레임의 `pose_action_ids`는 별도 pose 검수에서만 확정한다.
- `motion_phase`는 전 항목 `unknown`이다. 이름만 보고 준비·타격·회수 단계를 만들지 않았다.
- 행동명이 없던 CMU 35개 source clip은 이 매핑에서 제외됐다.

## 행동 ID가 없어도 검색되는 방식

- 활성 616개 semantic unit은 BVH에서 계산한 posecode 관절 atom과 한·영 자세 문서로 검색한다.
- 원본 이름은 `source_context`로 lexical/dense 후보 회수에 사용하되 행동 정답이나 hard filter로
  사용하지 않는다.
- 안전하게 판정된 domain·posture·style·prop만 canonical context 후보 신호로 추가한다.
- 비어 있는 행동 ID는 검색 제외 조건이 아니다. `unknown`도 해당 행동의 부재를 뜻하지 않는다.
- 따라서 사람 검수는 검색 가능 여부가 아니라 행동명 정확도를 높이기 위한 후속 작업이다.

## 대표 판정

| Source | 원본 이름 | canonical 결과 | 신뢰도 |
|---|---|---|---|
| `cmu:144_06` | Front_Kicking001 | `kick` | high |
| `cmu:81_09` | drag heavy object | `pull` | high |
| `cmu:131_06` | Start Duck Underneath Stop | facet: `transition`, `crouching` | medium |
| `cmu:80_12` | fishing | facet: `daily_activity` | medium |
| `local_action_raw:Floating` | Floating | facet: `airborne` | medium |
| `rokoko:BurstThroughDoor` | Burst Through Door | `push`, `run` | medium |
| `rokoko:MiddleFingers` | Middle Fingers | `dismiss_gesture` | high |

고신뢰 125개는 source context 자동 승인 후보이다. 중간신뢰 157개는 복합 이름 또는 근사 canonical
매핑이므로 CSV에서 검수하며, 저신뢰 46개는 행동 근거가 부족해 `unknown`을 유지했다.

## 주요 action domain

| Domain | Source 수 |
|---|---:|
| `locomotion` | 69 |
| `combat` | 43 |
| `dance` | 29 |
| `object_interaction` | 28 |
| `transition` | 28 |
| `social_interaction` | 17 |
| `sport` | 16 |
| `performance` | 16 |
| `gesture` | 15 |
| `exercise` | 14 |
| `acrobatics` | 13 |
| `idle` | 13 |
| `daily_activity` | 11 |

## 주요 canonical action

| Action | Source 수 |
|---|---:|
| `dance_step` | 27 |
| `walk` | 21 |
| `jump` | 20 |
| `turn` | 16 |
| `kick` | 11 |
| `run` | 10 |
| `hold_pose` | 10 |
| `jete` | 9 |
| `arabesque` | 8 |
| `climb` | 7 |
| `swordplay` | 7 |
| `cartwheel` | 7 |
| `punch` | 7 |
| `push` | 7 |
| `pick_up` | 6 |
| `stretch` | 6 |
| `jumping_jack` | 5 |
| `talk` | 5 |
| `pirouette` | 4 |
| `swim` | 4 |
| `block` | 4 |
| `clean` | 4 |
| `wave` | 4 |
| `squat` | 4 |
| `creep` | 4 |
| `reach` | 4 |
| `pull` | 4 |
| `threaten` | 4 |
| `salsa` | 4 |
| `step` | 3 |

## canonical 행동 미매핑 목록

- `cmu:102_03` — SuperTightLeft
- `cmu:104_18` — SpasticStop
- `cmu:105_07` — mummy8
- `cmu:120_02` — Gorilla
- `cmu:120_15` — Mickey Surprised
- `cmu:120_16` — Mickey Surprised
- `cmu:123_09` — 12.5 lbs
- `cmu:142_02` — Clumsy
- `cmu:27_11` — prairie dog (human subject)
- `cmu:28_15` — monkey (human subject)
- `cmu:54_01` — monkey (human subject)
- `cmu:54_02` — bear (human subject)
- `cmu:54_05` — pterosaur (human subject)
- `cmu:54_06` — pterosaur (human subject)
- `cmu:54_08` — roadrunner (human subject)
- `cmu:54_13` — dragon (human subject)
- `cmu:54_18` — squirrel (human subject)
- `cmu:54_23` — monkey (human subject)
- `cmu:54_25` — chicken (human subject)
- `cmu:55_03` — genie (human subject)
- `cmu:55_08` — monkey (human subject)
- `cmu:55_09` — monkey (human subject)
- `cmu:55_16` — panda (human subject)
- `cmu:55_17` — ghost (human subject)
- `cmu:55_19` — dragon (human subject)
- `cmu:55_24` — various animals (human subject)
- `cmu:55_26` — monkey (human subject)
- `cmu:79_49` — monkey
- `cmu:79_66` — movie and trial dont match
- `cmu:85_13` — BadStartSequnce
- `local_action_raw:Entry` — Entry
- `local_named:Finding` — Finding
- `local_named:Looking` — Looking
- `local_named:Reacting` — Reacting
- `local_named:Scared` — Scared
- `local_named:Surprised` — Surprised
- `rokoko:Contestant` — Contestant
- `rokoko:Flirty` — Flirty
- `rokoko:FootTapping` — Foot Tapping
- `rokoko:Host` — Host
- `rokoko:Judge` — Judge
- `rokoko:Lawyer_01` — Lawyer 01
- `rokoko:Lawyer_02` — Lawyer 02
- `rokoko:Shamwow_Guy` — Shamwow Guy
- `rokoko:StomacheIssue` — Stomache Issue
- `rokoko:Witness` — Witness

## 산출물

- `data/semantic/action_mapping.v2.jsonl`: 재현 가능한 전체 매핑 원장
- `data/semantic/action_mapping_review.v2.csv`: 사람이 보기 쉬운 검수표
- `data/semantic/action-mapping-summary.v2.json`: 집계와 fingerprint
- `config/semantic_action_mapping_rules.v2.json`: 사용한 결정적 매핑 규칙
- vocabulary fingerprint: `sha256:23b8e29b451a20228096b310eb86539f1197627f4b3ac362cf8b284664bca809`
- mapping rules fingerprint: `sha256:8dffdb50bb1671e186f40cab25cc32238392d83e222e236cd02ca073b9eebe68`

## 다음 단계

고신뢰 단일 행동은 source context 승인 후보로 사용할 수 있다. 중간·저신뢰 및 복합 이름도
posecode·원본 문맥으로 검색 가능하지만, pose-level action은 사람 검수 없이 자동 승계하지 않는다.
