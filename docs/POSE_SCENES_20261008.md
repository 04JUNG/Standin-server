# 2026-10-08 참고 그림 기반 포즈 추가

사용자가 제공한 세 장의 이미지를 동작 의도 참고용으로만 사용했다. 이미지는 라이브러리와 배포 번들에 포함하지 않는다. 몸·손가락은 Quaternius CC0 원본 리그를 바탕으로 3D 목표점에서 새로 제작했다. 단일 프레임 BVH와 Standin Master V2 4방향 미리보기를 생성했다.

| 포즈 ID | 동작 | 배치 |
| --- | --- | --- |
| `combat_pair_airborne_press_attacker` | 몸을 낮춰 공중에서 덮치는 공격자 | `combat-pair-20261008-r3` |
| `combat_pair_airborne_press_defender` | 등을 대고 누워 양팔로 막는 방어자 | `combat-pair-20261008-r3` |
| `story_seated_ledge_recline` | 턱에 기대 한쪽 다리를 뻗은 앉기 | `seated-story-20261008-r4` |
| `story_seated_ledge_knees_hug` | 턱에 앉아 무릎에 팔을 얹고 웅크리기 | `seated-story-20261008-r4` |
| `sports_recovery_hands_on_knees` | 숨을 고르며 양손을 무릎에 짚기 | `sports-recovery-20261008-r2` |
| `sports_skate_side_leg_reach` | 한 발 균형을 잡고 옆으로 든 다리에 손 뻗기 | `sports-recovery-20261008-r2` |

전투 2개는 `relationship=fighting`, 같은 `set_id=combat_pair_airborne_press`, 역할 `A/B`로 색인한다. 한 BVH에는 한 사람만 담는다. 현재 추론 파이프라인은 얽힌 2인 장면의 세트 검색을 지원하지 않으므로, 자동 매칭을 보장하지 않는다. 인물별 포즈를 수동으로 선택하고 클립 스튜디오에서 상대 위치를 맞춰야 한다. 좌석은 미리보기의 별도 가이드이며 BVH 관절이 아니다.

최종 6개 모두 단일 프레임·손가락 30관절·4방향 캐릭터 미리보기·목표 도달·허리 분절 회전·팔꿈치 방향·팔/몸통 관통 검사를 통과했다. 4방향을 직접 확인하고 현재 BVH/미리보기 해시에 검수 결과를 묶었다. 이 확인은 두 인물을 실제로 합성한 장면의 피부 접촉·상대 위치까지 검증하지 않는다.
