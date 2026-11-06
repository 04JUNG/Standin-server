# P0 그룹과 가시성 검수 후속

114쌍 전부를 4개 contact sheet와 모호한 대상의 원본 이미지로 시각 검수했다.
출처는 `agent_visual_review`다. 사람의 독립 검수나 직접 좌표 라벨 작성으로 표시하지
않으며, part visibility와 직접 관절 좌표 관측을 구분한다. 원본 파일은 보존했다.

## 분류와 가시성

| 표현 | 개수 |
|---|---:|
| 외곽 러프 | 66 |
| 일러스트·만화 | 25 |
| 렌더 참고 | 6 |
| 사진 | 6 |
| 과장 캐릭터 | 2 |
| 졸라맨·실루엣 | 9 |

전신 56, 상체 34, 일부 16, 얼굴만 6, 전역 대상 모호 2로 분류했다. 각 샘플에
머리·몸통·골반과 좌우 팔/손/다리/발 15부위의 visible/partial/occluded/cropped/unknown,
판독 이유와 이미지 SHA를 기록했다. 부위가 보인다고 관절 좌표가 확정된 것은 아니다.
073/077/097처럼 내부 팔꿈치·손목 구분이 없는 그림은 해당 관절 근거를 unknown으로
남겼다. 졸라맨·과장 캐릭터는 인체 외곽 피팅을 억제한다.

047.3/067.3/072.1/087.2/087.3/091.2는 얼굴/뒤통수만 있어 body-refine 대상으로
평가하지 않는다. 023.2/055/059/071/093.2는 대상 귀속이 모호하다. 특히 059는 두
인물이 겹치고, 093.2의 표시는 손/가방 영역이므로 별개의 전신 인물로 확정하지 않는다.
이 항목들도 원장에서 삭제하지 않는다. 모델 confidence나 GT에 본이 존재한다는
사실로 가려진 부위를 visible로 올리지 않았다. 생산용 pose authority는 모두 false다.

## 그룹과 분리

같은 원본 번호의 모든 인물을 함께 묶고 추가로 001–003, 005/006, 008/009,
014/015, 021/022, 065/084를 보수적인 시각 source-risk family로 연결했다.
이 연결은 같은 작가라는 인증이 아니라 유사 자료의 분할 누출을 줄이기 위한 것이다.
114샘플·100원본 컷이 93개 risk group에 속한다.

과거 v4 카탈로그와 이미지 SHA를 대조해 035/036/042/043/058/060의 실제 사용 이력을
찾았고 개발 split으로 고정했다. 과거 CB 번호와 현재 번호를 단순 대응하지 않았다.
기존 P1에서 평가한 001–010/041/074 및 앞서 검토한 사례도 개발에 둔다.

원본 컷 기준 개발/검증/평가 = 60/20/20, 샘플 기준 63/26/25다. 같은 원본과 risk
family가 split을 건너가지 않는지 자동 검사했다. 작가·원작 출처는 제공되지 않아
artist_id는 unknown이며 `artist_disjoint_verified=false`, `independent_blind=false`다.
이 분할은 다음 개발 실험에 사용할 탐색용 분할이며 독립 작가 blind 성능을 주장하지 않는다.

## 결과와 재현

- `evaluation/refine_v5/visual_review_v2.json`: 전량 판독·가시성·근거·기존 노출 이력.
- `evaluation/refine_v5/split-v2.json`: 고정 split과 risk group.
- `evaluation/refine_v5/p0-review-summary-v2.json`: 집계와 hash.
- private `data/curated_inventory_v2.json`: 기존 inventory와 최신 검수를 결합한 버전.

```sh
python -m experiments.refine_v5.curation --data PRIVATE_DATA \
  --review evaluation/refine_v5/visual_review_v2.json --out NEW_CURATED_JSON
```

같은 입력 hash로 재현하고, tamper·중복·잘못된 visibility·거짓 사람 인증은 거절한다.
기존 기계 감사 4개와 그룹·검수 계약 4개를 포함해 P0 8개 테스트가 통과한다.
