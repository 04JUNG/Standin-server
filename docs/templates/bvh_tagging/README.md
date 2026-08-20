# BVH 태깅 템플릿 v1

이 디렉터리에는 실제 BVH나 운영 태그를 넣지 않는다. 버전 관리할 **형식 예시**만 두고,
실제 산출물은 Git에서 제외된 `data/semantic/`에 생성한다.

```text
data/semantic/
├─ inventory.v1.jsonl       # 파일·해시·BVH QA·미러 관계, 재생성 가능
├─ source_clips.v1.jsonl     # 원본 clip별 제공처·공식 이름·제공처 번호
├─ pose_lineage.v1.jsonl     # BVH별 내부 번호·선택 frame·변환 이력
├─ library_numbers.v1.json   # append-only 내부 BVH 번호 registry
├─ proposals.v1.jsonl       # 자동 태깅 제안, append-only
├─ decisions.v1.jsonl       # 사람 승인·수정·거절, append-only
├─ review_queue.csv         # 위 JSONL에서 만든 검수용 파생 파일
├─ missing_action_names.csv # 행동명이 비어 있는 source clip 번호 목록
├─ provenance_review_queue.csv # 출처·번호·frame 검수용 파생 파일
├─ tagging_review.v1.db     # auto-verified/P0/P1 proposal·atom 작업 색인
├─ tagging-summary.v1.json  # 배치 집계·production blocker
├─ tagging-validation.v1.json # 구조 검증·남은 review item
├─ snapshots/               # 공식 catalog snapshot
└─ builds/<semantic_build_id>/
   ├─ semantic-build.json  # 승인 revision·passage/embedding 설정 고정
   └─ pose_semantics.db    # 재생성 가능한 dense/atom staging 색인
```

파일 역할:

- [`pose_member.v1.example.json`](pose_member.v1.example.json): `pose_member_inventory`, concrete BVH
  1개의 기술·출처·미러 관계
- [`source_clip.v1.template.json`](source_clip.v1.template.json): 원본 제공처·공식 이름·제공처 고유 번호
- [`pose_lineage.v1.template.json`](pose_lineage.v1.template.json): 각 BVH의 원본 연결·프레임·샘플 순번·변환 이력
- [`provenance_samples.v1.jsonl`](provenance_samples.v1.jsonl): CMU catalog와 로컬 원본을 사용한 실제 샘플
- [`tagging_review_card.cmu_05_03_00150.v1.example.json`](tagging_review_card.cmu_05_03_00150.v1.example.json):
  출처·번호·의미·기하를 합쳐 사람이 확인하는 실제 BVH 샘플 카드
- [`provenance_review_queue.v1.template.csv`](provenance_review_queue.v1.template.csv): 원본 정보 검수용 파생 컬럼
- [`semantic_proposal.v1.example.json`](semantic_proposal.v1.example.json): 원본/미러 공통 의미 atom과
  멤버별 posecode 측정값/observed atom 제안
- [`review_decision.v1.example.json`](review_decision.v1.example.json): 사람의 append-only 승인·수정 이력
- [`review_queue.v1.template.csv`](review_queue.v1.template.csv): 검수 화면/스프레드시트용 파생 컬럼

초기 인벤토리는 다음 명령으로 만든다.

```bash
python scripts/init_bvh_tag_inventory.py \
  --bvh-dir data/bvh \
  --output data/semantic/inventory.v1.jsonl
```

posecode·출처/계보·자동 proposal·검수용 작업 색인은 다음 명령으로 만든다.

```bash
python scripts/build_semantic_tagging.py \
  --inventory data/semantic/inventory.v1.jsonl \
  --bvh-dir data/bvh \
  --raw-dir data/_action_raw \
  --output-dir data/semantic \
  --cmu-catalog-html data/semantic/snapshots/cmu-search-20260814.html \
  --cmu-catalog-captured-at 2026-08-14

python scripts/validate_semantic_tagging.py --output-dir data/semantic
```

생성되는 `tagging_review.v1.db`는 관절 기반 자동 검증 결과와 P0/P1 제안을 조회하는 작업 색인이다.
CMU는 검수 화면에서 출처를 `CMU`로 표시하고, 원본·미러 bucket 차이는 원본 관찰 atom 기준으로
자동 정규화한다. 승인된 행동 문맥과 pinned embedding이 없으므로 제품 `pose_semantics.db` 또는
`/semantic-search` 입력으로 사용하면 안 된다.

`inventory.v1.jsonl`은 사람이 직접 고치지 않는다. 잘못된 자동 제안은 proposal을 덮어쓰지 말고
새 proposal revision 또는 decision의 `edits`로 남긴다. 상세 절차와 승인 게이트는
[`POSE_LIBRARY_EXPANSION_GUIDE.md`](../../POSE_LIBRARY_EXPANSION_GUIDE.md)의
§4 태깅 원장·검수 절을 따른다. 사람이 embedding용 문장을 별도로 작성하지 않으며, 승인된 내용에서
버전된 renderer가 text document를 만들고 pinned multilingual encoder가 색인을 생성한다. 검색용
atom·embedding 계약은 [`VLM_SEMANTIC_SEARCH.md`](../../NEXT_SPRINT/VLM_SEMANTIC_SEARCH.md) §5를 따른다.
`review_decision`은 `reviewed_fields`만 승인한다. 나머지 proposal 필드는 drop/unknown이며, nested
`generated | needs_review` 값의 최종 상태는 그 필드를 명시한 decision overlay로 판정한다.
