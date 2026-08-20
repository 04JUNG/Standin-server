# Semantic search document build 보고서

> 생성일: 2026-08-18  
> 상태: 최종 검색 문서 세트 생성 완료 · pinned embedding staging DB 생성 완료

## 결과

| 항목 | 수 |
|---|---:|
| 활성 source mapping | 328 |
| 최종 semantic unit 문서 세트 | 616 |
| 연결 pose member | 1232 |
| observed unit atom | 5044 |
| text document | 2892 |
| 행동 unknown이지만 검색 가능한 unit | 94 |
| 검색 공백 | 0 |
| 제외 source clip | 35 |
| 제외 semantic unit | 38 |
| 제외 unit 출력 혼입 | 0 |

## 문서 채널

- `posecode_render`: BVH에서 관찰한 방향 중립 자세. typed atom 제약에 사용 가능하다.
- `canonical_context`: source-level canonical 행동·domain·style 후보. pose truth와 hard filter가 아니다.
- `source_context`: 원본 이름의 문맥 검색용. candidate-only이며 좌우 이름은 `one side`로 중립화한다.
- 행동 ID나 이름이 없어도 `posecode_render`가 있으면 검색 대상에 남는다.

## 대표 문서

| Unit | 행동 상태 | 문서 구성 | 예시 검색 문맥 |
|---|---|---|---|
| `pose:cmu_05_03_00150` | mapped | posecode_render, posecode_render, canonical_context, canonical_context, source_context | dance - sideways arabesque, turn step, folding arms |
| `pose:rokoko_FootTapping_mixamo_00040` | unknown | posecode_render, posecode_render, source_context | Foot Tapping |
| `pose:cmu_144_10_02831` | mapped | posecode_render, posecode_render, canonical_context, canonical_context, source_context | one side Front Kicking001 |

## 재현성

- pose library version: `sha256:22eb5e9c24a954c11b68f684f327a71e42b694a9aed7e721589d30d84f724c76`
- semantic vocabulary: `sha256:23b8e29b451a20228096b310eb86539f1197627f4b3ac362cf8b284664bca809`
- passage template version: `1`
- semantic build input ID: `sha256:bbb1ce794f754aa82fdae9a352181ac68adb89321977e2b0d62ec7fd027fdb0a`

## 산출물

- `data/semantic/search_documents.v2.jsonl`: 616개 최종 문서 세트와 observed atom
- `data/semantic/search-document-summary.v2.json`: coverage·버전·fingerprint 요약
- `scripts/build_semantic_documents.py`: 재생성 builder
- `src/semantic_documents.py`: 결정적 renderer와 fail-closed 검증

## 남은 단계

이 문서를 입력으로 pinned E5 staging `pose_semantics.db`를 생성했다. 아직 production DB는
아니며 고정 평가셋과 semantic runtime/API 구현·승격 검증이 남아 있다.
