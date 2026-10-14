# 자연어 포즈 생성 초안 — 2026-10-14

사용자 요청에 따라 제한된 자연어 조건 → 새로운 1프레임 BVH 생성기를 구현했다.
`src/experimental/text_pose_generation.py`, `tests/test_text_pose_generation.py`를 별도 보존한다.
시맨틱 검색 세션 수입 요청으로 우선순위를 전환했으므로 완성된 생성 제품 기능으로 보고하지 않는다.

기본 리그 HIERARCHY/OFFSET만 이용하고 기존 프레임 motion은 0으로 초기화한다.
명시적 앉기/서기 + 팔짱/V자 팔 조합의 새 회전을 two-bone IK로 계산하며,
VLM이 좌표를 만들거나 기존 refine을 실패 러프에 실행하는 방식이 아니다.
기존 실제 Gemini T013 설명으로 생성기를 직접 실행했을 때 3개 변형이
팔 교차·무릎 90도·뼈 길이 및 팔-몸통/팔-다리/팔-팔 proxy 검사에 통과했다.
이는 피부 메시·손목 twist·손가락·관절 해부학 축·의자 접촉·3D 정답 검증이 아니다.

명시적 `Pipeline(pose_generator=...)` 주입에서만 성공적으로 수행된 빈 시맨틱 검색 뒤 실행한다.
search error/설명 오류/일반적인 자세/정상 geometric/명시적 bust·half·head/얽힘 차단에서는 실행하지 않는다.
생성물은 별도 `generation.candidates`에 연구 미리보기로 기록하고 기존 검색 후보는 그대로다.
release/refine/actual_mesh_validated는 항상 false다. 기본 API factory는 생성기를 주입하지 않는다.

현재 1995개 라이브러리에는 ext_seated_folded_arms가 있어, 기존 승인 8컷의 저장 설명을
현재 관계 검색기로 재실행하면 생성 hook이 실행되지 않는 것이 정상이다(11슬롯 not_requested).
이 결과를 “8컷 생성 성공”으로 보고하지 않는다. 생성기 단독 구조 실험과 새 검색 연결 검증을 분리한다.
스크립트 출력 `tmp/text-pose-generation-eight-20261014/results.json`은 이 미요청 상태를 기록한다.

단위 검사 34개와 회귀를 통과했다. 실제 인체 자산/실행 환경 검증 및 생성 미리보기 수용 검사는 남아 있다.
전체 자연어를 이해하는 학습형 text-to-motion 모델은 아니며 미지원/부정/비대칭/충돌 문장은 보류한다.
IK 회전 변환에는 기존 SciPy Rotation을 사용한다:
[SciPy Rotation 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html).
