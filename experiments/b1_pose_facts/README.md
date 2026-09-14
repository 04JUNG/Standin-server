# 실험 검색 0 → A → B1

이 디렉터리는 메인 스켈레톤 검색의 정확도 실험을 위한 문서 경계다. 구현 본체는
`src/experimental/`, 테스트는 `tests/experimental/`에만 둔다. 운영 검색에는
`src/pipeline.py`의 default-off 훅과 `src/config.py`의 플래그만 추가한다.

## 운영 경계와 0단계 기반

- 기존 geometry 엔트리의 **동일한 immutable snapshot**에서 A/B1 인덱스를
  메모리로 만든다. snapshot ID는 pose ID, projection view, float32 feature의
  SHA-256이다.
- 실험 코드는 `data/poses.db`, `data/semantic/`, BVH, 운영 semantic index를
  읽거나 쓰지 않는다. 별도 DB 빌드와 마이그레이션도 없다.
- A와 B1이 모두 `off`면 실험 모듈을 import/build하지 않고 trace도 만들지 않는다.
  반환 후보 객체, 순서, 거리, confidence/refine 판단은 기존 경로와 같다.
- 실험 인덱스 생성·평가가 실패하면 geometry 결과로 fail-open 한다. 요청 처리 전에
  끝난 confidence, fallback, crop, stability, refine 판정은 A/B1이 바꾸지 않는다.

파이프라인의 적용 순서는 다음과 같다.

```text
기존 geometry 검색과 모든 안전 판정
  → A 최소 구조 게이트
  → B1 포즈 팩트 리랭크
  → CutResult 조립
```

## A — 최소 구조 게이트

A v1의 값은 `stand | non_stand | unknown`뿐이다. 공중, 점프·낙하, 기대기,
접촉 물체, 바닥 종류, 행동 의미는 판정하지 않는다.

자동 게이트 조건은 의도적으로 좁다.

- 쿼리가 `valid + full coverage + full_body search`여야 한다.
- 양쪽 hip/knee/ankle이 모두 관측되어야 한다.
- 양쪽 무릎이 명확히 굽은 경우만 쿼리를 `non_stand`로 본다.
- 라이브러리 pose는 서로 다른 projection 최소 2개가 `stand`에 합의하고
  `non_stand` 표가 0개일 때만 제외한다. `unknown`은 항상 통과한다.
- 필터 후에도 기존과 같은 Top-K 수가 있어야 한다.
- 새 Top-1 또는 Top-K 평균 거리가 각각 기존 값의 기본 125%를 넘으면 기존
  후보로 복구한다.

A는 후보 집합을 다시 검색할 수 있으므로 `shadow` 평가를 먼저 거쳐야 한다.

## B1 — 포즈 팩트 리랭크

B1은 A가 넘긴 **현재 Top-K 안에서만** 순서를 바꾼다. 후보 추가·삭제, 거리 변경,
confidence/refine 재판정은 하지 않는다. 쿼리와 라이브러리 projection에 같은 2D
규칙 추출기를 사용한다.

현재 팩트는 다음의 보수적 부분집합이다.

- 좌·우 팔꿈치 굽힘
- 좌·우 손목의 어깨 대비 높이
- 좌·우 무릎 굽힘
- 몸통 좌우 기울기

각 임계값 사이에는 dead zone이 있다. 관절 결측, 퇴화 벡터, dead zone의 사실은
만들지 않으며 비교에서 `unknown`이 된다. 후보는 먼저 모순 수가 적은 순서,
그다음 일치 수가 많은 순서로 놓고, 동률이면 원래 geometry 순서를 유지한다.

## 플래그와 롤백

```env
EXPERIMENTAL_A_SUPPORT_MODE=off       # off | shadow | on
EXPERIMENTAL_A_MAX_DISTANCE_RATIO=1.25
EXPERIMENTAL_B1_POSE_FACT_MODE=off    # off | shadow | on
```

- `off`: 기존 결과 완전 보존, 실험 trace 없음
- `shadow`: 기존 결과를 반환하고 `descriptor.quality_trace`에 제안만 기록
- `on`: 안전 조건을 통과한 제안을 최종 후보에 반영

즉시 롤백은 두 mode를 모두 `off`로 바꾸고 프로세스를 재시작하면 끝이다. DB
재빌드, 파일 이동, 데이터 복원은 필요 없다.

권장 승격 순서는 `둘 다 off → A shadow → B1 shadow → A on → B1 on`이다. A/B1을
독립적으로 끌 수 있으므로 어느 단계에서 문제가 생겼는지 분리할 수 있다.

## 검증

```bash
.venv/bin/python tests/experimental/test_a_b1_search.py
.venv/bin/python tests/test_search_family_grouping.py
.venv/bin/python tests/test_smoke.py
```

현재 범위에는 C 카메라 전략, 감정, 2인 상호작용, 자연어 semantic endpoint,
family/mirror collapse 이전 후보 복구가 포함되지 않는다.
