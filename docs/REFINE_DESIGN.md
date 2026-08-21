# Refine 설계

> 상태: v1 fallback·공통 안전 게이트 기준 · 갱신일: 2026-08-18 · 기준 코드: `src/refine.py`, `src/collision.py`, `api/app.py`, `api/models.py`
>
> 제품 기본은 `REFINE_V2_ENABLED=1`, `REFINE_DEFAULT_MODE=aggressive`인 v2.5 safe aggressive다.
> 이 문서는 v1 fallback과 v2가 공유하는 기초 안전 게이트를 보존하며, 현재 v2.5 정책은
> `REFINE_V2_DESIGN.md`를 따른다.

## 목적

검색된 BVH 후보를 러프의 2D 스켈레톤에 맞춰 자동 미세조정한다. 3D를 새로 생성하지 않고, 검색 포즈가 가진 깊이와 해부학을 출발점으로 사용한다.

핵심 약속은 **좋아지거나, 그대로**다. 조정이 안전하지 않으면 베이스 BVH를 반환한다.

현재 제품 흐름은 다음과 같다.

```text
/analyze → 베이스 Top-5 표시 → 사용자 선택 → /refine → 조정본 또는 베이스
```

Top-5 선행 refine은 현재 API 기본 흐름이 아니다. 도입하려면 조정본 preview, 캐시, 영속 저장을 함께 구현해야 한다.

## 입력과 출력

입력:

- 선택한 `pose_id`, 매칭 `view`
- `/analyze`가 반환한 COCO-17 `keypoints`, 유효 `scores`
- 선택 후보의 `search_distance`
- `refine_allowed`, `refinable_limbs`

출력:

- `refined`: 실제 조정본 채택 여부
- `reason`: 적용·스킵·복구 이유
- `bvh_url`: 조정본 또는 베이스 BVH 경로
- `limbs`, `limb_decisions`: 최종 채택 사지와 안전 진단

`refined=false`는 오류가 아니다. 안전 게이트가 베이스를 유지한 정상 결과다.

## 조정 범위

BVH 회전 채널만 바꾼다.

| 부위 | 처리 |
|---|---|
| 어깨·팔꿈치 | 기본 조정 대상 |
| 고관절·무릎 | `REFINE_LIMBS=all`일 때만 후보 |
| 손목·손가락·발목 | 고정 |
| 루트 위치·힙·척추·목·머리 | 고정 |
| HIERARCHY·OFFSET | 원본 그대로 유지 |

기본값은 `REFINE_LIMBS=arms`다. 실제로 움직일 사지는 스켈레톤 품질 단계의 `refinable_limbs`와 관측 감도 게이트가 다시 제한한다.

러프에서 한 사지의 필요한 뼈가 온전히 보이지 않으면 그 사지 전체를 동결한다.

## 목적함수

전방 계산:

```text
BVH 회전값
→ FK로 3D 관절 계산
→ 검색 view로 2D 투영
→ 검색과 같은 normalize_skeleton
→ 2D 뼈 방향
```

최적화는 다음 항을 함께 최소화한다.

1. 러프와 투영 포즈의 뼈 방향 차이
2. 베이스 회전에서 멀어지는 비용
3. 관측하기 어려운 개별 축의 추가 정규화(P1a)
4. 관측하기 어려운 축 조합의 사지별 block-SVD 정규화(P1b)

scipy `least_squares`를 우선 사용하고, 사용할 수 없거나 진행하지 못하면 numpy LM으로 전환한다. 두 백엔드는 같은 안전 판정을 내야 한다.

## 안전 처리

### 적용 전

다음 경우 최적화를 시작하지 않는다.

- `REFINE_ENABLED=0`
- `refine_allowed=false`
- 스켈레톤 score 또는 유효 뼈 부족
- 검색 거리가 허용 임계값 초과
- 베이스가 멀티프레임 BVH
- 얽힘 `set_id`가 있는 포즈
- 베이스가 이미 충분히 러프와 가까움

### 최적화 중

- 파라미터는 베이스 회전 ± `REFINE_MAX_DELTA_DEG`로 제한한다.
- 사지별 관측 감도가 낮으면 해당 사지를 동결한다.
- P1a는 잘 보이지 않는 오일러 축을 베이스에 더 강하게 고정한다.
- P1b는 사지별 야코비안 block-SVD로 null-space 방향을 억제한다.

### 최적화 후

- 개선량이 부족하면 전체 결과를 폐기한다.
- P2는 3D 이동량을 항상 기록한다. 하드 게이트는 현재 기본 OFF다.
- P3는 베이스보다 새로 깊어진 손·전완-몸통 관통을 검사하고 실패 팔만 복구한다.
- 새 팔꿈치·무릎 굽힘 위반이 생기면 해당 사지만 복구한다.
- 사지 복구 뒤 전체 개선과 충돌 불변식을 다시 확인한다.

사지 하나가 실패해도 다른 사지의 유효한 조정은 유지할 수 있다. 모든 사지가 탈락하면 베이스 BVH를 반환한다.

## 주요 reason

| reason | 의미 |
|---|---|
| `disabled` | refine 비활성화 |
| `skeleton_policy` | 스켈레톤 품질 정책상 금지 |
| `low_skeleton_score` | 타깃 신뢰도 부족 |
| `base_mismatch` | 검색 실패 |
| `entangled_set` | 얽힘 세트라 개별 refine 금지 |
| `insufficient_target_bones` | 맞출 유효 뼈 부족 |
| `low_observability` | 관측 가능한 사지 없음 |
| `already_matched` | 베이스가 이미 충분히 가까움 |
| `no_gain`, `global_no_gain` | 개선량 부족 |
| `movement_gate` | P2 이동량 초과 |
| `collision_gate`, `collision_unresolved` | P3 충돌 판정 또는 복구 불변식 실패 |
| `joint_limit` | 새 해부학 위반 |
| `diverged` | 최적화 실패 |
| `ok`, `ok_partial` | 전체 또는 일부 사지 채택 |

정확한 HTTP 필드와 전체 enum은 [`API_CONTRACT.md`](API_CONTRACT.md)와 `api/models.py`를 따른다.

## v1 fallback 기본 설정

| env | 기본값 | 의미 |
|---|---:|---|
| `REFINE_ENABLED` | `1` | 전체 비상 스위치. production에서는 명시적으로 설정 |
| `REFINE_LIMBS` | `arms` | `arms` 또는 `all` |
| `REFINE_MIN_GAIN` | `0.95` | 필요한 최소 손실 개선 비율 |
| `REFINE_MIN_LOSS` | `0.01` | 이하면 이미 맞은 것으로 판정 |
| `REFINE_MAX_ITER` | `100` | 최적화 반복 상한 |
| `REFINE_LAMBDA` | `0.05` | 베이스 정규화 강도 |
| `REFINE_MAX_DELTA_DEG` | `45` | 채널별 베이스 이탈 상한 |
| `REFINE_OBS_GATE` | `1` | 사지별 관측 감도 게이트 |
| `REFINE_AXIS_OBS` | `1` | P1a 축별 정규화 |
| `REFINE_SVD_OBS` | `1` | P1b block-SVD 정규화 |
| `REFINE_MOVE_GATE` | `0` | P2 하드 게이트. 진단은 항상 기록 |
| `REFINE_COLLISION_GATE` | `1` | P3 팔-몸통 충돌 게이트 |
| `REFINE_MIN_BEND_DEG` | `20` | 팔꿈치·무릎 최소 굽힘각 |

세부 임계값의 최종 기준은 `src/config.py`다.

## 불변식

1. 검색 실패를 refine으로 억지 보정하지 않는다.
2. 스켈레톤 단계가 허용하지 않은 사지는 움직이지 않는다.
3. 쿼리·라이브러리·refine 전방 계산은 같은 피처 함수를 사용한다.
4. P1b는 사지별 block-SVD로 계산해 서로 다른 사지를 섞지 않는다.
5. 안전 게이트에서 탈락한 사지는 베이스 회전으로 정확히 복구한다.
6. HIERARCHY, 관절명, OFFSET, 채널 순서는 변경하지 않는다.
7. `refined=false`에도 사용 가능한 베이스 BVH를 반환한다.

## 검증 기준

자동 테스트:

- scipy와 numpy 판정 일치
- 검색 실패·스켈레톤 금지·얽힘 포즈 스킵
- 사지별 동결·복구와 반대 사지 유지
- 새 자기 충돌 검출과 베이스 접촉 보존
- 굽힘각·이동량 경계값
- 출력 BVH 재파싱과 HIERARCHY 보존

실 러프 검증:

- 매칭 view 하나만 보지 않고 여러 view에서 확인
- 2D 목적함수 감소를 독립 품질 지표로 오해하지 않음
- 3D 이동량, 충돌, 해부학, 사용자 눈 평가를 함께 기록
- 반복 튜닝한 calibration과 새 holdout을 분리

## 현재 한계와 통합 상태

- COCO-17만 사용하므로 손바닥·손가락·발 방향을 직접 추론하지 못한다.
- 머리·척추·루트 위치는 조정하지 않는다.
- 잘못된 베이스나 라이브러리 공백은 refine으로 해결할 수 없다.
- 조정본은 현재 추론 태스크 로컬 파일이다. 다중 태스크 운영 전 공유 영속 저장 또는 동일 태스크
  라우팅이 필요하다.
- BFF는 `/refine` 응답의 최종 `bvh_url`을 보존해야 한다. 기존 `/export-order`만 다시 호출하면
  조정본이 원본 URL로 되돌아갈 수 있다.

통합 조건은 [`REFINE_API_V25_BACKEND_BFF_HANDOFF.md`](REFINE_API_V25_BACKEND_BFF_HANDOFF.md)를 따른다.

## 구현 과정 기록

P1~P3의 실험 수치와 당시 판단은 [`archive/refine/`](archive/refine/)에 보관한다. 현재 동작을 바꿀 때는 새 단계 문서를 만들지 말고 이 문서와 코드를 함께 갱신한다.
