# 스켈레톤 추출 보완 파이프라인

> 목적: 웹툰 러프의 다인·가림 장면에서 RTMPose 스켈레톤의 누락·중복·인물 합침을 감지하고,
> 정상 컷의 처리시간은 거의 늘리지 않으면서 검색과 refine에 안전한 입력을 제공한다.
>
> 기준일: 2026-08-03 · 구현 브랜치 `feat/skeleton-extraction-improvement`
>
> 상태: **런타임 구현 완료, 실데이터 임계값 보정 대기.**
> mask-aware 거리, coverage, 라이브러리 검증, `PersonSlot` 전역 배정, 합침·소유권 구조 검사,
> 조건부 crop, pose-family A/B 안정성 진단, unstable partial의 사후 crop 재검사,
> 최종 좌→우 정렬, API 품질 메타데이터와 refine 사지 제한까지 연결됐다.
> 남은 일은 고정 평가셋으로 metric×coverage·구조 비율 임계값을 보정하는 것이다.

---

## 0. 결론

기존 검색 앞단을 다음처럼 보완한다.

```text
[선행] 최소 평가 라벨 고정
→ mask-aware 검색 순위 거리 구현
→ coverage confidence + metric×coverage 임계값 보정
[런타임] VLM → 라우팅
→ VLM PersonSlot 생성
→ RTMPose 전체 이미지 1회
→ 슬롯-스켈레톤 전역 일대일 배정
→ 조건을 만족하는 unmatched RTM은 provisional 슬롯으로 보존
→ 몸통·사지 품질 검사
→ 불량 관절 마스킹
→ missing/suspect 슬롯만 crop 재추론 1회
→ partial/suspect만 family-aware 검색 안정성 검사
→ unstable partial은 미재시도 상태일 때만 crop 1회 후 재검사
→ 최종 x1 기준 person_index
→ 피처화 → 검색 → refine
```

핵심 정책은 다음 다섯 문장이다.

1. **정상 컷에는 모델 호출을 추가하지 않는다.**
2. **검색은 관대하게, refine은 엄격하게 허용한다.**
3. **스켈레톤이 이상하다는 이유만으로 결과를 폐기하지 않고, 불량 부분을 제거한 뒤 검색 결과가 안정적인지 확인한다.**
4. **관절 마스킹보다 먼저 검색 거리와 score 게이트를 mask-aware하게 바꾼다.**
5. **검색 순위 거리는 후보를 정렬할 뿐이며, confidence는 거리와 coverage를 함께 보고 결정한다.**

---

## 1. 대원칙과 비목표

### 대원칙

1. 포즈 모델을 교체하지 않는다. 현재 `RTMPose Body(mode="performance")`를 유지한다.
2. 사용자가 캡처부터 Top-5/BVH를 받기까지 오래 기다리지 않게 한다.
3. VLM은 관절 좌표를 만들지 않는다. 사람 수·shot·대략 박스만 제공한다.
4. 쿼리와 라이브러리는 기존 `normalize_skeleton`과 거리 규약을 공유한다.
5. 잘못된 스켈레톤에는 refine을 돌리지 않는다.
6. Gemini 슬롯은 강한 힌트지만 GT가 아니다. 구조적으로 정상인 unmatched RTM을 무조건 버리지 않는다.
7. 서로 다른 관절 마스크에서 나온 raw 거리를 같은 confidence 척도로 비교하지 않는다.

### 이번 범위에서 하지 않는 것

- Wholebody/DWPose/새 검출기 등 모델 교체
- 모든 인물 무조건 crop
- 여러 padding으로 반복 추론
- 손·손가락 keypoint 추가
- 잘못된 관절 좌표 자동 생성·보간
- 실패 사유마다 새 슬롯 생성
- 무거운 외부 최적 배정 의존성 추가
- 모든 정상 슬롯에 안정성 kNN을 한 번 더 수행

---

## 2. 현재 구현의 문제

현재 실제 RTMPose 경로는 다음과 같다.

```text
Gemini 분석
→ RTMPose가 전체 이미지에서 자체 검출·포즈 추정
→ VLM 사람 수와 RTMPose 결과 개수만 비교
→ 개수가 부족하면 덜 겹치는 VLM 대략 박스로 crop 복구
→ RTMPose 반환 리스트 순서대로 person 0, 1, 2 부여
```

한계:

- 개수만 맞으면 합쳐진 스켈레톤과 중복 스켈레톤을 구분하지 못한다.
- Gemini 박스와 RTMPose 스켈레톤의 일대일 대응이 없다.
- 현재 기준 브랜치에서는 `person_index`가 좌→우 순서라는 보장이 없다.
- crop 복구 인물은 기존 리스트 뒤에 붙는다.
- 전체 평균 score만으로는 한쪽 사지가 다른 인물에게 붙은 상태를 잡기 어렵다.
- 결측 관절이 검색에 잘못 관여할 수 있다. `DISTANCE=angle`은 무효 뼈를 제외하지만,
  `pose_distance`는 별도 masked 처리가 필요하다.
- 현재 기본 검색 metric은 `DISTANCE=pos`다. 마스킹된 `(0,0)` 관절을 실제 힙 위치처럼
  비교하므로 sanitized score 도입 전에 반드시 수정해야 한다.
- `_search_one`은 전체 관절 평균 score를 사용한다. sanitized score의 0이 늘면 사용할 수 있는
  몸통과 사지가 남아 있어도 조기 fallback할 수 있다.
- `normalize_skeleton`은 score 마스킹 전에 hip과 shoulder로 중심·스케일을 계산한다.
  hip/shoulder가 불량한 슬롯은 normalize 이전에 차단해야 한다.
- 유효 관절만 평균해도 문제가 끝나지 않는다. 사지가 빠지고 정규화된 몸통만 남으면 검색 공간의
  정보량이 줄어 Top-1 거리가 오히려 작아지며, 정보가 적은 쿼리가 더 높은 confidence를 받는 역설이 생긴다.

좌→우 정렬은 `feat/left-to-right-person-tagging`의 커밋 `6d000c3`에 구현돼 있지만
기준 브랜치에는 병합되지 않았다. 이 정렬은 슬롯 복구가 끝난 뒤 최종 `person_index`를
부여하는 용도로 재사용하는 것이 맞다.

---

## 3. 용어

| 용어 | 뜻 |
|---|---|
| slot | Gemini가 제안한 인물 1명을 파이프라인 끝까지 관리하는 내부 자리 |
| slot_id | crop 재시도 중에도 유지되는 내부 ID |
| person_index | 최종 사용자에게 노출하는 좌→우 번호 |
| VLM box | Gemini가 반환한 0~1 대략 박스를 픽셀 좌표로 바꾼 값 |
| RTM candidate | 아직 슬롯에 배정되지 않은 RTMPose 스켈레톤 |
| provisional slot | Gemini가 놓쳤지만 구조적으로 정상인 unmatched RTM을 저신뢰로 보존한 슬롯 |
| sanitized scores | 불량 관절을 0으로 마스킹한 검색/refine용 score |
| rank distance | 같은 query mask 안에서 라이브러리 후보를 정렬하는 masked 평균 거리 |
| coverage class | 유효 몸통과 완성 사지 수로 구분한 `full/reduced/sparse/insufficient` 정보량 등급 |
| pose family | 원본과 `_mirror` 등 사용자 관점에서 같은 포즈 계열을 묶은 ID |
| soft fallback | 베이스 Top-5는 제공하지만 저신뢰로 표시하고 refine을 제한하는 결과 |
| hard fallback | 검색 자체를 신뢰할 수 없어 자동 후보/refine을 제공하지 않는 결과 |

`왼쪽/중앙/오른쪽`은 설명용 표현일 뿐 구현 데이터가 아니다. 실제 슬롯은 숫자 박스 좌표를 가진다.

---

## 4. 최소 PersonSlot

런타임 슬롯에는 제어에 필요한 값만 둔다.

```python
@dataclass
class PersonSlot:
    slot_id: int
    slot_origin: str                 # vlm | rtm_provisional
    vlm_box: BBox | None
    skeleton: Skeleton | None = None
    valid_joint_mask: np.ndarray | None = None
    state: str = "missing"
    skeleton_source: str = "none"   # none | full_image | crop_retry
    retry_count: int = 0
    reasons: tuple[str, ...] = ()
```

다음 값은 런타임 슬롯에 중복 저장하지 않는다.

- `skeleton_box`: `skeleton`에서 계산
- `limb_validity`: `valid_joint_mask`에서 계산
- `coverage_class`: `valid_joint_mask`와 몸통 구조 검사 결과에서 계산
- `retry_box`: VLM 슬롯은 `vlm_box + padding`으로 계산. provisional 슬롯은 기본적으로 crop하지 않음
- `person_index`: 최종 정렬 때 계산
- 상세 매칭 비용·품질 구성요소·시간: 평가용 `SlotTrace`로 분리

평가·디버그 환경에서는 다음 trace를 별도로 남긴다.

```python
@dataclass
class SlotTrace:
    slot_id: int
    assigned_rtm_index: int | None
    assignment_cost: float | None
    assignment_margin: float | None
    quality_components: dict
    coverage_class: str | None
    distance_metric: str | None
    rank_distance: float | None
    confidence_threshold: float | None
    before_state: str
    after_state: str
    retry_reason: str | None
    retry_elapsed_ms: float
```

---

## 5. 단계별 파이프라인

### 5-0. 선행 조건 — masked 순위 거리와 coverage confidence

이 단계는 슬롯·품질 검사·관절 마스킹보다 먼저 구현하고 보정해야 한다.

#### 문제 A — 결측 관절의 `(0,0)` 오염

현재 기본값 `DISTANCE=pos`의 `pose_distance`는 정규화된 몸통 12관절을 전부 평균한다.
결측 관절은 `(0,0)`이므로 “관절이 보이지 않음”이 아니라 “관절이 힙 중심에 있음”으로
비교된다. sanitized scores로 불량 관절을 더 많이 0으로 만들수록 이 오염이 커진다.

feature의 `(0,0)` 좌표만 보고 결측을 추론하지 말고 query mask를 검색 함수에 명시적으로 전달한다.

```python
knn_geometric(
    entries,
    query_feature,
    query_valid_mask,
)
```

위치 순위 거리는 유효한 query 관절만 평균한다.

```python
valid = query_valid_mask[BODY]
rank_distance = np.linalg.norm(A[valid] - B[valid], axis=1).mean()
```

`angle_distance`도 좌표가 `(0,0)`인지 추론하지 말고 `valid_joint_mask`에서 만든 뼈 마스크를
명시적으로 받는다. 단, BVH 관절이 매핑됐다는 사실과 특정 카메라에서 뼈 방향이 2D로 관측된다는
사실은 다르다. 투영 후 길이가 거의 0인 뼈는 결측이 아니라 해당 view의 `unobservable`로 분리하고,
query/library 양쪽의 관측 가능 뼈 교집합만 각도 거리에 사용한다.

- 거리 함수와 호출 인터페이스만 바꾸면 저장 feature 형식은 그대로이므로 `FEATURE_VERSION`을 올리지 않는다.
- library mask를 DB feature와 함께 저장하면 표현 규약이 바뀌므로 version을 올리고 재색인한다.
- `hybrid`도 masked `pose_distance`와 masked `angle_distance`를 사용한다.
- 레거시 `knn()`과 실제 경로 `knn_geometric()`의 결측 규약을 함께 수정한다.

#### 문제 B — 관절을 줄일수록 confidence가 올라가는 역설

자기 pose family를 제외한 20개 쿼리의 관측은 다음과 같다.

| 남긴 관절 | 유효 관절 | Top-1 거리 | 기존 폴백율 (`>0.45`) |
|---|---:|---:|---:|
| 전신 | 12 | 0.2216 | 5% |
| 상체 | 8 | 0.1072 | 0% |
| 몸통 | 4 | 0.0200 | 0% |

표본이 작아 최종 임계값을 확정할 수는 없지만, raw 거리 분포가 마스크에 따라 달라진다는
구조적 문제를 확인하기에는 충분하다. 원인은 단순히 평균 분모가 작아지는 것만이 아니다.
정규화된 몸통은 포즈 간 차이가 작고, 사지가 제거된 낮은 차원의 검색 공간에서는 라이브러리 중
우연히 가까운 후보를 찾기 쉬워진다.

따라서 masked 평균 거리는 **같은 쿼리 안에서 후보를 정렬하는 rank distance**로만 사용한다.
서로 다른 마스크 쿼리의 confidence를 raw 거리 하나로 비교하지 않는다.

```text
후보 순위 = masked rank distance
confidence = slot state + coverage class + metric별 보정 거리 + 검색 안정성
```

#### Coverage class

완성 사지는 팔 또는 다리의 두 뼈가 모두 유효한 경우다. 다음은 초기 안전 정책이며,
정확한 경계는 고정 평가셋으로 보정한다.

| coverage | 구조 조건 | high confidence 자격 | refine |
|---|---|---|---|
| `full` | 정상 몸통 + 완성 사지 3~4개 | 있음 | 유효 사지 허용 |
| `reduced` | 정상 몸통 + 완성 사지 2개 | 있음. 전용 임계값·안정성 통과 필요 | 유효 사지만 허용 |
| `sparse` | 정상 몸통 + 완성 사지 0~1개 | **없음** | 금지 |
| `insufficient` | 몸통 또는 전체 유효 뼈가 hard invalid 기준 미달 | 없음 | 금지 |

`sparse`는 Top-5가 작가에게 유용할 가능성이 있으므로 자동 hard invalid로 만들지 않는다.
다만 거리가 아무리 작거나 family stability가 높아도 `person_confidence=high`로 승격하지 않고,
베이스 Top-5만 제공하는 soft fallback으로 제한한다.

#### Confidence 판정과 임계값

단일 `FALLBACK_DISTANCE`는 폐기하고 최소한 다음 2차원으로 보정한다.

```python
threshold = FALLBACK_THRESHOLDS[distance_metric][coverage_class]
```

- `pos/full`, `pos/reduced`
- `angle/full`, `angle/reduced`
- 사용하는 경우 `hybrid/full`, `hybrid/reduced`

`sparse`와 `insufficient`에는 high confidence 임계값을 두지 않는다. 임계값은 특정 폴백률을
맞추는 방식이 아니라, 고정 평가셋의 `top5_usefulness`에서 요구 precision을 만족하도록 정한다.
`HYBRID_W`를 사용할 경우 pos/angle 상대 스케일도 같은 평가셋에서 함께 보정한다.

현재 `_search_one`의 “전체 score 평균 < 0.2”도 sanitized scores와 정합되지 않는다.
마스킹 이후 검색 가능 여부와 confidence는 다음을 1급 입력으로 사용한다.

```text
slot.state
coverage_class
유효 몸통 뼈 수
유효 전체 뼈 수
완성 사지 bitset
distance_metric별 rank distance와 threshold
```

refine의 `low_skeleton_score`도 effective scores와 `refinable_limbs` 기준에서 의도대로 동작하는지
함께 점검한다.

#### 라이브러리 불변식

현재 라이브러리 6,040개 projection은 body 관절 매핑 결측이 0개임을 확인했다. 이 가정을
암묵적으로 두지 않고 BVH를 읽은 직후 또는 `build_entries_from_pose()`에서 body score assertion을 건다.
`build_db()`는 이미 scores가 제거된 feature를 받으므로 `(0,0)`만 보고 결측 여부를 판단하면 안 된다.

```python
if not np.all(scores[COCO_BODY_INDICES] >= KPT_THRESHOLD):
    raise ValueError(f"{pose_id}: missing mapped BVH body joints")
```

DB 저장 단계에서는 feature shape, finite 여부와 명시적인 `body_complete` metadata만 검증한다.
다른 관절명을 쓰는 BVH 소스가 들어왔을 때 조용히 불완전한 라이브러리를 만들지 않는 것이 목적이다.

### 5-1. VLM 분석과 라우팅

기존과 동일하다.

```text
Gemini → num_people, shot, approx_boxes
face → skip
bust → 검색 스킵
full_half → 슬롯 파이프라인 진행
```

VLM 박스를 받으면 먼저 다음을 검증한다.

- 모든 좌표가 finite인가
- 이미지 범위로 clamp 가능한가
- `x1 < x2`, `y1 < y2`인가
- 너무 작은 박스가 아닌가
- 유효 박스 개수와 `num_people`이 일치하는가

불일치하면 익명 VLM 슬롯을 억지로 만들어내지 않고 `count_confidence=low`로 기록한다.
유효 VLM 박스가 0개여도 RTMPose 전체 추론은 실행하며, 이후 엄격한 조건을 통과한 RTM candidate를
provisional 슬롯으로 보존할 수 있다.

### 5-2. 임시 슬롯 생성

`full_half`에서 유효한 VLM 박스 하나당 `slot_origin="vlm"` 슬롯 하나를 만든다.

```text
slot 0: origin=vlm, vlm_box=A, skeleton=None, state=missing
slot 1: origin=vlm, vlm_box=B, skeleton=None, state=missing
slot 2: origin=vlm, vlm_box=C, skeleton=None, state=missing
```

이 시점의 `slot_id`는 요청 내부 추적용이다. Gemini 배열 순서를 사용자 인물 번호로 노출하지 않는다.

### 5-3. RTMPose 전체 이미지 1회

정상 경로의 모델 호출 수를 유지한다.

```python
skeletons = pose_model.estimate(image, None, img_w, img_h)
```

반환 스켈레톤에는 아직 사용자 인물 번호를 붙이지 않고 `RTM candidate 0..M-1`로 관리한다.

### 5-4. 슬롯-스켈레톤 전역 일대일 배정

각 RTM candidate에서 스켈레톤 박스와 몸통 중심을 계산한다.

초기 매칭 비용은 과설계하지 않고 다음 두 항목으로 시작한다.

```text
1. VLM box와 skeleton box의 IoU 비용
2. VLM box 중심과 skeleton 몸통 중심의 정규화 거리
```

규칙:

- 슬롯 하나에 스켈레톤 최대 하나
- 스켈레톤 하나가 슬롯 최대 하나
- 전체 조합 비용이 최소인 배정을 선택
- 비용이 임계보다 크면 배정하지 않음
- 1위와 2위 비용 차이가 작으면 `assignment_ambiguous`
- `x1` 순서는 매칭 비용으로 쓰지 않음

인물 수가 작으므로 scipy 없이 가능한 조합을 비교하거나 작은 전용 배정 함수를 구현할 수 있다.

배정되지 않은 슬롯은 `missing`, 배정되지 않은 RTM candidate는 `unmatched_skeletons`에 둔다.

Gemini가 사람을 놓쳤을 때 정상 RTM까지 영구 손실되지 않도록, unmatched candidate 중 다음 조건을
모두 만족하는 항목만 `slot_origin="rtm_provisional"` 저신뢰 슬롯으로 승격한다.

- torso 구조 정상
- 유효 몸통·전체 뼈 수 기준 통과
- 기존 VLM 슬롯 및 다른 RTM과 중복되지 않음
- 기존 슬롯 박스와 낮은 중첩
- keypoints가 finite이고 정규화 기준이 퇴화하지 않음

provisional 슬롯은 `person_confidence=low`로 표시하고 기본적으로 베이스 Top-5만 제공한다.
unmatched RTM이 남았다는 이유만으로 전부 슬롯으로 만들지는 않는다. RTMPose 과검출·중복을
사용자 결과로 승격하지 않기 위해서다.

개수 일치는 계속 컷 단위 신뢰도 신호로 사용하되, 인물별 신뢰도는 슬롯 배정과 구조 검사로 보완한다.

### 5-5. 몸통·사지별 품질 검사

전체 평균 score 하나로 판정하지 않고 다음 단위로 검사한다.

```text
torso
left_arm
right_arm
left_leg
right_leg
```

#### 유효 뼈

COCO-17 뼈의 양 끝 score가 모두 `0.3` 이상이면 해당 뼈를 유효하다고 본다.

#### 몸통 검사

- 어깨·골반을 이용해 몸통 방향을 계산할 수 있는가
- 어깨중점–골반중점 길이가 퇴화하지 않았는가
- 몸통 중심이 슬롯 박스와 지나치게 벗어나지 않는가
- 몸통 관절이 서로 다른 슬롯에 분산된 강한 증거가 있는가

hip/shoulder는 `normalize_skeleton`의 중심·스케일 기준이다. 이 관절들이 불량하면 다른 관절을
마스킹하는 것만으로는 위치 피처 오염을 막을 수 없으므로 normalize 이전에 `suspect/invalid_torso`로
분기한다.

초기 퇴화 기준:

```text
몸통 길이 < 인물 박스 높이의 5% → torso_degenerate
```

#### 사지 검사

- 상완-팔꿈치-전완, 대퇴-무릎-정강이 연결되는가
- 관절 score가 사지 내부에서 일관적인가
- 사지가 다른 슬롯 박스로 과도하게 넘어가는가
- 몸통 대비 뼈 길이가 극단적인가

웹툰 비율 과장을 고려해 뼈 길이 이상은 곧바로 hard invalid로 쓰지 않는다.
사지 마스킹 또는 `suspect`를 발생시키는 보조 신호로만 사용한다.

### 5-6. 불량 관절 마스킹

관절 좌표를 새로 만들거나 이동하지 않는다. 불량 관절의 검색/refine용 score만 0으로 만든다.

```python
sanitized_scores = original_scores.copy()
sanitized_scores[~slot.valid_joint_mask] = 0.0
```

원본 keypoints와 raw scores는 평가·디버깅을 위해 보존한다. 검색/refine에는
`valid_joint_mask`를 반영한 effective scores를 사용한다.

```text
raw_scores       = RTMPose 원본
valid_joint_mask = 구조 품질 판정
effective_scores = raw_scores, 단 invalid 관절은 0
```

효과:

- `pose_distance`: 명시적인 관절 mask로 불량 관절 제외
- `angle_distance`: 관절 mask와 view별 관측 가능 뼈 mask의 교집합으로 불량·퇴화 뼈 제외
- 검색: 정상 몸통·사지 정보 보존
- refine: 불량 사지 자동 동결

§5-0의 masked 순위 거리, coverage confidence와 metric×coverage 임계값 보정이 완료되기 전에
sanitized scores 마스킹을 활성화하면 검색 품질과 confidence가 모두 퇴행한다.

### 5-7. 조건부 crop 재추론

다음 경우에만 해당 슬롯을 crop한다.

- `missing`
- `assignment_ambiguous`
- `merge_suspected`
- `invalid_torso`

다음은 기본적으로 crop하지 않는다.

- 손목 하나의 낮은 score
- 한쪽 사지만 불량한 로컬 `partial`
- 라이브러리 공백으로 검색 거리가 높은 경우

규칙:

```text
slot당 최대 1회
컷 전체 최대 1~2명
padding 15~25% 단일 설정
추가 padding 반복 없음
실패 시 바로 다음 판정으로 진행
```

현재 `estimate_crop()`은 crop 안에서 평균 score가 가장 높은 스켈레톤을 선택한다.
이를 다음 기준으로 바꿔야 한다.

- 몸통 중심이 슬롯 중심과 가까운가
- 어깨·골반이 원래 슬롯 박스에 포함되는가
- 구조 품질이 원본보다 좋은가

crop은 새 슬롯을 만들지 않고 같은 슬롯의 `skeleton`, `state`, `skeleton_source`, `retry_count`를 갱신한다.

### 5-8. 검색 안정성 검사

스켈레톤 구조가 일부 이상해도 라이브러리 검색 결과는 유용할 수 있다.
따라서 구조 이상만으로 곧바로 fallback하지 않는다.

이 검사는 모든 정상 슬롯에 적용하지 않는다. 추가 검증이 필요한 `partial/suspect`에만 수행한다.

의심 사지를 포함한 검색과 제외한 검색의 Top-5를 비교한다.

```text
검색 A: 현재 유효 관절 전체
검색 B: 의심 사지까지 제거한 보수적 마스크
```

원본과 `_mirror`가 서로 다른 `pose_id`이므로 raw pose ID 교집합을 그대로 사용하지 않는다.
라이브러리의 `pose_family_id`를 기준으로 접어서 비교한다.

```python
family_overlap = len(
    set(top5_A.pose_family_ids) & set(top5_B.pose_family_ids)
) / 5
```

초기 MVP에서는 `_mirror` 접미사를 제거한 canonical ID를 사용할 수 있다. 장기적으로는
문자열 규칙 대신 DB metadata에 다음을 저장한다.

```json
{"pose_family_id": "Pose_A", "mirrored": true}
```

초기 기준:

| 공통 pose family | 판정 |
|---:|---|
| 3~5개 | stable |
| 1~2개 | ambiguous |
| 0개 | unstable |

family overlap만으로는 서로 다른 포즈가 비슷한 거리로 나온 경우를 충분히 구분하지 못할 수 있다.
Top-1 후보끼리의 `angle_distance`도 함께 기록한다.

```text
family overlap >= 3/5
그리고 Top-1 후보 간 angle distance가 보정 임계 이하
→ stable
```

추가 모델 호출은 없지만 브루트포스 kNN을 한 번 더 수행한다. “지연이 작다”고 가정하지 않고
단일 kNN과 A/B 안정성 검사의 p50/p95를 측정한다.

로컬 `partial`은 처음부터 crop하지 않는다. 다만 A/B 검색이 `unstable`이고 `retry_count == 0`이면
그때 `suspect`로 승격해 해당 슬롯만 crop 1회 후 품질 검사와 안정성 검사를 다시 수행한다.
이 경로도 슬롯당·컷당 재시도 상한을 그대로 적용한다.

### 5-9. 최종 person_index

배정·마스킹·crop 복구가 끝난 후 정렬 기준 박스의 `x1` 오름차순으로 사용자 번호를 부여한다.
정렬 기준은 배정된 스켈레톤 박스를 우선하고, 스켈레톤이 없는 VLM 슬롯은 검증된 `vlm_box`,
provisional 슬롯은 스켈레톤 박스를 사용한다.

```text
가장 작은 x1 → person_index 0
다음 x1       → person_index 1
```

`x1`은 사용자 번호와 결과 정렬에만 쓰고, 슬롯-스켈레톤 매칭 기준으로 쓰지 않는다.

---

## 6. 상태 정의와 invalid 기준

### 6-1. 상태

| state | 뜻 |
|---|---|
| `valid` | 슬롯 소유권과 주요 몸통·사지가 신뢰 가능 |
| `partial` | 몸통·소유권은 신뢰 가능하지만 일부 사지가 불량 |
| `suspect` | 합침·배정 모호·몸통 이상 가능성이 있어 재검사가 필요 |
| `missing` | 배정된 스켈레톤이 없음 |
| `invalid` | 복구·마스킹 후에도 검색 결과를 책임 있게 제공할 수 없음 |

검색 가능성과 refine 가능성은 별도다. 검색 최소 뼈 수를 통과해도 각 사지의 두 뼈가 모두
유효하지 않으면 `refinable_limbs=[]`일 수 있다. 이는 버그가 아니며 trace/API 메타데이터에
명시한다.

`state`는 스켈레톤 구조·소유권 상태이고, `coverage_class`는 검색에 남은 정보량이다.
둘을 합치지 않는다. 구조적으로 정상인 슬롯도 몸통만 남으면 `state=partial`,
`coverage_class=sparse`일 수 있다.

구조 이상을 발견하자마자 `invalid`로 확정하지 않는다.

```text
unassigned
  ├ valid
  ├ partial
  └ suspect
       ├ crop 성공 → valid/partial
       └ crop 실패
            ├ 검색 stable → partial 또는 soft fallback
            └ 검색 unstable → invalid
```

### 6-2. hard invalid

다음은 검색 입력으로 사용할 수 없는 강한 조건이다.

1. crop 재시도 후에도 스켈레톤이 없음
2. keypoints에 NaN/Inf가 있음
3. 전체 12개 뼈 중 유효 뼈가 4개 미만
4. 몸통 4개 뼈 중 유효 뼈가 2개 미만
5. 몸통 정규화 기준이 퇴화해 안정적으로 방향을 계산할 수 없음
6. 동일 스켈레톤 중복을 제거한 뒤에도 슬롯 소유권을 정할 수 없음
7. `full/reduced`에서 마스킹·crop 후 검색 결과가 `unstable`이고 Top-1 거리도 해당 metric×coverage 임계값을 넘음

3·4의 수치는 초기값이며 고정 평가셋으로 조정한다. 임계값을 바꿀 때는 변경 이유와 전후 결과를 문서화한다.

### 6-3. partial 또는 suspect로 남겨야 하는 조건

다음은 단독으로 hard invalid를 만들지 않는다.

- 한쪽 손목·발목 score 저하
- 한쪽 사지 누락
- 웹툰 과장으로 인한 긴 팔·다리
- 인물 박스 밖으로 뻗은 손·발
- 일부 관절이 다른 인물 박스와 겹침
- 스켈레톤 모양은 이상하지만 보수적 마스크에서도 Top-5가 안정적임
- 구조적으로 정상이고 기존 VLM 슬롯과 중복되지 않는 RTM provisional 인물
- 정상 몸통만 남아 `coverage_class=sparse`인 인물. 검색은 허용하되 high confidence와 refine은 금지

---

## 7. 검색·fallback·refine 결정

스켈레톤 추출 상태와 최종 제품 결정을 분리한다.

```python
slot.state     # valid | partial | suspect | missing | invalid
slot.decision  # accept | accept_base_only | retry | soft_fallback | hard_fallback
```

`decision`은 별도 필드로 영속할 필요는 없으며 검색 직전에 계산해도 된다.

| 상태 | coverage | 거리·안정성 | 결과 | refine |
|---|---|---|---|---|
| `valid` | `full/reduced` | 전용 임계값 통과 | 일반 Top-5 | 유효 사지 허용 |
| `partial` | `full/reduced` | 전용 임계값·안정성 통과 | Top-5 제공 | 유효 사지만 허용 |
| `valid/partial` | `full/reduced` | 전용 임계값 초과 | low confidence 베이스 Top-5 | 금지 |
| `valid/partial` | `sparse` | 거리·안정성과 무관 | soft fallback 베이스 Top-5 | 금지 |
| `partial` | `full/reduced` | unstable | crop 가능하면 1회, 이후 판단 | 보류 |
| `suspect` | `full/reduced` | stable | low confidence 베이스 Top-5 | 원칙적으로 금지 |
| crop 후 `suspect` | `full/reduced` | unstable | hard fallback | 금지 |
| `invalid`/`missing` | `insufficient` 또는 — | — | hard fallback | 금지 |
| `rtm_provisional` | 모든 검색 가능 coverage | stable | low confidence 베이스 Top-5 | 금지 |

### soft fallback

- 베이스 Top-5 BVH는 제공
- `person_confidence=low`
- refine 전체 또는 의심 사지 금지
- 사용자에게 직접 후보 확인을 요청

### hard fallback

- 자동 매칭 Top-5를 신뢰 결과로 제공하지 않음
- refine 금지
- 재캡처·직접 선택 등 사용자 확인 경로로 전환

검색은 자연스러운 라이브러리 BVH라는 강한 prior를 갖기 때문에 부분 노이즈를 흡수할 수 있다.
반면 refine은 잘못된 관절을 따라 베이스를 변형할 수 있으므로 더 엄격해야 한다.

---

## 8. CutResult 연결

`CutResult.person_candidates`와 `descriptors`는 최종 `person_index` 순서를 따른다.

```text
person 0: valid   → Top-5
person 1: suspect → soft fallback Top-5
person 2: partial → Top-5
person 3: invalid → []
```

중간 슬롯이 실패해도 뒤 인물의 번호를 앞으로 당기지 않는다.

응답에 포함되는 keypoints는 기존 좌표를 유지하고 기존 `scores`는 effective scores로 사용한다.
클라이언트가 이를 `/refine`에 되돌려주면 불량 사지가 다시 활성화되지 않는다.
평가·디버깅을 위해 `raw_scores`를 optional로 추가하거나 서버 trace에 보존한다.

필요하면 API에 다음 최소 메타데이터를 추가한다.

```json
{
  "person_index": 1,
  "slot_origin": "vlm",
  "skeleton_state": "partial",
  "skeleton_source": "full_image",
  "coverage_class": "reduced",
  "distance_metric": "pos",
  "rank_distance": 0.18,
  "valid_limbs": ["torso", "left_arm", "left_leg", "right_leg"],
  "refinable_limbs": ["left_arm"],
  "raw_scores": [
    0.91, 0.84, 0.88, 0.76, 0.82, 0.93, 0.89, 0.87, 0.74,
    0.85, 0.70, 0.95, 0.94, 0.91, 0.90, 0.88, 0.86
  ],
  "reasons": ["right_arm_mixed"]
}
```

---

## 9. 평가

슬롯은 런타임 관리 단위이자 평가 단위다. 단, Gemini 슬롯 자체는 GT가 아니다.

### 필요한 GT

최소:

- 실제 사람 수
- 인물별 GT 박스
- 인물별 결과 정성 등급 `good/usable/bad`

가능하면:

- COCO-17 keypoints
- 관절별 visibility

### 단계별 지표

| 단계 | 지표 |
|---|---|
| VLM | 사람 수 정확도, GT 박스 recall, 중복 슬롯률 |
| 배정 | slot fill rate, assignment accuracy, ambiguous rate, unmatched RTM 수, provisional precision |
| 구조 검사 | valid/partial/suspect/invalid 비율, coverage class 분포, 합침 검출 precision/recall |
| crop | retry rate, 복구율, 오복구율, 컷당 추가 추론 수 |
| 시간 | 단일 kNN·A/B kNN·전체 p50/p95, 정상 경로와 재시도 경로 시간 |
| 검색 | metric×coverage별 거리 분포·precision·fallback율, pose-family Top-5 stability, Top-1 후보 간 거리, Top-5 usefulness |
| refine | 적용률, refinable limb 수, 사지 동결률, P3 롤백률, 정성 개선률 |

스켈레톤 품질과 최종 결과 품질을 반드시 분리한다.

```text
skeleton_quality: good / partial / bad
top5_usefulness:  good / usable / bad
```

이 분리로 다음을 구분한다.

- 스켈레톤 bad + Top-5 good: 라이브러리가 추출 노이즈를 흡수
- 스켈레톤 good + Top-5 bad: 검색 또는 라이브러리 공백
- 스켈레톤 bad + Top-5 bad: 추출 실패

---

## 10. 구현 순서

### 0단계 — 최소 평가셋·라벨 고정

- 실제 사람 수
- 인물별 GT 박스
- `skeleton_quality=good/partial/bad`
- `top5_usefulness=good/usable/bad`
- 단일 kNN 및 현재 전체 파이프라인 기준 시간

임계값은 이 평가셋 없이 확정하지 않는다.

### 1단계 — mask-aware 순위 거리와 기존 score 게이트 수정

- `pose_distance`에 명시적인 query mask 전달
- `angle_distance`에 query/library 관절 mask와 view별 관측 가능 뼈 mask 전달
- `hybrid_distance`, `knn`, `knn_geometric`의 결측 규약 통일
- `_search_one`이 mask·coverage·유효 뼈 정보를 받도록 인터페이스를 수정
- 전체 평균 score 조기 fallback의 실제 교체는 6단계 품질 검사를 연결할 때 활성화
- refine `low_skeleton_score`와 effective scores의 정합 확인
- 이 단계에서는 masked rank distance만 계산하고 새 confidence 정책은 아직 활성화하지 않음

### 2단계 — coverage confidence와 임계값 보정

- 완성 사지 bitset과 `full/reduced/sparse/insufficient` 계산
- 검색 순위 거리와 confidence 판정 분리
- `distance_metric × coverage_class`별 거리 분포와 Top-5 usefulness 측정
- full/reduced의 fallback 임계값과 필요한 경우 `HYBRID_W` 보정
- sparse는 거리와 무관하게 high confidence·refine 금지
- BVH body mapping assertion과 DB feature shape/finite 검사 추가

0~2단계가 완료되기 전에는 sanitized score 마스킹을 활성화하지 않는다.

### 3단계 — 좌→우 결과 순서 복구

- `feat/left-to-right-person-tagging`의 정렬 헬퍼 검토
- 최종 슬롯 처리 뒤 `person_index` 부여로 위치 이동
- 기존 테스트를 슬롯 기준으로 조정

### 4단계 — 최소 슬롯과 전역 배정·trace

- `PersonSlot` 추가
- VLM 박스 검증
- RTM candidate 박스·몸통 중심 계산
- 작은 전역 일대일 배정 함수
- unmatched 슬롯/RTM trace
- 엄격한 RTM provisional 슬롯 조건
- 이 단계에서는 아직 crop·관절 마스킹을 켜지 않고 배정 실패 비율부터 측정

### 5단계 — pose family 정리

- 원본과 `_mirror`를 묶는 `pose_family_id`
- 가능하면 검색 Top-5도 family 기준으로 중복 제거
- 안정성 지표가 미러 중복에 오염되지 않게 준비

### 6단계 — 품질 검사와 마스킹

- 몸통·사지별 유효 뼈 검사
- `valid_joint_mask` 생성
- raw/effective scores 분리
- normalize 이전 torso anchor 검사
- coverage class 재계산

### 7단계 — 조건부 crop

- `missing/suspect` 트리거
- 슬롯당 1회, 컷당 최대 재시도 제한
- crop 결과 선택을 평균 score에서 슬롯 적합도+구조 품질로 변경

### 8단계 — family-aware 검색 안정성과 fallback

- `partial/suspect`에만 의심 사지 포함/제외 Top-5 비교
- pose-family overlap + Top-1 후보 간 angle distance
- sparse는 stability 결과와 무관하게 high confidence 금지
- soft/hard fallback 분리
- A/B kNN p50/p95 측정

### 9단계 — refine 가능 사지 연결과 최종 보정

- `refinable_limbs`를 valid mask에서 계산
- identity/merge suspect와 sparse coverage에는 refine 금지
- 정상 경로 처리시간과 실패 복구율 측정
- invalid·coverage 임계값 최종 보정

---

## 11. 필수 테스트

1. 결측 query 관절은 `pose_distance` 순위 평균에서 제외된다.
2. 좌표가 `(0,0)`이어도 explicit mask가 valid면 결측으로 오판하지 않는다.
3. `angle_distance`는 query/library 양쪽에서 유효하고 해당 view에서 관측 가능한 뼈만 사용한다.
4. 자기 자신 쿼리의 일부 관절을 마스킹해도 자기 pose family가 Top-K에 유지된다.
5. 유효 관절이 줄어 Top-1 거리가 작아져도 그 이유만으로 confidence가 올라가지 않는다.
6. 정상 몸통만 남은 `sparse` 쿼리는 거리·stability와 무관하게 high confidence가 되지 않는다.
7. `sparse`와 `insufficient`에는 refine을 실행하지 않는다.
8. `pos/angle/hybrid`와 `full/reduced` 조합이 각각 보정된 임계값을 사용한다.
9. 임계값 테스트가 고정 평가셋의 metric×coverage별 거리 분포와 usefulness 라벨에 일치한다.
10. sanitized score가 늘어도 유효 뼈 기준을 만족하면 전체 평균 score 때문에 조기 fallback하지 않는다.
11. hip/shoulder anchor가 불량하면 normalize/search 이전에 차단된다.
12. body mapping이 불완전한 BVH는 entry 생성 단계에서 DB 빌드 전에 실패한다.
13. DB에는 shape가 잘못됐거나 NaN/Inf가 있는 feature를 저장하지 않는다.
14. RTMPose 반환 순서가 섞여도 최종 `person_index`는 x1 순서다.
15. 슬롯 하나에 스켈레톤 둘이 배정되지 않는다.
16. 스켈레톤 하나가 슬롯 둘에 배정되지 않는다.
17. count가 같아도 중복/합침을 `valid`로 자동 확정하지 않는다.
18. 구조적으로 정상이고 비중복인 unmatched RTM만 provisional 슬롯이 된다.
19. VLM box가 0개여도 정상 RTM provisional 경로가 동작한다.
20. partial의 불량 관절은 effective scores에서 0이고 raw scores는 보존된다.
21. partial의 정상 사지는 검색과 refine에서 유지된다.
22. 같은 슬롯에 실패 사유가 여러 개여도 crop은 한 번만 실행된다.
23. crop 결과가 나쁘면 원본보다 우선하지 않는다.
24. 미러 pose ID가 달라도 같은 pose family로 stability를 계산한다.
25. suspect라도 보수적 마스크에서 Top-5가 stable이면 베이스 후보를 보존한다.
26. invalid·identity ambiguity에는 refine을 실행하지 않는다.
27. 검색은 가능하지만 refine 완성 사지가 없으면 `refinable_limbs=[]`로 보고한다.
28. 정상 컷의 RTMPose 호출 수는 정확히 1회이고 안정성 A/B 검색도 실행하지 않는다.
29. `CutResult`의 인물 순서와 descriptors/candidates/keypoints 순서가 일치한다.

---

## 12. 구현 시 지켜야 할 최종 판단

- VLM 박스당 슬롯 하나를 만들고, 엄격한 조건을 통과한 unmatched RTM만 provisional 슬롯으로 추가한다.
  실패 사유·crop 재시도마다 슬롯을 늘리지는 않는다.
- crop 재시도는 새 슬롯을 만들지 않고 기존 슬롯을 갱신한다.
- `x1`은 사용자 번호용이고 인물 매칭은 IoU+몸통 중심 거리로 한다.
- 고정 평가셋을 먼저 만들고, masked 거리 함수와 coverage 임계값을 보정한 뒤 관절 마스킹을 활성화한다.
- masked 평균 거리는 후보 순위용이다. 서로 다른 mask의 raw 거리를 confidence로 직접 비교하지 않는다.
- confidence는 `slot.state + coverage_class + metric별 거리 + stability`로 판정한다.
- `sparse`는 Top-5를 보존할 수 있지만 high confidence와 refine은 허용하지 않는다.
- 단일 fallback 임계값 대신 distance metric×coverage class별 임계값을 사용한다.
- 라이브러리 body mapping 완전성은 feature 생성 전에 assertion으로 보장한다.
- 구조 비율 이상은 웹툰 과장을 고려해 즉시 hard invalid로 쓰지 않는다.
- `invalid`는 “이상해 보이는 스켈레톤”이 아니라 **복구·마스킹 후에도 신뢰할 뼈가 부족하거나 검색이 불안정한 상태**다.
- 스켈레톤이 이상해도 Top-5가 안정적이면 결과를 살리고 refine만 제한한다.
- 안정성은 raw pose ID가 아니라 mirror를 접은 pose family 기준으로 본다.
- 검색 가능성과 refine 가능 사지 수는 별도로 보고한다.
- 정상 컷의 빠른 경로를 기준 성능으로 두고, 재시도 경로가 전체 p95를 과도하게 늘리지 않게 한다.

---

## 13. 구현 상태 (2026-08-05)

### 완료

- explicit query mask를 사용하는 `pos/angle/hybrid` 거리와 coverage별 confidence
- VLM 박스 검증, 슬롯-RTM 전역 일대일 배정, 좌→우 최종 인덱스
- IoU와 관절 유사도를 함께 보는 중복 검출
- 몸통 anchor·폭 비율·사지 segment 연속성·슬롯 소유권 검사
- 강한 길이 이상 관절 마스킹과 소유권 의심 사지 A/B 검색
- `missing/suspect` 선행 crop 및 `unstable partial` 사후 crop 1회
- crop 후 품질·검색 재검사와 soft/hard fallback 분리
- raw/effective score, valid/refinable limb, 안정성·거리·retry trace API 노출
- `/refine`의 `refine_allowed` 및 `refinable_limbs` 강제
- 정상 경로 추가 pose/crop 호출 없음 회귀 테스트

### 평가셋이 있어야 끝나는 항목

- `distance_metric × coverage_class`별 fallback 임계값 확정
- 몸통 폭·사지 길이·중복 관절 거리 임계값 보정
- `SLOT_STABILITY_TOP1_ANGLE_MAX` 활성값 확정
- provisional precision, crop 오복구율, 정상/재시도 p50·p95 측정

이 값들은 코드 미구현이 아니라 **라벨 없는 상태에서 임의 확정하면 안 되는 calibration 값**이다.
