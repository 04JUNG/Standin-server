# Refine aggressive 근접 공백 D0 평가 실행서

작성일: 2026-08-17  
상위 문서: `REFINE_V2_5_AGGRESSIVE_PROMOTION.md`  
대상: 현재 `in/`의 러프 19장  
목적: v2.5 구현 전에 다음 두 가지를 현재 보유 데이터로 확인한다.

1. `aggressive`가 `conservative`보다 근접 공백을 더 효과적으로 줄이는가
2. 그 개선이 RTMPose 수치뿐 아니라 작가가 본 러프 의도와도 일치하는가

---

## 0. 이 평가가 결정할 수 있는 범위

현재 러프는 이미 검색·refine 개발과 오류 분석에 사용됐다. 따라서 이 평가는 **D0 엔지니어링
증거**이며, 새로운 데이터에 대한 일반화나 최종 config 승격을 증명하지 않는다.

이 평가에서 결정할 수 있는 것은 다음뿐이다.

- 현재 러프에서 raw aggressive가 conservative보다 근접 공백을 더 줄이는지
- aggressive의 자동지표 이득이 작가의 직관적 선호와 같은 방향인지
- v2.5에서 보존해야 할 aggressive 이득과 차단해야 할 회귀가 무엇인지
- 현재 전략을 v2.5 구현 단계로 진행할지, objective부터 수정할지

이 평가만으로 `REFINE_DEFAULT_MODE=aggressive`를 기본값으로 바꾸지 않는다. config 승격은 v2.5
구현 후 별도 최종 정책 평가에서 결정한다.

---

## 1. 사전 가설

결과를 생성하거나 새 비교 화면을 보기 전에 다음 가설과 판정선을 고정한다.

### H1. 근접 공백 자동지표

같은 query skeleton과 같은 선택 base BVH에서 raw aggressive는 conservative보다 near-gap cohort의
공통 외부 평가 오차를 줄인다.

주지표:

```text
JointErrorReduction(C→A) =
  (mean_joint_nme_C - mean_joint_nme_A) / mean_joint_nme_C × 100
```

보조지표:

- `endpoint_nme`
- 활성된 `hand_pair_error`
- 활성된 `lower_pair_error`
- 활성된 `lap_contact_error`
- 단위별 `better / tie / worse`

### H2. 작가 의도

버전 정보를 숨긴 직접 비교에서 작가는 near-gap cohort의 aggressive를 conservative보다 더 자주
선호하며, aggressive의 확인된 major-worse는 0건이다.

### H3. 회귀의 분리 가능성

aggressive의 이득은 유지하면서 구조적 안전 회귀는 gate와 conservative/base fallback으로 분리해
차단할 수 있다. 현재 safety proxy 경보는 실제 눈으로 확인해 `confirmed`와 `false_positive`로 나눈다.

---

## 2. 고정 데이터

### 입력

- 이미지: `in/`의 PNG/JPG 19장
- 기존 frozen source:
  `out/eval/in_refine_auto_final_20260814/records.json`
- 기존 현황: RTMPose 검출 37명, 공통 evidence 유효 27명, evidence 부족 제외 10명
- pose DB와 base BVH는 기존 run의 hash와 동일한 것을 사용

### 평가 단위

```text
(image_sha256, person_index_left_to_right,
 frozen_keypoints, frozen_scores, selected_pose_id,
 selected_view, base_bvh_sha256)
```

기존 27개 평가 단위의 skeleton과 Top-1 base를 그대로 재사용한다. arm별로 RTMPose·검색을 다시
실행하지 않는다. 제외된 10명은 다른 인물로 교체하지 않고 제외 사유와 함께 분모 밖에 보존한다.

### 주의

같은 이미지 안의 여러 인물을 서로 완전히 독립된 이미지처럼 해석하지 않는다. 요약에는
`person n`과 `image cluster n`을 함께 기록한다.

---

## 3. 비교 arm

| Arm | 정의 | 목적 |
|---|---|---|
| `B0_base` | 선택한 원본 BVH | 공백과 절대 변화 확인용 |
| `C_v24_conservative` | v2 기능은 같고 `refine_mode=conservative` | aggressive의 직접 기준 |
| `A_v24_aggressive_candidate` | aggressive solve가 만든 gate 전 후보 | aggressive 자체의 효과 측정 |
| `A_v24_aggressive_final` | 현재 구현이 실제 반환한 aggressive/conservative/base 결과 | 현재 제품 정책 진단 |

H1과 H2의 주 비교는 `A_v24_aggressive_candidate vs C_v24_conservative`다. 기존 보고서의
`v1 vs v2 final` 비교로 aggressive 자체의 효과를 대신하지 않는다.

현재 코드가 gate 전 aggressive 후보 BVH를 저장하지 않는다면 평가 runner에서만 다음을 추가한다.

- conservative artifact 저장
- raw aggressive candidate artifact 저장
- 현재 gate 이후 final artifact 저장
- candidate가 탈락해도 artifact와 탈락 reason은 평가 폴더에 보존
- 제품 endpoint의 반환 규칙은 이 D0 계측 때문에 변경하지 않음

---

## 4. near-gap 사전 라벨

### 라벨 시점

`B0_base`의 target-view render와 원본 러프만 보고 라벨한다. conservative/aggressive 결과와 기존
NME·안전 결과는 보지 않는다.

### 정의

```text
near_gap:
  base가 같은 포즈 구조·방향·체중 지지·접촉 의도를 가지고 있으며,
  허용된 팔·다리·발의 제한된 관절 회전으로 러프에 접근할 수 있다.

structural_gap:
  다른 pose family, 큰 root 이동, 큰 몸통 방향 변화,
  다른 체중 지지 또는 다른 접촉 구조가 필요하다.

unknown:
  가림, 다인 소유권, RTMPose 오류 또는 러프 모호성 때문에 판단할 수 없다.
```

### 라벨 형식

```json
{
  "unit_id": "131112:p0",
  "gap_type": "near_gap|structural_gap|unknown",
  "target_parts": ["left_arm", "hand_pair", "lower_pair", "lap_contact", "foot"],
  "base_same_pose_intent": true,
  "reachable_by_allowed_joints": true,
  "reason": "한 문장 근거",
  "labeled_at": "ISO-8601"
}
```

`gap_type`과 `target_parts`는 arm 결과를 unblind한 뒤 변경하지 않는다. near-gap이 10개보다 적으면
현재 러프만으로 H1/H2는 `INCONCLUSIVE`로 판정하고 정성 사례만 남긴다.

---

## 5. artifact 생성 계약

모든 arm은 다음 조건을 공유한다.

- 동일한 frozen keypoints·scores·valid mask
- 동일한 base BVH content hash와 view
- 동일한 v2 lower-body/torso 설정
- torso default off
- 동일한 evaluator version과 metric mask
- 동일한 renderer·body·camera·scale
- arm 실행 순서는 unit마다 무작위화
- cache-off latency 기록

각 단위에 다음 파일을 남긴다.

```text
unit.json
B0_base.bvh
C_v24_conservative.bvh
A_v24_aggressive_candidate.bvh
A_v24_aggressive_final.bvh
metrics.json
lineage.json
```

`lineage.json` 필수 필드:

```text
mode_requested
mode_candidate
mode_applied
candidate_adopted
fallback_reason
base/conservative/candidate/final content hash
config/code/evaluator version
conservative/aggressive/final latency
```

---

## 6. 자동 평가

solver diagnostics의 내부 objective를 직접 비교하지 않는다. 네 artifact를 같은 외부 evaluator로
다시 파싱·FK·투영한다.

### near-gap 주 분석

`gap_type=near_gap`으로 사전 라벨한 전체 단위를 분모로 사용한다.

| 지표 | C conservative | A raw aggressive | A-C | better/tie/worse |
|---|---:|---:|---:|---:|
| Joint NME |  |  |  |  |
| Endpoint NME |  |  |  |  |
| Hand-pair error |  |  |  |  |
| Lower-pair error |  |  |  |  |
| Lap-contact error |  |  |  |  |

pair metric은 사전에 해당 `target_parts`와 query evidence가 활성화된 단위에서만 계산하고 `n`을 반드시
함께 표시한다.

### 안전 판정 분리

다음은 자동으로 확인 가능한 구조적 hard violation으로 둔다.

- BVH parse/FK 실패, NaN/Inf
- root·비허용 joint·채널 변경
- 신규 capsule/mesh 관통
- 신규 관절 제한 위반
- fallback artifact의 base identity 불일치

다음은 현재 evaluator의 **proxy alert**로 보고 눈으로 확정한다.

- `foot_direction_regression`
- `ground_contact_regression`
- `lap_contact_regression`

proxy alert를 자동으로 실제 hard issue라고 결론 내리지 않는다. 각 alert에
`confirmed_major / confirmed_minor / false_positive / uncertain`을 기록한다.

### D0 자동 판정선

near-gap `n >= 10`일 때 다음을 모두 만족하면 H1을 `PASS_D0`로 판정한다.

- Joint NME 상대 감소 `>= 5%`
- Endpoint NME가 conservative 대비 평균 non-regression
- 단위별 Joint NME `better > worse`
- 사전 활성 pair 지표 중 적어도 하나가 개선되고, 다른 활성 pair 지표에 큰 평균 회귀가 없음
- 신규 구조적 hard violation `0건`

수치 노이즈 tolerance는 evaluator 반복 실행 결과로 한 번 정한 뒤 고정한다. 결과를 본 뒤 tolerance를
바꾸지 않는다.

---

## 7. 작가 블라인드 직관 평가

### 화면 구성

주 비교는 `C vs A raw`의 2개 결과다. 좌우 위치를 unit마다 무작위화하고 다음 정보를 숨긴다.

- conservative/aggressive 이름
- pose ID·rank·search distance
- NME·solver loss·gate reason
- refined/adopted/fallback 여부
- 알려진 안전 회귀 목록

화면에는 다음만 표시한다.

1. 원본 러프
2. 좌·우 target-view render
3. 안전 확인 버튼을 누른 뒤 좌·우 front / three-quarter / side / back render

`B0_base`는 주 pair 판정이 끝난 후 별도 참고 화면에서만 보여준다.

### 판정 순서

1. 첫인상 5초 안에 러프와 더 가까운 결과를 선택
2. 동률 또는 둘 다 나쁨을 허용
3. 각 결과가 수정 없이 사용 가능한지 판정
4. 4-view에서 관통·발/지면·균형·해부학 확인
5. major/minor와 문제 부위 기록

### 라벨 형식

```json
{
  "pair_id": "blind:...",
  "unit_id": "131112:p0",
  "winner": "left|right|tie|both_bad",
  "left_usable": true,
  "right_usable": true,
  "left_issue": "none|minor|major",
  "right_issue": "none|minor|major",
  "issue_parts": ["hand", "arm", "leg", "foot", "torso", "contact", "collision"],
  "proxy_alert_judgment": "not_applicable|confirmed_major|confirmed_minor|false_positive|uncertain",
  "note": "한 문장 근거",
  "labeled_at": "ISO-8601"
}
```

### 자기 일관성 확인

- 전체 pair 중 최소 20%를 다음 날 좌우를 다시 섞어 재평가
- 처음 라벨은 수정하지 않고 두 판단을 모두 보존
- winner/major 판정 일치율을 별도로 보고
- 일치율이 80% 미만이면 H2를 `INCONCLUSIVE`로 둠

### D0 직관 판정선

near-gap 전체 분모에서 unblind 후 다음을 계산한다.

```text
WinRate_A       = aggressive wins / N_near
LossRate_A      = conservative wins / N_near
TieRate         = ties / N_near
BothBadRate     = both_bad / N_near
NetPreference_A = (aggressive wins - conservative wins) / N_near
SafeUsable_A/C  = usable이며 confirmed major가 없는 수 / N_near
```

다음을 모두 만족하면 H2를 `PASS_D0`로 판정한다.

- `NetPreference_A >= +20%p`
- `SafeUsable_A >= SafeUsable_C`
- aggressive의 confirmed major-worse `0건`
- 다음 날 반복 평가 일치율 `>= 80%`

---

## 8. 결과 해석 규칙

| 자동 H1 | 직관 H2 | 판단 | 다음 단계 |
|---|---|---|---|
| PASS | PASS | aggressive가 현재 near-gap에 실효성 있음 | 이득을 보존하는 v2.5 gate 구현 |
| PASS | FAIL | pseudo target 과적합 가능성 | objective·mask 재검토, config 승격 금지 |
| FAIL | PASS | evaluator가 작가 의도를 못 측정할 가능성 | 사례 감사 후 metric 수정 여부 결정 |
| FAIL | FAIL | aggressive 전략 근거 부족 | aggressive 강화보다 검색/pose family 우선 |

confirmed major가 1건이라도 있으면 H1/H2와 별개로 현재 raw aggressive의 직접 반환은 금지한다.
다만 해당 문제를 deterministic gate가 검출하고 conservative/base로 정확히 복구할 수 있다면 v2.5
안전 cascade 구현 후보로 남긴다.

---

## 9. 결과 폴더 계약

```text
out/eval/v25_current_rough_near_gap_d0_YYYYMMDD/
├─ manifest.json
├─ frozen_units.jsonl
├─ gap_labels.jsonl
├─ arms/
│  └─ <unit_id>/
│     ├─ B0_base.bvh
│     ├─ C_v24_conservative.bvh
│     ├─ A_v24_aggressive_candidate.bvh
│     ├─ A_v24_aggressive_final.bvh
│     ├─ metrics.json
│     └─ lineage.json
├─ blind/
│  ├─ pairs.jsonl
│  ├─ mapping.hidden.json
│  └─ renders/
├─ self_labels.jsonl
├─ summary.json
└─ REPORT.md
```

`mapping.hidden.json`은 직관 평가를 모두 완료하기 전에는 열지 않는다.

---

## 10. 최종 보고 형식

`REPORT.md` 첫 부분은 반드시 다음 순서를 따른다.

1. D0이며 승격 증명이 아니라는 한계
2. near-gap/structural-gap/unknown 실제 `n`
3. raw aggressive vs conservative 자동지표
4. blind `W/T/L/BB`, Safe Usable과 반복 일치율
5. 구조적 violation과 proxy alert 확정 결과
6. aggressive candidate → final fallback funnel
7. H1/H2 `PASS_D0 / FAIL_D0 / INCONCLUSIVE`
8. v2.5 구현에서 보존할 이득과 새로 넣을 gate 목록

허용되는 결론:

> 현재 러프의 사전 라벨 near-gap N개에서 raw aggressive는 conservative 대비 Joint NME를 X% 줄였고,
> 블라인드 직관 평가는 W/T/L/BB=a/b/c/d였다. 확인된 major-worse는 M건이었다.

금지되는 결론:

- `aggressive가 모든 러프의 정확도를 X% 높인다`
- `현재 D0 통과만으로 기본 config 승격이 완료됐다`
- proxy alert를 사람 확인 없이 실제 안전사고로 단정
- aggressive가 적용된 사례만 골라 전체 성공률처럼 보고

---

## 11. 실행 체크리스트

- [ ] 기존 27개 frozen unit과 base hash 검증
- [ ] 결과를 보기 전에 gap type과 target parts 라벨 완료
- [ ] conservative와 raw aggressive candidate를 별도 artifact로 저장
- [ ] arm-independent evaluator로 모든 artifact 재평가
- [ ] 구조적 violation과 proxy alert 분리
- [ ] blind pair와 hidden mapping 생성
- [ ] 1차 직관 평가 완료
- [ ] 다음 날 20% 반복 평가 완료
- [ ] 라벨 잠금 후 unblind
- [ ] H1/H2와 해석 규칙에 따라 REPORT 작성
- [ ] 통과한 이득과 실패 사례를 v2.5 구현 요구사항으로 전달
