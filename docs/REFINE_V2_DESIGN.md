# Refine v2 설계 (v2.4 통합 승인·구현 기준)

> 상태: v2.4 보수적/공격적 feature flag 구현·engineering probe 완료 · production 승격 대기
> v2.2 승인일: 2026-08-11
> v2.3 최종 승인·구현일: 2026-08-12
> v2.4 범위 승인일: 2026-08-12
> v2.4 구현일: 2026-08-12
> No-refine/v1/v2.4 3-arm 평가 전략 정리일: 2026-08-13
> 현재 기준 코드: `src/refine.py`, `src/refine_v2.py`, `src/collision.py`, `src/config.py`, `src/skeleton_extraction.py`, `src/pipeline.py`, `api/app.py`, `api/models.py`
> 현재 동작 기준: `docs/REFINE_DESIGN.md`
>
> 이 문서는 Refine v2의 단일 후속 설계 문서다. v2가 검증·승격될 때까지 실제 동작의
> 단일 기준은 `docs/REFINE_DESIGN.md`와 실행 코드다.

`REFINE_V2_ENABLED=0`이 기본값이므로 현재 production 동작은 v1을 유지한다. 구현 완료는
코드 경로와 자동 테스트가 준비됐다는 뜻이며, 고정 holdout의 `worse=0`과 단계별 `better>=1`을
통과했다는 뜻은 아니다. v2는 해당 승격 관문을 통과하기 전 production 기본값이 되지 않는다.

이 문서에서 **v2.2**는 2026-08-11 기준 초기 feature-flag 코드를 뜻한다. **v2.3**은 아래 두
실사용 피드백을 반영해 2026-08-12 최종 승인 후 구현한 보정 버전이다.

1. 양손을 모으는 상체 개선은 있었지만 손·전완이 다리를 관통했다.
2. 앉은 자세에서 양다리를 모아야 했지만 단축 투시를 구조 오류로 오판해 하체 refine이 실행되지 않았다.

v2.3은 기존 v2의 positive-gain·부위별 non-regression·정확한 베이스 복구 계약을 약화하지 않는다.
하체 eligibility를 넓히는 변경과 손-다리 충돌 게이트는 반드시 같은 구현 묶음으로 들어간다.
**v2.4**는 v2.3을 보수적 모드로 고정하고, 그 결과에서 `hand_pair`, `lap_contact`, 강화된
`lower_pair`, 제한적 발목 counter-rotation을 추가로 시도하는 공격적 모드를 구현한다.

## 1. 목적과 확정 범위

검색된 BVH를 러프에 더 가깝게 조정하면서 다음 조건을 동시에 만족한다.

- 2D 뼈 방향뿐 아니라 관절·말단 위치도 개선한다.
- 보이지 않는 깊이 방향으로 관절이 임의로 움직이지 않는다.
- 안전하지 않은 변경은 부위별 또는 전체 베이스로 정확히 복구한다.
- v1에서 기본 비활성이던 하체 refine을 몸통 확장보다 먼저 검증한다.
- 앉기·쪼그리기에서 정상적인 2D 단축 투시를 소유권 오류와 구분해 유효 하체를 보존한다.
- 양쪽 무릎·발목의 상대 관계를 사용해 목표가 요구할 때 두 다리를 함께 모은다.
- 손·전완과 허벅지·무릎·정강이의 신규 3D 관통을 차단하되 의도된 얕은 접촉은 허용한다.
- 몸통은 해부학적으로 안전하고 실제 이득이 있을 때만 제한적으로 방향 회전을 허용한다.
- 자동 채택 결과는 **조금이라도 좋아지고, 어느 평가축에서도 나빠지지 않아야 한다.**

v2의 목표는 검색 실패를 생성적으로 복구하는 것이 아니다. 같은 자세 계열 안에서 부족한
국소 변형을 안전하게 보완하는 것이다.

### 1-1. 제품 흐름

이번 v2는 최종 승인된 현재의 선택 후 refine 흐름을 유지한다.

```text
/analyze → 베이스 Top-5 표시 → 사용자 후보 1개 선택
→ /refine 1회 → 조정본 또는 베이스 BVH → export
```

Top-N/Top-5 후보를 사용자에게 보여주기 전에 전부 refine하거나, 완료되는 순서대로 preview를
교체하거나, refine 결과로 Top-5를 재정렬하는 기능은 이번 범위에 포함하지 않는다.

`/analyze`는 베이스 Top-5와 기존 베이스 썸네일을 먼저 반환하며 선행 refine 완료를 기다리지
않는다. 사용자가 후보를 선택하면 그 후보 하나만 refine하고, 반환된 `bvh_url`을 클라이언트 3D
뷰어에서 확인한 뒤 export한다. refined BVH 생성만으로 refined 썸네일이 생기지는 않으므로,
서버의 후보별 refined PNG 생성은 이번 v2의 필수 범위로 두지 않는다.

post-click 지연이 실제 사용자 이탈 원인으로 측정되면 Top-1 또는 hover 후보의 저우선 백그라운드
prefetch를 별도 최적화로 실험한다. 이 경우에도 베이스 Top-5 노출을 막지 않고, 사용자가 선택한
작업을 우선하며, 미완료·timeout·worker 포화 시 기존 선택 후 refine 또는 베이스 결과로 복구한다.

### 1-2. 라이브러리 공백 정의

| 종류 | 정의 | 처리 |
|---|---|---|
| **근접 공백 (`near_gap`)** | 목표 자세 가족은 있으나 팔·다리 각도, 굽힘, 제한된 몸통 정렬 같은 변형이 부족함 | refine이 보완할 대상 |
| **구조 공백 (`structural_gap`)** | 목표 자세 가족 자체가 없음 | 라이브러리 확장으로 해결 |
| **미분류 (`unknown`)** | family GT가 없거나 판정 근거가 부족함 | near-gap 분모에는 넣지 않되 unknown 비율을 함께 보고 |

검색 실패를 고정 비율로 복구한다는 목표는 사용하지 않는다. 검색 실패와 근접 공백은 같은
개념이 아니며, 검색 거리 하나만으로 둘을 동일시하지 않는다.

근접 공백 개선률은 병렬 측정 트랙 F에서 먼저 측정한다. 고정된 수치 목표는 분모와 baseline을 실측한 뒤
별도 승인으로 정한다.

```text
near_gap_improvement_rate =
  안전 위반 없이 human-rated better가 된 labeled near_gap pair 수
  / 전체 labeled near_gap pair 수
```

### 1-3. `pose_family_id` 선행조건의 정확한 의미

현재 라이브러리에 검증된 pose-family 메타데이터가 없으므로, 실제 러프를 자동으로
`near_gap`과 `structural_gap`으로 나누거나 개선률을 주장하려면 다음이 먼저 필요하다.

1. 라이브러리 provenance 복구
2. 버전된 `pose_family_id` 또는 동등한 사람 GT 부여
3. family schema와 라이브러리 manifest hash 고정
4. 사람 감사셋으로 공백 분류 정확도 검증

이 메타데이터는 **공백 분류와 성능 측정의 선행조건**이지, 하체 refine 프로토타입이나
합성 보정 실험을 시작하기 위한 선행조건은 아니다. 메타데이터가 준비되기 전에는 합성 pair와
사람이 직접 라벨한 pair로 하체·몸통 feasibility를 검증하되, 자동 공백 분류 성능을 주장하지 않는다.

`gap_type`과 적용 결과는 섞지 않는다.

```text
gap_type: near_gap | structural_gap | unknown
refine_outcome: improved | unchanged | reverted | not_attempted
```

## 2. 현재 구현 기준 (v1 production / v2.4 flag)

| 항목 | v1 production 기본 | v2.4 feature flag 현재 코드 |
|---|---|---|
| 조정 부위 | 기본 양팔, `REFINE_LIMBS=all`이면 다리 | 양팔 + 조건부 하체, 몸통은 별도 default-off |
| 고정 부위 | 손목·손가락·발목·힙·루트·척추·목·머리 | 몸통 flag OFF 기준 손목·손가락·발목·힙·루트·척추·목·머리 |
| 모드 | 단일 v1 | `conservative` 기본, 명시적 `aggressive` |
| 목적함수 | 사지별 두 뼈의 2D 단위 방향 | direction + robust endpoint + pair/contact + 3D move + collision + anatomy |
| 뼈 가중치 | API 기본 `1.0` | effective score·valid mask 기반 |
| 뼈 유효 조건 | 양 끝 score 각각 `0.3` 이상 | 양 끝 score·구조 mask·허용 사지·관측 감도 통과 |
| 기본 정규화 | `REFINE_LAMBDA=0.05` | 같은 base lambda + P1a/P1b 관측 정규화 |
| 회전 상한 | 모든 채널 베이스 ±`45°` | 관절·축별 trust region, 전역 상한 ±`45°` |
| 이미 일치 | 평균 각도 손실 `0.01` 이하 생략 | hybrid가 수치 epsilon 안에서 일치하면 생략 |
| 자동 채택 | 최소 5% direction 개선 | epsilon을 넘는 positive hybrid gain + 블록 non-regression |
| 부분 채택 | 사지 롤백 | 블록별 `1/.75/.5/.25` alpha 후 정확한 롤백 |
| 충돌 | 팔·손-몸통 | 팔-몸통·다리-몸통·다리-다리·손/전완-다리 |
| 손/전완-다리 | 없음 | 4 pair base-relative hard gate + adopted final 재측정 |
| 양다리 관계 | 없음 | signed lower_pair loss + 공동 alpha + 안전한 per-leg fallback |
| 양손 관계 | 없음 | aggressive에서 signed hand_pair + 공동 alpha + per-arm fallback |
| 손-무릎 접촉 | 없음 | aggressive에서 2D intent + 3D signed lap-contact band |
| 발 방향 | 베이스 방향 gate | 같은 gate + aggressive에서 Foot local counter-rotation 최대 `18°` |
| 굽힘 제한 | 신규 팔꿈치·무릎 `20°` 미만 복구 | 동일 + 하체/몸통 후 전체 안전 재검사 |

`REFINE_MIN_GAIN=0.95`는 v1의 5% 개선 문턱으로 그대로 유지한다. v2는 의미가 다른 값을
억지로 alias하지 않고 별도 `REFINE_GAIN_EPSILON`을 사용한다. v2의 채택 기준은 “5% 이상”이
아니라 **수치 오차 `epsilon`을 넘는 양의 개선**이다. v1·v2 병행 기간에는 각 feature flag가
자기 설정만 읽고, v2가 완전히 승격될 때 v1 env의 폐기 일정을 정한다.

### 2-1. v2.3 보정 근거 — 2026-08-11 sitting probe

입력 `in/스크린샷 2026-07-09 124702.png`은 한 사람이 정면으로 앉아 양손·양무릎·양발을
가깝게 모은 컷이다. 이 사례는 engineering/calibration probe이며 sealed holdout이 아니다.

스켈레톤 추출과 v2.2 실행에서 다음이 확인됐다.

| 관측 | 결과 |
|---|---|
| 좌·우 발목 raw score | `0.8295`, `0.8224` — 추론 신뢰도는 낮지 않음 |
| 좌·우 leg 최대 segment/torso | `1.095`, `1.073` — 절대 길이 상한 `2.5` 안쪽 |
| 좌·우 인접 segment 비율 | `5.197`, `3.852` — 전역 상한 `3.5` 초과 |
| v2.2 정상 경로 | `left_leg_length_outlier`, `right_leg_length_outlier`; 다리 solver 0회 |
| Top-1에서 raw leg score를 보존한 진단 실행 | 양다리 모두 `alpha=0.5` 채택, hybrid `0.416 → 0.243` |
| 정규화 무릎 간격 | base `0.861` → refined `0.596`, target `0.294` |
| 정규화 발목 간격 | base `0.932` → refined `0.602`, target `0.190` |

따라서 하체 미실행의 1차 원인은 solver 미수렴이 아니라 **인접 비율 하나가 정상적인 seated
foreshortening을 소유권·구조 오류로 hard mask한 것**이다. 마스크만 완화해도 양다리는 유의하게
모이지만 목표 간격에는 충분히 도달하지 않아, 2차로 양다리 관계 목적함수와 공동 채택이 필요하다.

같은 probe의 후보 #2에서는 v2.2가 채택한 팔 결과에 대해 right-hand/right-thigh 신규 관통 깊이
`0.025407 torso`가 측정됐다. 현재 충돌 검사는 팔-몸통·다리-몸통·다리-다리만 보므로 이 경로를
검사하지 않는다. 하체 eligibility를 넓히면 손-다리 충돌 표면이 더 늘어나므로 **손-다리 게이트를
먼저 또는 같은 PR에서 도입하지 않고 하체만 여는 변경은 금지한다.**

### 2-2. v2.4 확정 범위 — 보수적 모드와 공격적 모드

v2.4는 하나의 solver를 임계값만 바꿔 두 이름으로 부르지 않는다. 보수적 모드는 검증된 v2.3
계약을 그대로 보존하고, 공격적 모드는 보수적 snapshot 이후 명시적인 추가 목표와 제한된 자유도만
여는 두 번째 단계다. **공격적이라는 말은 충돌·해부학·non-regression 게이트를 완화한다는 뜻이
아니다.** 두 모드는 같은 hard safety gate를 통과해야 한다.

| 항목 | `conservative` | `aggressive` |
|---|---|---|
| 기본값 | 예 | 아니오, 요청에서 명시 선택 |
| 시작점 | 선택한 base BVH | 보수적 adopted BVH, 보수적 변경이 없으면 안전한 base snapshot |
| 손 | v2.3 팔별 direction·endpoint | `hand_pair` 공동 목표 추가 |
| 손-다리 | 신규 깊은 관통 차단 | 같은 차단 + `lap_contact` 표면 band 목표 |
| 다리 | v2.3 `lower_pair` | 가중치를 높인 `lower_pair` + 공동 채택 우선 |
| 발 | 베이스 방향·접촉 hard gate | 제한된 발목 counter-rotation으로 베이스 발 방향 복원 |
| 실패 fallback | base | conservative, conservative가 없으면 base |
| 몸통 | 별도 `REFINE_V2_TORSO=0` 기본 | 자동 활성화하지 않음 |

API 요청은 `refine_mode: conservative | aggressive`를 받고 기본값은 `conservative`다. 응답 진단에는
`mode_requested`, `mode_applied`, `aggressive_attempted`, `aggressive_reason`을 남긴다. 캐시 content hash와
sidecar에도 mode를 포함해 보수적 artifact가 공격적 요청에 재사용되거나 그 반대가 되지 않게 한다.
두 단계는 같은 요청 timeout 예산을 공유한다.

```text
base
  └─ conservative adopted 또는 안전한 unchanged snapshot
       └─ aggressive accepted → aggressive artifact
            aggressive 실패 → conservative artifact
  conservative 구조·소유권·안전 실패 → base
```

#### `hand_pair`

양 손목이 모두 유효하고 같은 인물 소유권에 있으며 양팔이 `refinable_limbs`에 있을 때만 활성화한다.
개별 손목 위치 손실 외에 손목 signed relative vector와 두 손목 중점을 함께 맞춘다.

```text
L_hand_pair =
    Huber(((right_wrist - left_wrist) - target_wrist_vector))
  + Huber((wrist_midpoint - target_wrist_midpoint))
```

공동 alpha `1.0 → 0.75 → 0.5 → 0.25`를 먼저 시도한다. 양팔 손실·hand_pair·전신 손실과 모든
arm-torso/arm-leg gate가 non-regression일 때만 채택한다. 실패하면 hand_pair 관계를 악화시키지 않는
범위에서만 팔별 fallback을 허용한다. COCO-17은 손가락·손바닥 방향을 제공하지 않으므로 v2.4의
완료선은 **손목 위치와 손 실루엣의 근접**이며 손가락 맞물림을 보장하지 않는다.

#### `lap_contact`

target 2D에서 손목이 고관절-무릎 segment 또는 무릎의 사전 고정 거리 안에 있을 때만 손-허벅지
접촉 의도가 있다고 본다. 각 손은 target 2D에서 가장 가까운 좌·우 허벅지 하나에만 배정한다.
3D에서는 손 capsule과 허벅지/무릎 capsule의 signed surface clearance를 사용한다.

```text
contact band = 얕은 허용 overlap ~ 작은 surface gap
too far       → contact residual
band 안       → residual 0
깊은 관통     → 기존 hand-leg hard gate가 거부
```

lap contact는 모든 손을 다리에 붙이는 규칙이 아니다. 2D 접촉 증거, 양쪽 관절 score, 소유권과
해당 arm/leg 허용 조건을 모두 통과해야 한다. 메시가 없는 서버의 capsule proxy이므로 production
승격 전에는 실제 CSP 메시 holdout에서 접촉·부유·관통을 별도로 평가한다.

#### 강화된 `lower_pair`

활성 조건과 signed target vector는 v2.3을 유지한다. 공격적 모드에서는 pair 가중치를 별도 설정으로
높이되 target 간격 자체를 축소하거나 모든 다리를 무조건 모으지 않는다. 공동 alpha 채택과 각 다리
non-regression, leg-leg/leg-torso/arm-leg gate도 그대로 유지한다.

#### 제한적 발목 counter-rotation

다리를 더 모은 뒤 발 방향 hard gate 때문에 유용한 하체 결과가 탈락하는 경우, `LeftFoot`·`RightFoot`
로컬 회전만 작은 trust region에서 보정해 공격적 단계 시작점의 world-space 발 방향을 복원한다.
루트·골반 translation, 발 위치 이동, planted-foot IK, 카메라 최적화는 열지 않는다. counter-rotation은
COCO 발목 위치 손실을 개선하기 위한 자유도가 아니라 **이미 개선된 다리를 유지하면서 발 방향 회귀를
줄이기 위한 후처리**다. 보정 뒤에도 ground/contact와 모든 충돌을 다시 검사한다.

## 3. v2 실행·채택 안전 계약

### 3-1. 실행 전 차단

다음 안전 게이트에 걸리면 solver를 실행하지 않고 베이스를 반환한다.

- `REFINE_ENABLED=0`
- v2 정책으로 다시 계산한 `refine_allowed=false`
- 스켈레톤이 `missing`·`invalid`·`insufficient`이거나 인물 소유권을 신뢰할 수 없음
- 조정할 유효 뼈 또는 `refinable_limbs`가 하나도 없음
- 베이스 BVH가 멀티프레임임
- 얽힘 `set_id`가 있는 포즈처럼 단일 인물 solve로 관계를 보존할 수 없음

요청·자산 자체가 잘못된 경우는 사용할 베이스가 없으므로 안전 게이트 결과로 위장하지 않는다.
잘못된 keypoint·score shape 또는 NaN/Inf는 `422`, 모르는 `pose_id`는 `404`, 등록된 BVH 파일
누락·파싱 불가는 `409` 계열 오류로 명시적으로 실패한다.

v2에서는 **검색 거리 초과나 낮은 검색 순위만으로 미리 `base_mismatch` 처리하지 않는다.**
검색 거리·metric·coverage·confidence threshold는 진단과 공백 분석에 남기되, 사용자가 선택한
베이스와 유효한 스켈레톤이 있으면 보수적으로 solve를 시도할 수 있다.

이에 따라 v2의 `refine_allowed`는 검색 거리·순위의 동의어가 아니다. 유효 스켈레톤·인물
소유권·조정 가능 부위와 BVH 안전 조건을 합친 실행 허가로 다시 정의한다. 현재 파이프라인의
거리 기반 `refine_allowed=false` 정책은 v2 feature flag 안에서만 변경하며, v1 동작은 유지한다.

현재 파이프라인은 `refine_allowed=false`일 때 하위 호환 안전장치로 effective score를 전부 0으로
만든다. v2.3에서는 전신 단위의 all-or-nothing score zeroing을 사지 단위 정책으로 바꾼다.

- 검색 거리·순위·검색 안정성 low만으로 구조적으로 유효한 사지 score를 0으로 만들지 않는다.
- 한 사지가 실제 구조·소유권 검사를 실패해도 독립적으로 검증된 다른 사지는 보존한다.
- `missing`·`invalid`·`insufficient`, NaN/Inf, 몸통·인물 소유권 실패, 조정 가능 사지 0개에서는
  기존처럼 전부 차단한다.
- 길이 상한 초과·cross-slot·낮은 endpoint score 같은 실제 오염은 해당 사지만 fail-closed한다.
- balance-only 단축 투시는 §5의 별도 refine 신뢰도로 전달하고 검색 mask와 섞지 않는다.

`refine_allowed`는 “모든 사지가 안전함”이 아니라 “적어도 하나의 `refinable_limbs`가 안전하게
시도 가능함”을 뜻한다. 실제 실행 부위는 `refinable_limbs`와 부위별 score가 제한한다.

단, 검색 실패가 근접 공백이라는 뜻은 아니다. `gap_type`은 v2에서 평가·분석 라벨이며 런타임
실행 게이트로 사용하지 않는다. 사람 GT나 검증된 family 분류에서 `structural_gap`으로 확인된
사례는 near-gap 성과에 포함하지 않고, 안전한 개선이 관측되더라도 라이브러리 보강 목록에 남긴다.

### 3-2. 자동 채택 — positive gain, zero regression

solver가 결과를 만들었다는 사실만으로 채택하지 않는다. **실제로 반환할 BVH**가 다음을 모두
만족해야 한다.

1. 전체 `hybrid_loss_adopted < hybrid_loss_base - epsilon`
2. 실제로 변경한 각 팔·다리·몸통 블록의 손실이 베이스보다 악화되지 않음
3. 신규 팔-몸통·손/전완-다리·다리-몸통·다리-다리 충돌과 관절 제한·지면/접촉 위반이 없음
4. 3D 이동량과 관절별 trust region을 통과
5. 부분 복구 후 전체 손실과 모든 안전 불변식을 다시 통과

`epsilon`은 부동소수점·렌더 노이즈만 거르는 작은 수치 허용치다. 제품 의미의 최소 5% 개선
문턱으로 사용하지 않는다. 개선이 `epsilon` 이하이면 `unchanged`로 보고 베이스를 반환한다.

탈락한 블록은 베이스 회전으로 정확히 복구한다. 유효한 변경이 하나도 남지 않으면 원본 베이스
BVH와 동일한 geometry를 반환한다. URL 또는 handle만 같은 것을 동일성으로 보지 않고 재파싱한
관절·채널 값 또는 콘텐츠 hash로 확인한다.

교차 블록 충돌의 원인이 하나로 확정되지 않으면 최근 변경 블록부터 작은 alpha를 재시도하고,
팔-only·다리-only counterfactual 중 안전하면서 손실이 작은 결과를 택한다. 어느 단일 복구도
안전을 증명하지 못하면 관련 블록을 모두 복구한다. 복구 후 `final_depth`는 실패한 solved frame이
아니라 실제 반환할 adopted frame에서 다시 측정한다.

### 3-3. 사람 평가의 `worse=0`

런타임은 하이브리드 손실과 안전 게이트로 non-regression을 판정한다. v2 승격 시에는 별도로
고정 holdout의 blind `base vs refined` 사람 평가에서 accepted 결과의 `worse=0건`을 요구한다.
이는 유한 표본에서 관측된 0건이라는 뜻이며 미래 오류 확률이 수학적으로 0이라는 주장은 아니다.

holdout에서 `better`가 최소 1건도 없으면 복잡도를 늘릴 이득이 없으므로 승격하지 않는다.

## 4. v2 목적함수

```text
L = L_direction
  + alpha * L_endpoint_2d
  + eta   * L_lower_pair
  + theta * L_hand_pair       # aggressive only
  + kappa * L_lap_contact     # aggressive only
  + beta  * L_move_3d
  + L_P1a_axis
  + L_P1b_nullspace
  + gamma * L_collision
  + delta * L_anatomy
```

### `L_direction`

현재의 2D 뼈 방향 손실을 유지한다.

### `L_endpoint_2d`

검색과 같은 `normalize_skeleton`·유효 mask를 사용해 팔꿈치·손목·무릎·발목 위치 오차를
추가한다. 웹툰 체형 과장을 억지로 맞추지 않도록 Huber 등 robust loss를 사용한다.

### `L_lower_pair`

양쪽 무릎·발목이 모두 유효하고 target 간격이 base보다 유의미하게 좁을 때만 활성화한다. 단순
거리 대신 signed relative vector를 사용해 좌우 순서와 방향을 보존한다.

```text
L_lower_pair =
    Huber(((right_knee - left_knee) - target_knee_vector))
  + Huber(((right_ankle - left_ankle) - target_ankle_vector))
```

두 다리 중 하나의 소유권·endpoint 신뢰도를 증명하지 못하면 이 항을 끈다. 이 항은 모든 자세에서
다리를 무조건 모으는 규칙이 아니며, 벌린 자세의 target에는 target relative vector 자체를 따른다.

### `L_hand_pair`, `L_lap_contact`

`L_hand_pair`는 양손목의 signed relative vector와 중점을 함께 맞춘다. `L_lap_contact`는 target 2D에서
손목-허벅지 접촉 의도가 검출된 pair만 3D signed surface band로 유도한다. 두 항은 aggressive에서만
활성화하며, 각 손별 contact loss가 이전 snapshot보다 커지면 평균 손실이 개선돼도 해당 채택을
거부한다. 깊은 관통은 목적함수와 별개로 기존 arm-leg hard gate가 다시 차단한다.

### `L_move_3d`

베이스 대비 중간관절·말단 이동량을 몸통 길이로 정규화해 소프트 비용으로 사용한다.

```text
move(joint) = |joint_refined - joint_base| / torso_length
```

### `L_collision`, `L_anatomy`

약한 최적화 페널티와 채택 전 하드 안전 게이트를 구분한다. 베이스에 이미 존재하던 접촉은
refine이 새로 만든 오류로 판단하지 않는다.

v2.3의 `L_collision`은 기존 팔-몸통·다리-몸통·다리-다리에 다음 네 arm-leg pair를 추가한다.

```text
left_arm  × left_leg
left_arm  × right_leg
right_arm × left_leg
right_arm × right_leg
```

전완은 팔꿈치→손목 capsule, 손은 손목→손끝 capsule(손끝 매핑이 없으면 손목 sphere), 다리는
허벅지·무릎·정강이 capsule/sphere로 근사한다. 깊이는 몸통 길이로 정규화한다. soft residual은
베이스 대비 악화 깊이만 비용으로 넣고, hard gate는 모든 alpha와 최종 adopted frame에서 다시
검사한다.

hand-leg 초기 engineering 상한은 신규 깊이 `0.01 torso`로 두며, 이는 위 probe의 실패값
`0.025407`보다 작다. 이 값은 production 상수가 아니라 합성·실메시 holdout으로 보정할 초기값이다.
얕은 overlap은 `shallow_contact`, 베이스에 있던 overlap은 `existing_penetration`, 새로 깊어진
overlap은 `new_penetration`으로 진단하되 기존 `status` 필드의 하위 호환 의미는 유지한다.

### 합성 보정 루프

가중치 `alpha`, `beta`, `gamma`, `delta`를 반복 사용한 소수 러프에 맞춰 정하지 않는다.

```text
라이브러리 포즈 A → 렌더·키포인트화 = 2D 쿼리, 3D GT 보유
포즈 A의 이웃 포즈 B = 베이스
B를 A의 2D로 refine
→ 채택 결과와 A의 3D 오차·안전 위반 측정
```

합성 loop는 가중치와 trust region의 1차 스크리닝에 사용한다. 웹툰 도메인과 2D→3D 모호성을
대체하지 못하므로 사람 평가는 최종 승격 관문으로 유지한다.

## 5. 관절·뼈별 신뢰도

```text
bone_weight = valid_mask
            * structural_quality
            * inference_stability
            * calibrated_keypoint_reliability
```

- 검색의 `valid_joint_mask`와 refine의 부위별 score/weight를 분리한다. 검색 피처 공간의 대칭성은
  바꾸지 않고, refine만 projection ambiguity를 별도로 표현한다.
- 절대 segment/torso 길이 초과·cross-slot·낮은 endpoint score·NaN/Inf로 마스킹된 관절은
  refine에서도 사용하지 않는다.
- `adjacent_segment_ratio`만 초과하고 다음을 모두 만족하면 hard mask 대신
  `foreshortening_ambiguous`로 기록한다.
  - 각 segment의 절대 torso 비율은 기존 길이 상한 안쪽
  - root·middle·endpoint raw score가 유효
  - 몸통·owner anchor가 정상
  - distal 관절이 owner box 안에 있고 peer box 소유 증거가 없음
- `foreshortening_ambiguous` 사지는 endpoint 위치 score를 보존하되 압축된 뼈의 direction weight를
  낮춘다. 정확한 감쇠값은 합성 보정 loop로 정하며 전역 길이 임계값을 단순히 올리지 않는다.
- full-image와 crop/A-B 추론이 모두 있을 때 관절 위치 안정성을 반영한다.
- RTMPose raw score만 단독 신뢰도로 사용하지 않는다.
- 실제 구조/소유권 실패로 한 사지의 두 뼈가 유효하지 않으면 그 사지만 동결한다.
- 각 사지는 전체 gain뿐 아니라 자기 사지의 하이브리드 손실도 악화되지 않아야 한다.
- pose backend나 모델 버전이 바뀌면 reliability calibration과 config version을 함께 갱신한다.

## 6. V2-1 우선 과제 — 조건부 하체 refine

v1에서 다리는 코드상 선택 가능하지만 기본값은 팔만 활성화되어 있다. v2는 파트 조합이나 몸통
확장보다 먼저 하체를 실제 제품 조건에서 검증한다.

하체는 다음 조건을 모두 만족하는 선택 후보에서만 활성화한다.

- full shot 또는 하체가 충분히 보이는 half shot
- 고관절·무릎·발목이 구조 검사 통과 또는 §5의 `foreshortening_ambiguous` soft eligibility 통과
- coverage가 `full` 또는 별도 평가에서 승인된 `reduced`
- 각 다리에 유효한 두 뼈와 충분한 관측 감도가 있음
- 다리별 2D direction·position non-regression gate 활성화
- 양다리가 모두 유효하면 target 무릎·발목 signed relative vector를 쓰는 `lower_pair` gate 활성화
- 3D 이동량 정규화와 관절별 해부학 DOF·범위 활성화
- 신규 다리-다리·다리-몸통·손/전완-다리 충돌 검사가 통과
- 베이스 발 방향 보존 또는 검증된 지면 접촉 정책이 있음

발 방향은 COCO-17에서 직접 추정하지 않는다. 지면 증거가 불충분하면 베이스의 world
orientation과 접촉을 보존한다. 다리 하나가 실패하면 그 다리만 복구하되, 부모 관절 영향까지
포함한 최종 전신 pose를 다시 검사한다.

양다리가 모두 유효하면 solver는 기존처럼 하나의 파라미터 벡터로 풀되, 채택은 `lower_pair`를
먼저 시도한다. 공통 alpha `1.0 → 0.75 → 0.5 → 0.25` 중 pair loss·각 다리 loss·전신 loss와
모든 안전 게이트를 만족하는 가장 큰 값을 채택한다. 공동 채택이 실패하면 pair 관계를 악화시키지
않는 범위에서만 기존 다리별 partial fallback을 시도한다.

v2.3 보정 범위에서는 루트·골반 translation, 발목/발 IK, 카메라 최적화를 새로 열지 않는다.
위 sitting probe에서 현행 hip·knee 회전만으로도 간격 개선이 확인됐으므로 먼저 mask·pair loss·
채택·충돌의 누락을 고친다. 그래도 holdout에서 목표 간격에 도달하지 못할 때만 constrained foot
counter-rotation 또는 planted-foot IK를 후속 단계로 검토한다.

하체 단계가 고정 holdout의 `worse=0`, 안전 위반 0, `better>=1`을 통과하기 전에는 몸통 회전을
production에서 활성화하지 않는다.

## 7. V2-2 — 제한된 몸통 방향 회전

몸통 solve는 하체 단계 이후 별도 default-off feature flag로 도입한다.

- 루트 위치와 global root rotation은 고정한다.
- HIERARCHY·OFFSET·채널 순서는 보존한다.
- allowlist에 명시된 골반·가슴의 로컬 회전 채널만 작은 trust region 안에서 푼다.
- shoulder line, hip line, torso side가 모두 유효할 때만 시도한다.
- 목·머리는 매핑이 충분히 검증될 때까지 고정한다.

몸통 후보는 다음 조건을 모두 만족할 때만 채택한다.

1. 몸통 direction·position 손실이 `epsilon`보다 크게 개선
2. 전체 하이브리드 손실이 개선
3. 이미 안전하게 채택된 팔·다리 손실이 악화되지 않음
4. 신규 충돌·관절 제한·지면/접촉 위반이 없음
5. 회전량이 해부학 allowlist와 trust region 안에 있음

몸통 stage의 “베이스”는 원본 BVH가 아니라 **안전하게 채택된 V2-1 사지 결과의 입력 snapshot**이다.
몸통은 사지의 부모이므로 하나라도 실패하면 몸통 채널을 이 snapshot으로 복구하고, 기존 사지의
local channel과 world pose가 허용 오차 안에서 snapshot과 같은지 확인한 뒤 전신 손실·충돌·접촉을
다시 검사한다. 복구 동일성을 증명하지 못하면 V2-1 채택본으로, 그것도 불가능하면 원본 베이스로
전체 fallback한다.

카메라 오차를 root rotation으로 대신 기록하지 않는다. 카메라 yaw·pitch·roll 최적화는 숫자형
카메라 메타데이터를 API·클라이언트·export가 함께 소비할 수 있을 때까지 후속 연구로 둔다.

## 8. 안전한 부분 채택과 제한된 재최적화

사지 또는 몸통 solve가 안전 게이트를 통과하지 못해도 바로 전체 결과를 버리지 않는다.

```text
blend alpha = 1.00 → 0.75 → 0.50 → 0.25 → 0.00(base)
```

각 블록에서 조건을 모두 만족하는 가장 큰 `alpha`를 채택한다. 블록별 채택 뒤 전신 상태에서
전체·부위별 non-regression과 안전 게이트를 다시 검사한다. 안전한 중간값이 없을 때만 해당
블록을 완전히 베이스로 복구한다.

`lower_pair`는 두 다리의 공동 블록으로 먼저 채택한다. 팔은 다리가 베이스일 때 안전했더라도
이후 다리가 움직이며 손을 관통할 수 있고, 반대로 다리도 이미 채택된 손을 관통할 수 있다.
따라서 팔 채택 시 양다리, 다리/`lower_pair` 채택 시 양팔을 모두 재검사한다. 몸통 단계 뒤에도
같은 arm-leg 네 pair를 다시 검사한다.

교차 충돌 발생 시 처리 순서는 다음과 같다.

1. 가장 최근 변경 블록의 더 작은 alpha 재시도
2. 안전한 alpha가 없으면 해당 블록만 base channel로 복구
3. 원인 귀속이 모호하면 arm-only·leg-only counterfactual 비교
4. 남은 블록의 positive gain과 모든 충돌을 adopted frame에서 재검사
5. 어느 조합도 증명되지 않으면 관련 블록 전체, 필요하면 원본 BVH로 fallback

충돌 사지 재최적화(P3b)는 부분 채택만으로 유용한 개선을 보존하지 못할 때만 추가한다.
모든 충돌을 제거하는 물리 시뮬레이션으로 확장하지 않는다.

## 9. 결과 진단 계약

v2 결과에는 최소한 다음 값을 남긴다.

- 전체·팔·다리·몸통별 `direction_loss_base/solved/adopted`
- 전체·팔·다리·몸통별 `position_loss_base/solved/adopted`
- `hybrid_loss_base/solved/adopted`
- 실제 사용한 뼈별 가중치와 제외 사유
- 검색 mask와 refine score/weight의 차이, `foreshortening_ambiguous` 판정 근거
- 중간관절·말단 3D 이동량
- `lower_pair`의 target/base/solved/adopted 무릎·발목 relative vector와 간격
- 관측 감도와 P1a/P1b 진단
- 팔-몸통·손/전완-다리·다리-몸통·다리-다리 충돌과 해부학·지면·접촉 판정
- arm-leg pair별 `part`, `base_depth`, `solved_depth`, 실제 반환 BVH의 `final_depth`, `relation`
- 블록별 부분 채택 `alpha`
- 몸통 단계 적용 여부와 로컬 회전 변화량
- 최종 채택·복구 reason
- refine config·code·feature·pose-library version
- `distance_metric`, `coverage_class`, `search_distance`, 당시 confidence threshold
- 평가용 `gap_type`과 `refine_outcome`을 분리한 값

현재 `RefineResult`와 HTTP 응답은 기존 필드 의미를 유지하고 진단 필드를 선택적으로 추가한다.
`loss_solved`와 실제 반환 BVH의 `loss_adopted`를 구분한다.

## 10. No-refine / v1 / v2.4 비교 평가 전략

### 10-1. 목적과 세 실험군

평가 목적은 검색 결과를 바꾸는 것이 아니라 **동일한 선택 포즈를 refine하지 않았을 때, v1으로
조정했을 때, v2.4 공격적 모드로 조정했을 때 사용자가 받는 최종 포즈의 차이**를 측정하는 것이다.
실제 러프에는 유일한 3D 정답이 없는 경우가 많으므로 실데이터의 대표 표현은 막연한 `정확도`가
아니라 `안전 사용 가능률`, `사람 평가 개선률`, `회귀율`로 한다. 합성 3D GT가 있는 평가에서만
normalized MPJPE 같은 3D 오차를 정확도 지표로 사용한다.

| Arm | 고정 정의 | 실제 평가 artifact |
|---|---|---|
| `B0_no_refine` | 사용자가 선택한 base BVH를 그대로 사용 | 선택 BVH의 content hash와 geometry |
| `B1_v1` | `REFINE_ENABLED=1`, `REFINE_V2_ENABLED=0`, 당시 production v1 config | gate·no-gain·실패 fallback까지 거친 최종 BVH |
| `B2_v24_aggressive` | `REFINE_V2_ENABLED=1`, 요청 `refine_mode=aggressive`, 기본 `REFINE_V2_TORSO=0` | aggressive·conservative·base fallback 중 실제 반환 BVH |

`B2`에서 aggressive 단계가 탈락해 conservative 결과가 반환돼도 이를 `B1`로 재분류하지 않는다.
v2.4 conservative는 v2.4 내부 기여를 분석하는 선택적 네 번째 ablation일 뿐, 사용자가 요청한 세
실험군의 주 비교에는 넣지 않는다. v1의 `REFINE_LIMBS`, v2.4의 모든 가중치·trust region과 timeout은
각 arm의 release config hash로 manifest에 고정한다.

`B0`은 `/analyze`를 `REFINE_ENABLED=0`으로 다시 실행한 결과가 아니다. `/analyze`는 refine을 실행하지
않으므로, 사전에 선택한 **동일한 base artifact**가 no-refine 기준이다.

최종 대체 판정의 주 비교는 `B2 vs B1`이다. `B1 vs B0`은 v1 자체의 부가가치, `B2 vs B0`은
v2.4의 절대 부가가치를 설명하는 보조 비교다.

### 10-2. 평가 단위와 고정 조건

한 평가 단위는 다음과 같다.

```text
(dataset_id, artist/project, scene_group_id, person_id,
 query image/preprocess hash, fixed skeleton, selected_base)
```

세 arm 실행 전에 다음 값을 봉인한다.

- 동일한 원본 러프와 전처리 결과
- 동일한 raw keypoints·scores, target valid-joint mask와 인물 소유권
- 동일한 `pose_id`, view, selected base BVH content hash
- 동일한 pose-library/feature version과 DB/BVH manifest
- 동일한 body/avatar, camera, lighting, renderer와 render version
- 동일한 target view와 3D 안전 확인용 고정 three-quarter 또는 side view
- 동일한 서버 하드웨어·동시성·timeout 조건
- 실제 사용자 선택 또는 deterministic 선택 중 하나로 사전 고정한 base 선택 규칙

버전별로 VLM·RTMPose·검색을 다시 실행하거나 Top-1을 다시 고르면 검색·추출 변화가 refine 효과에
섞인다. live fixture는 한 번만 캡처하고 세 arm이 같은 fixture와 base hash를 소비해야 한다. 서로 다른
서버 프로세스를 쓰면 각 응답이 실제로 같은 base BVH hash를 사용했는지 다시 검증한다.

Top-5 전체를 진단용으로 실행할 수는 있지만 같은 사람의 다섯 후보를 독립 표본 다섯 개로 세지
않는다. 제품 주지표에는 실제 사용자 선택 1개 또는 사전 고정한 rank 1개만 넣고, Top-5 결과는
candidate-level 진단과 실패 원인 분석으로 따로 보고한다.

### 10-3. 분모 계약

주 분모 `N_eval`은 arm 실행 전에 등록한 모든 `(person, selected_base)`다. 다음 결과도 제외하지 않는다.

- refine 미시도 또는 version-specific eligibility 차단
- gate·no-gain·부분/전체 rollback
- aggressive → conservative fallback
- exact-base fallback
- timeout·오류·파싱 불가 artifact

이는 제품이 실제로 반환하는 결과를 보는 ITT(intent-to-treat) 분모다. exact-base fallback은 품질상
no-refine과 동일한 결과로 남고, hash가 같으면 blind pair에서 자동 tie가 될 수 있다. timeout·오류처럼
사용 가능한 artifact를 제시간에 제공하지 못한 경우는 tie가 아니라 운영 실패로 남긴다.

| 분모 | 용도 |
|---|---|
| `N_eval` | 제품 수준 주지표의 고정 분모 |
| `N_common_eligible` | 버전과 독립적으로 정의한 공통 solve 가능 cohort의 진단 |
| `N_attempted,m` | arm `m`에서 solver가 실제 실행된 수 |
| `N_changed,m` | arm `m`의 최종 geometry가 base와 다른 수 |
| `N_fallback_required,m` | gate·실패로 복구가 필요했던 수 |
| `N_feature_active` | 고정 target evidence상 hand/lap/lower-pair가 필요한 수 |

`refined=true`, 변경된 블록, 특정 feature가 활성화된 사례만 모은 수치는 원인 분석에는 유용하지만
headline 개선률의 분모로 쓰지 않는다. 쉬운 결과만 남기는 선택 편향이 생기기 때문이다.

용어도 구분한다.

- 검색의 `accepted_candidate`는 작가에게 유용한 후보라는 뜻이다.
- solver의 `accepted/adopted`는 geometry 블록을 자동 채택했다는 뜻이다.
- runtime `refine_outcome=improved`는 내부 손실 판정이며 사람의 `better` 판정이 아니다.

### 10-4. 블라인드 사람 평가

먼저 B0/B1/B2 artifact를 각각 독립적으로 평가한다.

```text
overall_usability:
  direct | reference | unusable

reject_reason:
  pose_mismatch | anatomy | collision | contact | feet_ground |
  balance | silhouette | ownership | other
```

`direct`와 `reference`를 제품의 `human_usable=1`로 합칠지는 holdout 개봉 전에 rubric으로 고정한다.
그다음 같은 평가 단위에서 다음 세 blind pair를 독립적으로 무작위화한다.

```text
B1 ↔ B0   # v1의 base 대비 가치
B2 ↔ B0   # v2.4의 base 대비 가치
B2 ↔ B1   # v2.4가 v1을 대체할 수 있는가
```

라벨 UI는 버전·mode·rank·distance·pose ID·내부 loss·refined 여부를 숨기고 같은 body·camera·renderer를
사용한다. target view만으로 2D 맞춤을 보고, 고정 추가 view로 관통·균형·발 접촉 같은 3D 안전을 함께
본다. 저장 형식은 버전에 종속되지 않게 한다.

```json
{
  "pair_id": "pair:...",
  "left_artifact_id": "sha256:...",
  "right_artifact_id": "sha256:...",
  "winner": "left|right|tie|both_bad",
  "severity": "minor|major",
  "body_part": "overall|arm|hand|leg|foot|torso",
  "safety_violation": "none|anatomy|collision|contact|ground|other",
  "labeler_id": "artist-..."
}
```

`both_bad`를 tie에 합치지 않는다. exact geometry hash가 같은 두 artifact는 자동 tie로 기록하되 분모에는
남긴다. D1 파일럿은 작가 2명이 30–50 pair를 독립 평가하고 불일치는 제3자가 판정한다. 본 평가도
15–20% 중복 라벨과 5% 숨은 반복으로 inter/intra-rater drift를 측정하며 원래 개인 라벨을 보존한다.

### 10-5. 주지표와 직접 선호 지표

제품 품질의 주지표는 **안전 사용 가능률(Safe Usable Rate, SUR)**이다.

```text
SUR_m =
  count(human_usable_m = 1 AND hard_safety_violation_m = 0)
  / N_eval

v1의 순증          = SUR_B1 - SUR_B0
v2.4의 절대 순증   = SUR_B2 - SUR_B0
v2.4의 v1 대비 순증 = SUR_B2 - SUR_B1   # confirmatory primary effect
```

항상 `x/n`, 비율, 차이 `%p`, paired 95% CI를 함께 보고한다. 상대 개선율을 쓰려면 절대 `%p` 차이를
먼저 적는다. 예를 들어 `50% → 60%`는 `+10%p`, 상대 lift는 `+20%`다.

각 contrast의 직접 선호는 다음과 같이 낸다.

```text
WinRate_all(A>B)  = wins_A / N_eval
LossRate(A<B)     = wins_B / N_eval
TieRate           = ties / N_eval
BothBadRate       = both_bad / N_eval
NetPreference     = (wins_A - wins_B) / N_eval

SafeBetterRate(A>B) =
  count(A가 B보다 선호되고 A에 신규 hard violation이 없음)
  / N_eval
```

tie를 뺀 `wins/(wins+losses)`는 보조 지표로만 사용하고 반드시 raw `win/tie/loss/both_bad`와 함께
제시한다. `B2 vs B1`의 safe win·loss와 SUR 차이가 최종 대체 근거이고, B0 비교는 각 refine 세대의
절대 부가가치를 보여준다. 이미 잘 맞는 base를 안전하게 유지한 tie는 실패가 아니다.

### 10-6. 세 arm 공통 자동 정량 지표

v1 direction loss와 v2 hybrid loss는 목적함수·mask·가중치가 다르므로 직접 비교하지 않는다. 세 최종
BVH를 동일한 **외부 evaluator**로 재투영하고 다음 metric을 다시 계산한다.

| 지표 | 고정 계산 |
|---|---|
| 2D joint NME | 유효 관절의 target 대비 거리 평균 / target torso length |
| limb direction error | 팔·다리 뼈의 target 대비 각도 또는 단위벡터 오차 |
| endpoint NME | 손목·무릎·발목 target 거리 / target torso length |
| hand-pair error | 두 손목 signed relative vector와 midpoint의 target 오차 |
| lower-pair error | 양 무릎·발목 signed relative vector와 midpoint의 target 오차 |
| lap-contact error | target evidence가 있는 손-허벅지 pair의 surface band 이탈량 |
| synthetic 3D error | 같은 rig GT에서 root-aligned torso-normalized MPJPE |

target valid-joint mask와 hand/lap/lower 활성 cohort는 세 arm을 보기 전에 query evidence로 한 번만
계산한다. 각 버전의 diagnostics가 활성화한 관절만 골라 평가하면 안 된다.

```text
ErrorReduction(B→A) = (error_B - error_A) / error_B × 100
```

2D 오차는 깊이·가림·단축 투시 모호성을 해결하지 못하므로 사람 평가와 안전 검사를 대체하지 않는다.
합성 3D GT는 알고리즘 스크리닝 근거이며 웹툰 도메인 holdout의 작가 평가를 대체하지 않는다.

### 10-7. 안전·복구·실행·latency guardrail

solver가 남긴 diagnostics만 신뢰하지 않고 실제 최종 BVH를 공통 evaluator로 재파싱·FK해 독립 검사한다.

- 신규 arm-leg, arm-torso, leg-leg, leg-torso 관통
- 관절 제한·NaN/Inf·BVH parse/FK 실패
- 발 방향·지면 접촉 회귀와 의도된 lap contact의 부유
- root/비허용 joint 이동과 인물 ownership 오류
- fallback artifact의 base content hash·geometry 불일치

```text
NewViolationRate_m     = 신규 hard violation / N_eval
ChangedViolationRate_m = 신규 hard violation / N_changed,m
ExactFallbackRate_m    = exact-base 복구 / N_fallback_required,m
```

`B2`는 다음 funnel을 별도로 낸다.

```text
eligible → attempted → geometry_changed
                     ├─ aggressive applied
                     ├─ conservative fallback
                     ├─ exact-base fallback
                     ├─ timeout
                     └─ error
```

각 arm의 eligibility·attempt·changed/adopted·partial rollback·gate reason·timeout/error 비율을 공개한다.
no-refine은 즉시 준비된 base 기준선이며 v1/v2 endpoint latency percentile에 억지로 섞지 않는다.
v1/v2는 같은 서버·동시성에서 실행 순서를 무작위 interleave하고 cache-off post-click p50/p95와 실제
time-to-ready를 측정한다. timeout은 completed latency 표본에서 사라지지 않게 별도 비율로 함께 낸다.

안전 0건은 `0/N observed`라고 표현한다. 미래 오류율이 0이라는 뜻이 아니므로 정확한 이항 95% 상한
또는 근사 `3/N`도 함께 보고한다.

### 10-8. 통계·데이터 분리·cohort

세 arm이 같은 평가 단위를 공유하므로 paired 분석을 사용한다.

- raw `x/n`, person micro-average와 작가별 macro-average
- artist/project 또는 `scene_group_id` clustered bootstrap 95% CI
- binary SUR 차이의 McNemar 검정은 보조로 사용
- win/loss/tie와 연속 오차는 paired clustered bootstrap
- 효과 크기와 CI를 우선하고 p-value만으로 결론 내리지 않음
- confirmatory primary는 `B2 vs B1` 하나로 사전 고정
- `B1 vs B0`, `B2 vs B0`와 여러 slice는 secondary/exploratory로 구분

D0의 반복 사용 이미지는 engineering probe로만 쓴다. D1의 30–50 pair 파일럿에서 discordant pair
비율·예상 SUR 차이·작가/project 내 상관과 라벨러 변동을 측정하고, D2를 열기 전에 최소 의미 개선폭
(MCID)과 표본 수를 고정한다. 약 `+10%p` 차이를 안정적으로 주장하려는 현재 계획값은 12–16명 작가,
350–500 person pair이며 파일럿의 cluster design effect로 다시 계산한다. 새 작가 또는 새 프로젝트를
sealed D2 holdout으로 분리한다.

다음 cohort는 사전 고정해 각각 `n`, SUR, win/tie/loss/both_bad, 안전 위반, geometry-changed 비율을 낸다.

- hand-pair, lap-contact, lower-pair target evidence
- 팔·다리·몸통 block
- standing, sitting, crouching 등 pose type
- front, three-quarter, side, back
- full/reduced coverage와 foreshortening ambiguity
- selected rank와 search-distance band
- aggressive applied / conservative fallback / base fallback
- `near_gap`, `structural_gap`, `unknown`

작은 slice는 탐색 결과로만 표시한다. family GT가 없으면 near-gap 개선률은 `INCONCLUSIVE`로 두며,
자동 라벨을 KPI 정답처럼 쓰거나 `unknown`을 조용히 제외하지 않는다.

### 10-9. 승격 기준과 보고 형식

D2 holdout을 열기 전에 다음 값을 고정한다.

```text
Primary value:
  SUR_B2 - SUR_B1 >= 사전 MCID
  paired 95% CI가 사전 우월/비열등 기준 통과

Human guardrail:
  B2 changed 결과의 base 대비 major worse = 0/N observed
  B2 vs B1 major regression = 0/N observed

Safety:
  신규 구조 violation = 0/N observed
  exact fallback identity = 100%

Operations:
  cache-off post-click p95가 제품 예산 이내
  timeout/error가 사전 허용치 이하

Worst slice:
  사전 핵심 cohort에서 허용 폭 이상의 회귀 없음
```

표본·라벨이 부족하거나 CI가 판정선을 가로지르면 `PASS`나 `FAIL` 대신 `INCONCLUSIVE`다. 결과표는
최소 다음 형태를 유지한다.

| Arm | SUR x/n | SUR | Δ vs B0 | human W/T/L/BB vs B0 | 공통 2D NME | 신규 위반 | changed | cache-off p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 no-refine |  |  | 기준 | 기준 |  |  | 0 | 즉시 base |
| B1 v1 |  |  |  |  |  |  |  |  |
| B2 v2.4 aggressive |  |  |  |  |  |  |  |  |

별도로 `B2 vs B1`의 `W/T/L/BB`, SUR paired 차이와 clustered 95% CI를 headline에 둔다.

허용되는 표현 예시는 다음과 같다.

> 동일한 N개 선택 포즈에서 안전 사용 가능률은 no-refine A%, v1 B%, v2.4 aggressive C%였다.
> v2.4는 v1 대비 `+(C-B)%p`였고 clustered bootstrap 95% CI는 `[L, U]%p`였다.
> v2.4/v1 직접 비교는 `W/T/L/BB=x/y/z/q`, 신규 구조 위반은 `0/N observed`였다.

사람 선호만으로 `v2.4 정확도 30% 향상`, 서로 다른 내부 loss로 품질 향상, changed-only 결과를 전체
성공률처럼 표현하는 것은 금지한다. 합성 GT에서는 `3D error가 X% 감소`라고 별도로 표현한다.

### 10-10. 현재 하네스 상태와 필요한 구현

2026-08-14 기준 위 계약은 기존 engineering probe인 `refine-pairs`와 분리된
`standin_eval run refine-eval`로 구현됐다.

| 단계 | 구현 위치 | 산출물 |
|---|---|---|
| frozen query/base와 B0 | `standin_eval/refine_three_arm.py` | `frozen_units.jsonl`, `frozen_manifest.json`, B0 BVH |
| v1/v2 capability·config 검증 | `api/app.py`, `refine_three_arm.py` | `/healthz.refine`, server manifest |
| B1/B2 실행과 lineage | `refine_three_arm.py` | `refine_arms.jsonl` |
| 공통 자동 evaluator | `standin_eval/refine_evaluator.py` | arm별 `external_evaluation`, 공통 metric·안전 결과 |
| 고정 blind render와 세 contrast | `standin_eval/refine_render.py` | 독립 item, 3 pair, private provenance, SVG |
| SUR·W/T/L/BB·CI·guardrail | `standin_eval/refine_report.py` | `refine_evaluation_report.json/.md` |

하네스는 B2 요청에 `refine_mode=aggressive`를 명시하고, aggressive→conservative→base
lineage를 펼쳐 저장한다. source의 `refine_allowed`가 false이거나 요청·gate·fallback·timeout이
발생해도 사전 등록한 `N_eval`과 세 arm 행에서 제외하지 않는다. 양쪽 서버가 같은 base content를
제공하지 않거나 v1/v2 flag·code/config 정체성이 다르면 실행 전에 실패한다. 동일 최종 geometry는
blind pair에서 자동 tie로 남되 분모에서 제거하지 않는다.

승격용 threshold·rubric·통계 seed·bootstrap 반복 수·report version은
`run refine-eval --promotion-criteria ...` 시점, 즉 서버 접촉과 holdout 결과 확인 전에
`promotion_criteria.frozen.json`으로 봉인한다. reporter는 manifest와 frozen manifest의 hash를
다시 확인하며, 사후 전달한 다른 기준은 거부한다. arm 결과·BVH·SVG·blind provenance·라벨 assignment도
`result_manifest.json`에 봉인한다. 사람 평가 assignment는 기본 20% 이중 라벨과 5% 숨은 반복을 포함하고,
최종 승격에서는 최소 2명·15% 유효 이중 라벨·5% 유효 숨은 반복을 완화할 수 없다.

추출 실패·잘못된 query도 ITT 세 arm에 남긴다. target skeleton을 그릴 수 없는 경우에는 target 부재를
명시한 동일 버전 safety-view blind card를 만들며, artifact를 제시간에 전혀 제공하지 못한 transport
timeout/error만 자동 운영 실패로 판정한다. solver timeout 응답이 usable exact-base artifact를 제공한
경우 품질 비교는 geometry tie로 남기되 timeout 운영 guardrail에는 계속 센다.

현재 공통 3D 안전 검사는 COCO-17/capsule proxy이므로 실제 CSP mesh 안전 holdout을 대체하지 않는다.
승격 reporter는 arm row의 자기 주장 flag를 mesh 증거로 인정하지 않는다. 별도 versioned
`mesh_safety_evidence.jsonl`이 각 `unit_id × arm`의 봉인 artifact/geometry와 일치하고, CSP/avatar body
version·검사 완결성·absolute/new hard-violation ID를 제공해야 한다.
구현된 `mesh-checks-v2`는 `parse_fk`, `ownership`, `anatomy`, `collision`, `contact`, `ground`,
`foot_direction`을 필수로 고정한다. 실패 check는 같은 prefix의 stable hard-violation ID가 필요하고,
동일 unit의 B0가 통과한 check를 B1/B2가 실패하면 같은 prefix의 신규 violation ID도 필수다.
또한 러프의 유일한 3D 정답이 없으므로 실제 데이터의 SUR와 직접 선호에는 완성된 blind 사람 라벨이
필요하다. 라벨·사전 승격 기준·ownership 또는 mesh 안전 근거가 부족하면 reporter는 수치를 조작해
PASS로 만들지 않고 `INCONCLUSIVE`를 반환한다. 실행·라벨 절차는 `evaluation/README.md`를 따른다.

### 10-11. v2.3 결함 회귀 테스트

#### 단위 — 손/전완-다리 충돌

- clear·얕은 접촉·깊은 관통과 base 기존 관통/악화 판정을 분리
- 좌·우 손 × 좌·우 허벅지·무릎·정강이 pair 검사
- translation·scale·mirror 불변성
- 손끝 관절이 없는 BVH의 wrist/hand sphere fallback
- `alpha=1.0`은 관통하지만 더 작은 alpha는 접촉인 합성 사례에서 가장 큰 안전 alpha 채택
- 원인 팔 또는 다리만 롤백하고 실제 반환 frame의 `final_depth`가 안전한지 확인
- sitting 후보 #2의 base depth `0`, refined depth 약 `0.025407` fixture를 신규 관통으로 검출

#### 단위 — seated foreshortening과 `lower_pair`

- 절대 길이 정상·raw score 유효·owner 정상·peer 오염 없음인 balance-only 단축 투시는
  hard mask하지 않고 soft eligible로 보존
- 절대 segment 상한 초과·낮은 endpoint score·cross-slot·NaN/Inf는 기존처럼 차단
- 전역 ratio 경계 `3.5±epsilon`, 좌우 mirror, 이미지 scale parameterized 회귀
- target이 모은 자세일 때 무릎·발목 relative vector와 간격이 개선
- 벌린 target에서는 임의로 다리를 모으지 않음
- 공동 alpha 실패 시 안전한 per-leg fallback 또는 exact base 복구

#### 통합 — 실제 파이프라인

테스트가 `refine_bvh(... allowed_limbs=...)`를 직접 호출하는 데서 끝나면 안 된다. sitting fixture가
스켈레톤 분석 → descriptor → `refinable_limbs` → `/refine` → adopted/base BVH 전체 경로를 지나야 한다.

- 양다리가 hard mask되지 않고 lower-body solver가 실제 시도됨
- 관측 가능한 각 다리에 `accepted` 또는 구체적 안전 reason이 존재
- 채택 다리·`lower_pair`·전신 hybrid가 모두 non-regression
- 최종 BVH 재파싱 후 모든 hand-leg/leg-leg/leg-torso/ground/contact 재검사 통과
- 실패 시 해당 블록 채널 또는 전체 BVH가 콘텐츠 hash 기준 베이스와 동일

현재 sitting 5-pair 합성 probe에서는 5개가 2D 기준 채택됐지만 3D GT는 3개만 개선했고 2개는
각각 `+0.000962`, `+0.002919 torso` 회귀했다. 합성 승격 gate는 accepted 결과의 3D GT 회귀
`0건`을 요구하며 수치 허용 오차는 `1e-3 torso` 이하로 사전 고정한다. `+0.002919` 사례는 반드시
실패해야 한다.

#### 시각 holdout

- 앉기·손을 무릎에 얹기·팔짱·쪼그리기를 포함
- 같은 body·mesh·camera로 target view와 고정 3/4 또는 side view를 함께 확인
- accepted 결과의 보이는 손/전완-다리 관통 `0건`
- 의도된 손-무릎 접촉이 손이 공중에 뜨는 결과로 바뀐 회귀 `0건`
- 하체 eligible 결과의 실제 solver attempt와 pair 채택/롤백 reason을 전부 분모에 포함

### 10-12. 결함 회귀 검증용 데이터 분리

- 반복 사용한 기존 러프는 engineering/calibration으로만 사용한다.
- 합성 보정 loop는 가중치·trust region 탐색에 사용한다.
- 새 작가 또는 프로젝트 데이터를 sealed holdout으로 분리한다.
- holdout 확인 전에 `worse=0`, `better>=1`, 안전 위반 0과 latency 예산을 고정한다.

### 10-13. 기능 단계별 ablation 기준

- **V2-1 하체**: 같은 config의 `legs off vs legs on`을 하체 eligible cohort에서 비교한다. 실제
  다리 블록이 채택된 결과에서 `worse=0`, `better>=1`이어야 한다.
- **V2-2 몸통**: `V2-1 adopted vs torso on`을 비교한다. 실제 몸통 블록이 채택된 결과에서
  `worse=0`, `better>=1`이어야 한다.
- **최종 대체**: `current refine vs final v2`에서도 accepted 결과의 회귀가 0이어야 한다.

평가 manifest에는 `(query, selected_base)`를 실제 사용자 선택으로 고정할지, deterministic 규칙으로
고정할지를 기록한다. 같은 비교 안에서 base 선택 규칙을 바꾸지 않는다.

## 11. API·BFF·export 범위

이번 v2는 `/analyze` 후보 계약이나 Top-5 순서를 바꾸지 않는다. `/refine`은 사용자가 선택한
후보 하나만 처리한다. 따라서 다음 기능을 새로 설계할 필요가 없다.

- Top-5 다섯 파일의 선행 생성과 영속 저장
- 일부 후보만 먼저 끝났을 때 비동기 순서 교체
- 선행 refine timeout 후 Top-5 전체를 복구하는 정책
- 합성 후보를 위한 새 candidate identity

후보 카드의 정적 썸네일은 베이스 포즈를 나타낸다. refine 결과의 확인 기준은 `/refine`이 반환한
`bvh_url`을 사용한 선택 후보의 3D preview다. refined 썸네일을 저장·공유해야 하는 요구가 생기면
BVH와 동일한 artifact identity 및 body·camera·renderer version을 사용하는 별도 계약으로 설계한다.

기존 선택 후 `/refine` 경로에서는 응답의 `bvh_url`이 조정본 또는 베이스를 가리킨다는 계약을
유지한다. BFF·클라이언트·export는 사용자가 실제 선택한 결과의 `bvh_url`을 끝까지 보존해야 한다.

현재 legacy 추론 서버의 `/export-order`는 `pose_id + view`만 받아 항상 베이스
`/pose/{id}/bvh`를 다시 만든다. 따라서 **이 endpoint는 v2 조정본 export 경로로 사용하지 않는다.**
제품 기준인 BFF candidate export가 `/refine`의 반환 `bvh_url` 또는 refined handle을 선택 결과에
저장해 전달한다. legacy endpoint까지 v2에 포함하려면 optional refined artifact 필드를 추가하는
별도 API 계약 승인을 먼저 받는다.

v2 config를 켤 때 현재 선택 1개 캐시 키에는 최소한 query keypoints·scores·mask, base BVH content
hash, pose-library/feature/config version, view와 허용 부위를 포함한다. 이는 새 비동기 기능이 아니라
같은 입력에 오래된 v1 조정본이 재사용되는 것을 막기 위한 기존 캐시 무효화 보강이다.

## 12. 구현 순서

### 현재 구현 상태

| 단계 | 코드 상태 | 남은 승격 증거 |
|---|---|---|
| V2-0 계약·측정 | 구현됨 | sealed holdout의 사전 판정 기준 고정 |
| No-refine/v1/v2.4 3-arm 평가 | runner·공통 evaluator·blind label/report 구현됨 | sealed D1/D2 작가 라벨·실 CSP mesh 안전 증거 |
| 병렬 트랙 F pose family | 미착수 | provenance·versioned family GT·사람 감사셋 |
| V2-1 하체 v2.2 | feature flag 구현됨 | sitting probe에서 단축 투시 hard mask와 pair 목적함수 누락 확인 |
| v2.3 C1 손-다리 안전 | feature flag 구현·단위/실입력 proxy 검증 완료 | 실메시 holdout 관통 0 |
| v2.3 C2 단축 투시 | 검색/refine mask 분리·실제 sitting 양다리 eligibility 확인 | sealed 오염 holdout 회귀 0 |
| v2.3 C3 `lower_pair` | signed loss·공동 alpha·per-leg fallback 구현 | 합성 3D GT·human worse 0·better≥1 |
| v2.4 보수적/공격적 모드 | 구현·모드별 cache·정확 복구 단위검증 완료 | 고정 blind pair의 공격적 accepted worse 0 |
| v2.4 `hand_pair`·`lap_contact` | 공동/개별 채택·접촉 band·각 pair non-regression 구현 | 손목 간격/중점 human 개선·접촉 부유/관통 0 |
| v2.4 강화 lower pair·발목 보정 | 공격적 가중치·Foot-only `18°` 보정 구현 | 목표 간격 human 개선·발 방향/지면 회귀 0 |
| V2-2 몸통 | 별도 default-off flag 구현됨 | V2-1 승격 후 독립 blind pair 평가 |
| V2-3 제품 품질 | 선택 1개 cache·timeout fallback·진단 구현됨 | BFF/클라이언트/export 실제 환경 E2E와 cache-off p95 |

자동 검증은 `tests/test_refine_v2.py`, 합성 보정 loop는
`scripts/eval_refine_v2_synthetic.py`가 담당한다. 검색과 분리된 예전 engineering probe는
`standin_eval refine-pairs`, 승격용 B0/B1/B2 blind 평가는 `standin_eval run refine-eval`과
`standin_eval report`가 담당한다.

v2.3 C1–C3는 사용자 최종 승인 후 같은 구현 묶음으로 반영했다. C2/C3만 켜지고 C1이 빠지는
중간 상태는 허용하지 않는다. 구현은 `REFINE_V2_ENABLED=1` 아래에서만 동작하며 v1과 production의
v2 기본 OFF는 유지한다.

`REFINE_V2_CODE_VERSION`은 `v2.4.0`이며 C1–C3와 v2.4 mode·pair/contact·발목 설정은 refine
content hash와 진단 `config_version`에 포함된다. mode도 hash에 들어가 conservative artifact가
aggressive 요청에 재사용되거나 그 반대가 되지 않는다. 이전 v2 캐시·artifact는 v2.4 결과로
재사용하지 않는다.

### v2.3 구현 직후 sitting E2E 확인 — 2026-08-12

`in/스크린샷 2026-07-09 124702.png`을 실제 Gemini·RTMPose와 Top-5 전체 refine으로 다시 실행했다.
감사 산출물은 `out/refine_v23_sitting_20260812_live/`에 있다.

- 검색 mask는 두 발목을 기존처럼 제외했지만 refine mask는 양쪽 고관절·무릎·발목을 보존했다.
- `foreshortened_limbs=[left_leg,right_leg]`, `coverage=reduced`, `refine_allowed=true`가 기록됐다.
- Top-5 모두 안전한 부분 결과를 반환했다. #1·#2·#3·#5는 `lower_pair` 공동 alpha를 채택했고,
  #4는 target이 base보다 좁지 않아 pair 항을 열지 않고 안전한 단일 다리 fallback만 채택했다.
- 예를 들어 #2의 정규화 무릎·발목 간격은 `[0.861, 0.932] → [0.599, 0.592]`로 줄었다.
- 다섯 결과의 arm-leg 네 pair는 proxy 기준 모두 `clear`, 실제 adopted `final_depth=0`이었다.
- 별도 10-pose 합성 보정 loop에서는 10건 모두 3D GT 오차가 개선됐고 accepted 3D regression은
  0건이었다(`out/refine_v23_synthetic_10/manifest.json`).

이는 구현·engineering probe 통과 증거이며, sealed holdout의 사람 평가나 실메시 충돌 검증을
대체하지 않는다. 따라서 production 승격 상태는 여전히 대기다.

### v2.4 구현 직후 sitting Top-5 E2E 확인 — 2026-08-12

같은 `124702` 입력을 실제 Gemini·RTMPose로 다시 추출하고 Top-5 전부에 명시적 `aggressive`를
실행했다. 감사 산출물은 `out/refine_v24_sitting_20260812_aggressive/`에 있다.

- Top-5 모두 `mode_requested=aggressive`, `mode_applied=aggressive`의 안전한 조정본을 반환했다.
- #1의 정규화 무릎·발목 간격은 `[0.728, 0.762] → [0.399, 0.349]`로 줄었고, 오른발은 Foot local
  counter-rotation으로 기준 방향 오차가 `13.338° → 2.058°`가 됐다.
- #4는 손목 간격 `0.453 → 0.315`, hand-pair loss `0.01755 → 0.00351`, lap-contact loss
  `0.11970 → 0.02023`, lower-pair loss `0.01775 → 0.00085`로 함께 개선됐다.
- lap-contact가 활성화된 #3·#4·#5는 모든 pair가 양의 clearance를 유지해 신규 관통이 없었고,
  각 contact loss가 aggressive 시작 snapshot보다 감소했다.
- `tests/test_refine_v2.py`의 모드/cache/contact/Foot-only/정확 복구를 포함한 16개와 기존 smoke·
  skeleton extraction·evaluation 88개, 총 104개 자동 테스트가 통과했다.

다만 #3·#4·#5의 일부 손은 아직 목표 contact band까지 도달하지 못하고 부유가 남는다. 따라서 이
결과는 v2.4 구현과 방향성의 engineering 증거이지, 사용자 예시 수준 달성 또는 production 승격
승인이 아니다. 다음 판정은 고정 blind pair에서 육안 `worse=0`, 손·다리 개선, 접촉 부유 0을 본다.

### V2-0 — 계약과 측정 고정

1. 전체·부위별 base/solved/adopted 손실과 설정 버전 기록
2. positive-gain·zero-regression reason과 정확한 베이스 복구 테스트 고정
3. v2 `refine_allowed`를 스켈레톤·소유권·BVH 안전 기반으로 재정의
4. v2 전용 `REFINE_GAIN_EPSILON`과 v1 `REFINE_MIN_GAIN`의 병행·폐기 일정 정의
5. 합성 보정 loop와 blind pair 평가 스키마 구축

### 병렬 측정 트랙 F — pose family

이 트랙은 V2-1 하체 구현을 막지 않는다.

1. 라이브러리 provenance·pose-family 복구
2. family GT가 준비된 뒤 near/structural/unknown 분류 baseline 측정
3. 사람 감사셋으로 분류 규칙 검증
4. near-gap 개선률과 unknown 비율 보고

### V2-1 — 하체 우선

1. C1: 전완·손 × 허벅지·무릎·정강이 3D 충돌 프록시와 base-relative hard gate
2. C1: 팔·다리·몸통 채택 뒤 arm-leg 네 pair 최종 재검사와 실제 adopted `final_depth`
3. C2: 검색 mask와 refine score/weight 분리, balance-only foreshortening soft eligibility
4. C2: 전체 score zeroing을 사지별 fail-closed 정책으로 교체
5. C3: robust 2D endpoint loss에 `lower_pair` signed relative-vector loss 추가
6. C3: 양다리 공동 alpha 채택 후 안전한 per-leg fallback
7. 기존 관절별 해부학 DOF·이동량·다리-다리·다리-몸통·발 방향·지면 접촉 게이트 유지
8. 실제 sitting E2E·합성 3D·sealed visual holdout 승격 판정

### V2-2 — 제한된 몸통 회전

1. allowlist 기반 골반·가슴 로컬 회전
2. 몸통별 positive-gain·non-regression gate
3. 몸통 변경 후 전 사지 재계산·복구 검증
4. 하체 단계 대비 독립 blind pair 평가

### V2-3 — 복구와 제품 품질

1. 필요할 때만 충돌 블록 재최적화
2. 뼈별 신뢰도 보정과 target-independent 관측 감도 검증
3. 선택 후 `/refine → bvh_url → export` E2E 검증
4. cache-off p95와 timeout 시 베이스 복구 검증

### V2-4 — 보수적/공격적 사용자 선택

1. `refine_mode` API·cache·sidecar·진단 lineage 추가, 기본 `conservative`
2. conservative 결과 또는 안전한 base snapshot을 aggressive 시작점으로 고정
3. `hand_pair` 공동 solve·공동 alpha·팔별 pair non-regression fallback
4. 2D 접촉 증거 기반 `lap_contact`와 3D surface band residual
5. 공격적 전용 lower_pair 가중치와 기존 target-relative 조건 유지
6. `LeftFoot`·`RightFoot` 제한적 counter-rotation 및 최종 전신 안전 재검사
7. aggressive 실패 시 conservative artifact의 내용·URL 일관성 검증
8. 같은 base·query에서 conservative/aggressive blind pair 평가

V2-1 이후 각 제품 단계는 이전 단계의 `worse=0`, 구조 안전 위반 0, `better>=1`을 통과한
경우에만 다음 단계로 간다. 병렬 family 측정 트랙의 완료는 하체 feasibility를 막지 않는다.
개선이 없으면 복잡도를 더하지 않고 현행 refine을 유지한다.

## 13. 후속 연구로 연기

- 상·하체 파트 조합과 rig retarget
- Top-N/Top-5 선행 refine과 재정렬
- Top-1/hover 후보 저우선 background prefetch
- progressive refined preview
- 검색 거리만으로 자동 선택하는 공격 모드(명시적 v2.4 사용자 선택은 현재 범위)
- 카메라 yaw·pitch·roll 최적화
- 2인 상호작용 세트 refine
- COCO-17만으로 손가락·발 방향 직접 최적화

후속 연구 항목은 현재 v2 완료 조건이나 near-gap 개선률의 분자에 포함하지 않는다.

## 14. 당장 하지 않을 것

- 회전 상한만 키워 더 세게 맞추기
- 정규화만 낮춰 2D 손실을 강제로 줄이기
- `SKELETON_ADJACENT_SEGMENT_RATIO_MAX` 전역값만 올려 단축 투시와 오염을 함께 허용하기
- 검색 mask를 그대로 풀어 피처 공간 대칭성과 기존 검색 결과를 조용히 바꾸기
- hand-leg 안전 게이트보다 먼저 foreshortening 하체를 활성화하기
- target 관계를 보지 않고 모든 자세의 양다리를 무조건 모으기
- 하체 안전 게이트 없이 `REFINE_LIMBS=all`을 기본 활성화
- 몸통 allowlist·복구 검증 없이 골반·척추 전체를 풀기
- 스켈레톤 추출 실패를 검색 거리나 solver로 억지 보정
- 구조 공백을 큰 관절 변형으로 메우기
- 반복 사용한 소수 러프만 보고 가중치와 임계값 확정
- pose-family GT 없이 공백 자동 분류 성능이나 고정 개선률 주장

## 15. 완료 조건

v2는 다음을 모두 만족할 때 현재 refine을 대체한다.

- frozen `(query, selected_base)`의 B0/B1/B2 ITT artifact·라벨 완결성이 100%
- `SUR_B2 - SUR_B1`이 D2 전에 고정한 MCID·paired CI 판정선을 통과
- 유효하지 않은 스켈레톤·인물 소유권 실패에서 solver 실행 0
- 실제 반환한 모든 조정본의 전체·변경 블록 손실이 non-regression gate 통과
- 고정 holdout의 accepted 결과에서 human-rated `worse=0건`
- 같은 holdout에서 `better>=1건`
- `current refine vs final v2` 비교에서도 accepted 결과의 회귀 0
- 신규 충돌·관절 제한·지면/접촉 위반 0
- accepted 결과의 손/전완-허벅지·무릎·정강이 신규 관통 0
- hand_pair 변경이 채택된 aggressive 결과에서 손목 간격·중점 중 최소 하나가 conservative보다 개선
- lap_contact 활성 결과의 깊은 관통 0, 의도된 접촉의 과도한 부유 0
- 발목 counter-rotation 적용 결과의 발 방향·지면 접촉 회귀 0
- aggressive 실패 시 conservative 또는 base geometry로 정확히 복구
- seated foreshortening fixture에서 양다리 solver attempt가 진단에 남고, 안전한 pair 또는 per-leg
  개선이 최소 1건 이상 채택됨
- 합성 accepted pair의 3D GT 회귀 0(`1e-3 torso` 이하의 사전 고정 수치 오차만 허용)
- 손을 무릎에 얹는 target에서 관통을 피한다는 이유로 의도된 접촉이 공중 부유로 바뀐 회귀 0
- 안전 실패 시 해당 블록 또는 전체 베이스 geometry로 정확히 복구
- 하체와 몸통 단계가 각각 독립 승격 관문 통과
- 선택 후 `/refine` 흐름에서 BFF·클라이언트·export가 조정본 또는 베이스를 일관되게 소비
- post-click p95가 사전 합의한 제품 예산 안에 있음
- family GT가 준비된 경우 near-gap 개선률과 unknown 비율을 함께 보고

family metadata가 없으면 near-gap 개선률 평가는 `INCONCLUSIVE`로 남긴다. 이것은 하체·몸통
feasibility 구현을 막지는 않지만, 근접 공백 커버 성능을 주장할 수 없다는 뜻이다.

현재 `REFINE_DESIGN.md`는 위 조건을 통과해 v2가 승격될 때까지 실행 동작의 단일 기준으로 유지한다.
