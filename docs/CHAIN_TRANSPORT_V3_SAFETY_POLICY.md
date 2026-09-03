# CHAIN_TRANSPORT_V3 수학·동결 정책과 Safety V1 폐기 기록

> 문서 상태: 기준 — V3 수학·기준선 동결 / Safety V1 **REJECTED**
> 작성일: 2026-08-26 · 최종 판정 반영: 2026-08-27
> 기준 코드: `qa/retarget/CHAIN_TRANSPORT_V3/converter/retarget.py`
> 목적: 살아남은 체인 수학과 동결 범위를 고정하고, 실패한 메시 안전 선택기를 재승격하지
> 않도록 실험과 최종 판정을 함께 기록한다.

이 문서는 현재 운영 코드의 승격 선언이 아니다. 수학 코어와 frozen V3 기준선은 유지하지만,
이 문서의 §5~§9에 기록된 Safety V1 선택기는 원본 FBX 육안 gate에서 실패해 폐기됐다.
해당 절은 후속 구현 지시가 아니라 실패 분석을 보존하는 기록이다.

---

## 0. 결론

CHAIN_TRANSPORT_V3의 중심 수학인 **순차 최소회전 체인 수송**과 사용자가 합격시킨 frozen
V3를 유지한다. 실제 메시 기반 적용량 선택기 Safety V1은 대리 지표 일부를 개선했지만 골반과
발목의 실제 형상을 개선하지 못했고, CMU에서는 기존 120° foot 보호 조건을 우회해 발 방향을
회귀시켰다. 따라서 Safety V1은 **REJECTED**이며 임계 조정이나 부분 재사용 대상이 아니다.

현재 허용 구조는 다음과 같다.

~~~text
frozen CHAIN_TRANSPORT_V3
          │
          ├─ 기존 120° foot 보호 조건 유지
          ├─ 기존 terminal-follow 유지
          └─ 사용자 육안 합격본을 기준선으로 사용
~~~

동결 결정의 한 문장 정의는 다음과 같다.

> **frozen V3의 수학과 기존 보호 조건을 함께 보존한다. 메시 대리 지표가 좋아져도 해부학적
> 방향이나 원본 FBX 육안 품질이 나빠지면 후보를 승격하지 않는다.**

파일명·프로파일명 예외는 계속 금지한다. 다만 frozen V3에 이미 포함되어 사용자가 합격시킨
120° foot 보호 조건은 기준선의 일부이며, 후속 wrapper가 이를 우회해서는 안 된다.

---

## 1. 상태 표기

| 표기 | 의미 |
|---|---|
| **FROZEN-MATH** | 수학 코어. 후속 안전 정책이 바꾸면 안 된다. |
| **FROZEN-BASELINE** | 비교 기준인 QA 후보 코드와 산출물. 제품 승격을 뜻하지 않는다. |
| **FROZEN-POLICY** | 설계 원칙과 폴백 의미. 구현 중 자의적으로 바꾸면 안 된다. |
| **PROVISIONAL** | 현재 사례로 만든 탐색값. discovery 뒤 고정해야 한다. |
| **UNRESOLVED** | 별도 실험이 필요한 영역. 현재 정책으로 일반화하지 않는다. |

---

## 2. 기준선과 용어

### 2.1 QA 기준선

~~~text
variant: CHAIN_TRANSPORT_V3
retarget.py SHA-256:
2701cc44a9ec6584722c194002d505a21bbf4fe199695325bb3b660c2818985e
~~~

경로:

~~~text
qa/retarget/CHAIN_TRANSPORT_V3/converter/retarget.py
~~~

이 해시는 **FROZEN-BASELINE**이다. 해시 안에는 최종 동결되지 않은 120° foot 제한과
terminal hand 정책도 포함돼 있다. 따라서 “파일 해시 동결”과 “파일 안 모든 정책의 제품
동결”을 동일하게 해석하면 안 된다.

### 2.2 Legacy 결과 C

소스 본 s, 타깃 본 d에 대해 기존 회전 결과를 다음처럼 쓴다.

~~~text
Δ_s = rot(PoseWorld_s · RestWorld_s⁻¹)
R_C = Δ_s · rot(RestWorld_d)
~~~

C는 안전 후보 부재, 측정 불가, 수치 퇴화 시 되돌아가는 정확한 기준이다. 임의의 중간
회전이나 parent-follow를 C라고 부르지 않는다.

### 2.3 좌표와 edge

체인의 관절점으로부터 다음 단위 방향을 만든다.

~~~text
e_s[i] = normalize(source_pose_point[i+1] - source_pose_point[i])
e_d[i] = normalize(target_rest_point[i+1] - target_rest_point[i])
~~~

미러 경로에서는 소스 점과 방향을 X 반사한 동일 프레임에서 계산한 뒤 같은 수식을 적용한다.
방향 길이가 유효하지 않으면 해당 체인은 수치 퇴화다.

---

## 3. 중심 수학 — 순차 최소회전 체인 수송

### 3.1 최소회전

단위벡터 a, b에 대해 다음을 계산한다.

~~~text
φ = acos(clamp(dot(a,b), -1, 1))
axis = normalize(cross(a,b))
H = Rotation(axis, φ)
~~~

규칙:

- φ < 0.5°: H = I
- φ > 175°: 축이 불안정하므로 수치 퇴화
- |a| 또는 |b| ≈ 0: 수치 퇴화
- 퇴화 시 임의 축을 선택하지 않는다.

최소회전에는 본 축 둘레의 추가 twist가 없다. 이 성질이 타깃 rest의 roll을 최대한 보존한다.

### 3.2 누적 수송 공식

각 체인에서 부모가 이미 운반한 타깃 rest edge를 현재 소스 pose edge에 맞추는 최소회전만
누적한다.

~~~text
Q[-1]        = I
predicted[i] = Q[i-1] · e_d[i]
H[i]         = min_rotation(predicted[i] → e_s[i])
Q[i]         = H[i] · Q[i-1]
R_out[i]     = Q[i] · R_target_rest[i]
~~~

이 공식은 **FROZEN-MATH**다.

### 3.3 이 공식이 사용하지 않는 것

- 거의 일직선인 타깃 T-pose의 bend-plane normal
- 소스의 절대 rest frame을 타깃 roll 기준으로 가져오는 전송
- source local twist를 새 타깃 roll로 사용하는 절대 회전 전송
- 파일명이나 source profile에 대응하는 rest 상수표

V2는 불안정한 타깃 bend-plane normal을 사용해 방향이 맞아도 허벅지·발목을 축 둘레로
회전시키고 메시를 찢을 수 있었다. V3는 그 기준을 제거한다.

### 3.4 적용 순서 불변식

- 모든 desired world rotation을 먼저 계산한다.
- 타깃 본에는 부모→자식 순서로 적용한다.
- desired.translation은 현재 pose world translation을 그대로 유지한다.
- 루트 이동은 기존 apply_root_translation 계약을 그대로 유지한다.
- 후속 wrapper는 위 수식·적용 순서와 frozen V3 보호 조건을 우회하면 안 된다.

---

## 4. 동결 범위

### 4.1 지금 동결하는 것

다음은 **FROZEN-MATH** 또는 **FROZEN-POLICY**다.

1. 순차 최소회전 누적 공식
2. 최소회전의 0.5° identity와 175° 수치 퇴화 의미
3. 임의 축·bend-plane normal·absolute source roll을 사용하지 않는 원칙
4. hips 회전은 정확한 legacy C 유지
5. 루트 translation 계약 유지
6. 부모→자식 적용 순서와 translation 보존
7. foot incremental rotation이 120°를 넘으면 foot solve를 적용하지 않는 frozen V3 동작
8. 위 foot 보호 시 terminal-follow를 유지하는 동작
9. 후속 wrapper가 7~8번 보호 조건을 우회하지 않는 불변식
10. 파일명·프로파일별 예외 금지

spine·neck·head·shoulder는 frozen V3 baseline의 legacy 경로를 유지한다. 별도 후보와 육안
gate 없이 이 범위를 넓히지 않는다.

### 4.2 아직 동결하지 않는 것

다음은 **PROVISIONAL** 또는 **UNRESOLVED**다.

- 120°를 모든 리그에 적용할 보편 제품 임계로 일반화하는 것
- 발목 GREEN/AMBER/RED 경계값
- 골반 α와 발목 μ 후보 간격
- 골반·발목 메시 임계의 최종 수치
- rest compatibility 10° 활성화 경계
- terminal hand가 forearm을 그대로 따르는 정책
- palm roll 복원 정책
- 보편 승격 후보와 운영 반영 여부

### 4.3 손 정책

Phase 12 추가 12건에서는 손이 모두 정상으로 관찰됐지만, 이전 UAL2 Hook에서 손목 roll 실패가
확인됐다. UAL2 BVH에는 유의미한 hand 회전 채널이 존재하는 사례도 있다.

따라서 손은 다음처럼 취급한다.

- 체인 수학을 폐기할 근거로 사용하지 않는다.
- 전역 hand 보정을 추가하지 않는다.
- terminal hand 정책은 **UNRESOLVED**로 남긴다.
- 후속 손 전용 후보는 source hand motion과 finger rest geometry를 함께 사용해 forearm 축
  주위 palm roll만 조건부로 복원해야 한다.

---

## 5. 골반 안전 적용 정책 — REJECTED 실험 기록

### 5.1 문제의 정확한 위치

여기서 “골반 보정”은 hips를 회전시키는 것이 아니다. hips는 계속 legacy C다.

골반 구멍·찢김은 Hips와 UpLeg에 함께 웨이트된 사타구니 전이부가 서로 다른 회전을 받으면서
발생하는 LBS 변형이다. 따라서 조절 대상은 각 다리의 시작점인 upleg.L/R의 V3 적용량이다.

### 5.2 각도는 진단 신호일 뿐이다

단일 swing cutoff는 품질 판정에 사용할 수 없다.

- g1-move1: root 보정 각도가 약 65–67°인데 육안 실패
- g1-move7: 한쪽이 약 125°인데 육안 정상
- κ 또한 g1-move1과 g1-move7을 단독으로 구분하지 못함

각도 등급이나 BVH/프로파일 이름으로 selector를 분기하지 않는다. 모든 active leg는 같은 실제
메시 후보 평가를 통과한다. swing·twist·requested angle은 원인 분석을 위해서만 기록하며 결과를
직접 자르지 않는다.

### 5.3 후보 생성

각 다리에 대해 C와 full V3를 먼저 계산한다. full V3의 사타구니 메시가 안전하면 아무 제한 없이
그대로 사용한다.

후보 격자 분할 수는 외부 정책 파일에서 읽는다. 현재 QA 정책은 다음 α 후보를 만든다.

~~~text
α ∈ {1.00, 0.75, 0.50, 0.25, 0.00}
exact fallback C = 별도 후보
~~~

UpperLeg 시작 누적 회전을 SO(3) 최단 경로로 보간한다.

~~~text
Q0_C   = R_C,upleg   · R_target_rest,upleg⁻¹
Q0_V3  = R_V3,upleg  · R_target_rest,upleg⁻¹
Q0(α) = Q0_C · Exp(α · Log(Q0_C⁻¹ · Q0_V3))
~~~

즉 C와 V3의 world rotation 자체를 섞는 것이 아니라, 같은 target rest frame에 작용하는 두
누적 수송 회전을 비교·보간한다.

그 seed에서 shin과 foot 방향은 §3의 순차 최소회전 수식으로 다시 계산한다.

α=0에서 자식을 재계산한 결과는 전 본 exact C와 같다고 보장되지 않으므로, exact C는 반드시
별도 후보로 유지한다.

### 5.4 양쪽 다리 공동 평가

사타구니 중앙에는 좌우 다리와 Hips의 공유 웨이트가 존재한다. 따라서 좌우 α를 독립적으로
선택한 뒤 합치는 방식은 금지한다.

- αL × αR 조합을 최종 메시에서 함께 평가한다.
- source 구조가 좌우 대칭이면 동일 α를 우선하거나 강제한다.
- 중앙 공유 패치에서 접힘·교차가 발생하면 양쪽 다리를 exact C로 복귀한다.
- 독립 선택을 허용할 때도 최종 판정은 결합된 pelvis patch에서 수행한다.

좌우 대칭 판정의 5° 같은 세부 임계는 **PROVISIONAL**이다.

### 5.5 메시 패치

대상은 Hips와 해당 UpLeg 웨이트가 모두 존재하는 전이부다.

초기 선택 규칙:

~~~text
weight(Hips)  >= 0.05
weight(UpLeg) >= 0.05
transition score >= 0.08
위 정점을 포함하는 face + 한 겹 인접 face
~~~

측정 항목:

- 새 non-adjacent self-intersection
- 심한 fold와 normal flip
- 최소 stretch singular value
- rest 대비 triangle area ratio
- edge-length relative deformation p95
- condition number κ p95 — 진단값이며 단독 gate 금지

### 5.6 외부 메시 정책

사례를 보고 `40°`, `0.011`, `0.65` 같은 복합 임계를 코드에 넣는 방식은 폐기했다. selector는
다음 QA 정책 파일을 읽고 경로·SHA-256·전체 payload를 report에 남긴다.

~~~text
out/rest-safety-qa/variants/CHAIN_TRANSPORT_V3_SAFETY_V1/
  converter/mesh_safety_policy.json
~~~

현재 골반 정책은 사타구니 전이 패치의 p05 단면 반경이 target rest의 **2/3 이상** 남아야
한다는 하나의 해석 가능한 재료·해상도 계약이다. 이 값은 BVH명·source profile·각도와 무관하다.

~~~text
radius_p05(candidate) / radius_p05(target_rest) >= 2 / 3
~~~

다음 값은 후보 선택의 숨은 임계가 아니라 독립 관측치로 모두 기록한다.

- log principal strain 기반 membrane energy
- rest 대비 adjacent-normal 변화 기반 bending energy
- 단면 반경·면적 log energy
- non-adjacent triangle piercing 쌍
- fold, area, edge, condition number 분포

자기교차 수는 극단 포즈의 정상적인 신체 접촉과 g1 실패를 단독으로 분리하지 못했으므로 현재
hard gate로 사용하지 않는다.

### 5.7 선택과 폴백

1. 외부 메시 정책 통과 후보만 남긴다.
2. `αL + αR`가 가장 큰 후보를 선택한다.
3. 적용량이 같은 좌우 조합끼리만 pose RMSE로 tie-break한다.
4. 한쪽 전이부만 실패하면 해당 다리 exact C를 허용한다.
5. 중앙 공유 패치 또는 양측 상호작용 실패면 양쪽 다리 exact C로 복귀한다.
6. 혼합 웨이트가 실제로 없으면 이 LBS 전이 위험은 없으므로 full V3를 허용한다.
7. 웨이트는 있으나 유효 면이 부족하거나 측정이 불가능하면
   pelvis_safety_unmeasurable을 기록하고 exact C로 복귀한다.

유효 에너지·단면을 계산할 수 없으면 측정 불가로 처리하며 조용히 통과시키지 않는다.

---

## 6. 발목 안전 적용 정책 — REJECTED 실험 기록

### 6.1 요청 각도 관측

V3가 solved shin에서 source foot 방향까지 요구하는 최소회전 각도를 θ라 한다.

현재 관찰:

| 사례 | θ | 육안 결과 |
|---|---:|---|
| UAL2 Slide L | 109.9° | 명백한 과굴곡 |
| g1 Move1 L | 96.4° | 경미한 꺾임 |
| UAL2 SwordHeavy R | 95.1° | 경고 수준 |

θ는 리포트에 남기지만 90°·105° 같은 사례 기반 routing cutoff는 사용하지 않는다. 수치 퇴화는
frozen V3의 최소회전 계약대로 처리하고, 그 밖의 적용량은 실제 발목 메시 후보가 결정한다.

### 6.2 같은 축에서 각도만 제한

full V3가 만든 foot 최소회전 Hfoot의 축은 유지한다. 새로운 twist나 roll 기준을 만들지 않는다.

~~~text
μ ∈ {1.000, 0.875, 0.750, 0.625, 0.500, 0.375, 0.250, 0.125, 0.000}
Hfoot(μ) = Rotation(axis(Hfoot), μ · θ)
~~~

- μ=1은 정확한 full V3다.
- toe는 선택된 foot의 최종 Q를 따른다.
- 모든 후보는 같은 순차 최소회전 체인에서 나온다.
- 임의의 Euler clamp나 축별 제한을 사용하지 않는다.

### 6.3 발목 메시 패치와 단면

대상은 Leg와 Foot 웨이트가 함께 존재하는 발목 전이부와 한 겹 인접 면이다.

측정 항목:

- 새 self-intersection, fold, normal flip
- edge deformation p95
- 20% 이상 변형된 면의 비율
- min-stretch p05
- condition number κ p95
- 관절 띠를 shin/foot 축 직교 평면에 투영한 단면 covariance 고유값
- 최소 반지름, 단면적·둘레 proxy

마지막 단면 지표는 방향이 맞더라도 발목이 종잇장처럼 얇아지는 실패를 잡기 위해 필요하다.

### 6.4 외부 발목 정책

현재 QA 정책은 Leg/Foot 전이 패치의 p05 최소 주신장이 **1/3 이상** 남아야 한다고 요구한다.

~~~text
principal_stretch_min_p05 >= 1 / 3
~~~

이 조건은 UAL2 Slide L의 full V3 `0.3284`를 거부하고 μ=0.875의 `0.3864`를 선택했다.
CMU·Rokoko·g1 normal·UAL2 holdout은 이름 예외 없이 full V3를 유지했다. edge·condition·단면·
self-intersection·surface energy는 계속 리포트하지만 단독 cutoff로 사용하지 않는다.

### 6.5 선택과 폴백

1. 정책 파일이 정한 μ 격자를 모두 평가한다.
2. 메시 정책을 통과하는 가장 큰 μ를 선택한다.
3. 어떤 μ도 통과하지 못하면 foot+toe를 exact C로 복귀한다.
4. parent leg나 pelvis patch 자체가 안전하지 않으면 해당 다리 전체를 exact C로 복귀한다.
5. requested/applied/residual angle과 폴백 이유를 반드시 기록한다.

---

## 7. 실행 순서 — REJECTED 실험 기록

골반 적용량은 downstream foot 예측 방향을 바꾼다. 따라서 순서를 뒤집으면 안 된다.

~~~text
1. 입력 검증·mapping·mirror 좌표 확정
2. exact C 전 본 결과 계산
3. full V3 체인 결과 계산
4. hips와 root translation 불변 확인
5. pelvis α 후보를 좌우 공동 평가·선택
6. 선택된 pelvis seed에서 shin/foot 체인 재계산
7. ankle μ 후보 평가·선택
8. toe가 선택된 foot Q를 추종
9. 부모→자식 순서로 타깃 pose 적용
10. export 전·재임포트 후 정책 및 메시 지표 기록
~~~

---

## 8. 리포트 계약 — REJECTED 실험 기록

조용한 제한과 조용한 폴백은 금지한다.

### 8.1 골반

~~~text
mesh_safety.policy_file / policy_sha256 / policy_payload
mesh_safety.pelvis.selected_alpha {L,R}
mesh_safety.pelvis.full_v3_policy_pass
mesh_safety.pelvis.full_v3_policy_reasons
mesh_safety.pelvis.selection
mesh_safety.pelvis.candidates[]
  - alpha / policy_pass / policy_reasons
  - pose_rmse / mesh_energy / metrics
hips_legacy_exact
root_translation_unchanged
~~~

### 8.2 발목

~~~text
mesh_safety.ankle.{L,R}
  - requested_deg / selected_mu / applied_deg / residual_direction_deg
  - status / selection / legacy_metric
  - candidates[].mu / policy_pass / policy_reasons / metric
~~~

### 8.3 사건 목록

~~~text
safety_limited_bones
safety_fallback_bones
numerical_degenerate_bones
safety_unmeasurable_bones
~~~

numerical_degenerate_bones에 설계상 제외나 메시 안전 제한을 섞지 않는다.

---

## 9. 검증과 임계 동결 — REJECTED 실험 기록

### 9.1 비활성 경로 동치성

안전 wrapper가 활성화되지 않은 경우 결과는 frozen V3와 의미상 동일해야 한다.

~~~text
rotation <= 0.001°
head/tail/vertex <= 1e-6 × rest scale
solver policy와 report의 선택 경로 일치
Mixamo compatible 경로 exact 유지
~~~

### 9.2 Discovery

이미 본 Phase 11의 5건과 Phase 12의 12건, 총 D17은 discovery로 영구 분리한다.

비교 variant:

~~~text
B   = frozen V3
OFF = wrapper 강제 비활성 — B와 exact 동치여야 함
P   = pelvis selector만
A   = ankle selector만
PA  = 둘 다
~~~

정책이 활성화된 사례는 report의 `policy_pass`만으로 승격하지 않는다. 알려진 실패 부위에서 정책이
실제로 제한을 발동하고 독립 surface 지표가 제한 방향과 일치해야 하며, 원본 FBX 육안 A/B가
필수다. 현재 자동 결과는 다음과 같다.

- g1-move1: full pelvis는 radius-retention 정책 실패, `{L:1.0,R:0.5}` 선택,
  hip.R radius ratio `0.5321 → 0.7615`
- UAL2 Slide L: full ankle min-stretch `0.3284` 정책 실패, `μ=0.875` 선택,
  min-stretch `0.3284 → 0.3864`, condition `2.7748 → 2.5034`
- D9 정상군 중 나머지는 full V3 유지
- 별도 H8 holdout은 8/8 full V3 유지

### 9.3 Negative self-test

최소한 다음 실패를 독립 검증기가 잡아야 한다.

- 본 head가 같아도 foot roll을 +10° 주입
- 발목 단면을 30% 축소
- 사타구니 fold/self-intersection 주입
- head RMSE에 숨는 pelvis 구조 프레임 +10° 오류
- 좌우 swap, mirror, zero vector, 176° 수치 퇴화
- topology 또는 weight 변조
- selector가 잘못된 후보를 골랐을 때 독립 audit 실패

### 9.4 사용자 육안 gate

렌더만으로 승인하지 않는다. 원본 FBX를 같은 DCC에서 frozen V3와 정책 후보로 A/B 비교한다.

필수 discovery 육안군에는 최소 다음을 포함한다.

~~~text
Mixamo control
CMU
Rokoko
UAL2 main
MakeHuman
UAL2 Slide
UAL2 SwordHeavy
g1 Move1
~~~

골반 구멍·fold, 발목 과굴곡·단면 붕괴, 손발 axial flip, 팔꿈치 pinching 중 하나라도 새로
보이면 실패다. Mixamo control은 눈에 띄는 변화가 없어야 한다.

### 9.5 Blind holdout

후보 구조와 모든 임계를 고정한 뒤 미사용 H32를 한 번만 연다. holdout을 본 뒤 후보·임계·
selector를 바꾸면 기존 holdout은 소진된 것으로 처리한다.

운영 승격은 다음을 모두 통과한 뒤 별도 승인으로 결정한다.

- 회귀 28/28
- 수학·mirror·negative self-test
- D17 discovery 정량 gate
- 원본 FBX discovery 육안 gate
- H32 blind 정량 gate
- blind 원본 FBX 육안 표본
- production 파일 최종 해시와 rollback 검증

---

## 10. 근거와 현재 판단

자동 근거:

- Phase 11: 실물 5건, synthetic mirror/non-mirror, 독립 transport 검증, 회귀 28/28
- Phase 12: 추가 12건 artifact 생성 및 독립 transport 검증
- Phase 12 최대 본 방향 오차: 0.0321°
- topology/weights integrity 통과
- Safety V1 discovery 9/9 변환·재임포트 surface 검증 통과
- Safety V1 holdout 8/8에서 제한 없이 full V3 유지
- g1-move1: pelvis `α={L:1.0,R:0.5}` 자동 선택
- UAL2 Slide: ankle `μL=0.875` 자동 선택
- 임시 물리 복제본 회귀 28/28
- `CHAIN_SAFETY_FORCE_OFF=1`과 frozen V3 pre-export 수치 최대 차이 0
- Safety V1은 CMU에서 frozen V3의 120° foot 보호 조건을 우회해 `141.715°`를 전량 적용
- g1 Move1 RMSE `0.182824 → 0.185781`, UAL2 Slide `0.558751 → 0.559014`로 악화

사용자 육안 근거 — frozen V3:

- 다수 사례에서 발 방향 반전이 제거돼 중심 수학의 효과 확인
- g1-move1: 골반 뒤틀림과 발목 경미 실패
- UAL2 Slide: 발목 과굴곡
- UAL2 SwordHeavy: 발목 경미 경고
- 추가 12건의 손은 정상
- 별도 UAL2 Hook 사례에서 손목 roll 실패

사용자 육안 근거 — Safety V1:

- g1 Move1 골반은 실제로 개선되지 않음
- UAL2 Slide 발목 과굴곡이 유의미하게 개선되지 않음
- CMU 발 방향이 다시 회귀함
- 따라서 대리 지표 개선과 무관하게 원본 FBX gate 실패

따라서 현재 판정은 다음과 같다.

~~~text
V3 중심 수학                  : 유지 / FROZEN-MATH
V3 QA 기준선 해시             : 유지 / FROZEN-BASELINE
골반·발목 Safety V1 selector   : 폐기 / REJECTED
외부 메시 정책 파일           : 폐기 / 승격 금지
terminal hand 정책            : 미동결 / UNRESOLVED
production 승격               : 없음
~~~

---

## 11. 구현 금지선

- 수학 코어를 Euler 축 clamp로 바꾸지 않는다.
- target rest bend-plane normal을 다시 도입하지 않는다.
- source absolute rest roll을 타깃에 직접 이식하지 않는다.
- UAL2, g1, Mixamo 이름을 selector 분기로 사용하지 않는다.
- 안전 제한을 통과시키려고 임계를 결과 확인 후 완화하지 않는다.
- κ, 방향 오차, head RMSE 하나만으로 메시 품질을 판정하지 않는다.
- 골반 문제를 해결하기 위해 hips를 V3로 회전시키지 않는다.
- 발목 문제를 숨기기 위해 toe나 foot을 조용히 제외하지 않는다.
- 손 한 건 때문에 모든 hand에 전역 roll 보정을 적용하지 않는다.
- blind holdout을 후보 탐색에 사용하지 않는다.
- 별도 승인 전에 production 파일을 수정하거나 승격하지 않는다.

---

## 12. 근거 파일

이 문서가 요약한 기준선과 실측의 직접 근거:

~~~text
<RECOVERY_QA_ROOT>/
  manifests/phase-11-candidate-design.json
  manifests/phase-12-extra-tests.json
  manifests/state.json
  reports/phase-11-report.md
  reports/phase-11-user-original-fbx-gate.md
  reports/phase-12-report.md
  metrics/phase-11-summary.json
  variants/CHAIN_TRANSPORT_V3/converter/retarget.py
~~~

상위 복구·검증 절차:

~~~text
docs/REST_QUALITY_BUG_FIX_PLAN.md
~~~

서로 충돌할 경우 우선순위:

1. 사용자의 최신 명시적 결정
2. 이 문서의 FROZEN-MATH·FROZEN-POLICY
3. Phase 11·12의 실행 manifest와 원시 metric
4. 과거 계획 문서의 아직 실행되지 않은 제안
