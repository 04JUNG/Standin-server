# Chain Transport V3.2.1 Palm Roll QA

> 상태: **`mu=0.50` 사용자 승인 / QA 후보 / 운영 미연결**
> 작성일: 2026-08-30
> 유일한 부모: 사용자 승인 `CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY`
> 범위: `hand.L`·`hand.R`의 조건부 palm roll만 복원
> 비범위: 손가락 애니메이션, 팔꿈치·전완 방향, 골반·다리·발목, 검색·refine, 운영 converter

## 0. 결정

손가락 뿌리의 rest 구조로 손바닥 평면을 만들고, frozen V3.2가 만든 손 결과에
**손의 길이축 주위 roll만** 조건부로 추가한다.

이 후보는 소스의 absolute rest roll을 타깃에 복사하지 않는다. 손가락 끝의 posed 위치도
사용하지 않는다. 주먹·손가락 굽힘이 손목 roll로 잘못 해석되는 것을 막기 위해, 첫
손가락 뼈의 **rest base geometry**와 source hand pose delta만 사용한다.

손바닥 프레임이 측정 불가능하거나 후보가 실제 메시 안전 조건을 통과하지 못하면 해당
손만 frozen V3.2 `terminal_follow`로 정확히 복구한다. 반대쪽 손과 나머지 본은 영향을
받지 않는다.

## 1. 부모와 격리 계약

### 1.1 frozen parent

승인된 부모는 Git 객체 `f39ca3b`의 다음 디렉터리다.

```text
qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY/
```

핵심 해시는 다음과 같다.

```text
V3.2 retarget.py
692e975d32f41e3406144763c7c0b7dbf0a586ff07732f09c4365e3233b13693

V3.1 ankle_policy.json
79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f
```

작업 시작 시 Git 객체에서 별도 QA 디렉터리로 복원한다.

```bash
git archive f39ca3b \
  qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY \
  | tar -x -C "$QA_STAGING"
```

복원 직후 부모 `SHA256SUMS`를 통과하지 못하면 중단한다.

### 1.2 새 후보 위치

```text
qa/retarget/CHAIN_TRANSPORT_V3_2_1_PALM_ROLL_QA/
```

다음은 수정하지 않는다.

- frozen V3.2 스냅샷
- 운영 `converter/`, `converter_api/`, `api/`, `src/`
- 원본 BVH와 등록 character FBX
- V3.1 발목 정책과 V3.2 골반 경계
- 검색·refine·pose library

## 2. 현재 문제와 관측 근거

V3.2 팔 체인은 upperarm과 forearm 방향을 순차 최소회전으로 맞춘 뒤, hand에는
마지막 forearm 수송을 그대로 적용한다.

```text
R0_hand = Q_forearm @ R_target_hand_rest
mode    = terminal_follow
```

이 방식은 손목 위치와 팔 체인을 안정적으로 보존하지만, source와 target의 손바닥 rest
방향이 다르면 source wrist roll을 잃는다. 결과적으로 손의 길이 방향은 맞지만 손바닥이
뒤집혀 보일 수 있다.

현재 frozen 코드에는 다음 보조 함수가 이미 있으나 실제 solve에서 호출하지 않는다.

- hand descendant에서 index/pinky/thumb landmark 탐색
- 두 landmark ray로 palm frame 구성

이번 후보는 이 아이디어를 사용하되, posed finger tail 대신 finger rest base를 사용하도록
안전 경계를 다시 정의한다.

### 2.1 데이터 가용성

- 운영 BVH 1,248개 중 1,222개가 양손에서 `index+pinky` 또는 `index+thumb` landmark pair를
  구성할 수 있다.
- 나머지 26개는 손별 exact V3.2 fallback 대상이다.
- 남성 `standin-master-v2.fbx`와 여성 `standin-female-v3.3.1.fbx`는 각각 Mixamo finger
  30본을 모두 가진다.
- UAL2 Melee Hook 샘플은 hand와 finger 회전 채널이 존재하지만 값은 약 `1e-6°`로 사실상
  rest다. 채널값이 0이어도 finger base offset이 palm frame을 제공하므로 유효 입력이다.
- 제거 작업 이후의 육안 라벨에는 손목 방향 문제가 61개 library entry, 106개 성별 review
  row로 남아 있다. 강한 실패 14행, 약한 보류 92행이다.

## 3. 불변식

V3.2.1 후보가 지켜야 할 불변식은 다음과 같다.

1. `hips`, spine, neck, head, shoulder, upperarm, forearm, leg, foot, toe 결과는 frozen
   V3.2와 동일하다.
2. hand head 위치와 root translation은 frozen V3.2와 동일하다.
3. 수정 가능한 자유도는 각 hand의 길이축 주위 roll 하나뿐이다.
4. 한 손의 실패가 반대쪽 손이나 팔 체인의 fallback을 유발하지 않는다.
5. 후보 평가는 매번 frozen V3.2 상태에서 시작한다. 회전을 반복 누적하지 않는다.
6. 파일명, clip명, 성별명, BVH family명에 따른 보정량 분기를 만들지 않는다.
7. mirror는 별도 공식이 아니라 반사된 source geometry에서 같은 공식을 실행한다.
8. 측정 불가·비유한 값·퇴화는 성공으로 간주하지 않는다.
9. 후보가 없으면 FBX를 막지 않고 exact V3.2를 반환한다.

## 4. Landmark와 palm frame

### 4.1 landmark 우선순위

각 hand의 첫 phalanx base를 이름과 hierarchy로 찾는다.

```text
1순위: index + pinky
2순위: index + thumb
없음 : exact terminal_follow fallback
```

소스와 타깃은 반드시 같은 semantic role pair를 사용한다. 한쪽은 index+pinky, 다른 쪽은
index+thumb인 상태로 프레임을 비교하지 않는다.

### 4.2 tail과 fingertip을 사용하지 않는 이유

posed tail이나 fingertip은 finger local rotation에 영향을 받는다. 손가락을 구부린 자세에서
이를 사용하면 finger curl을 wrist roll로 오판한다.

따라서 기본 landmark는 첫 phalanx의 **head/base**다. head가 hand origin과 겹쳐 ray 길이가
0이면 다음 순서로 처리한다.

1. 더 바깥쪽의 유효한 finger base가 있는지 탐색
2. source hand pose transform에 rest-local base vector를 적용
3. 그래도 두 개의 독립 ray를 만들 수 없으면 fallback

finger의 posed local rotation은 palm normal 계산에 사용하지 않는다.

### 4.3 palm frame

hand origin을 `o`, 두 landmark를 `p_a`, `p_b`라 한다.

```text
a = normalize(p_a - o)
b = normalize(p_b - o)

n = normalize(a × b)          # palm normal
f = normalize(a + b)          # palm forward / hand longitudinal direction
s = normalize(n × f)

F = [f, s, n]
```

`|a × b|`가 너무 작거나 `F`가 proper rotation을 만들지 못하면 frame은 측정 불가다.
좌우는 semantic role 순서를 고정하고 determinant가 `+1`인지 검사한다.

## 5. Roll-only 수학

### 5.1 source desired palm normal

source rest hand 회전과 pose hand 회전을 각각 `R_s_rest`, `R_s_pose`라 한다.

```text
D_s      = R_s_pose @ inverse(R_s_rest)
n_s_pose = D_s @ n_s_rest
```

mirror에서는 source rest landmark와 hand delta를 X축 반사 프레임으로 옮긴 뒤 같은 계산을
수행한다. source absolute rest 회전을 target에 직접 복사하지 않는다.

### 5.2 frozen V3.2 baseline

frozen V3.2의 hand 결과를 `R0`, target hand rest를 `R_d_rest`라 한다.

```text
Q0 = R0 @ inverse(R_d_rest)

n0 = Q0 @ n_d_rest
u  = normalize(Q0 @ f_d_rest)
```

`u`는 현재 출력 손의 길이축이다. 보정은 이 축을 바꾸지 않는다.

### 5.3 signed roll

두 palm normal을 `u`에 수직인 평면으로 투영한다.

```text
a = normalize(n0       - dot(n0,       u) * u)
b = normalize(n_s_pose - dot(n_s_pose, u) * u)

theta = atan2(dot(u, a × b), dot(a, b))
theta ∈ [-pi, pi]
```

적용량 `mu`에 대한 후보는 다음과 같다.

```text
R(mu) = Twist(mu * theta, u) @ R0
```

`mu=0`은 exact V3.2다. 후보 ladder는 QA 정책 파일에 두며 모든 source와 character에 동일하게
적용한다. 특정 clip을 통과시키기 위한 source별 `mu`는 금지한다.

초기 QA 후보군:

```text
mu ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

이는 제품 임계로 동결된 값이 아니다. 실제 cohort 결과와 반복 오차를 측정한 뒤 유지·축소한다.

## 6. 퇴화와 exact fallback

다음 중 하나라도 참이면 해당 hand는 `mu=0`, 즉 frozen V3.2를 정확히 반환한다.

- source 또는 target에서 같은 landmark role pair를 만들 수 없음
- landmark ray 길이가 수치 epsilon 이하
- 두 ray가 거의 일직선
- palm frame determinant가 proper rotation이 아님
- normal의 roll-plane projection이 수치 epsilon 이하
- `theta` 또는 중간 행렬이 non-finite
- mirror 후 frame handedness 불일치
- requested roll이 near-180°라 연속적인 방향을 안정적으로 선택할 수 없음
- 후보의 hand/wrist mesh gate가 측정 불가 또는 실패
- export/reimport 뒤 적용량·행렬·메시 판정이 재현되지 않음

`|theta| < 0.5°`는 identity로 처리한다. near-180°의 구체 경계는 기존 수치 퇴화 의미와
맞춰 QA에서 우선 `175°`로 시작하되, 제품 임계로 승격하기 전 실제 분포를 기록한다.

fallback은 경고 없이 숨기지 않는다. report에 hand, 원인, landmark coverage를 남긴다.

## 7. 후보 선택과 메시 안전

### 7.1 선택 목적

안전 후보 중 source palm normal과의 roll 오차가 가장 작은 후보를 선택한다.

```text
E_palm(mu) = angle_about_axis(n_candidate(mu), n_s_pose, u)
```

동점이면 작은 `mu`를 선택한다. palm 오차를 줄였다는 이유만으로 메시 gate를 우회하지 않는다.

### 7.2 구조 gate

- hand 외 canonical bone matrix: frozen V3.2와 수치 계약 내 동일
- hand head 위치: 동일
- hand longitudinal direction: 동일
- forearm→hand parent relation에서 roll 외 swing 증가 없음
- 좌우·mirror frame determinant: `+1`
- pose RMSE: frozen V3.2 비회귀

### 7.3 실제 메시 gate

등록 character의 실제 evaluated mesh에서 hand/wrist ROI를 평가한다. ROI는 이름이 아니라
`forearm`, `hand`, finger weight의 실제 영향 정점으로 만든다.

- 새 non-adjacent self-intersection 없음
- wrist/hand 최소 triangle area와 최소 edge ratio 악화 없음
- edge strain p99 악화 없음
- 연결 압축·접힘 component 증가 없음
- 반대쪽 손과 손목 밖 정점 이동 없음
- bake 전과 export/reimport 뒤 같은 판정

절대 임계가 아직 없으면 후보 간 상대 비악화만 기록하고 운영 승격하지 않는다. 수치상 개선과
실제 형상이 다를 수 있으므로 최종 승격에는 실제 FBX 육안 gate가 필수다.

## 8. Report 계약

QA report에 최소한 다음 필드를 추가한다.

```text
palm_roll_requested_deg   {hand.L: deg, hand.R: deg}
palm_roll_applied_deg     {hand.L: deg, hand.R: deg}
palm_roll_mu              {hand.L: float, hand.R: float}
palm_landmark_roles       {hand.L: [role, role], hand.R: [...]}
palm_plane_sin            {hand.L: float, hand.R: float}
palm_frame_status         {hand.L: str, hand.R: str}
palm_roll_mode            {hand.L: corrected|identity|fallback, ...}
palm_fallback_reason      {hand.L: str|null, hand.R: str|null}
palm_roll_bones           [canonical]
palm_roll_fallback_bones  [canonical]
```

선택 전 모든 `mu` 후보의 palm 오차와 메시 gate 결과도 QA 상세 report에 남긴다.

## 9. 검증 계획

### 9.1 순수 수학 control

- palm roll `0°, ±30°, ±90°, ±179°`
- source/target rest palm 방향이 다른 경우
- 좌우 semantic role 순서
- mirror conjugation
- finger curl이 있어도 rest-base frame이 불변인 경우
- landmark 누락, zero ray, collinear ray, projection zero
- 한 손 퇴화 시 반대쪽 손은 정상 계산
- `mu=0`이 frozen V3.2와 행렬 단위로 동일

### 9.2 converter 회귀

```bash
"$BLENDER_BIN" --background --python tests/make_fixtures.py
"$BLENDER_BIN" --background --python tests/test_convert.py
```

기존 28/28을 유지한다. rigged_rest, rigged_anim, static_mesh, mirror를 모두 포함한다.

### 9.3 실제 코호트

1. **양성 시험대**: `UAL2__Melee_Hook_Rec__p01_f0002`와 mirror
2. **대조군**: Mixamo 계열의 육안 정상 손목
3. **라벨 코호트**: 현재 library에 남은 `wrist_direction` 61개 entry
4. **fallback 코호트**: palm pair를 만들 수 없는 26개 source
5. **캐릭터**: 승인 남성·여성 target FBX 각각

라벨 원본:

```text
data/library-review-2026-08-29-reason-removals/
  source/v33-reason-review-response.json
```

### 9.4 정량 지표

- source↔output palm roll error
- hand longitudinal direction error
- hand head/tail endpoint error
- forearm·elbow·나머지 canonical matrix delta
- pose fidelity RMSE
- hand/wrist ROI surface 지표
- original↔mirror 적용량 대칭성
- report↔artifact 차이

### 9.5 실제 FBX 육안 gate

숫자만으로 승격하지 않는다. 같은 카메라와 같은 pose에서 V3.2/V3.2.1 actual FBX를 비교한다.

- 손바닥이 source 의도와 같은 방향인가
- 손등/손바닥이 뒤집히지 않았는가
- 손목이 과하게 꺾이거나 가늘어지지 않았는가
- 팔꿈치와 전완 실루엣이 V3.2와 동일한가
- 손가락이 뒤틀리거나 벌어지지 않았는가
- Mixamo 정상군이 사실상 동일한가
- original/mirror가 좌우 대칭인가

강한 실패 14행은 전부 확인한다. 약한 보류는 original/mirror·source family를 층화해 확인하고,
승격 전에는 전체 61개 entry를 일괄 front/back/hand-closeup viewer로 검토한다.

## 10. 단계별 실행 순서

1. Git 객체 `f39ca3b`에서 frozen V3.2를 QA staging에 복원
2. 부모 `SHA256SUMS` 검증
3. helper만 추가하고 `mu=0` exact 동치성 검증
4. rest-base landmark extractor와 palm frame unit control
5. roll-only candidate 생성과 hand별 fallback
6. UAL2 Hook + Mixamo control + mirror 소규모 실행
7. 남녀 actual FBX hand closeup 육안 확인
8. 28/28 converter 회귀
9. fallback 26개와 wrist-direction 61개 전수 실행
10. export/reimport verifier 및 반복 결정론 검사
11. 실제 FBX viewer로 사용자 판정
12. 승인 전까지 운영 미연결 유지

각 단계는 산출물·입력 해시·Blender 버전·실제 import 경로를 manifest에 기록하고 다음 단계 전에
멈춘다.

## 11. 승격 조건

다음을 모두 만족해야 QA 후보를 동결할 수 있다.

- 부모 V3.2 해시 일치
- 28/28 회귀 통과
- non-hand 결과 비회귀
- `mu=0` exact fallback 증명
- 손별 퇴화 fallback과 mirror 통과
- UAL2 Hook 실제 FBX 손목 방향 개선
- Mixamo 대조군 육안 비회귀
- wrist-direction 강한 실패 전수에 새 악화 0건
- 61개 전체 actual FBX 사용자 육안 gate 완료
- 남녀 target 모두 wrist/hand mesh hard failure 0건
- export/reimport 뒤 같은 판정

한 항목이라도 실패하면 운영 승격하지 않는다. 실패 후보와 산출물만 폐기하고 frozen V3.2로
돌아간다.

## 12. 하지 말 것

- source hand absolute rest rotation을 target에 직접 복사
- posed fingertip/tail로 palm roll 계산
- full palm frame을 hand에 절대 전송
- forearm·elbow를 손목 보정 때문에 다시 solve
- clip명·성별명·BVH family별 각도 하드코딩
- 한 손 퇴화 때문에 양팔 전체 fallback
- 손가락 애니메이션까지 해결했다고 주장
- 수치 palm error만으로 실제 FBX 육안 gate 생략
- frozen V3.2 또는 운영 converter 직접 수정
- 원본 BVH 수정

## 13. 완료 정의

V3.2.1 Palm Roll QA의 완료는 코드 작성이 아니라 다음 증거 묶음이다.

- frozen parent 해시와 후보 SHA256SUMS
- 수학 control 결과
- 28/28 회귀 로그
- UAL2/Mixamo/mirror paired report
- palm pair 불가 26개 exact fallback 증명
- wrist-direction 61개 남녀 전수 결과
- report/artifact 독립 검증
- 실제 FBX 비교 viewer와 사용자 판정 JSON
- 운영 미연결 확인

이 증거가 없으면 상태는 계속 `QA 후보 / 운영 승격 금지`다.

## 14. 2026-08-30 구현 상태

격리 후보는 다음 위치에 구현했다.

```text
qa/retarget/CHAIN_TRANSPORT_V3_2_1_PALM_ROLL_QA/
```

현재 확보된 증거:

- 순수 수학 control 22/22
- converter 회귀 28/28
- `mu=0` frozen V3.2 재임포트 본 52개와 메시 정점 exact
- Mixamo 대조군 `mu=0`/`mu=1` exact
- UAL2 남녀·mirror roll 계산 및 hand 계층 밖 비회귀
- 실제 weight 기반 wrist/hand ROI ladder 측정

UAL2 full `mu=1`은 일부 손목 triangle area를 크게 압축했다. 사용자는 남녀 원본 FBX의
정성평가에서 `mu=0.50`을 승인했고, 독립 export/reimport 측정에서 원본 BVH palm roll 평균
일치율 73.239232%, V3.2 오차 감소 50.000125%를 확인했다. 따라서 격리 QA 후보 기본값은
`mu=0.50`으로 고정한다. 명시적 `mu=0`과 모든 퇴화 fallback은 frozen V3.2 exact를 유지한다.
61개 entry 전수와 제품 mesh gate가 끝날 때까지 운영 converter에는 연결하지 않는다. 세부
수치는 후보 폴더의 `PALM_ROLL_QA_REPORT.md`를 단일 구현 보고서로 사용한다.
