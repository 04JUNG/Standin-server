# CHAIN_TRANSPORT_V3_2_1_PALM_ROLL_QA

> 상태: **`mu=0.50` 사용자 승인 / QA 전용 / 운영 미연결**
> 부모: 사용자 승인 `CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY`
> 수정 자유도: `hand.L`, `hand.R`의 palm longitudinal axis 주위 roll만

## 무엇을 구현했나

frozen V3.2의 arm/forearm solve와 hand `terminal_follow`를 기준으로, 손가락 첫 마디의
**rest head/base** 두 개로 source/target palm frame을 만든다. source hand pose delta로
source rest palm normal을 옮기고, frozen V3.2 hand 길이축 주위의 signed roll만 추가한다.

```text
D_s      = R_s_pose @ inverse(R_s_rest)
n_s_pose = D_s @ n_s_rest

R0 = frozen V3.2 hand result
Q0 = R0 @ inverse(R_d_rest)
n0 = Q0 @ n_d_rest
u  = normalize(Q0 @ f_d_rest)

theta = atan2(dot(u, a × b), dot(a, b))
R(mu) = Twist(mu * theta, u) @ R0
```

landmark pair는 양쪽 리그에 같은 semantic role이 있을 때만 다음 순서로 선택한다.

1. `index + pinky`
2. `index + thumb`
3. 없거나 퇴화하면 해당 손만 exact V3.2 fallback

posed finger tail/fingertip, finger local rotation, source absolute rest rotation은 쓰지 않는다.

## 승인된 QA 기본값

`converter/palm_roll_policy.json`의 공통 ladder는 다음과 같다.

```text
mu = 0.00, 0.25, 0.50, 0.75, 1.00
default = 0.50
```

사용자 원본 FBX 정성평가와 독립 BVH 각도 비교를 근거로 QA 후보의 기본값을 `0.50`으로
고정했다. 운영 converter에는 아직 연결하지 않았다. 좌우는 같은 ladder 안에서 독립 선택할
수 있지만 clip명·성별명·BVH family명별 보정은 없다. frozen V3.2 exact 비교·fallback은
명시적으로 `palm_roll_mu=0.0`을 사용한다.

```python
convert(...)  # QA default: 0.50
convert(..., palm_roll_mu=0.0)  # exact frozen V3.2
convert(..., palm_roll_mu={"hand.L": 0.25, "hand.R": 0.50})
```

`|theta| < 0.5°`는 identity, `|theta| >= 175°`는 exact fallback이다. 측정 불가,
non-finite, common landmark pair 부재도 손별 exact fallback이다.

## 구성

```text
converter/retarget.py             roll-only 후보와 report/fallback
converter/convert.py              QA opt-in 인자 전달
converter/palm_roll_policy.json   공통 mu ladder
tools/run_case.py                 단일 actual FBX 생성
tools/test_palm_roll_math.py      순수 수학 control
tools/compare_fbx_artifacts.py    export/reimport 본·정점 비교
tools/measure_palm_mesh_delta.py  실제 weight 기반 wrist/hand ROI 상대 측정
tools/verify_palm_artifact.py     원본 BVH↔export FBX palm roll 독립 비교
PALM_ROLL_QA_REPORT.md            현재 증거와 미완료 gate
```

## 현재 판정

- 순수 수학: 22/22 PASS
- converter 회귀: 28/28 PASS
- `mu=0`: frozen V3.2와 52개 본 행렬·모든 메시 정점 exact
- Mixamo 대조군: `mu=0`과 `mu=1` actual artifact exact
- UAL2: roll 계산과 mirror 경로 정상, hand 계층 밖 본 exact
- UAL2 전체 ladder: 원본 BVH 대비 V3.2 palm roll 오차가 `mu`에 따라 0/25/50/75/100% 감소
- 실제 메시: full `mu=1`은 일부 손목 면적을 과도하게 압축하므로 **승격 불가**

`measure_palm_mesh_delta.py`는 현재 상대 지표만 측정하며 합격 임계를 만들지 않는다. UAL2
원본 FBX 육안 판정으로 `mu=0.50`을 선택했지만, 실제 mesh gate와 61개 wrist-direction
cohort가 끝나기 전에는 운영에 연결하지 않는다.

## QA 산출물

렌더를 만들지 않았고 원본 FBX 후보만 남겼다.

```text
/Users/dowon/dev/Standin-server/out/palm-roll-v321-qa-20260830/
```

세부 결과는 `PALM_ROLL_QA_REPORT.md`를 본다.
