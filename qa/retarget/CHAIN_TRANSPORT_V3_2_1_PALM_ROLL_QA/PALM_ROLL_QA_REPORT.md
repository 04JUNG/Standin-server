# V3.2.1 Palm Roll QA — 구현 결과

## 판정

**사용자 정성평가와 독립 각도 검증을 근거로 QA 기본값을 `mu=0.50`으로 승인했다. 운영
converter에는 아직 연결하지 않는다.**

명시적 `mu=0`은 frozen V3.2를 exact 보존하며, 측정 불가·퇴화 때의 hand별 fallback도
동일하다. UAL2에서 full roll은 실제 손목 메시를 과도하게 압축했지만 `mu=0.50`은 사용자의
원본 FBX 육안 gate를 통과했고 V3.2 palm roll 오차를 50.000125% 줄였다. 따라서 격리 QA
후보의 no-argument 기본값만 `0.50`으로 고정한다. 전수 cohort와 제품 mesh gate는 미완료다.

## 부모·격리

- Git object: `f39ca3b`
- parent retarget SHA-256: `692e975d32f41e3406144763c7c0b7dbf0a586ff07732f09c4365e3233b13693`
- ankle policy SHA-256: `79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f`
- 운영 `converter/`, `api/`, `src/`: 수정하지 않음
- 원본 BVH/FBX: 수정하지 않음
- 렌더: 생성하지 않음

## 구현 경계

- rest first-phalanx base만 사용
- source/target 공통 semantic pair만 사용
- source hand pose delta로 source palm normal 계산
- V3.2 hand longitudinal axis 주위 roll 하나만 변경
- 좌우 hand별 독립 `mu`와 독립 fallback
- scalar 또는 `{hand.L, hand.R}` QA 인자 지원
- common ladder 외 값 거부
- report 필드와 모든 ladder palm residual 기록
- actual weight 기반 wrist/hand ROI 측정 도구 추가

actual mesh 도구는 아직 상대 지표만 기록한다. 임계가 동결되지 않아 `PASS`를 만들지 않고
`MEASURED_NOT_GATED`를 반환한다.

## 자동 검증

| 항목 | 결과 |
|---|---:|
| Blender | 5.2.0 LTS |
| 순수 수학 control | 22/22 PASS |
| converter 회귀 | 28/28 PASS |
| `mu=0` parent report 핵심값 | exact |
| `mu=0` export/reimport bone matrix | 52/52 exact, max 0 |
| `mu=0` baked mesh vertices | max distance 0 |
| no-argument QA default vs 명시적 `mu=0.50` | 52/52 bone exact, 모든 mesh vertex exact |
| Mixamo `mu=0` vs `mu=1` | bone/mesh exact |
| UAL2 `mu=1` 비-hand 본 | exact |
| UAL2 ROI 밖 정점 이동 | 0 |
| UAL2 새 ROI 내부 비인접 self-intersection | 0 |

FBX binary hash는 exporter metadata 때문에 달랐지만, 독립 재임포트 비교에서 본 행렬과 메시
정점은 exact였다.

## UAL2 Hook roll

### 남성 target

| hand | requested | mirror 반대손 requested |
|---|---:|---:|
| L | +119.957° | mirror R −118.687° |
| R | −74.076° | mirror L +75.365° |

### 여성 target

| hand | requested | mirror 반대손 requested |
|---|---:|---:|
| L | +120.587° | mirror R −116.975° |
| R | −70.736° | mirror L +74.174° |

target 좌우 rest/weight의 작은 비대칭 때문에 magnitude가 완전 동일하지는 않지만, mirror에서
손이 교환되고 부호가 반전되는 패턴은 유지됐다.

## 실제 메시 ladder

아래 값은 frozen V3.2 대비 actual baked FBX의 weight-derived forearm/hand/finger ROI다.
`p99`는 `|log(edge length ratio)|`, `min area`는 후보/기준 최소 triangle area ratio다.

### 남성

| mu | applied L/R | p99 L/R | min area L/R |
|---:|---:|---:|---:|
| 0.25 | +29.99° / −18.52° | 0.157 / 0.085 | 0.627 / 0.384 |
| 0.50 | +59.98° / −37.04° | 0.290 / 0.169 | 0.309 / 0.354 |
| 0.75 | +89.97° / −55.56° | 0.404 / 0.257 | 0.270 / 0.428 |
| 1.00 | +119.96° / −74.08° | 0.502 / 0.335 | **0.026** / 0.334 |

### 여성

| mu | applied L/R | p99 L/R | min area L/R |
|---:|---:|---:|---:|
| 0.25 | +30.15° / −17.68° | 0.123 / 0.064 | 0.679 / 0.790 |
| 0.50 | +60.29° / −35.37° | 0.244 / 0.129 | 0.332 / 0.584 |
| 0.75 | +90.44° / −53.05° | 0.345 / 0.193 | 0.261 / 0.428 |
| 1.00 | +120.59° / −70.74° | 0.430 / 0.259 | **0.131** / 0.314 |

이 결과는 full correction을 안전하다고 볼 수 없음을 보여준다. `mu=0.25`는 훨씬 완만하지만,
실제 손바닥 방향이 충분히 개선되는지 사용자가 원본 FBX에서 판단하기 전에는 선택하지 않는다.

## 전체 `mu` ladder 원본 BVH 손목 각도 일치율

`verify_palm_artifact.py`가 production retarget 수학을 import하지 않고 원본 BVH와 export/reimport
FBX를 각각 다시 읽어 독립 계산했다. source/target 양쪽에서 `index + pinky` 첫 마디 base로
손바닥 normal을 만들고, 실제 artifact의 손 길이축 주위 잔여 roll을 비교했다.

두 퍼센트는 의미가 다르므로 함께 기록한다.

```text
source match = 100 * (1 - |source-output roll error| / 180°)
V3.2 error reduction = 100 * (V3.2 error - candidate error) / V3.2 error
```

남성·여성 target의 좌우 손 네 개를 합산한 결과다. 메시 열은 같은 네 actual FBX에서 가장
나쁜 손의 값을 골랐다. `p99 edge strain`은 작을수록, `min area ratio`는 1에 가까울수록 좋다.

| `mu` | 원본 BVH 평균 일치율 | V3.2 오차 감소 | 평균 잔여 오차 | 최악 p99 edge strain | 최악 min area ratio |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 46.478331% | 0.000000% | 96.339004° | 0.000 | 1.000 |
| 0.25 | 59.858761% | 25.000023% | 72.254231° | 0.157 | 0.384 |
| **0.50** | **73.239232%** | **50.000125%** | **48.169382°** | **0.290** | **0.309** |
| 0.75 | 86.619597% | 75.000026% | 24.084726° | 0.404 | 0.261 |
| 1.00 | 99.999960% | 99.999925% | 0.000072° | 0.502 | **0.026** |

각도 일치율은 `mu`에 따라 거의 선형으로 좋아지지만 메시 변형은 선형 안전성을 보장하지
않는다. 특히 `mu=1`은 각도는 사실상 완전 일치하지만 최소 면적비가 0.026까지 무너진다.
사용자가 정성적으로 고른 `mu=0.50`은 이 UAL2 표본에서 각도 오차를 절반 줄이는 대신,
full correction보다 메시 붕괴를 크게 줄인 중간점이다.

### `mu=0.50` 손별 상세

| target | hand | V3.2 오차 | `mu=0.50` 오차 | source match | V3.2 오차 감소 |
|---|---|---:|---:|---:|---:|
| 남성 | L | 119.956830° | 59.978329° | 66.678706% | 50.000072% |
| 남성 | R | 74.076103° | 37.037952° | 79.423360% | 50.000134% |
| 여성 | L | 120.586931° | 60.293154° | 66.503803% | 50.000258% |
| 여성 | R | 70.736152° | 35.368092° | 80.351060% | 49.999978% |

- 남성 양손 평균 source match: **73.051033%**
- 여성 양손 평균 source match: **73.427432%**
- 남녀 네 손 평균 source match: **73.239232%**
- 남녀 네 손 가중 V3.2 오차 감소: **50.000125%**
- 평균 절대 오차: V3.2 **96.339004°** → `mu=0.50` **48.169382°**

측정기 sanity control로 같은 artifact에 `mu=1`을 넣었을 때 네 손 평균 source match는
**99.999960%**였고, 손별 잔여 오차는 최대 **0.000195°**였다. 따라서 `mu=0.50`의 정확히
절반 감소는 report 공식의 자기검증이 아니라 export된 FBX에서 재현된 값이다.

`source match` 73.24%는 180°를 최악으로 둔 절대 방향 유사도다. 이 수치만으로 메시 안전성을
뜻하지 않는다. 사용자의 정성평가와 실제 메시 지표를 함께 보면 `mu=0.50`은 UAL2 Hook에
대해 full correction의 절반을 전달하면서 `mu=1`의 심한 면적 붕괴를 피한 후보로 해석한다.
현재 표본은 BVH 1개와 target 2개이므로 전 라이브러리 승격 근거로는 아직 부족하다.

## Mixamo 대조군

팔 rest 방향과 wrist frame이 이미 호환되어 palm 후보를 활성화하지 않았다.
`mu=0`과 `mu=1` 산출물을 재임포트했을 때 52개 본 행렬과 모든 메시 정점 차이가 0이었다.

## 남은 gate

1. 안전 임계와 손별 candidate selector 동결
2. export/reimport 뒤 선택 결과 재현성
3. fallback 26개 전수 exact V3.2 확인
4. wrist-direction 61개 entry 남녀 전수
5. 강한 실패 14행 새 악화 0건

하나라도 실패하면 운영 승격하지 않고 해당 손을 `mu=0` exact V3.2로 복구한다.

## 원본 FBX 경로

```text
/Users/dowon/dev/Standin-server/out/palm-roll-v321-qa-20260830/
  ual2-male-mu0/output.fbx
  ual2-male-mu0_25/output.fbx
  ual2-male-mu0_50/output.fbx
  ual2-male-mu0_75/output.fbx
  ual2-male-mu1/output.fbx
  ual2-female-mu0/output.fbx
  ual2-female-mu0_25/output.fbx
  ual2-female-mu0_50/output.fbx
  ual2-female-mu0_75/output.fbx
  ual2-female-mu1/output.fbx
  mixamo-male-mu0/output.fbx
  mixamo-male-mu1/output.fbx
```
