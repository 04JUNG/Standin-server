# CHAIN_TRANSPORT_V3_SAFETY_V1 QA 보고서

> 날짜: 2026-08-26
> 최종 판정: **REJECTED — 원본 FBX 육안 gate 실패 / frozen V3로 롤백 완료**
> 기준 코드: `qa/retarget/CHAIN_TRANSPORT_V3/converter/retarget.py`

## 1. 최종 결과 요약

`CHAIN_TRANSPORT_V3_SAFETY_V1`은 메시 대리 지표 일부를 개선했지만 실제 형상 품질을
개선하지 못했고, CMU 오른발에서는 frozen V3의 보호 조건까지 우회해 명백한 회귀를
만들었다. 따라서 후보를 폐기하고 사용자가 육안으로 합격시킨 frozen V3를 현재 기준선으로
복원했다.

- `CHAIN_TRANSPORT_V3_SAFETY_V1`: **REJECTED**, 승격 금지
- `CHAIN_TRANSPORT_V3`: **FROZEN-BASELINE** 유지
- production 파일: 수정·승격 없음
- 안전 후보 FBX와 variant: 활성 QA 경로에서 제거
- 실패 원인 로그·매니페스트: 감사 기록으로만 보존

이하의 discovery·holdout 수치는 실패한 선택기가 무엇을 최적화했는지 남기는 실험 기록이며,
제품 품질 개선이나 승격 근거로 해석하지 않는다.

## 2. 구현했던 후보

Frozen V3의 순차 최소회전 체인 수학은 변경하지 않았다. 실제 target skin을 Blender depsgraph로
평가해 골반 `α`와 발목 `μ`를 자동 선택하는 wrapper를 QA variant 안에 구현했다.

- 사례명·BVH명·source profile을 selector에서 사용하지 않는다.
- 정책 수치는 Python 코드가 아니라 별도 JSON에 있고 report가 경로·SHA-256·payload를 기록한다.
- 정책을 통과하는 후보 중 **V3 적용량이 가장 큰 후보**를 선택한다.
- 같은 적용량 후보끼리만 pose RMSE로 tie-break한다.
- 측정 불가·수치 퇴화·허용 후보 부재는 exact legacy C로 명시적 폴백한다.
- `CHAIN_SAFETY_FORCE_OFF=1`은 frozen V3와 pre-export 수치 최대 차이 0이다.

## 3. 실제 메시 측정

각 target mesh에서 Hips/UpLeg 및 Leg/Foot 혼합 웨이트 전이 패치를 자동 추출한다. 모든 후보를
실제 armature deformation으로 평가하며 다음 값을 report에 남긴다.

- edge strain, triangle principal stretches, area ratio, condition number
- adjacent-triangle fold 변화
- 단면 p05 반경과 covariance area proxy
- dimensionless membrane/bending/cross-section energy
- 비인접 triangle piercing 진단

자기교차·energy·각도는 discovery에서 정상 극단 포즈와 실패를 단독으로 분리하지 못했으므로
숨은 hard cutoff로 사용하지 않았다.

## 4. 외부 정책

정책 파일:

`out/rest-safety-qa/variants/CHAIN_TRANSPORT_V3_SAFETY_V1/converter/mesh_safety_policy.json`

현재 QA 계약은 두 개뿐이다.

1. pelvis p05 단면 반경 유지율 `>= 2/3`
2. ankle p05 최소 주신장 `>= 1/3`

후보 격자는 pelvis 1/4 간격, ankle 1/8 간격이다. 이 값들은 QA 정책이며 production 상수가
아니다. 사용자 FBX 육안 gate 전에는 동결하거나 승격하지 않는다.

## 5. Discovery 9건

| 사례 | pelvis α L/R | ankle μ L/R | 판정 |
|---|---:|---:|---|
| Mixamo control | inactive | inactive | compatible 경로 유지 |
| CMU | 1 / 1 | 1 / 1 | full V3 |
| Rokoko | 1 / 1 | 1 / 1 | full V3 |
| UAL2 Hook | 1 / 1 | 1 / 1 | full V3 |
| MakeHuman | 1 / 1 | 1 / 1 | full V3 |
| g1 Move1 | **1 / 0.5** | 1 / 1 | 오른쪽 pelvis 제한 |
| g1 Move7 | 1 / 1 | 1 / 1 | full V3 control |
| UAL2 Slide | 1 / 1 | **0.875 / 1** | 왼발목 제한 |
| UAL2 SwordHeavy | 1 / 1 | 1 / 1 | full V3, 육안 경고 유지 |

독립 post-export surface integrity는 9/9 통과했다.

### g1 Move1

- full V3 hip.R radius retention: `0.5321` → 정책 실패
- 선택: `α={L:1.0,R:0.5}`
- 선택 후 hip.R radius retention: `0.7615`
- hip.R condition p95: `7.5307 → 6.6462`
- ankle.R condition p95도 downstream 효과로 `2.3057 → 1.6015`
- hip edge p95는 감소하지 않았으므로 숫자만으로 육안 합격을 선언하지 않는다.

### UAL2 Slide

- full V3 ankle.L min-stretch p05: `0.3284` → 정책 실패
- 선택: `μ=0.875`, 적용 각도 약 `96.17°`
- min-stretch p05: `0.3284 → 0.3864`
- condition p95: `2.7748 → 2.5034`
- edge p95: `39.8399% → 37.2743%`

### UAL2 SwordHeavy

full V3가 현재 실제 메시 정책을 통과해 자동 감쇠하지 않았다. 사용자가 본 경미한 과굴곡은
현재 정책상 메시 붕괴가 아니라 pose plausibility 문제다. 이 사례를 줄이려면 메시 안전 임계를
결과에 맞춰 강화하지 말고 별도의 관절 plausibility 계약을 설계해야 한다.

## 6. Holdout 8건

개발 선택에 쓰지 않은 다음 8건을 별도 manifest로 고정했다.

- g1 Fence, Move4, Move9, Move17
- UAL2 FoldArms, Throw, Climb, NinjaJump

결과는 변환 8/8, post-export integrity 8/8이며 모든 사례가 pelvis `1/1`, ankle `1/1` full V3를
유지했다. 조용한 제한이나 폴백은 없었다.

## 7. 추가 검증

- 회귀: 임시 물리 복제본 `<TMP_QA_ROOT>`에서 `28/28`
- force-off: frozen V3와 RMSE·delta·본 정책·chain diagnostics 동일
- force-off vs frozen V3 pre-export canonical numeric 최대 차이: `0`
- real mirror g1: 제한이 `R`에서 `L`로 이동 (`α={L:0.5,R:1.0}`)
- mirror post-export surface integrity: PASS
- production retarget/schemas와 `.bak` 해시 불변

## 8. 왜 최종 실패인가

### 8.1 CMU의 frozen V3 보호 조건 우회

Frozen V3는 오른발 incremental ankle rotation이 `141.72°`로 `120°` 보호선을 넘으면
foot solve를 적용하지 않고 terminal-follow로 남긴다. 안전 후보는 다음 조건으로 safety mode가
켜진 동안 이 보호선을 무조건 우회했다.

~~~python
solve_foot = h_foot is not None and (
    _QA_SAFETY_ENABLED or foot_increment <= _QA_FOOT_INCREMENT_MAX_RAD
)
~~~

그 결과 CMU `foot.R`에 `141.715°`, `μ=1.0`이 전량 적용돼 사용자가 확인한 발 방향 회귀가
발생했다. 이는 단순한 metric 선택 실패가 아니라 frozen V3의 안전 동작을 깨뜨린 구현 결함이다.

### 8.2 수치와 육안 판정의 불일치

| 사례 | frozen V3 RMSE | Safety V1 RMSE | 대리 지표 | 최종 판정 |
|---|---:|---:|---|---|
| g1 Move1 | `0.182824` | `0.185781` | 단면 유지율 일부 개선 | RMSE·골반 육안 악화 |
| UAL2 Slide | `0.558751` | `0.559014` | 최소 주신장 일부 개선 | RMSE 악화, 과굴곡 유지 |
| CMU | `0.277728` | `0.256644` | 위치 RMSE 개선 | 141.72° 강제 적용으로 발 방향 회귀 |

CMU의 RMSE 감소는 frozen V3가 의도적으로 거부한 큰 회전을 강제로 적용해 얻은 값이므로
품질 개선으로 인정할 수 없다. 단면·condition·RMSE만으로 해부학적 방향과 사용자가 보는
메시 품질을 대신할 수 없다는 것이 최종 결론이다.

## 9. 산출물 처리와 롤백

안전 후보의 FBX·중간 출력·variant는 활성 QA 경로에서 제거했다. 로그와 매니페스트는 실패
원인 재현을 위한 감사 기록으로만 남겼다.

복원한 V3 source snapshot:

`qa/retarget/CHAIN_TRANSPORT_V3/`

`retarget.py` SHA-256:

~~~text
2701cc44a9ec6584722c194002d505a21bbf4fe199695325bb3b660c2818985e
~~~

보존 snapshot과 복원본은 파일 4개 전체 `diff -qr` 차이 0, SHA-256 일치로 검증했다. 당시
승인 대상 FBX 세트도 원본 보존 산출물과 byte 동일한 복사본만 사용한다.

## 10. 남은 범위

Terminal hand roll은 frozen V3의 별도 `UNRESOLVED` 항목이다. Safety V1을 고쳐 재사용하거나
임계를 조정해 통과시키지 않는다. 새 후속 후보가 필요하면 frozen V3의 기존 보호 조건을 hard
invariant로 두고 별도 설계·별도 육안 gate로 시작한다.
