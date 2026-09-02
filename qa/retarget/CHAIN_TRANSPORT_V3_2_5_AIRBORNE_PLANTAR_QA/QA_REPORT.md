# V3.2.5 Airborne Plantar QA Report

상태: **정량 QA·전체 활성 라이브러리 스캔 통과 / 전군 육안 검토 대기 / 운영 미연결**

실행 환경: Blender 5.2.0 LTS, frame 0, `rigged_rest`, 남성 승인 캐릭터.

## 구현 경계

- V3.2.4 artifact를 먼저 완성하고 그 SHA-256을 rollback 기준으로 고정한다.
- source toe `|pitch| <= 8°` 또는 V3.2.4 contact correction이 선택된 발은 공중 발
  selector가 소유하지 않는다.
- 공중 발 목표는 source rest 대비 종아리–toe 상대 굽힘을 target rest의 실제 메시
  plantar–shin 관계에 더해 만든다.
- target foot/toe에 같은 최소회전만 적용하며 proximal chain과 toe 상대회전을 보존한다.
- 후보는 목표 오차, correction hard guard, twist, proximal, 실제 메시 절대 gate와
  사용자 승인 V3.2.4 contact deformation envelope를 모두 통과해야 한다.

## V3.2.4 exact rollback

| 케이스 | 사유 | parent/final SHA | 판정 |
|---|---|---|---|
| `cmu_05_10_00400` | 승인 접지 발, contact 소유 | 동일 | PASS |
| `Talking On Phone_02` | 정상 대조군, 상대 굽힘 호환 | 동일 | PASS |
| `cmu_05_13_00603` | 안전 후보 없음 | 동일 | PASS |
| `cmu_75_16_00244` + kill switch | `FORCED_EXACT_V324` | 동일 | PASS |
| 강제 post-parent 예외 | `RuntimeError` 주입 | 동일 | PASS |

강제 예외에서도 `fallback_reason=EXCEPTION_AFTER_V324_PARENT`,
`exact_parent_artifact_restored=true`가 기록되었다. 후보 탐색 전·중·후 실패는 이미 완성된
V3.2.4 FBX를 byte-for-byte 복사한다.

## 비접지 표시 전군

발목 문제 리뷰의 원본 22개 중 source 양발 모두 수평 ±8° 밖인 비접지 15개를 전수 실행했다.

- 안전 선택: 2/15
- exact V3.2.4 rollback: 13/15
- 변환 실패: 0

| 케이스 | 선택 | 목표 오차 | clearance | area p01 | edge min | strain p99 | dihedral p99 | fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmu_05_11_00542` L | gain 0.90 | 1.662° | 0.0632 | 0.9045 | 0.8835 | 0.0887 | 2.978° | 0 |
| `cmu_75_16_00244` L | gain 0.65 | 0.280° | 0.0821 | 0.9409 | 0.9210 | 0.0577 | 1.176° | 0 |
| `cmu_75_16_00244` mirror R | gain 0.65 | 0.488° | 0.0806 | 0.9244 | 0.8951 | 0.0721 | 1.364° | 0 |

mirror에서 선택 side가 L→R로 이동하고 gain 0.65가 유지되었다.

선택된 두 원본 케이스에서 proximal matrix 오차와 post-bake proximal 오차는 모두 0,
toe 상대회전 오차는 0이었다. correction twist 최대치는 `4e-7°` 미만으로, 보정이 foot/toe
공통 swing에만 제한됨을 확인했다. 실제 적용 swing은 각각 `17.39°`, `8.45°`였다.

## 판정

수학 positive control, 접지/정상 대조군 보호, actual mirror, 실제 메시 gate, 강제 예외 exact
rollback을 통과했다. `cmu_05_11_00542`는 사용자 비교에서 포즈 가독성과 접지 복원이
긍정적으로 확인됐지만, 전체 선택 범위가 더 넓으므로 단일 사례만으로 승격하지 않는다.
따라서 이 모듈은 QA-only로 유지하며 운영 solver와 V3.2.4 동결 파일을 수정하지 않는다.

## 전체 활성 라이브러리 스캔

2026-09-01에 `data/poses.db`의 활성 포즈 1,248개를 남녀 캐릭터에 대해 검사했다. 삭제·격리
아카이브가 아니라 현재 제품이 제공하는 활성 DB를 입력 단일 소스로 삼았다.

| 성별 | 활성 포즈 | source prefilter 통과 | V3.2.5 선택 | exact V3.2.4 fallback | 실패 |
|---|---:|---:|---:|---:|---:|
| 남성 | 1,248 | 664 | 109 | 555 | 0 |
| 여성 | 1,248 | 664 | 159 | 505 | 0 |

- non-selected 전군에서 최종 FBX가 V3.2.4 부모 artifact와 byte-for-byte 동일했다.
- 변환 실패, 조용한 미분류, 빈 artifact는 0건이었다.
- 여성 선택 159건은 남성 선택 109건보다 넓으므로 성별 결과를 합쳐 추론하지 않는다.
- 선택된 268 `(pose, gender)` 항목에 V3.2.4/V3.2.5 FBX 536개와 동일 정면 비교 이미지
  536개를 생성했다. 누락 및 0-byte 파일은 0건이다.
- 육안 검수기는 미검수/보류/실패, 메모, JSON import/export를 제공한다. 명시적 합격은
  기록하지 않으며 정상으로 보이는 항목은 미검수로 유지한다.

전체 스캔은 selector와 exact fallback의 기계적 안전성을 확인한 것이다. 선택된 268건의
메시 외형 안전성과 포즈 의도 보존은 육안 검토가 끝나기 전까지 미승인이다.
