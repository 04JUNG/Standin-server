# 포즈 자동 해부학 검수 (plausibility)

사람이 하기 어려운 포즈를 자동으로 **보류/제외 권장**하고, 가능한 경우 **수정안 BVH**를 만든다.
코드: `pose_curation/plausibility.py`(검사·수정), `scripts/audit_pose_plausibility.py`(전수 실행),
`pose_curation/artifact_review`(검수 화면 연동), 테스트 `tests/test_pose_plausibility.py`.

```bash
python scripts/audit_pose_plausibility.py --base <검수 base> \
  --out <base>/plausibility/report.json --corrected-dir <base>/plausibility/corrected
```

검수 화면은 `<base>/plausibility/report.json`이 있으면 자동으로 읽는다(서버 재시작 필요).
원본 BVH·FBX·S3는 바꾸지 않는다. 수정안은 `corrected/`에만 쓴다.

## 판정 방법

- **골격 무관 관절각.** 각 분절의 해부학 좌표계를 레스트 포즈의 관절 위치에서 한 번 측정하고
  관절의 월드 회전으로 옮긴다. Mixamo·CMU·100STYLE·UAL 등 9종 골격이 같은 각도 정의를 쓴다.
- **2단 한계.** 정상(임상 능동 ROM) 밖이면 보류, 극단(+15~30°) 밖이면 제외.
  - 팔꿈치·무릎: 굽힘/과신전, **옆꺾임(힌지 평면 이탈)**, 비틀림.
  - 어깨·고관절: 방향 의존 스윙 한계(reach cone)와 비틀림 구간(swing-twist).
    비틀림은 팔·다리가 120° 넘게 들리면 검사하지 않는다(Codman 역설로 기준이 불안정).
  - 발목 배굴 한계는 **무릎 굽힘에 따라 넓어진다**(비복근, 20°→40°).
    무릎을 편 다리 들기도 별도로 제한한다(햄스트링, 100°).
  - **견갑-상완 리듬**: 팔을 122° 넘게 올렸는데 쇄골이 안 올라가면 보류.
  - 몸 관통: 기존 `src/collision.py` 캡슐 프록시.
  - 무게중심 vs 지지면(de Leva 질량비): 정보용으로만 기록. 점프·기대기 포즈가 정상적으로 걸리기 때문.
- **수정안.** 근위→원위 순으로 정상 범위 안에 투영한다. 하위 관절의 로컬 회전은 유지한다.
  팔꿈치·무릎 옆꺾임은 위팔·허벅지를 자기 축으로 돌려 맞추므로 **관절 위치가 변하지 않는다.**
  수정 후 재검사가 나빠지거나 새 관통이 생기면 수정안을 버린다.

## 검증(2026-11-02, 5,862개)

| 항목 | 결과 |
|---|---|
| 사람이 보류·제외·메모한 44건 | 44건 모두 자동 보류/제외 |
| 전수 판정 | 통과 3,238 · 보류 권장 1,768 · 제외 권장 856 |
| 수정안 재검사 | 2,624건 중 2,596건 통과 |

대량 원인은 데이터 계통 문제다.
- CMU 리타깃 팔꿈치는 힌지 평면이 일정하게 약 28° 어긋나 있다.
- Rokoko 계열은 약 -28° 어긋나 있다.
- 생성 포즈 일부는 발이 173° 뒤로 돌아가 있다(`gen_fl_knees_up` 계열).
- CMU·webtoon·combat 계열은 쇄골이 전혀 움직이지 않는다.

## 문헌과 채택 여부

| 문헌·도구 | 쓴 것 | 안 쓴 이유 |
|---|---|---|
| Akhter & Black, Pose-Conditioned Joint Angle Limits, CVPR 2015 | 이웃 관절 조건부 한계라는 발상(발목↔무릎, 고관절↔무릎) | 데이터 라이선스(비상업, 상업 모델 학습 금지) |
| Wilhelms & Van Gelder 2001 reach cone; Baerlocher & Boulic 2001 swing-twist | 어깨·고관절 스윙 원뿔과 비틀림 구간 | — |
| AAOS ROM, Soucie et al. 2011(CDC), WA DSHS 13-585A, Roaas 1982 | 정상 단계 수치 | — |
| Bennell et al. 1998 체중 부하 런지 검사 | 무릎 굽힘 시 배굴 한계 확장 | — |
| Inman 1944 견갑-상완 리듬 | 팔 올림 시 쇄골 거상 기대값 | — |
| de Leva 1996 분절 질량 | 무게중심(정보용) | — |
| SMPLify GMM, VPoser, Pose-NDF, NRDF, DPoser, HuMoR | — | SMPL/AMASS 비상업 라이선스, SMPL 피팅·torch 필요 |
| TUCH/selfcontact, IPMAN, PhysCap | 접촉·지지면 개념만 | 메시·비상업 코드 |
| GFPose(Apache-2.0, 관절 위치 기반) | — | 자체 데이터로 재학습해야 함. 다음 단계 후보 |

## 한계와 다음 단계

- 손가락은 검사하지 않는다(손 복구 파이프라인 소관).
- 쇄골 각도 자체는 출처마다 레스트 기준이 달라(ACCAD +25°, SD -18°) 범위 검사에 쓰지 않는다.
- 수정안 뼈대는 FBX/GLB로 다시 변환해야 3D 미리보기에 반영된다.
- "가능하지만 어색한" 포즈는 검수 완료된 채택 포즈로 자체 GMM/kNN 사전분포를 학습해 잡을 수 있다.
