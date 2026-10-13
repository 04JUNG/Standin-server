# T013 팔짱 팔 체인 + 앉기 베이스 — 오프라인 probe

2026-10-13 사용자 요청으로 상·하체 분리 검색 후 합성을 실험했다. 현재는 연구 산출물이며
`Pipeline`, HTTP API, 원본 DB/BVH/태그, 기하 피처·distance·refine에 연결하지 않는다.
`docs/REFINE_V2_DESIGN.md` §13의 운영용 파트 조합/retarget 연기는 그대로이며 이번 probe가
그 승격을 의미하지 않는다.

## 범위와 결과

- 같은 T013 러프, 인물 슬롯 2의 **기존 실제 Gemini 3.8 Flash/d2 설명**을 고정했다.
  새 T013 실호출의 upstream_error는 해결한 것이 아니다. 이번 추가 VLM 호출·이미지 전송은 0회다.
- 상체: 가슴 앞 팔짱으로 별도 검색하여 reviewed fold_arms AND 실제 BVH 손 교차가 검증된
  `UAL2__Idle_FoldArms_Loop__p01_f0073_mirror`를 사용했다(원본/미러 중복 제외).
- 하체: 관찰 문장의 `허벅지를 수평으로 뻗어 무릎을 직각으로 굽힌 채 앉아 있다.` 부분으로
  sitting 태그 236개에서 별도 검색했다. arms 관찰·행동은 제거했다. 이번 문장 분리는 사례 한정
  명시적 분리이며 일반 자연어 자동 분해 모델의 성능 검증은 아니다.
- 하체 chair-like 검사: 무릎 70~115°, 허벅지 수평 대비 기울기 <=30°, 정강이 아래 방향 분율 >=0.65.
  의자 mesh/contact는 없으며 이 값은 실험 기준이지 물리적 지지 보증이 아니다.
- 검색순으로 조건을 통과한 하체 12개에 대해, 단순 Euler 값 복사와 방향 retarget를 비교했다.
  단순 복사는 팔짱 0/12개 통과, 방향 retarget는 구조 검사 11개 통과/1개 제외다.
  **단일 T013의 구조 검사이며 accuracy/실사용 성공률 11/12가 아니다.**
- 기존 v3 전신 검색은 스트레칭을 반환했고 v4 핵심 검색은 검증 후보 0개였다. 이번 합성은
  원래 없던 조합을 별도 synthetic identity/BVH로 만든 것이다. 기존 후보 0개 기록을 고치지 않았다.

## 이식과 보호 경계

첫 단계는 torso 전체 교체가 아니라 **팔 체인 이식**이다. 앉기 베이스의 목·머리·쇄골·척추·골반·
다리·모든 position 채널을 유지한다. LeftArm/RightArm 이하의 회전 채널만 바꾼다.

`src/experimental/part_composition.py`는 기존 BVH FK에서 donor 팔·손가락 방향을 구하고
각 포즈의 어깨/힙 기반 몸통 frame으로 맞춘다. target 리그의 원래 OFFSET·계층·뼈 길이를
보존한 상태에서 target Euler 순서로 회전행렬을 변환한다. 단일 ray는 shortest-arc,
여러 손가락 ray는 강체 최소제곱이다. UE 계열 손가락 이름과 Mixamo 이름을 명시적으로 매핑한다.

메모리 합성에서 보호 관절/하체 6관절 위치 변화는 0, 뼈 길이 오차는 최대 `5.49e-16 torso`였다.
팔-몸통/팔-다리 capsule proxy의 신규 깊은 관통을 확인했다(허용 증가 0.02 torso, 실험 기준).
성공 BVH 11개는 저장/재읽기에 통과했다. HIERARCHY 원문을 보존했고 보호 관절 변화 최대는
6자리 직렬화 오차 `4.03e-08 torso`(검사 상한 1e-6)였다. 모든 source BVH SHA는 불변이다.

`sd_cmu_114_05_f01284`는 팔 방향 오차가 거의 0이어도 체형의 어깨 폭/팔 길이 비율 때문에
가슴 앞 손 교차가 성립하지 않아 `chest_fold_lost`로 제외했다. 뼈 길이를 늘리거나 조건을
낮추지 않았다. 방향 일치와 손 접촉/팔짱 관계 일치는 서로 다른 검사다.

## 재현

선택적 SciPy가 필요한 오프라인 실험이다. 서버 기본 의존성/프로덕션 경로는 바꾸지 않았다.

```sh
python scripts/eval_part_composition.py \
  --results tmp/routing-eval-live-v3-eight-20261013/results.json \
  --tags <viewer>/dist/data/search-v2.json \
  --body-index tmp/semantic-body-v1.json --shape-index tmp/semantic-shapes-v1.json \
  --bvh-root <bundle>/bvh --out tmp/part-composition-T013-v1

python scripts/report_part_composition.py \
  --experiment tmp/part-composition-T013-v1/results.json \
  --images <testset-root> --thumbs <bundle>/thumbs \
  --attribution <bundle>/ATTRIBUTION.md
```

출력은 `results.json`, `REPORT.md`, `ATTRIBUTION.md`, 성공 조합별 BVH와 진단 PNG다.
조합 ID는 `experiment_T013_fold_sitting_XX`이며 라이브러리에 등록하지 않는다.
PNG는 같은 베이스 몸통 frame의 정면/양쪽 사선/측면/뒤에서 만든 COCO17 관절 마네킹이다.
원본 러프나 기존 mesh JPG를 수정하지 않았고 실제 인체 FBX mesh 렌더로 표시하지 않는다.

## 미검증과 후속 게이트

- 실제 캐릭터 FBX/Blender 런타임이 이번 로컬 환경에 없어 skinning, 손목 twist, 손가락 형태,
  arm-arm/chest 피부 접촉, 관절 limit, CSP import는 미검증이다. 모든 조합은 `release_allowed=false`다.
- 관절 방향만으로 twist를 유일하게 정할 수 없고 손바닥 여러 방향은 근사다. 따라서 proxy 통과를
  실제 mesh 안전 또는 시각적 자연스러움으로 일반화하지 않는다.
- 상하체 전체 연결/척추 seam 문제를 해결한 것은 아니다. 손 접촉 목표와 관절 제한이 있는
  contact-aware IK 및 원본 캐릭터 mesh 렌더가 후속 검증이다.
- 라이브러리 공백을 막는 연구 후보 생성 계층으로 분리한다. 기존 geometric 후보/distance와
  섞거나 스켈레톤 없는 semantic 결과에 기존 refine을 허용하지 않는다.
- 실제 모델 확보 후 같은 donor/base/composed를 같은 render 설정으로 보고, 정면/측면/뒤 시점
  및 작가 검수로 개선·회귀를 평가해야 한다. 12개는 독립 러프 12건이 아니다.
- 원본 ATTRIBUTION을 보존했으며 외부 배포 전 donor/base의 파생 이용 조건 확인이 필요하다.
  PR/push/공유 파일 통합/운영 배포는 하지 않았다.

검증: 단위 21개, 서버 893 passed/8 skipped/22 subtests, 스모크 54/54, compileall/diff-check 통과.
Blender 전용 `tests/converter`는 해당 런타임 부재로 제외했다.
전후 비교 화면은 성공/실패 조합×4시점의 이미지/선택 동작 및 320/360/736px 표시를 검증했다.
