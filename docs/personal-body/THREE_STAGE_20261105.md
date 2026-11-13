# 골반 변형 3단계 순차 실험 — 2026-11-05 KST

기준 코드/제품 runtime은 #142 `1332d598be58aec68540137e887fea4dc5fb9991`의 personal-converter/10이다. 시작 시 develop `efa8d68ff`와 기존 PR head를 다시 fetch했다. 리타겟팅/국소 스키닝/표면 마무리를 각각 별도 worktree와 `codex/personal-body-stage{1,2,3}-…-20261105` 브랜치에서 순서대로 시험했다. 제품 converter 코드는 변경하지 않았고, 기존 입력·웨이트·runtime을 읽기 전용으로 사용했다. 신규 유료 검색/평가·운영 설정 변경·배포·병합은 없다.

## 1. 골반·허벅지 회전 대조: 완료, 채택 0

실제 남성 CS3C·여성 FBX·아동 FBX × 기존 두 포즈에서 102개 대조를 실행했다. 원래 스킨 전체 influence로 LBS를 독립 재구성한 후, 관절 머리와 하위 bone world frame을 고정하고 pelvis/thigh의 axial roll만 바꿨다. 머리 위치 오차 최대 약 1.50µm다. 무변경 matrix 재대입만으로 경계 근처 급접힘 수가 1개 달라질 수 있어 그 차이를 개선으로 해석하지 않았다.

여성 CMU 허벅지 local twist는 좌 약0.000015°/우0.000007°로 이미 거의0이었다. 이를 제거하거나 부모 기준 최소 swing으로 다시 구성해도 최대 엣지 배율219.9가 유지됐다. pelvis local twist142.9°를 하위 frame 고정 상태에서 절반/전부 제거하면 최대 배율221.8/222.7로 악화했고 다른 접힘도 증가했다. pelvis 각도는 리그/좌표계와 입력 회전에 의존하므로 숫자만으로 과회전이라고 판정하지 않는다. 이 ablation은 원인 분리이며 원본 포즈를 보존하는 제품 대체 해법이 아니다.

**판정:** 여성 CMU의 문제를 단순 허벅지 축 과회전으로 설명할 근거는 없다. 이 시험이 모든 리타겟팅 축/의미의 올바름을 인증하지는 않는다. [102개 수치](three-stage-rotation-20261105.json).

## 2. 국소 스키닝 보완: 완료, 채택 0

원래 rest relief와 pose 대응점의 공간적 평균/공분산으로 국소 강체 움직임을 least-squares fit하고, 원래 세부 형상을 되붙이는 실험을 했다. 실제 제품의 원본 웨이트/리그/재질은 수정하지 않았다. Direct Delta Mush의 정식 구현이나 동등한 품질 인증으로 부르지 않는다. rigid transform으로 움직인 조형의 복원이 1e-7 오차 내 유지되는 단위 대조를 통과했다.

여섯 출력에서 반경·강도를 달리한42개 후보를 평가했다. 추가로 여성 CMU의 기하 주름 pin만 해제한6개 대조를 평가했다. 저작자가 지정한 sharp/boundary anchor는 유지했다. 별도5개 필터 그룹은 수렴 budget에 실패하여 기록했으며 통과 결과로 대체하지 않았다.

여성 CMU의 최대 배율은219.9→47.7까지 줄었지만 급접힘1819개, 접힘1106개로 커졌다(기존 v10은783/482). 압축213개와 면적1% 미만 면3개도 남고, 새 붕괴 면3개가 생겼다. 원래 손상 support 밖 정점 이동은0이지만, support 안의 다른 변형이 심해져 채택할 수 없다. 기하 주름 pin을 푼 대조도 회귀를 해결하지 못했다. [후보와 수렴 실패 기록](three-stage-skin-20261105.json).

## 3. 작은 표면 마무리: 완료, 채택 0

2단계에서 승인된 결과가 없으므로 제품 경로로 승격하지 않았다. 각 출력의 진단 후보에만 작은 마무리 세 강도를 적용해18개를 비교했다. 마무리 이동 상한은 체형 extent의0.3%다. 여성 CMU에서 실제 최대5.04mm였다.

| 여성 CMU | 기존 v10 | 2단계 진단 후보 | 3단계 최대 강도 |
|---|---:|---:|---:|
| 최대 엣지 배율 | 198.629 | 47.698 | 47.698 |
| 99분위 배율 | 1.807 | 1.913 | 1.872 |
| 2배 초과 엣지 | 1785 | 2291 | 1908 |
| 1/5 미만 압축 엣지 | 180 | 213 | 58 |
| 기준 면적1% 미만 면 | 1 | 3 | 1 |
| 새60° 급접힘 | 783 | 1819 | 1619 |
| 새90° 접힘 | 482 | 1106 | 928 |

작은 마무리로 일부 압축은 줄지만 급접힘과 전체 변형 회귀를 만회하지 못했다. 여섯 출력의18개 후보 모두 기존 v10 대비 한 가지 이상 회귀가 남았다. [3단계 전체 수치](three-stage-finish-20261105.json).

## 실제 모델 렌더·재임포트와 인수

`render_three_stage.py`는 여섯 출력의 v10/국소 fit/마무리를 실제 모델로 앞뒤 같은 카메라·조명·원래 재질 조건에서 렌더한다. 형상만 비교하기 위해 **세 열 모두 동일한 기하 노멀을 재계산한 진단 대조**다. 원래 custom normal 보존을 인증하는 제품 출력이나 CSP 사진으로 부르지 않는다. 진단 FBX에는 원래 weight와 기존 hierarchy가 남는지, world 정점 재임포트 오차가20µm 미만인지 별도로 검사한다. 원본 모델/FBX/NPZ/Blender/decoder/렌더는 Git에 올리지 않는다.

공용 `artifacts/personal-deforme-followup-20261105/converter-three-stage/`에 작업 위치·코드 commit·실제 input/output hash·전체 raw 결과와 이미지·재임포트 기록을 남긴다. 기존 v10 runtime manifest `eaac3841c8b9e693ff88c959eabaceaf1ecc298e547eede748e83ad4a0de1d70`과 제품 결과는 유지한다. 이번 단계에서 새 runtime을 만들거나 검증되지 않은 후보를 앱에 연결하지 않았다.

**결론:** 세 단계의 대조는 마쳤지만, 여성 CMU 골반 약199배 변형의 제품 수정은 아직 해결되지 않았다. 추가 평활화 강도 조절로 해결됐다고 보지 않는다. 다음 구현 후보는 원래 crease를 위치 고정점으로만 취급하지 않고 형태 제약으로 보존하면서, 실제 손상 부위의 변형 전환을 함께 푸는 국소 solver다. 이번의 단순 pin 해제나 강체 fit이 그 solver를 구현한 것은 아니다. 같은 pose의 제작 도구 기준 출력이 있으면 원본 skin과 retarget 의미를 구분하는 추가 정답 대조로 사용해야 한다. CSP 실제 import/save/reopen은 기존 클라 담당 검증 경계를 유지한다.

재현: stage1 `experiment_rotation_stages.py`는 pinned Blender에서 원래 등록 결과와 두 BVH를 받는다. stage2 `experiment_local_motion_fit.py`는 기존 `verify_followup_geometry.py`의 stage NPZ를 받으며, pin 대조는 `--allow-geometric-crease-motion --female-cmu-only` 옵션으로 재현한다. stage3 `experiment_small_finish.py`는 stage2 결과를 받고, `render_three_stage.py`가 진단 FBX/렌더를 생성한다. 어떤 스크립트도 제품 source model/weight 또는 기존 runtime을 변경하지 않는다.
