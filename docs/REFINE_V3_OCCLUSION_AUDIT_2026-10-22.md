# 운영 가림 인식과 부분 관측 리파인 조사

조사일: 2026-10-22. 운영 설정·이미지 식별자는 AWS 읽기 전용 조회, 동작은 동일 코드와 로컬 합성 검증으로 확인했다. 실제 사용자 러프 재분석·앱/E2E·배포·운영 설정 변경은 하지 않았다.

## 1. 결론

부분 가림에서도 **관측된 부위를 더 비슷하게 보정하는 것은 가능**하며 현행 v2.5에도 일부 경로가 있다. 그러나 가려진 관절의 실제 3D 정답을 복원했다는 의미는 아니다. 현행은 골반/사지 완전성·관측 출처를 인물 단위로 엄격하게 검사하므로 개선할 수 있는 팔까지 막히는 경우가 있다. v3의 우선 개선점은 가린 좌표를 더 많이 생성하는 것보다 **관절별 출처·확실성 보존 + 관측 가능한 chain만 제한 보정**이다.

실제 러프에서 통계적으로 유의한 품질 개선이 나오는지는 아직 검증하지 않았다. 아래 합성 검증은 현재 허용 정책과 하체 동결 동작의 확인이다.

## 2. 운영 배포 확인

- 서비스: `StandinApp-InferenceService1C7A7625-KPZkcW87EjUE`
- task definition: `StandinAppInferenceTask679E29CF:39`, running 1 / desired 1, rollout COMPLETED, HEALTHY.
- 시작: 2026-10-21 23:56:29 KST.
- 배포 커밋: `5317e972c603d2dec6044e3ff99e7adb8ee578c7`.
- 실제 실행 digest: `sha256:08e19643d4f7a521ebe022c5ce6614fb71b100662d670f3cff55573c0ccc9354`. ECR의 해당 커밋 태그와 일치.
- 명시된 환경: `APP_ENV=production`, `VLM_PROVIDER=gemini`, `POSE_BACKEND=rtmlib`, `POSE_MODEL_VARIANT=humanart-m`, `POSE_CANARY_STAGE=canary-100`, `ROUTING_FULLBODY_ENABLED=1`, `REFINE_ENABLED=1`, `VLM_PERSON_TAGS=1`, `BODY_MATCHING_MODE=auto`.
- `SKELETON_REPAIR`, `REFINE_V2_ENABLED`, `REFINE_V2_LOWER_BODY`, `REFINE_DEFAULT_MODE`는 task 환경에서 미지정. 해당 이미지 코드 기본값은 각각 auto, 1, 1, aggressive. 별도 environmentFiles/해당 이름의 secret override 없음. Dockerfile의 override도 없고 `.env`는 이미지 제외 대상이다.
- 손 판독 `VLM_HAND_SHAPES` 역시 task 환경 미지정이며 코드 기본값 0이다. 이것은 가림 조사 중 확인된 설정이며 이번에 바꾸지 않았다.
- 읽은 핵심 가림/리파인 파일은 설계 worktree의 기준 develop과 운영 main 사이 차이가 없음을 확인했다. 운영 요청별 repair 실행 여부를 로그로 추적한 결과는 아니다.

## 3. 현재 러프 인식 흐름

1. Gemini 기본 분석에서 인원·구도·박스·하체 가시성을 받는다.
2. 운영 Human-Art 모델(YOLOX + RTMPose-M)이 인물과 17개 관절 좌표/점수를 추출한다. 모델 점수가 높다는 것만으로 직접 보이는 관절임을 인증하지는 않는다.
3. `analyze_skeleton`이 유효 값/점수, 몸통 anchor, 사지 길이, 인물 소유권을 검사하고 마스크와 coverage를 만든다. 다른 사람의 팔이나 비정상 길이는 보정 대상에서 제외한다.
4. 필요한 슬롯은 crop 재인식/설정된 rescue 경로를 거친다. 운영 primary가 humanart-m이므로 모든 요청을 current-X→Human-Art cascade로 설명하면 틀리다.
5. fullbody + Gemini 구성에서는 불완전 슬롯에 skeleton repair가 조건부 실행된다. 원본·관절 overlay·현재 좌표를 함께 보고 제한적으로 보완한다.
6. 이후 검색은 사용할 수 있는 관절과 기하/의미 경로로 후보를 찾는다. 후보가 나왔다는 사실과 자동 refine 자격은 별개다.

### 이미 있는 VLM 가림 보완의 정확한 의미

현재 hybrid prompt는 `visible`, `occluded_estimated`, `not_present`를 구분한다. 가렸어도 주변 연결로 위치를 좁힐 수 있을 때 `occluded_estimated` 좌표를 제안할 수 있다. 따라서 **현재 서버 전체에서 VLM이 좌표를 전혀 만들지 않는다는 설명은 정확하지 않다.** 기본 분석·새 v3 의미 근거와 별개로, 기존 repair 경로에 제한된 좌표 보완이 이미 있다.

`_skeleton_from`은 보완된 좌표에 낮은 점수를 부여해 Skeleton으로 변환한다. `visible`과 `occluded_estimated`의 상세 provenance는 이 좌표/점수 배열만으로 충분히 보존되지 않는다. 보완이 채택되면 슬롯은 `skeleton_source=vlm_repair`가 되고, 현행 정책은 그 인물의 자동 리파인을 전부 차단한다. 추정 좌표를 정답처럼 보정 목표로 쓰지 않기 위한 장치다.

## 4. 현재 자동 보정의 허용 범위

구조 정책은 `state=valid|partial`, `coverage=full|reduced`, 보정 가능한 사지 존재, VLM 슬롯에 귀속, `skeleton_source=full_image`를 요구한다. 최종 후보의 원본 BVH hash·기하 카메라 binding도 필요하다. 정책 통과가 모든 후보의 실제 보정/개선을 보장하지 않는다.

- full/reduced는 대략 정상 몸통 anchor와 완전한 사지 3개 이상/2개를 바탕으로 정해진다.
- 어깨·팔만 있고 골반이 없으면 `upper_only`: 어깨 폭 기준 검색은 가능하지만 현재 보정은 금지된다.
- 완전한 사지가 부족한 `sparse`도 현재 보정 금지다.
- 팔꿈치나 손목이 빠지면 그 팔의 두 뼈가 모두 유효하지 않아 그 팔 전체가 제외된다.
- 하체가 관측됐다는 명시적 근거가 없으면 다리 전체를 고정한다.
- `crop_retry`, `fallback_full_image`, `vlm_repair` 출처는 현행 자동 보정에서 차단된다.
- 검색 카메라 binding도 양어깨·양골반을 요구하므로 upper_only는 정책 하나만 바꿔 해결되지 않는다.

## 5. 로컬 확인 결과

같은 코드의 합성 COCO17 관절에서 일부 관절 score를 0으로 만들었다. 실제 그림의 가림을 모델이 얼마나 잘 알아채는지 측정한 실험이 아니다. 아래 허용은 full_image 출처의 구조 정책 기준이며 후보 binding/최종 selector 이전이다.

| 조건 | coverage | 보정 가능한 부분 |
|---|---|---|
| 전부 유효 | full | 양팔·양다리 |
| 오른손목만 결측 | full | 왼팔·양다리, 오른팔 제외 |
| 오른팔·오른다리 결측 | reduced | 왼팔·왼다리 |
| 무릎·발목 결측, 골반 유효 | reduced | 양팔 |
| 골반·하체 전부 결측 | upper_only | 없음, 검색만 가능 |
| 왼팔만 완전 | sparse | 없음 |
| 왼팔꿈치만 결측 | full | 오른팔·양다리, 왼팔 제외 |

같은 7개 입력의 출처를 vlm_repair로 두면 모두 구조 정책에서 차단됨을 확인했다.

기존 두 테스트도 직접 실행해 통과했다:

- `test_v252_unobserved_lower_body_freezes_every_leg_channel`: 합성 BVH에서 팔 보정 성공, 모든 다리 rotation channel이 원본과 동일함을 확인. fixture는 관측성 gate를 끄므로 운영의 최종 채택률을 입증하지 않는다.
- `test_v252_pipeline_removes_all_legs_when_vlm_marks_half_body`: 반신 판단이면 허용 사지에서 다리를 제거하고 all_lower_frozen을 기록함을 확인.

raw 출력과 실행 스크립트는 같은 보고서 파일명에 `.result.txt`, `.check.py`로 보존했다.

## 6. v3에서 가림을 유용하게 활용하는 설계 보완

### A. 먼저 할 것: 관절별 provenance

각 관절에 원래 full_image 관측, crop 관측, VLM 추정, 결측을 구분하고 원본 좌표·수정 좌표·수정 이유를 별도로 보존한다. 일부 다리만 보완됐다고 직접 관측된 팔까지 추정으로 취급하지 않도록 **chain별 새 자격 심사**를 실험한다. 기존 global source 금지를 단순 해제하지 않는다. 추정 좌표는 hard target이나 관측성 증명으로 쓰지 않는다.

### B. 첫 확장: 골반 없는 상체의 팔 보정

양어깨와 같은 인물의 팔이 확실히 보이는 경우, 어깨 중심/폭을 고정 기준으로 팔만 보정할 수 있는 별도 local-arm 경로를 만든다. 현재 검색에 있는 shoulder_frame을 참고하되 검색 피처를 바꾸지 않는다. 몸통·골반·하체·카메라는 고정하고, 평가 normalization도 baseline에서 고정해 점수를 속이지 못하게 한다.

기존 동적 binding이 골반을 요구하므로, 원본 후보/관측/부분 투영·카메라 신뢰도를 묶는 새 버전 계약이 필요하다. 단순히 가짜 골반을 채우거나 카메라를 임의 지정해 기존 계약을 통과시키면 안 된다. 완전 측면처럼 양어깨 투영 폭이 거의 0이면 별도 검증된 scale이 없는 한 제외한다.

### C. 다음 확장: 팔꿈치가 가렸지만 손목이 보이는 팔

어깨·손목의 관측, 실제 체형의 팔 길이, 선택 포즈의 팔꿈치 굽힘/앞뒤 방향, VLM의 검증 가능한 관계를 사용해 가능한 해를 제한한다. 손목을 러프에 가깝게 옮기되 팔꿈치 위치는 하나의 정답으로 확정하지 않는다. 선택 포즈의 굽힘 평면·twist에서 최소로 움직이는 후보를 우선하고 실제 메시 검사로 관통을 제거한다.

보이는 endpoint에 영향을 주는 자유도만 열고, 효과가 없는 depth/twist는 고정 또는 강한 prior를 둔다. 다른 가능성이 비슷하게 남으면 작은 보정 또는 원본 유지가 맞다. 이 경로는 현재 없으며 새 holdout/반례 평가 전에는 제품 자격을 주지 않는다.

### D. 끝까지 확정할 수 없는 것

팔 전체가 가렸거나 손목·팔꿈치가 모두 없으면 자세의 정확한 3D를 알 수 없다. 선택 포즈를 유지하고, 확인 가능한 접촉/안전 조건만 검사한다. 러프 유사도 향상이나 실제 가림 부위 복원 성공으로 집계하지 않는다. 작은 관통 수정도 선택 동작·메시 기준으로 별도 평가한다.

### E. 가림 검증의 성공 기준

- 개발: 완전 관측 러프의 관절 일부를 마스킹한 반사실 비교 + 실제 가림 러프를 함께 사용.
- 판정: 관측된 부분의 실제 target FBX 오차 감소, 숨은 부위의 불필요한 회전/branch flip 없음, 새 관통 없음.
- 독립 평가: VLM 추정점으로 최적화하고 같은 추정점으로 성공을 평가하지 않음. 가림 위치 정답이 없으면 가림 복원 정확도는 unknown으로 둠.
- 비교: 현재 정책 / 관절 provenance만 / local-arm / endpoint-only를 같은 선택 후보·체형·카메라로 비교.
- 골반 없음, 한쪽 팔 결측, 팔꿈치만 결측, 다른 인물에 가림, 극단적 단축투영, 화면 밖, 좌우 불명, 모델의 높은 점수 오인식을 분리 보고.
- 지원 권한은 기존 전신 경로와 별도 capability/버전으로 연다. 기존 search-only/semantic/generated 후보 자격을 일괄 승격하지 않는다.

## 근거

운영 SHA 고정 소스:

- [슬롯·가시성·보정 정책](https://github.com/04JUNG/Standin-server/blob/5317e972c603d2dec6044e3ff99e7adb8ee578c7/src/pipeline.py)
- [가림 마스크·coverage](https://github.com/04JUNG/Standin-server/blob/5317e972c603d2dec6044e3ff99e7adb8ee578c7/src/skeleton_extraction.py)
- [상체 전용 관측/검색](https://github.com/04JUNG/Standin-server/blob/5317e972c603d2dec6044e3ff99e7adb8ee578c7/src/partial_pose.py)
- [VLM 관절 보완](https://github.com/04JUNG/Standin-server/blob/5317e972c603d2dec6044e3ff99e7adb8ee578c7/src/skeleton_repair.py)
- [현재 보정 자격](https://github.com/04JUNG/Standin-server/blob/5317e972c603d2dec6044e3ff99e7adb8ee578c7/src/refine_policy.py)
- [고정 검색 카메라 binding](https://github.com/04JUNG/Standin-server/blob/5317e972c603d2dec6044e3ff99e7adb8ee578c7/src/dynamic_refine.py)
