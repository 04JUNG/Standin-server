# 시점 확대·원근 검색의 런타임 연결

2026-10-15. PR #88의 후속 구현. **코드 연결이며 운영 배포·머지는 수행하지 않았다.**

## 활성화와 롤백

`POSE_SEARCH_MODE=perspective`가 새 코드의 기본값이다. 새 버전이 배포되고
Pipeline이 생성되면 아래 지원 범위에서 `/analyze`의 후보 검색에 사용된다.
`POSE_SEARCH_MODE=legacy`로 설정하고 프로세스를 다시 시작하면 기존 4시점
검색만 사용하며 새 BVH/카메라 색인을 로드하지 않는다. 배포 설정은 수정하지 않았다.

지원 범위는 **refine v2에서 한 명인 컷의 정상 full coverage·full_image·전신 pos 검색**이다.
다인물(슬롯 1개만 남았어도 VLM 인원수가 2 이상이면 포함), upper_only,
부분/의심/복구 슬롯, Human-Art rescue의 보수적 검색, angle/hybrid 검색은
기존 검색과 안전 정책을 유지한다. 인물별 윤곽 분리 없이 다인 컷 전체를
각 인물에게 매칭시키지 않는다. RTMPose/Human-Art 모델이나 XY 추론은 바꾸지 않는다.
`REFINE_V2_ENABLED=0`의 v1은 기존 검색·거리 confidence·보정 경로를 함께 유지한다.

## 실제 호출 경로

`Pipeline.process_cut → _process_slots → _evaluate_slot → _search_one`
에서 `PerspectiveSearchIndex.search`를 호출한다. 별도 실험 실행이 필요 없다.

1. Pipeline 생성 시 저장 투영과 같은 원본 BVH를 읽어 COCO17 XYZ를 확보한다.
   모든 포즈/투영의 대응·유효 body·저장 feature 일치를 확인한다. 하나라도
   누락/불일치면 일부 포즈만 삭제하지 않고 **검색 전체를 legacy로 유지**한다.
   원본 DB/BVH를 수정하거나 외부 번들을 다운로드하지 않는다.
2. 13개 yaw × 3개 pitch × 직교/거리4/거리8 = 117개 조합을 색인한다.
   좌우는 전체 몸의 두 라벨 해석으로 비교하며, 마스크도 함께 대응시킨다.
3. 거리2/거리3의 78개를 추가한 195개 시작 카메라에서 미세 조정한다.
   결과 점수는 기존117 비용 25% + 조정 비용 75%다.
4. 검증 당시와 같은 Top20 후보 풀에서 첫 후보까지 모두 윤곽 비용을 계산한다.
   흰 정방 여백·256px 면적 축소·minRGB<200·대칭 거리 비용 상한0.5·비중1을
   유지한다. **이전 첫 후보도 순위가 내려갈 수 있다.**
5. 어느 후보 하나라도 표면 렌더가 실패하거나 입력 선이 부족하면 해당 입력의
   모든 윤곽 비용을 버리고 같은 카메라 혼합 점수로 정렬한다. 카메라 검색 자체가
   실패하면 전체 legacy 검색으로 복귀한다. 4096px 초과 변은 큰 정방 패딩을
   할당하지 않고 윤곽만 복귀한다. 새 렌더러는 점수 계산 내부용이며 화면 인형이 아니다.

quarantine은 요청마다 후보 풀을 만들기 전에 적용한다. 마지막에 기존의 pose
family 정책(원본/mirror 묶음)을 적용하고 뒤 후보로 채운다. 입력/색인은 변경하지
않는다. 새 라이브러리는 새 Pipeline을 생성해야 반영된다. 후보의 반환 순위가 정본이다.

## 신뢰도·refine·표시 경계

최초 연결 커밋의 일괄 refine 차단은 제거했다. 새 최적화/윤곽 합산 점수에 기존
`FALLBACK_DISTANCE=0.45`를 적용하지 않아 confidence=low, confidence_threshold=null은
유지하지만, **refine_allowed는 기존 v2 구조/소유권/관측 게이트로 결정**한다. 새 점수가
작아졌다는 이유로 high를 주지 않는다. 정상 입력의 유효 scores도 일괄 0으로 만들지 않는다.
원래 refine 금지 입력이나 카메라 바인딩 미완성 입력은 계속 차단한다.

`CandidateCamera v2`는 선택한 fit 카메라의 회전·원근/직교·몸통 길이 기준 거리·
전체 몸의 labels_swapped·원본 BVH SHA·pose_id·query SHA·측정 mask를 전달한다.
검색은 두 해석 중 실제 선택한 값을 저장하며, API는 다른 표시 카메라를 다시 맞추지 않는다.
리파인은 회전 정렬된 BVH에서 **원래 골반 중심과 3D 몸통 길이로 고정한 카메라**를
쓴다. 좌우가 바뀌면 타깃 XY·scores·allowed_limbs·선택 bone_weights도 함께 대응시킨다.
물리적인 BVH 미러는 하지 않는다. 결과는 기존과 같이 원래 BVH 좌표계로 복원한다.

`src/refine_projection.py`의 요청별 ContextVar 범위에서 library.pose_to_feature가
고정 원근 투영을 사용한다. v1/v2 전방계산·v2.5 최종 selector의 공통 오차가 같은
카메라를 사용하며, 다른 요청/예외/중첩 호출 이후에는 원래 투영으로 복원된다.
거리와 골반 원점은 solver 반복이나 몸통 변화에 맞춰 재피팅하지 않는다. near-plane을
넘는 제안은 depth clamp로 숨기지 않고 camera_projection_invalid로 원본을 유지한다.
기존 충돌·이동량·관절 제한·비회귀·selector·시간 제한 게이트를 완화하지 않는다.

camera=null/v1 요청은 기존 방식 그대로다. v2는 필수 projection/좌우/바인딩 필드
누락, 다른 포즈/쿼리, stale BVH, 측정되지 않은 관절 추가를 거부한다. 클라이언트는
camera 전체와 distance_metric=perspective_pos를 그대로 /refine에 재전송해야 한다.
카메라가 탈락한 새 검색 요청을 가까운 4방향으로 조용히 대체하지 않는다. 이 검사는
계약/좌표 대응 검사이지 클라이언트 입력의 암호학적 인증은 아니다.

기존 view/썸네일은 가까운 라이브러리 시점이다. 표시/썸네일 파이프라인은 회전을
받아도 아직 직교이므로 **최종 화면의 원근까지 동일하게 재현했다는 뜻은 아니다.**
이 PR은 서버 API 연결이며 외부 BFF/앱의 v2 전체 필드 전달·화면·운영 부하/배포는
별도 통합 확인이 필요하다. 검색 fit은 혼합 점수의 한 성분이지 전체 blend의 단일 카메라가 아니다.

Z 추정·팔꿈치/교차 특징 보조점수·수동 러프별 조건·새 roll/이웃 점수는 연결하지 않았다.
실험 모듈을 런타임에서 import하지 않는다. 동결 실험 파일/해시를 보존하기 위해 검증된
기하·윤곽·내부 렌더 함수의 런타임 사본을 두고 parity 테스트로 일치를 확인한다.

## 로컬 확인

- 커밋 대상 기존 테스트와 새26개 camera/refine 계약 테스트: 1009개 통과·8개 조건부 skip.
  로컬 모의 HTTP 서버 권한이 필요한 기존4개는 별도 실행해 모두 통과했다
  (총1013개 통과, converter/Blender 테스트 제외). src/api/scripts 컴파일과 diff 검사 통과.
  외부 유료 API·모델 추론은 호출하지 않았다.
- 추가 계약 테스트: 기하/카메라·내부 렌더·윤곽의 동결 코드 동등성, 첫 후보 이동,
  전체 윤곽 복귀, quarantine/family backfill, BVH/투영 불일치 거부, 입력 불변,
  두 요청 동시 실행, legacy 롤백, 예외 복귀, 전신 컷 연결 및 v2 구조 게이트 유지,
  다인물 컷 윤곽 차단 등을 확인했다. 추가로 실제 v2 솔버의 직교/원근 두 구도 ×
  직접/좌우 교환 × conservative/aggressive 12조건에서 알려진 팔 각도 차이가 줄어드는지,
  원점·뼈 길이 보존/원본 복원, 같은 투영의 최종 selector, HTTP analyze→refine 바인딩과
  near-plane 원본 복귀, stale/누락/다른 query 거부, 스레드/예외 격리를 검사했다.
- 기존 검수 러프10장의 **저장된 측정 XY만** 사용한 구성요소 재실행:
  동일한 1995 포즈 번들, 검색가능9장 모두 런타임과 동결 정책의 Top5 ID·점수 일치.
  rough_178의 기존 관절 기준 미달은 분모에 남겼다. 원본3D 정답이 없어 정확도는 null이다.
- 1회 로컬 재실행의 검색 중앙값: legacy 1.308ms, 새 경로514.408ms.
  초기 색인 준비1547.506ms는 요청 전에 한 번이다. 워밍업/반복/부하 시험이 아니며,
  모델 추론·API·표시 카메라 fitting·네트워크를 포함한 운영 응답시간이 아니다.
- 원본 DB는 `mode=ro`로만 읽고 DB/BVH/그림/측정 XY의 실행 전후 해시를 확인했다.
  출력은 ignored `out/perspective-runtime-replay-20261015/results.json`에만 보관한다.
  리파인 연결 후에도 `out/search-refine-runtime-replay-20261015/results.json`에서 9장 모두
  검색 순위/점수 동등성을 재확인했다. 새 카메라 메타데이터 추가로 검색 순위를 바꾸지 않았다.

재현:

```sh
python -m pytest -q tests/test_perspective_runtime.py
python -m pytest -q tests/test_search_camera_refine.py tests/test_candidate_camera.py
python scripts/check_perspective_runtime.py \
  --library-root /path/to/frozen-bundle \
  --review-root /path/to/rough-test-perspective-review-20261013 \
  --out out/new-runtime-replay
```

소스 러프/라이브러리/생성 결과는 커밋하지 않는다. 이 재실행은 기존 측정치의 정책
이식 검사이지 실제 모델을 포함한 새로운 종단간 정확도 검증이 아니다.
