# 인물별 출력 범위와 관측 영역 검색

`shot`은 기존 검색 라우팅 태그로 유지하고, 출력 구도는 별도 메타데이터로 관리한다.
라이브러리 BVH, 리그, 검색 벡터를 자르거나 다시 만들지 않는다.

## 범위 정의

| 값 | 표시 | 입력 구도 |
|---|---|---|
| full | 전신 | 머리부터 발까지 |
| half | 반신 | 허리·골반·허벅지 부근에서 잘림 |
| bust | 흉상 | 가슴 위·어깨·머리 |
| head | 두상 | 머리·얼굴 중심, 목 일부 포함 |

VLM `body_scopes`는 `approx_boxes`와 같은 순서/길이다. 불확실한 값은 null.
다른 물체에 가려진 하체, 낮은 관절 점수, `lower_body_observed=false`는 반신 판별의 근거가 아니다.
배열 길이가 다르면 전부 unknown, 항목이 잘못되면 해당 자리만 unknown으로 유지한다.
슬롯 ID로 연결한 뒤 화면 좌→우로 정렬한다. 잠정 검출 인물에는 VLM 값을 추측해서 붙이지 않는다.
옛 VLM 응답은 단인 bust/face만 `legacy_shot`으로 사용할 수 있다.
명시적 null을 옛 shot으로 덮어쓰지 않는다. 전신과 반신이 묶인 full_half는 unknown이다.

## 추론 API

`POST /analyze`의 `people[].output_scope`:

```json
{"detected": "half", "source": "vlm_person"}
```

`detected`: full/half/bust/head/null. `source`: vlm_person/legacy_shot/unknown.
흉상은 2단계부터 실제 관측 관절로 검색한다. 두상만 있는 컷은 인물/박스/범위를
유지하고 후보는 빈 배열, 관절은 null, refine은 금지다. 검색 미지원을 사람 미검출로 표현하지 않는다.

## BFF와 앱

`Standin-app-server/src/output-scope/`가 해석·저장·HTTP 경계를 담당한다.
`GET /v1/analysis/jobs/{jobId}/result`의 `candidatesByPerson[].outputScope`:

```json
{
  "selection": "auto",
  "detected": "half",
  "detectionSource": "vlm_person",
  "resolved": "half",
  "resolutionSource": "auto"
}
```

- `selection`: auto/full/half/bust/head. 사용자가 정하는 값.
- `detected`는 분석 시점 값을 보존. 수동 선택 후 auto로 되돌릴 수 있다.
- auto + null은 full로 폴백하며 `resolutionSource=fallback`. UI에 판별 불가라고 표시.
- 수동 선택은 `resolutionSource=user`. 기존 검색/refine 안전정책은 변경하지 않는다.
- `PUT /v1/analysis/jobs/{jobId}/people/{personIndex}/output-scope`에 `{ "selection": "half" }`.
  설치 소유권과 완료 상태/인물 번호/enum 검사 후 Job `result_json`에 저장한다.
  행 잠금 트랜잭션으로 서로 다른 인물의 동시 변경을 보존한다. DB 스키마 변경은 없다.
- 후보 화면과 플로팅 바는 동일한 선택 컴포넌트 사용. 성공한 서버 값만 Query cache에 반영.
  저장 중 중복 변경/포즈 확정을 막고 실패 시 이전 값과 재시도 안내를 유지한다.
- 작업 기록에서 복원, 새 작업은 auto. 후보 선택/refine에 종속되지 않는다.
- 구 BFF는 `capabilities.outputScopeSelection`이 없으므로 선택 UI를 숨긴다.

## 현재 적용 범위와 다음 단계

1단계는 판별과 설정 저장이다. **`outputScopeCropping=false`**이며 화면에도
"범위 설정만 저장됩니다. 현재 미리보기와 파일은 전신입니다."를 표시한다.
2단계 관측 영역 검색은 아래와 같다. 카메라 크롭, FBX 메시 절단은 아직 구현하지 않았다.
현재 BVH/FBX export 파라미터나 converter의 고정된 worker 계약에는 범위를 넣지 않는다.

다음 출력 단계는 full rig를 유지한
semantic/rest-space 메시 크롭을 미리보기와 FBX에 동일하게 적용한다.
converter 버전·manifest·캐시 키에 범위를 명시해야 실제 출력 기능을 켤 수 있다.

## 2단계: 관측 영역 검색

출력 설정과 관측 관절은 별개다. 수동으로 반신/두상을 선택해도 추출 관절을
새로 만들거나 후보 순위를 바꾸지 않는다. 새 분석 요청부터 다음 검색 경로를 쓴다.

| 관측 | 검색/출력 정책 |
|---|---|
| 골반·몸통 기준 유효 | 기존 hip/torso 피처와 거리·A/B 검사 유지 |
| 골반 기준 불가, 다리 말단 비관측, 양쪽 어깨와 팔 선분 2개 이상 유효 | `coverage_class=upper_only`, 어깨 기준 검색 |
| 어깨만 있거나 한 팔 선분뿐, 어깨 투영 붕괴, 소유권 모호 | 자동 후보를 제공하지 않는 기존 복구/폴백 |
| 인물별 두상 판별 | `head_search_unsupported`, 후보 없음; 혼합 컷의 다른 인물은 검색 |
| 얽힘 관계 | 기존 세트 검색 미지원 폴백 유지 |

`src/partial_pose.py`는 관측 상체의 기하 검사와 `shoulder_frame` 변환을 담당한다.
양 어깨 중점으로 이동하고 어깨 폭으로 나눈다. COCO 5–10번만 비교한다.
쿼리와 저장된 라이브러리 투영 모두 같은 함수를 거치므로 기존 hip/torso
정규화의 이동·스케일이 상쇄된다. DB `FEATURE_VERSION=1`과 BVH는 그대로다.
사용한 변환은 `quality_trace.normalization=shoulder_width_v1`로 구분한다.

안전 기준:

- 점수 임계/유한 좌표, 양 어깨, 팔 선분 2개 이상을 요구한다. 팔 하나 전체 또는
  양쪽 상완이 필요하며, 얼굴의 가상 Head 점은 검색에 쓰지 않는다.
- 관측된 무릎/발목이 있으면 새 경로로 우회하지 않는다. 전신 검사를 유지한다.
- 어깨가 소유 인물 박스를 벗어나거나 다른 인물 중심 영역과 겹치면 거부한다.
  다른 인물 박스의 팔, 소유 박스 바깥 팔은 제외한다. 기존 배정 모호성도 유지한다.
- 팔 선분 길이가 어깨 폭의 0.05–3배 범위를 벗어나거나 인접 선분 비가 기존
  `skeleton_adjacent_segment_ratio_max`를 넘으면 제외한다. 웹툰 과장을 위한 넓은
  탐색 기준이며 해부학적 정확도 판정이 아니다. 어깨가 겹치는 완전 측면은 제한된다.
- 격리 포즈 제외·pose family 중복 제거·안정 정렬은 기존 검색과 동일하다.
- 새 거리 `upper_pos`는 전신 임계값과 비교하지 않는다. 항상 `confidence=low`,
  `confidence_threshold=null`, `refine_allowed=false`, `refinable_limbs=[]`,
  반환 scores=0. v1/v2와 구 클라이언트 모두 자동 보정을 우회할 수 없다.
- BFF/앱은 `upper_only`를 참고 후보로 표시한다. 두상은
  `candidateShortfallReason=HEAD_SEARCH_UNSUPPORTED`로 인물별 안내한다.

두상 방향 검색은 미구현이다. 현재 BVH COCO 얼굴 5점은 Head의 대용 좌표여서
실제 눈·코·귀와 대칭적으로 비교할 수 없다. 별도 얼굴 관측 및 3D 머리 방향
색인을 검증한 뒤 켜야 한다. 현재 상체 후보가 얼굴 표정·시선을 맞춘다고 약속하지 않는다.

### 2단계 검증

`tests/test_partial_search.py`: 좌표 이동/크기 불변, 숨은 골반·다리 독립성,
masked 손목 독립성, 실제 관측 부족·소유권·길이 이상·퇴화 투영 제외,
격리/family 처리, 혼합 구도, v1/v2 보정 차단, HTTP 직렬화.
기존 전신 검색/개수 신뢰도/refine·cascade 회귀도 검사한다.

로컬 캐시 439장/704명에서는 소유권 박스를 제외한 기하 검사상 상체 32명이
추가 검색 대상이 됐고 모두 유한한 후보를 반환했다. 라이브러리 1,568포즈/
6,272투영에서 조회 중앙값 3.58ms, 샘플 투영의 변환 대칭성 121/121 통과.
이는 **캐시 기반 검색 가능성 검사**이며 실 VLM 재추론·소유권 확인·시각적
정답률 평가가 아니다. 사용자 원본/식별자는 보고서에 내보내지 않았다.
집계는 `data/dev/body-scope/partial-search-eval.json`에 있다.

검증 결과: Python 관련 196개, BFF 관련 35개, 앱 포즈 기능 62개 통과.
기존 refine의 2초 deadline 테스트 1건은 병렬 검사 부하 중 timeout으로 실패한 뒤
단독 실행에서 1.28초에 통과했다. 테스트 기준이나 구현의 deadline은 바꾸지 않았다.
앱/BFF TypeScript, 변경 영역 ESLint, Python compileall, 앱 Vite 빌드 통과
(Vite의 500kB 초과 번들 안내는 남아 있다).
실제 후보 UI 컴포넌트의 격리 Mock 화면에서 상체·두상 안내를 확인했으며,
720×460 플로팅 바 스크린샷은 `data/dev/body-scope/partial-search-bar.png`다.
운영 배포 및 실제 PostgreSQL/VLM/네이티브 앱 검증은 포함하지 않았다.

## 검증

- `tests/test_body_scope.py`: 누락/오류/혼합 구도, 슬롯 정렬, 조기 종료 API,
  범위 변경 전후 검색 순위/거리/refine 정책 유지.
- BFF `src/output-scope/outputScope.test.ts`: 자동↔수동, 매핑, 소유권/입력 검증,
  저장 왕복·다른 인물 값 보존·행 잠금 SQL. 저장소 테스트는 DB client 대역 사용.
- 앱 `OutputScopeSelect.test.tsx`: 인물별/작업별 cache 격리, 자동 복원, 오류 재시도,
  저장 중 잠금, 구 BFF 게이트. HTTP adapter 기존 계약 테스트 병행.
- 실 VLM 판별 정확도, 운영 PostgreSQL, Tauri 네이티브·macOS는 별도 실환경 검증 대상.

2026-10-02 로컬 검증: Python 105개, BFF 42개, 앱 포즈 기능 59개 통과.
HTTP 공통 어댑터 경계에서 JSON 이중 직렬화를 수정한 뒤 관련 6개 테스트를 재확인했다.
앱/BFF TypeScript 검사, 앱 변경 영역 ESLint와 Vite 빌드, Python compileall 통과.
Windows 브라우저에서 실제 후보 화면 컴포넌트를 독립 Mock fixture로 구동해
앱 최소 크기 960×640과 플로팅 바 720×460에서 인물별 변경/자동 복원/인물 이동을 확인했다.
검수용 파일·프로세스는 정리했고 스크린샷은
`data/dev/body-scope/output-scope-ui.png`, `output-scope-bar.png`에 보관했다.
운영 배포·DB 변경·S3 업로드는 하지 않았다.
