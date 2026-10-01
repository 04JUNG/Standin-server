# 인물별 출력 범위 — 1단계

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
흉상·두상 조기 종료에서도 인물/박스/범위 정보를 반환한다. 후보는 빈 배열,
관절은 null, refine은 금지다. 검색 미지원을 사람 미검출로 표현하지 않는다.

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

이번 단계는 판별과 설정 저장이다. **`outputScopeCropping=false`**이며 화면에도
"범위 설정만 저장됩니다. 현재 미리보기와 파일은 전신입니다."를 표시한다.
부분 구도 검색, 카메라 크롭, FBX 메시 절단은 구현하지 않았다.
현재 BVH/FBX export 파라미터나 converter의 고정된 worker 계약에는 범위를 넣지 않는다.

다음 단계는 관측 영역 검색과 흉상/두상 경로를 따로 검증하고, 이후 full rig를 유지한
semantic/rest-space 메시 크롭을 미리보기와 FBX에 동일하게 적용한다.
converter 버전·manifest·캐시 키에 범위를 명시해야 실제 출력 기능을 켤 수 있다.

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
