# BFF·앱 인계: 동적라우팅 3D 미리보기

계약: `api/pose_presentation.py`, `api/models.py::PersonOut`, `docs/API_CONTRACT.md`의
`dynamic-pose-presentation-v1`. 서버 수정만으로 구 앱이 새 목록을 읽지는 않는다.
BFF PR #83·앱 PR #88의 기존 연결은 유지하면서 새 버전을 우선 처리한다.

## 소비 순서

1. 지원하는 `people[i].pose_presentation.version`이면 이 목록·상태가 표시 기준이다.
2. 미포함/null인 구 서버만 기존 `dynamic_search` 연결을 사용한다. 그것도 없으면 legacy
   `candidates`다. 알 수 없는 새 버전/검색 오류를 다른 목록으로 덮지 않는다.
3. `person_index=people[i].index`. `library_candidates`는 검색 조건 통과,
   `review_candidates`는 근접/기하/구도 참고, `generated_candidates`는 생성 검수다.
   `basic_model`은 아직 렌더되지 않은 기본 인체 계획이며 포즈 후보가 아니다.
4. 그룹 안의 rank를 유지한다. `candidate_id`는 인물·BVH 바이트·시점·출력 범위에 바인딩된
   식별자이며 job 접근권한을 대신하지 않는다. 생성 `pose_id=null`이고 library lookup은 불가다.

| 표시 상태 | 추가 확인 |
| --- | --- |
| ready | 검색 후보 표시; 자동 적용/export 승인 아님 |
| review_ready | 참고·생성 검수 표시, 정상 검색 정답으로 집계 금지 |
| basic_reference | 출력 계획 표시/별도 기본 모델 렌더 구현 필요 |
| no_result | search_status와 generation_status/reason도 표시 |
| generation_failed | generation_status가 error/rejected/timeout/unsupported인지 구분 |
| partial_error/error | 남은 그룹만 표시하고 오류/미리보기 불가 안내; 생성으로 덮지 않음 |
| blocked/undetermined/not_run | 원래 검색 정책/근거 부족/미실행 사유 유지 |

## 파일과 카메라

- library_bvh: 고정 추론 서버의 내부 `asset.bvh_url`을 BFF가 인증된 프록시로 중계.
  원본 SHA-256을 검증한다. 프론트에 추론 내부 주소나 임의 URL fetch 권한을 주지 않는다.
- inline_bvh: `encoding=utf-8`, `asset.bvh`의 UTF-8 바이트로 해시 검증. 생성 pose_id로
  라이브러리 조회하지 않는다. 기존 분석 결과 접근·삭제·보관 기간 정책 안에서만 보관/재사용.
- library URL은 서명 URL/TTL 링크가 아니다. 재시작 후 같은 파일·정책이 유지될 때만 유효.
  생성 본문은 서버 임시파일과 무관하다. 새 보관 기간·공개 파일 서버·라이브러리 등록은 없다.
- 조회/해시 실패는 미리보기 오류로 표시하고 다른 BVH를 대신 쓰지 않는다. 기존 검사 근거도
  다른 체형·재베이킹 손 프리셋 등 바뀐 변환 결과의 안전 인증으로 재사용하지 않는다.
- `render_camera`는 `pose-preview-camera-v1`이며 정규화 투영 수식/행렬/단위/기준점을 제공한다.
  BVH에 표시 카메라 회전을 굽지 않았다. source root/관절 회전은 보존하고 표시 회전만 한 번 적용.
  fixed_117만 검색 방향·투영을 재현하며 픽셀 구도 적합까지 인증하지 않는다.
  semantic_preview/default는 참고 시점이다. 원본 `camera`나 legacy CandidateCamera와 혼용 금지.
- output_scope를 적용한다. unobserved_body_is_default는 true/false/null 원문 의미를 보존한다.
  정면 썸네일은 검색 시점 이미지가 아니며 null이어도 3D BVH는 정상일 수 있다.
- 모든 신규 후보는 lookup_allowed=true(미리보기만), review_required=true,
  auto_apply_allowed/refine_allowed/export_allowed/release_allowed/actual_mesh_validated=false다.

## 사례·실제 파일 전달

공개 저장소 규칙상 라이브러리 BVH/인라인 BVH JSON은 커밋하지 않는다. 내부 전달 ZIP은
`scripts/build_pose_presentation_examples.py`로 승인된 로컬 BVH 두 개에서 생성한다.
이는 실제 API 모델로 직렬화한 **오프라인 계약 사례**이며 새 Gemini 호출/실측 검색 결과가 아니다.
BVH는 실제 매핑된 인체 파일이며 파싱·FK·SHA-256을 검사한다. 앱 3D 뷰어 E2E/피부 안전 인증과는 다르다.

```bash
python scripts/build_pose_presentation_examples.py \
  --library /private/approved/library_pose.bvh \
  --generated /private/approved/generated_pose.bvh \
  --output /private/team-handoff/pose-presentation
```

출력 폴더/ZIP:

- `01-library.json`: 실제 BVH 경로·SHA, 검색 카메라·scope와 검색 후보.
- `02-near-only.json`: 정식 후보 없이 근접 참고.
- `03-generated.json`: UTF-8 BVH 본문으로 미리보기 가능한 생성 검수.
- `04-empty.json`: 성공 빈 경계 사례(현재 생성기가 실제로 no_result를 냈다는 기록 아님).
- `05-search-error.json`, `06-generation-error.json`, `07-generation-rejected.json`.
- `08-mixed-people.json`: 세 인물의 검색/생성/오류 혼합.
- `09-basic-plan.json`: 어깨~머리 기본 인체 계획; 가짜 artifact URL 없음.
- `10-legacy-null.json`, `11-legacy-omitted.json`: 구버전 호환.
- `assets/library.bvh`, `assets/generated.bvh`: 앞 JSON과 해시가 일치하는 실제 입력.
- `openapi.json`: CutResultOut JSON Schema(HTTP path 전체 OpenAPI는 서버 `/openapi.json`).
- `manifest.json`: 모든 파일 해시·기준 커밋·dirty 여부·오프라인 사례 표시.

오프라인 뷰어는 01의 내부 URL을 `assets/library.bvh`로 매핑할 수 있다. 이것은 현재 staging의
같은 URL이 같은 파일을 준다는 보증이 아니다. 배포 환경에서는 실제 GET 바이트를 반드시 대조한다.
ZIP에는 raw BVH를 포함하므로 팀 내부 전송만 허용하고 공개 PR/채팅 링크에 업로드하지 않는다.

## 확인 및 배포 순서

추론 feature branch → develop PR → 팀원 계약 확인/병합 → 추론·BFF·앱 순차 배포 및 E2E.
이 PR은 BFF/앱 코드, 상하체 검색·합성, 라이브러리 업로드, 권한 승격을 변경하지 않는다.
API factory는 자연어 0.6·기하 0.4다. 직접 생성하는 과거 실험 기본값 0.35는 보존한다.
`/healthz.dynamic_routing`과 배포 preflight의 ranking/semantic_weight/geometry_weight로 실제 값을 확인한다.

완료 기준은 같은 러프의 **인물·그룹별 서버 수=앱 표시 수**, library/inline 실제 3D 입력 모두
열림, 해시·카메라·scope 확인, 검수 후보 자동 적용/refine/export 계속 차단이다.
인계 JSON/HTTP 단위 테스트만으로 앱 E2E까지 완료했다고 보고하지 않는다.
