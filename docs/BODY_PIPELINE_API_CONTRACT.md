# 체형 감지·자동선택 API 계약 v1

## 범위

이번 구현의 HTTP 변경은 `POST /analyze`의 선택적 `body_matching` 필드뿐이다. 컷 이미지 입력과 기존 `people[].candidates` 결과는 유지한다. 렌더·수동 변경·캐릭터 저장·Export endpoint는 추가하지 않는다.

`BODY_MATCHING_MODE=off`(기본)에서는 `{}`. `shadow`에서는 관측·추천만 반환하고 적용 요청은 하지 않는다. `auto`에서는 자동선택한 체형을 외부 소비자가 사용할 수 있도록 참조를 반환한다. BFF나 클라이언트의 기존 수동 선택을 직접 변경하지 않는다.

Python 응답은 `BodyMatchingOut | EmptyBodyMatchingOut`이다. JSON은 object이며 코어 `CutResult.body_matching`은 dict다. Python 직접 호출자는 속성 또는 `.model_dump()`로 접근한다.

## 응답 필드

| 필드 | 의미 |
|---|---|
| schema_version | body-match.v1 |
| mode | shadow / auto |
| status | ok / partial / unavailable / not_applicable |
| input_sha256 | PIL RGB 크기 문자열+픽셀 bytes의 hash. PNG 파일 hash와 다름 |
| catalog_version / catalog_sha256 | 선택에 사용한 카탈로그 snapshot |
| provider_requested / provider_actual / model / is_mock | 요청/실제 adapter 정보. model은 요청 모델 이름이며 alias가 가리킨 실제 provider model version과 같다는 보장은 없음 |
| prompt_version / elapsed_ms | 관측 프롬프트 버전·처리 시간 |
| people[].person_index | 기존 people[].index와 결합 |
| people[].person_id | input_sha256:p{person_index}. 현재 분석 범위의 ID이며 장기 캐릭터 ID가 아님 |
| observations | 속성 값·가시성·근거·coverage·소유권·캐릭터 디자인 표현 |
| auto_body_id / selected_asset | 추천 체형 및 body/rig/measurement 버전·asset hash |
| applied_body_id | auto에서 추천 ID, shadow에서 null. 외부 적용 대상으로 선택됐다는 뜻이며 메시/썸네일 적용 완료를 뜻하지 않음 |
| selection_source | auto_best_effort / auto_presentation_default / auto_default. presentation_default는 명확한 인물 디자인 단서에 맞는 체형군 안에서 기본 체격을 선택. auto_default는 시각 근거 없는 기본값 |
| diagnostic / reason_codes | 근거 부족·provider 실패·자산 없음 등 |
| candidates / tied_body_ids | 상위 체형 최대 3개 및 동점 목록. 포즈 Top-5와 별개 |
| presentation_selection | 디자인 계열 관측으로 후보 제한/동점/기본값을 결정한 기록 |
| pose_bindings | 기존 Top-K와 같은 순서의 pose_id/view/pose_sha256(nullable) |
| render_required | auto에서 체형과 pose가 선택됐으므로 외부 렌더가 필요하다는 신호 |
| rendering_executed | 항상 false |
| acceptance_probability | 항상 null. rank_score는 미보정 거리이며 일치율이 아님 |

문자열 어휘/정수 범위/인물 및 자산 일관성은 `api/body_models.py`와 생성 JSON Schema가 정본이다. body metadata의 추가 진단 필드는 확장될 수 있다. 선택 자산은 다운로드 URL이나 서버의 로컬 파일 경로를 반환하지 않는다.

## 보존 조건과 실패 처리

- 기존 pose_id/view/후보 순위·개수·검색·refine 정책을 수정하지 않는다.
- 사람별 body pose_bindings가 기존 후보와 다르거나 인물 ID가 어긋나면 `body_contract_invalid` unavailable로 축소한다. 기존 people/포즈 응답은 유지한다.
- core 외 route에는 not_applicable. 후보 없는 인물에는 자동 체형을 적용하지 않는다.
- 관측 timeout/파싱 오류는 provider_error를 기록하고 승인된 기본값이 있으면 auto_default를 쓴다. 실패를 시각 관측 성공으로 바꾸지 않는다.
- 카탈로그 파일 오류는 unavailable, 빈/부적합 자산 목록은 인물별 unavailable 및 컷 partial.
- draft/qa_pending/retired 자산, FBX/QA hash 불일치, 지원되지 않은 pose는 선택하지 않는다.
- off는 체형 VLM·카탈로그 로드를 실행하지 않는다. body 단계 예외는 성공한 포즈 결과를 파괴하지 않는다.

## 파일·검증

- [응답 JSON Schema](contracts/body/BodyMatchingOut.schema.json)
- [auto 예제](contracts/body/body.auto.example.json)
- [shadow 예제](contracts/body/body.shadow.example.json)
- [off 예제](contracts/body/body.off.example.json)

예제는 합성 계약 fixture이며 실제 러프 정확도·사용 가능한 FBX·렌더 증거가 아니다. `python scripts/export_body_api_contract.py`로 생성한다. 전체 OpenAPI는 실행 서버 `/openapi.json` 및 `/docs`에서 확인한다.

[구현·설정·하드코딩 점검·실제 평가 한계](BODY_MATCHING_IMPLEMENTATION.md).

### 디자인 단서만 관측된 경우

`auto_presentation_default`는 소유권이 명확하고 visible 여성형/남성형 디자인 단서로 compatible_candidates를 구성했지만 비교 가능한 체격·투영 점수가 없을 때 사용한다. 체격·근육·등신을 감지했다는 뜻이 아니며 일치율은 계속 null이다. 약한 단서·헤어/의상만의 단서·소유권 불명·호환 체형 부재는 이 출처로 승격하지 않는다. BFF는 이 출처와 presentation 진단을 검증하여 선택을 유지한다. 사용자 manual/fixed_default 우선순위는 유지한다. 기존 BFF는 새 출처를 unavailable로 처리하므로 BFF 지원 후 감지 변경을 활성화한다.
