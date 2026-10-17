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
| auto_body_id / selected_asset | 자동 선택한 체형 및 body/rig/measurement 버전·asset hash. 보류 시 둘 다 null |
| suggested_body_id / suggested_asset | 적용 정책과 분리된 후보 제안. BFF 자동 선택으로 소비하지 않음 |
| decision | auto_apply / presentation_default / suggest_only / insufficient_evidence / asset_unavailable |
| policy_version / application_policy | 자동 적용 허용·명확한 축·독립 근거 그룹·보류 사유. calibrated=false, acceptance_probability=null |
| applied_body_id | auto에서 추천 ID, shadow에서 null. 외부 적용 대상으로 선택됐다는 뜻이며 메시/썸네일 적용 완료를 뜻하지 않음 |
| selection_source | auto_best_effort / auto_presentation_default / auto_default. presentation_default는 명확한 인물 디자인 단서에 맞는 체형군 안에서 기본 체격을 선택. auto_default는 시각 근거 없는 기본값 제안이며 자동 적용하지 않음 |
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
- 관측 timeout/파싱 오류·약한 근거·동점은 auto/selected/applied를 null, selection_state를 unavailable, render_required를 false로 반환한다. 추천/기본값은 suggestion으로만 남긴다. 기존 BFF는 수동 선택 → fixed_default → 유효 auto → 사용자 기본값 → catalog 기본값 우선순위를 유지한다. 명시적 auto 요청에 유효 auto가 없으면 BFF의 기존 오류 동작을 유지한다.
- 새 decision과 policy의 허용 여부·출처·버전, suggestion ID와 자산 참조의 ID가 다르면 body_contract_invalid로 축소한다. 이전 decision 없는 v1 응답도 허용한다.
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

`auto_presentation_default`는 소유권이 명확하고 visible 여성형/남성형 디자인 단서로 compatible_candidates를 구성했지만 비교 가능한 체격·투영 점수가 없을 때 사용한다. 체격·근육·등신을 감지했다는 뜻이 아니며 일치율은 계속 null이다. 약한 단서·헤어/의상만의 단서·소유권 불명·호환 체형 부재는 이 출처로 승격하지 않는다. BFF는 이 출처와 presentation 진단을 검증하여 선택을 유지한다. 사용자 manual/fixed_default 우선순위는 유지한다. 2026-10-16 확인한 스테이징 BFF(`ee0ba879`)는 이 출처를 지원한다. 이 경로는 `decision=presentation_default`, `application_policy.allowed=false`로 구분하며 체형 판독 성공으로 계산하지 않는다.

### 형태 근거와 적용 정책 (P1)

`body-attributes.v5.regional-evidence` / `body-observation.v5.regional-evidence`는 부위별 `representation`과 `surface_detail`, 축별 `evidence_regions`를 관측한다. 새 provider 스키마는 이 필드를 요구하지만 이전 관측 payload는 읽을 수 있다. coverage/representation 없는 legacy 관측은 형태 자동 적용을 허용하지 않는다.

coverage가 visible이고 representation이 outline/shaded인 **동일 부위**만 명확한 형태 근거다. 스틱 라인·숨은 볼륨·잘리거나 단축된 등신을 제외한다. 팔 윤곽만 없는 전신에서 등신을 버리지는 않는다. 세부 근육선을 생략한 그림의 muscle=0은 미상 처리하며, 보이는 외곽 근육 볼륨은 보존한다.

`body-apply.structural-evidence.v2`는 최소 2개 독립 근거 그룹과 전체/명확한 근거 순위의 동일한 유일한 1위를 요구한다. 같은 윤곽의 여러 라벨은 1그룹이다. 명시적인 torso 프레임과 arms 볼륨처럼 서로 다른 부위의 근거는 2그룹이 될 수 있다. 투영 가설별 동점에 1위가 포함되면 허용하되, 다른 후보를 단독 지지하는 모순은 보류한다. 거리나 점수 차이는 확률·보정된 신뢰도로 표시하지 않는다.

[최신 20종 명세·리뷰 결과·BFF 회귀 검증](BODY_SELECTION/P0_P1_REVIEW.md).

## VLM 대분류 추천 전략 (coarse)

`BODY_CLASSIFICATION_STRATEGY=coarse`는 남녀 계열을 먼저 정하고 해당 계열 BASE를 기본값으로 추천한다. 명확한 근육·체격·비율 특징만 프리셋을 변경한다. `people[].coarse_classification`은 판독 결과, `people[].coarse_recommendation`은 모든 인물의 추천 ID·추천 방식·대안·이유다. 기존 자동 적용용 `candidates=[]`, `selected_asset/suggested_asset/auto_body_id/applied_body_id=null`을 유지하고 렌더를 실행하지 않는다. UI는 추천을 `coarse_recommendation.recommended_body_id`에서 읽어야 한다. 기본 legacy 전략의 계약은 바뀌지 않는다. [규칙·실제 28명 평가·활성화 예시](BODY_SELECTION/COARSE_RECOMMENDATION.md).


## 첫 호출 통합 전략

`integrated`는 첫 `VLM.analyze`의 인물별 관측을 재사용하며 추가 체형 API 호출·crop 전송·계열 재검토를 하지 않는다. 추천 정책과 20종 카탈로그는 coarse와 공유한다. `classification_strategy=integrated`, `prompt_version=body-integrated.v3.compact`; provider/model은 별도 BODY_VLM_*가 아닌 주 VLM 출처다. `observations.crop_box=null`은 crop을 생성하지 않았다는 뜻이다. `elapsed_ms`는 체형 후처리만 측정하며 첫 VLM 시간이나 전체 분석 시간이 아니다.

이 전략은 `auto`에서도 추천만 반환한다. 자동 적용·렌더·BFF 기본값 우선순위를 변경하지 않는다. 원본 VLM 슬롯에 체형을 연결하고 출력 인물 순서로 옮긴다. 검출기에서 추가된 슬롯처럼 연결 근거가 없으면 unknown 관측과 명시적 기본 추천을 반환한다. 연결 불명인 관측을 가까운 다른 사람에게 전용하지 않는다. 얼굴/skip 경로에도 descriptor가 있으면 기본 추천을 제공할 수 있다.

`VLM_BODY_THINKING_LEVEL`은 Gemini 통합 첫 호출의 선택적 추론 설정이다. 빈 값은 모델 기본을 유지하며 지원하지 않는 값은 provider 오류가 된다. 별도 BODY_TIMEOUT_SECONDS 대신 기존 주 VLM 요청·총 예산과 재시도 설정을 공유한다. 전신 포즈 설명 등 기존 추가 호출은 유지한다. [실측 및 제한](BODY_SELECTION/INTEGRATED_FIRST_CALL.md).
