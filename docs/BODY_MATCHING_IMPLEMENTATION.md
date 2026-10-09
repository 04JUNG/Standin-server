# 러프 체형 감지·자동선택 파이프라인

## 범위와 상태

러프 PNG → 기존 인물 영역/포즈 추출 → 인물 crop별 체형 관측 → 승인된 FBX 카탈로그 순위 계산 → 인물별 자동선택 결과를 반환한다. `feat/body-selection`의 이번 PR 범위는 이 추론 구간이다.

- 기본 `BODY_MATCHING_MODE=off`. 기존 포즈 검색과 refine 정책은 유지한다.
- 출력은 `CutResult.body_matching`, HTTP에서는 선택적 `/analyze.body_matching`이다.
- UI·BFF의 수동 선택·영속 저장·converter·Top-5 새 이미지 렌더·Export를 실행하지 않는다.
- 이전 렌더 통합 작업은 commit `b9cfb72` 및 로컬 `codex/body-render-integration-snapshot`에 보존한다. 이 PR의 최종 diff에서 제외했다.
- 운영 카탈로그는 빈 목록이다. 평가용 9종과 정규화 중인 모델을 승인 없이 활성화하지 않는다.

## 실행 흐름

1. 기존 `Pipeline._process_cut`에서 인물·스켈레톤·Top-K를 생성한다.
2. `process_cut`은 off가 아닐 때 body sidecar를 추가한다. 기존 포즈 배열·순서·카메라를 수정하지 않는다.
3. 결과의 인물 박스를 이미지 범위로 자른다. 인물 소유권이 불명확하거나 박스가 크게 겹치면 관측을 생략한다.
4. Gemini는 인물별로 라벨이 붙은 crop을 한 번의 batch로 받는다. 관절 좌표·실제 키/근육량·나이·생물학적 성별·FBX ID를 생성하지 않는다.
5. 가시성, 잘림, 단축, 소유권 조건으로 쓸 수 없는 체형 속성을 마스킹한다.
6. 승인/해시/현재 pose 지원 조건을 통과한 카탈로그 후보를 대상으로 같은 속성 축을 비교한다.
7. 형태 점수와 디자인 단서로 자동선택한다. 근거가 없으면 카탈로그의 기본값 또는 결정적인 품질/ID 순서를 사용하며 `auto_default`를 명시한다.
8. 오류는 체형 측 unavailable/기본값으로 처리하고 성공한 포즈 결과는 반환한다.

## 관측과 선택

관측 축: 등신 범주 h3~h8, 체격, 프레임 폭, 팔다리 비율, 보이는 근육 표현 0~4, 부드러운 볼륨 0~4, 볼륨 분포. 이는 정밀 신체 측정이 아닌 그림의 형태 단서다.

- 부위별 coverage(head/torso/arms/legs)를 보고 가려진 형태를 제외한다.
- 잘린/단축된 인물에서 등신을 아동형 근거로 사용하지 않는다.
- 남성형/여성형/중성형 그림 디자인은 `presentation`으로 분리한다. 근육량이나 마름을 성별 단서로 치환하지 않는다.
- 뚜렷한 얼굴/윤곽 디자인 단서가 있으면 동일 계열·공용 자산을 우선 후보로 제한한다. 머리/의상만의 단서는 weak로 낮추고 동점/기본값 선택에만 사용한다.
- 호환 계열 자산이 없으면 `matching_style_unavailable` 진단을 남긴다. 반대 계열 선택을 계열 검출 성공이라고 표시하지 않는다.
- 기하 비교는 동일 pose/view/BVH hash와 목표 FBX hash의 투영 자료가 모든 후보에 있을 때만 쓴다. 없으면 속성 선택으로 남긴다.
- `rank_score`는 미보정 거리이며 일치 확률이 아니다. `acceptance_probability=null`이다.

## 하드코딩 점검

| 구분 | 현재 상태 |
|---|---|
| 모델 개수·ID·파일 경로·남성 기본값 | 런타임 코드에 고정하지 않음. 카탈로그 JSON에서 읽음 |
| 관측 provider/model/catalog/timeout/max_people | 환경변수 또는 생성자 인자로 변경 가능 |
| 품질 우선순위·계열별 기본 체형 | 카탈로그에서 설정 |
| 인물 겹침 > 작은 박스 면적의 0.35 | 코드의 보수적 소유권 게이트. 실데이터로 보정할 알고리즘 상수 |
| uncertain 가중치 0.25, 단축 비율 축 가중치 0.25 | 선택 알고리즘 상수. 정확도 보정값이 아님 |
| crop 최대 768px, 관측 캐시 기본 128개 | 비용/메모리 상한. 모델 목록과 무관 |
| 속성 어휘, COCO17 차원, SHA256 규격 | 버전 관리되는 스키마·데이터 계약 |
| 9종/`male-base`/특정 평가 폴더 고정 스크립트 | 과거 실험 코드이므로 이번 PR에서 제외 (`b9cfb72`에 보존) |

임의 ID와 1·9·12개 자산 카탈로그, 기본값 변경·자산 추가를 테스트한다. 모델 파일을 추가하는 것만으로는 승인되지 않으며 카탈로그 등록과 해당 포즈 QA가 필요하다.

## 설정과 실행

```bash
# API의 기존 동작 유지
BODY_MATCHING_MODE=off uvicorn api.app:app

# 관측·추천만 기록; applied_body_id는 null
BODY_MATCHING_MODE=shadow BODY_VLM_PROVIDER=gemini \
BODY_VLM_MODEL=gemini-flash-lite-latest \
BODY_CATALOG_PATH=config/body_catalog.v1.json uvicorn api.app:app

# 명시적인 offline mock 데모: 모델/실제 VLM 없이 실행
python scripts/match_body.py rough.png --synthetic --provider mock \
  --catalog config/body_catalog.v1.json --output /tmp/body-result.json
```

실제 Gemini에는 `GEMINI_API_KEY`가 필요하다. crop 외 원본 전체 패널은 전송하지 않는다. mock/provider 초기화 실패는 시각 근거로 주장하지 않고 provider_actual/is_mock/provider_error에 기록한다.

환경변수:

- `BODY_MATCHING_MODE`: off/shadow/auto (기본 off)
- `BODY_CATALOG_PATH`: 기본 config/body_catalog.v1.json
- `BODY_VLM_PROVIDER`: mock/gemini (기본 mock)
- `BODY_VLM_MODEL`: 기본 gemini-2.5-flash; 평가 당시 alias는 별도 기록
- `BODY_TIMEOUT_SECONDS`: 기본 12, 양의 유한값
- `BODY_MAX_PEOPLE`: 기본 8, 1~20

## 카탈로그 관리

예시는 `config/body_catalog.example.json`이며 draft 상태다.

```bash
python scripts/body_catalog.py config/body_catalog.v1.json validate
python scripts/body_catalog.py catalog.json add --asset asset.json --catalog-version my.v2
python scripts/body_catalog.py catalog.json approve --body-id my-body --qa reviewed-qa.json --catalog-version my.v3
python scripts/import_body_projections.py catalog.json projections.json --catalog-version my.v4
```

eligible에는 body/rig/measurement version, FBX·QA report hash, 정확한 supported_pose_ids가 필요하다. 누락/불일치 자산은 선택에서 제외된다. 새 catalog는 매 컷 읽으므로 모델 추가에 서버 재시작이 필요하지 않다.

## 실제 러프 평가 이력과 한계

2026-10-07 개발셋: 러프 17컷·28인물, 평가 체형 9종(통통형 2종 제외). Gemini 요청 alias `gemini-flash-lite-latest`, 실제 응답 모델 `gemini-3.5-flash-lite`. 기존 RTMPose/cascade 인물·Top-5는 hash 검증 후 재사용했다.

- 사용자가 지적한 남성형/여성형 오류 8건에서 presentation 보완 후 5건 개선, 3건 미해결.
- 동일한 새 관측에서 presentation 신호를 제거한 비교는 0/8이었다.
- 알려진 오류를 개선한 개발셋이므로 독립 정확도나 전체 BodyHit@1이 아니다.
- 후보 이미지 직접 비교는 순서 변경에 안정적이지 않아 이번 런타임/PR에서 제외했다.
- 원본 이미지·FBX·provider raw 응답·비밀값은 PR에 포함하지 않는다. 3건에 대한 원본 전체 패널 전송도 수행하지 않았다.
- 운영 활성화 전 새로운 작가/캐릭터의 독립 검증셋과 승인된 자산이 필요하다.

## 검증

```bash
python tests/test_body_matching.py
python tests/test_body_matching_v2.py
python tests/test_body_presentation.py
python tests/test_body_api_contract.py
python tests/test_body_pipeline.py
python tests/test_smoke.py
python -m pytest tests/ --ignore=tests/converter -q
python -m compileall src api scripts pose_curation
python scripts/export_body_api_contract.py
```

[HTTP 필드·실패 계약](BODY_PIPELINE_API_CONTRACT.md). BFF·클라이언트 연결 전 필수 충돌 검토는 마스터 독스의 `체형 선택/필수_수정사항_2026-10-07.md`를 따른다. 이번 PR은 그 제품 연동의 완료를 주장하지 않는다.

2026-10-08 보완: 명확한 디자인 단서로 체형군을 좁힌 뒤 기본 체격을 선택하면 `auto_presentation_default`로 구분한다. BFF가 근거 없는 `auto_default`와 혼동해 반대 체형 기본값으로 되돌리지 않도록 한다. 상세 계약은 BODY_PIPELINE_API_CONTRACT.md를 따른다.
