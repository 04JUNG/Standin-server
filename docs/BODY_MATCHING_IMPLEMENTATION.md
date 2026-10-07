# 러프 체형 탐지·자동 결정 v1 구현

작성: 2026-10-06. 근거: [파이프라인 전략 §7](NEXT_SPRINT/BODY_MATCHING_PIPELINE_STRATEGY_2026-10-02.md), [팀 3D 인체 분류 지침](../artifacts/body-model-test/2026-10-05-adults/team-body-classification-guide.md).

## 구현 범위

`기존 인물 추출·포즈 Top-5 → 인물 crop의 VLM 속성 + 관절 비율 → catalog 비교 → 인물당 FBX 하나 결정`을 구현했다. `/analyze`의 기존 `people[].candidates`를 그대로 두고 `body_matching`을 추가한다. 인물·포즈·카메라 순서를 바꾸지 않는다. `BODY_MATCHING_MODE=off`가 기본이며 배포 설정은 변경하지 않았다.

- **모델 확장:** 모델 이름/개수를 코드에 고정하지 않는다. JSON catalog를 원자적으로 갱신하면 다음 컷에서 읽는다. 기존 컷에는 당시 catalog hash·자산 버전이 남는다.
- **분류:** 등신 계열 h3~h8, 일곱 체형, 근육·연부 볼륨·분포·골격·사지 비율. 등신은 VLM의 거친 계열 판정이며 정밀 측정값이 아니다. 나이대는 자산 metadata로 보관하며 자동 필터로 쓰지 않는다. 캐릭터 디자인의 남성형·여성형 표현은 체격과 별도 관측하며, 명확한 단서가 있을 때 호환 모델 계열을 제한한다.
- **관측:** Gemini가 속성·가림·단축만 JSON으로 반환한다. 관절 좌표·cm·kg·body ID는 VLM에서 받지 않는다. 기존 COCO17의 유효 관절 마스크로 비율을 별도 계산한다.
- **선택:** 모든 실행 가능 후보에 공통으로 있는 관측 축만 비교한다. 투영 데이터가 있으면 같은 pose ID·view·BVH hash에서 얻은 target FBX 비율을 결합한다. 없는 투영은 점수에서 제외하고 이유를 남긴다. canonical 3D 폭을 2D 러프 폭과 비교하지 않는다.
- **출처:** 현재 점수는 보정 전이므로 `auto_best_effort` 또는 `auto_default`를 반환한다. `auto_match`·일치 확률·library_gap 판정 임계값은 실제 러프 보정 이후 승격 항목이다. 기본값도 먼저 적용할 수 있으나 탐지 성공으로 세지 않는다.
- **예외:** unknown과 0 구분, 의상에 가린 볼륨 제외, 겹친 인물 crop의 보수적 제외, 중복/누락된 VLM ID의 전체 batch 거부, provider 오류 후 기하 또는 기본값, 검수 자산 0개 시 명시적 unavailable.

FBX 렌더·리타게팅은 외부 변환기/뷰어 담당이다. 응답의 `render_required=true`는 렌더 주문 가능 상태이며 렌더 완료를 뜻하지 않는다. 이 구현은 `rendering_executed=false`를 반환한다. 캐릭터 저장·재사용은 전략서의 v2.0 후속 단계다.

## 코드 지도

| 경로 | 역할 |
|---|---|
| `src/experimental/body_matching/schema.py` | 속성 단일 어휘·응답 schema·엄격한 값 검증 |
| `observation.py` | Gemini/명시적 mock 어댑터, 공용 2D 투영 비율 함수 |
| `catalog.py` | catalog snapshot, FBX/QA hash 및 지원 포즈 검증 |
| `selection.py` | 공통 관측 축·가설 비교, 결정론적 순위와 기본값 |
| `service.py` | 인물별 crop·단일 batch·관측 캐시·결정 연결 |
| `src/pipeline.py` | 기존 결과 뒤에 선택적 sidecar 실행 |
| `scripts/body_catalog.py` | draft 추가·검수 승인·검증 |
| `scripts/import_body_projections.py` | 변환기 투영 자료 → 공용 비율 → catalog |
| `scripts/match_body.py` | PNG 입력으로 CLI 실행 |
| `tests/test_body_matching.py` | 탐지·선택·예외·기존 포즈 불변 계약 검증 |

## 실행

기존 VLM/포즈 실행 환경에 아래를 추가한다. Gemini SDK와 Pillow는 기존 requirements에 포함되어 있다. `GEMINI_API_KEY`는 기존 비밀 관리 경로로 설정한다.

```bash
BODY_MATCHING_MODE=auto \
BODY_VLM_PROVIDER=gemini \
BODY_CATALOG_PATH=config/body_catalog.v1.json \
uvicorn api.app:app --reload
```

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `BODY_MATCHING_MODE` | `off` | off / shadow / auto. shadow는 결정만 기록하고 applied_body_id=null |
| `BODY_CATALOG_PATH` | `config/body_catalog.v1.json` | 추가 중인 FBX catalog |
| `BODY_VLM_PROVIDER` | `mock` | mock / gemini. mock은 시각 속성을 만들지 않음 |
| `BODY_VLM_MODEL` | `gemini-2.5-flash` | 별도 체형 관측 모델. 기존 shot/count 모델과 독립 |
| `BODY_TIMEOUT_SECONDS` | `12` | VLM 요청 timeout, SDK 재시도 1회 시도만 설정 |
| `BODY_MAX_PEOPLE` | `8` | 컷당 VLM crop 상한, 최대 20. 초과 인물은 기하/기본값 |

인물당 포즈마다 재호출하지 않고 요청당 최대 한 번의 VLM batch를 사용한다. 관측 캐시는 최대 128개이며 입력 해시·인물 ID·crop 좌표·provider·model·prompt 버전으로 분리한다. 카탈로그 변경 시 VLM 관측은 재사용하고 선택은 새로 계산한다. 캐시는 프로세스 메모리이며 캐릭터 영속 저장 기능이 아니다.

```bash
# 실제 PNG + 기존 DB/인물 추출 설정 + 체형 감지
python scripts/match_body.py path/to/rough.png \
  --db data/poses.db --catalog config/body_catalog.v1.json \
  --provider gemini --output out/body-match.json

# API 키 없이 경로/오류 처리만 점검. 인물/포즈는 명시적 합성 입력
python scripts/match_body.py path/to/rough.png --synthetic --provider mock \
  --output out/body-match-mock.json
```

## 모델을 추가할 때

현재 배포용 catalog는 **빈 목록**이다. 제작 중인 unrigged 모델을 자동으로 검수 통과 처리하지 않았다. `config/body_catalog.example.json`은 작성 형식을 보여주는 draft 예시이며 실물 모델이 아니다. 실제 검수가 끝난 모델이 등록되면 선택이 시작된다. 모델 추가는 다음 순서다.

1. 예시의 `assets[0]` 형식으로 모델 하나의 JSON을 만든다. `body_id`는 유일한 안정 ID다. 실제 관측 가능한 형태를 기입하고 모르는 축은 null로 둔다. `age_design`과 `presentation_style`은 metadata이며 등신/체형에 묶지 않는다. FBX 경로는 **catalog 파일 기준 상대 경로**다.
2. draft로 추가한다. 승인된 기존 항목에 덮어쓰기하지 않는다.

```bash
python scripts/body_catalog.py config/body_catalog.v1.json add \
  --asset new-body.json --catalog-version bodies.v2
```

3. 실제 FBX의 리그·스킨·최종 메시를 검수한 담당자가 아래 QA 보고서를 만든다. `supported_pose_ids`는 검수된 실제 포즈의 ID이며 `*`로 전체 지원을 주장할 수 없다. report의 passed는 이 코드가 메시를 검사했다는 뜻이 아니라 외부 검수 결과다.

```json
{
  "status": "passed",
  "body_id": "regular-h7",
  "body_version": "v1",
  "asset_sha256": "실제 FBX의 64자리 SHA-256",
  "rig_version": "mixamo-v1",
  "supported_pose_ids": ["실제-pose-id-1", "실제-pose-id-2"]
}
```

```bash
python scripts/body_catalog.py config/body_catalog.v1.json approve \
  --body-id regular-h7 --qa path/to/qa.json --catalog-version bodies.v3
python scripts/body_catalog.py config/body_catalog.v1.json validate
```

CLI는 QA의 body/version/rig/hash 일치를 확인하고 QA 보고서 자체의 hash도 저장한다. 실행 시 파일이 없어졌거나 바뀌면 그 자산은 제외된다. 현재 컷의 모든 포즈를 지원하는 체형 중 하나를 선택한다. 지원 체형이 없으면 `asset_incompatible`이며 카드별로 다른 체형을 끼워 넣지 않는다. 기본값은 catalog의 `default_body_id`로 지정하며, 사용 불가하면 검수 우선순위와 안정 ID로 공통 대안을 고른다.

현재 버전에서는 catalog·자산 hash를 요청마다 검증한다. 큰 FBX가 늘어나면 읽기 비용을 측정해 버전 고정 snapshot 로더로 최적화할 수 있다. 최적화 때문에 변조·미완성 파일을 사용하지 않도록 해야 한다.

## 변환기에서 기하 자료 연결

형태 속성만 등록해도 VLM 기반 후보 비교가 실행된다. 기하 비교를 사용하려면 **그 FBX에 해당 포즈를 리타게팅한 뒤, 검색 후보와 같은 카메라로 투영한** COCO17을 변환기에서 전달한다. 정면 rest 포즈 한 장을 모든 포즈에 재사용하면 안 된다.

입력 JSON은 `body_id, body_version, asset_sha256, rig_version, retarget_version, renderer_version, projections`를 가진다. projections의 각 항목은 다음을 가진다.

- `pose_id`, `view`: 기존 후보와 같은 값.
- `pose_sha256`: 해당 원본 BVH의 SHA-256. 버전이 달라지면 비교에서 제외.
- `keypoints`: 실제 target FBX에서 투영한 17×2 좌표.
- `valid_joint_mask`: 해당 투영의 관측 가능한 관절 17개 boolean. 좌표나 마스크를 VLM이 만들지 않는다.

```bash
python scripts/import_body_projections.py config/body_catalog.v1.json \
  target-fbx-projections.json --catalog-version bodies.v4
```

쿼리와 catalog 모두 `geometry_ratios()`를 사용한다. 몸통 길이로 정규화한 어깨·골반 폭과 좌우 팔·다리 길이이며, 이동·균등 배율에 불변이다. 같은 후보 집합에 자료가 공통으로 있는 가설만 사용하고 중앙값으로 집계한다. 현재 윤곽 segmentation/정수리·턱 검출은 구현하지 않았으므로 정밀 등신·실루엣 손실을 생성하지 않는다.

## `/analyze` 소비 계약

기존 `people`와 별도로 `body_matching.people`를 `person_index`로 연결한다. index는 **현재 컷 안의 기존 정렬**이며 캐릭터 ID가 아니다. cut 이미지 hash를 붙인 person_id를 요청 식별에 사용한다.

| 필드 | 처리 |
|---|---|
| `auto_body_id`, `applied_body_id` | auto에서는 같은 ID. shadow는 applied=null |
| `selected_asset` | body/version/hash/rig/measurement 참조. 서버 파일 경로는 응답에서 제외 |
| `pose_bindings` | 기존 순서대로 pose_id/view/BVH hash. 5개 미만이면 실제 개수 유지 |
| `selection_source` | auto_best_effort / auto_default. 아직 검증된 auto_match 없음 |
| `diagnostic` | uncertain / insufficient_evidence / provider_error / catalog_empty / catalog_error / asset_incompatible / no_pose_candidates |
| `observations` | 속성·근거·unknown·crop 변환·기하 비율·provider 오류 |
| `candidates` | 내부 체형 Top-3, 실제 점수와 evidence_axes. 사용자 사전 선택 단계가 아님 |
| `render_required` | 선택한 체형으로 기존 Top-5를 외부 렌더러가 그릴 필요가 있음 |
| `catalog_version/sha256`, `prompt_version`, `scorer_version` | 재현·캐시·평가의 기준 |

BFF는 먼저 `applied_body_id` 하나로 기존 포즈 후보들을 렌더하고, 이후에 체형 변경 UI를 제공한다. 기존 `thumbnail_url`은 이전 포즈 썸네일이므로 새 체형 렌더로 오인해서 표시하지 않는다. 수동 체형 변경·렌더 묶음 교체·최종 FBX 생성은 해당 팀의 연동 단계다.

## 검증과 남은 승격

```bash
python tests/test_body_matching.py
python tests/test_smoke.py
```

자동 테스트는 인공 FBX 바이트/관측 fixture를 사용한다. 실제 FBX의 메시 품질이나 실제 Gemini의 러프 정확도를 증명하지 않는다. 실제 모델 등록과 표본 러프 평가 후 점수 보정·공백 임계값·속도 예산·auto_match 범위를 확정한다. 나중의 인물별 기억 기능은 `CharacterProfile` 계약을 별도 구현한다.

### 실제 Flash-Lite·9종 평가 (2026-10-06)

[러프별 검수 화면](../artifacts/body-matching/2026-10-06-flash-lite-9/review.html)과 [상세 보고서](../artifacts/body-matching/2026-10-06-flash-lite-9/REPORT.md)에 실제 실행 증거를 저장했다. 통통형을 제외한 FBX 9종과 실제 러프 17컷·28인물을 `gemini-flash-lite-latest`(실제 응답 `gemini-3.5-flash-lite`)로 비교했다. 기존 인물 추출·포즈 결과는 해시 검증 후 재사용했다.

- 수정 후 단일 최저점 5명, 동점 13명, 정보 부족 기본값 10명. 정답 주석이 없어 정확도는 계산하지 않았다.
- `scripts/eval_body_real_rough.py`는 실제 응답·모델 버전·자산/입력 hash·인물별 순위를 저장한다. `compare_body_shapes()`는 같은 선택 점수로 형태만 비교하는 실험 경로이며 QA 승인이나 렌더 권한을 부여하지 않는다.
- 실제 응답의 `null + uncertain` 또는 `값 + unknown` 불일치는 해당 축만 unknown으로 처리하도록 수정했다. 잘못된 enum·ID·batch 구조는 계속 거부한다. 회귀 테스트 20개 통과.
- 전신 가시성에 따른 등신 제한, 모델 치수 검수, 동일 포즈 투영 자료가 필요하다. 배포용 catalog는 계속 비어 있으며 실제 체형을 입힌 Top-5 렌더는 이번 평가 범위에 포함하지 않았다.


## 2026-10-07 분리 워크트리 개선

브랜치: `codex/body-matching-improvements`. 원래 작업 디렉터리의 미커밋 체형 구현을 복사해 시작했으며 `artifacts/body-matching/2026-10-07-improvements/baseline-code.json`에 복사된 코드 해시를 남겼다. 원래 checkout의 소스·모델·배포 설정은 수정하지 않는다.

### 구현

1. `schema.py` / `observation.py`: 부위별 coverage, full_body_visible, 잘림/단축 등신 제외, 가려진 몸통과 드러난 팔의 정보 분리. 기존 v1 입력 호환 유지. Gemini v2는 추가 필드를 필수로 요청한다.
2. `scripts/measure_body_fbx.py`: 원본 FBX 해시를 확인하고 rest 관절 길이·몸통 단면·머리 메시 extent proxy 총 12개 비율과 측면 렌더 생성. 소스 FBX를 저장하거나 수정하지 않는다. 단면은 skin weight와 수직 band로 계산한 근사치이며 사람 검수를 거친 해부학적 실측으로 주장하지 않는다.
3. `visual_selection.py`: 9종 정면·측면 이미지와 실측 프로필을 함께 비교. 모델 이름은 익명 ID로 바꾼다. 순서를 뒤집고 ID를 회전한 두 응답이 같은 단독 후보를 지지할 때만 채택한다. 전신이 없으면 머리 비율을 선택 근거로 쓰지 못하게 schema/후처리로 제한한다. 부분 러프의 머리 비율을 명시적으로 사용한 영문 근거는 보수적으로 거부한다. 이는 문장 의미 검증을 완전히 해결하는 장치가 아니다.
4. `service.py`: 선택적 visual_selector 주입, QA/포즈 호환 자산만 비교, 오류 시 원래 결정 복구. 기존 포즈 순서 보존. 기본 API 생성 경로에서는 추가 비교가 비활성이다.

### 실행

기존 9종 manifest와 semantic catalog를 별도 출력 폴더에 복사한 후 실행한다. source-root는 원본 FBX·러프·기존 평가 데이터가 있는 checkout이다. `.env`는 기존 source-root에서 읽으며 출력에 쓰지 않는다.

```bash
# Blender: measurements.json + front/side reference renders
/Applications/Blender.app/Contents/MacOS/Blender -b -t 2 --python scripts/measure_body_fbx.py -- \
  --source-root /Users/dowon/dev/Standin-server --out artifacts/body-matching/2026-10-07-improvements

# Real Gemini attribute observations, cached real person/pose inputs
python scripts/eval_body_real_rough.py --source-root /Users/dowon/dev/Standin-server \
  --out artifacts/body-matching/2026-10-07-improvements --workers 1

# Two-order real candidate comparison, explicitly experimental
python scripts/eval_body_visual.py --source-root /Users/dowon/dev/Standin-server \
  --out artifacts/body-matching/2026-10-07-improvements --workers 1
```

각 단계는 입력/코드 해시를 고정한다. 코드가 바뀌면 새 폴더 또는 새 `--stage`에서 평가해야 한다. 저장된 결과를 반복 실행해 독립 표본으로 세지 않는다. 429 및 파싱/근거 검증 오류는 자동 선택 실패와 별도 기록한다.

같은 데이터로 개선했으므로 이번 결과는 개발셋 평가다. 정답 주석이 없어 BodyHit@1을 계산하지 않으며, 통통형 2종과 실제 체형별 포즈 렌더는 평가에 포함하지 않는다. 새 작가/캐릭터의 허용 체형 주석을 확보한 뒤 별도 검증해야 한다.


## 캐릭터 남성형·여성형 디자인 분리 (2026-10-07)

사용자가 지정한 오류 8건을 기준으로 `body-observation.v3.presentation`을 추가했다. `presentation`은 value(feminine/masculine/androgynous/null), visibility, cues(face_design/body_contour/hair_design/costume_design), evidence를 가진다. 실제 인물의 성 정체성·생물학적 성별이 아닌 가상 캐릭터의 그림 디자인을 위한 관측이다. 체격·근육·등신과 분리한다.

- 명확한 여성형/남성형 관측: QA/포즈 호환 자산 중 같은 계열 또는 unisex만 비교한다. 라이브러리 제작자가 `metadata.presentation_style`을 등록한다. 허용값은 feminine/masculine/unisex/unspecified이다.
- 약한 관측: 기존 체형 점수를 바꾸지 않고 동점/정보 부족 기본값 선택에만 반영한다. 헤어·의상만의 단서는 자동으로 uncertain으로 낮춘다.
- `presentation_defaults`는 catalog 루트에서 계열별 기본 body ID를 지정한다. 해당 ID의 계열 메타데이터가 일치해야 하며 검수/포즈 호환을 통과하지 못하면 사용하지 않는다. 예: `{"feminine":"female-base","masculine":"male-base"}`.
- 정보 부족/소유권 불명은 기존 기본값을 유지하되 `presentation_selection.mode=unresolved`로 기록한다. 같은 계열의 적격 자산이 없으면 `matching_style_unavailable`이다. 기본값 선택을 성별 인식 성공으로 세지 않는다.
- 관측 v1/v2·메타데이터 없는 기존 catalog는 계속 읽을 수 있다. 추가적인 후보 이미지 비교에도 같은 계열 게이트를 적용한다.

사용자 지정 정답은 평가에만 썼으며 모델 입력에 보내지 않았다. [평가 보고서](../artifacts/body-matching/2026-10-07-presentation/REPORT.md)와 [비교 화면](../artifacts/body-matching/2026-10-07-presentation/review.html)을 참조한다. 8건은 오류를 보고 개선한 개발셋이므로 독립 평가 정확도가 아니다.


## API·제품 연동 계약

[전체 체형 파이프라인 계약](BODY_PIPELINE_API_CONTRACT.md)을 정본으로 사용한다. `/analyze`의 체형 응답은 `api/body_models.py`로 타입 검증하며 기존 people/포즈는 유지한다. `api/body_handoff.py`는 자동 렌더 주문, refine 후 pose 교체, 늦은 렌더 결과 차단, 체형 포함 Export envelope를 정의한다. 체형 렌더는 `POST /body/render`로 연결했다. BFF 영속 상태·Export envelope의 외부 처리는 별도 구현 대상이다.


## develop 분리 및 체형 렌더 연결 (2026-10-07)

`codex/body-render-api`는 최신 develop에서 시작해 체형 코드만 옮겼다. 기존 검색·refine 실험은 포함하지 않는다. API 상세는 `BODY_PIPELINE_API_CONTRACT.md` §10을 따른다. 과거 평가 보고서·원본 러프·FBX는 기존 실험 워크트리에 보존하며 PR에는 넣지 않는다.
