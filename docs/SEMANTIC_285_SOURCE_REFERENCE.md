# 직접 자연어 포즈 검색

자연어 설명으로 기존 라이브러리의 포즈 후보와 해당 JSON을 반환한다. 직접 검색, 검색 품질 개선, 선택적 사용자 선호, 번들 검증과 로컬 검수 도구만 포함한다.

러프 이미지 판독, VLM 프롬프트, 동적 라우팅, `/analyze`의 응답 확장, 팔짱·V자 팔의 BVH 관계 색인, 카메라 선택은 포함하지 않는다. 동적 라우팅과의 어댑터 연결은 후속 작업이다. 기하 검색과 refine solver는 그대로이며 시맨틱 후보는 `refine_allowed=false`다.

## 요청과 응답

`POST /semantic-search`

```json
{"query":"야구공을 던지는 자세","source":"user","top_k":5}
```

`query`는 최대 500자다. `source`는 `user` 또는 `vlm`으로 임계값만 선택한다. `vlm`을 지정해도 이 경로에서 외부 VLM을 호출하지 않는다. `tags`는 사전 값의 `field:value` 목록이며, 문장 없이 검증된 태그만 전달할 수도 있다. 동적 라우터의 관찰·관계 정보는 아직 이 입력에 연결하지 않았다.

| 상태 | HTTP | 의미 |
|---|---:|---|
| ok | 200 | 조건과 임계값을 통과한 후보가 있음 |
| no_result | 200 | 현재 색인·조건·임계값에서 후보를 확인하지 못함 |
| error | 422 또는 503 또는 500 | 잘못된 입력, 준비되지 않은 색인, 내부 오류 등 |

`no_result`는 자산 부재의 증명이 아니다. 호출자는 결과 없음과 시스템 오류를 다르게 처리해야 한다. `error.code`는 입력/색인/인코더/ID 정합성 오류를 구분한다.

후보에는 `rank`, `pose_id`, `score`, `score_components`, `evidence`, `mirror_alternates`, `refine_allowed`, `pose`가 있다. 포즈 JSON에는 같은 `pose_id`, 태그, 설명, `bvh_url`, 시점별 썸네일 URL과 라이브러리 정체성이 담긴다. 후보 ID와 포즈 JSON의 ID가 다르거나 JSON이 없으면 오류다.

## 점수와 조건

개념·동작 태그 일치, 상하체 설명, 글자 TF-IDF 유사도, 내용어 덮음으로 점수를 계산한다. E5 문장 임베딩은 선택 성분이다. 원본/미러는 한 가족으로 묶어 다양성을 확보한다.

상대의 행동을 찾는 인물의 행동과 혼동하지 않고, 명시적인 부정·시간 전환을 처리한다. 종목·도구·핵심 행동, 바닥 손 지지, 양손 검 등의 조건을 확인한다. 악수와 단순 팔 내밀기, 팔굽혀펴기와 수영/포복, 상대를 돕기와 스스로 일어나기를 구분한다. 검끝 방향과 복부 반응도 설명 근거를 요구한다.

이는 태그·설명 기반 판정이다. 실제 접지·접촉·기하적 자세 적합성을 증명하지 않으며 잘못된 태그와 누락된 설명은 별도로 보완해야 한다.

## 명시적 사용자 선호

`SEMANTIC_REVIEW_PREFERENCES_PATH`를 지정할 때만 서비스 초기화에서 선호 파일을 읽는다. 기본은 비활성이다. 예시 파일은 `config/semantic_review_preferences.v1.json`이다.

정규화된 동일 문장의 선호 후보를 조건·임계값 통과 후 우선한다. 새 표현에 선호를 임의 전파하거나 자산을 재태깅하지 않는다. 검색 점수는 그대로이므로 최종 순위는 점수 내림차순과 다를 수 있다. `score_components.human_preference=1` 및 `query.review_preference`로 근거를 표시한다. 파일 해시도 색인 정체성에 포함된다.

명시적 거절 후보는 해당 문장에서 감점한다. 누락·격리된 포즈를 선호 파일로 복구할 수 없고, 잘못된 선호 파일은 색인 로드 오류다. 로컬 평가 도구는 예시 선호 파일을 사용하며 이는 학습된 일반화 성능이 아니다.

## 설정과 번들

`.env.example`의 `SEMANTIC_*` 항목을 참고한다. `SEMANTIC_ENABLED=1`, `SEMANTIC_REQUIRED=0`가 기본이다. 색인이 없어도 기존 `/analyze`·`/refine`은 유지된다. `/semantic-search`는 색인 오류를 반환하고 `/healthz.semantic`에 원인을 표시한다. `SEMANTIC_REQUIRED=1`이면 색인 준비 실패도 전체 health를 unhealthy로 한다.

기본 파일은 `DATA_DIR/semantic/search-v2.json`과 `search-taxonomy-v2.json`이다. 태그의 `source_db_sha256`와 DB, 포즈 ID, 사전 정합성을 확인한다. 검색 색인은 기존 파이프라인의 허용 ID로 제한해 격리 정책을 공유한다.

`build_pose_bundle.py attach-semantic`과 `deploy_pose_library.py --require-semantic`은 태그 파일·manifest 검증을 지원한다. 태그 해시는 별도로 기록하고 기존 BVH 내용 해시와 라이브러리 버전은 바꾸지 않는다. 상세는 [POSE_LIBRARY_BUNDLE.md](POSE_LIBRARY_BUNDLE.md)를 참고한다. 이 PR을 머지하거나 코드 파일을 올리는 것만으로 라이브러리가 업로드되거나 운영 설정이 켜지지는 않는다.

임베딩은 기본 비활성이다. `SEMANTIC_DENSE_ENABLED=1`, 모델 디렉터리와 가중치를 설정해야 한다. 모델 가중치·토크나이저·태그·DB·BVH·이미지는 Git에 넣지 않는다. 로컬 평가에는 weight 0.5, lo 0.72, hi 0.88을 사용했으며 운영 기본값으로 승격하지 않았다.

## 로컬 검색과 검수

```bash
python scripts/serve_semantic_lab.py --library-dir <pose-library-viewer> --port 4181
# 선택: --dense-model-dir <multilingual-e5-small>
```

127.0.0.1에만 바인딩하며 VLM 호출과 이미지 업로드 UI가 없다. `/`는 직접 검색, `/review`는 저장된 100문장 실험을 한 문장씩 확인하는 도구다. 결과 파일을 고정하고 사람 검수는 별도 append-only JSONL에 저장한다. 모델 판정과 사람 판정을 합산하지 않는다. 화면은 평가 연구용이며 인증된 공개 서비스용 UI가 아니다.

사용자 입력과 검수 로그는 `out/semantic-feedback/`에 저장되며 Git에 포함하지 않는다. 운영 로그의 원문 기록은 `SEMANTIC_LOG_QUERY_TEXT=0`으로 기본 비활성이다.

## 재현과 검증

```bash
python -m pytest tests/test_semantic_search.py tests/test_semantic_relevance.py \
  tests/test_semantic_review.py tests/test_semantic_bundle.py tests/test_semantic_text_scope.py -q
SEMANTIC_REAL_DATA_DIR=<pose-library-viewer> python tests/test_semantic_search.py
python scripts/eval_semantic_metrics.py --library-dir <pose-library-viewer> \
  --queries evaluation/semantic/dev-20261013.v3.json
python scripts/run_semantic_input_collection.py --library-dir <pose-library-viewer> \
  --queries evaluation/semantic/natural-inputs-100-20261013.json \
  --dense-model-dir <multilingual-e5-small> --out out/semantic-experiment.json
```

2026-10-13 실험은 [SEMANTIC_TEXT_SEARCH_REVIEW.md](SEMANTIC_TEXT_SEARCH_REVIEW.md)에 정리한다. 개발 세트와 사용자 선호 재실행을 새 입력의 독립 정확도로 해석하지 않는다.
