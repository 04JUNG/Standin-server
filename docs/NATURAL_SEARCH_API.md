# 자연어 상·하체 검색 API — UI/BFF 연동 초안 v1

브랜치 `feat/natural-part-text-search-api`. PR·배포·S3 업로드 없이 코드와 계약을 공유하는 **preview** 구현이다. 기존 `/analyze`, `/refine`, `/export-order` 동작은 변경하지 않는다. 검색 품질이 완성됐다는 뜻이 아니다.

## 입력과 흐름

```text
텍스트 UI → POST /natural-search/text → Gemini 상·하체 분리 ┐
이미 읽은 설명 → POST /natural-search/parts ───────────────┤
                                                        ↓
상·하체 독립 검색 → 원본/미러 좌우 확인 → 합성 → 구조 검사 → 최대 5개 BVH
```

`/parts`는 이미지·관절·Gemini 키가 필요 없다. 러프 자체 인식은 기존 `/analyze` 영역이다. **운영 러프 파이프라인을 자연어 우선으로 전환하지는 않는다.** 러프 설명을 이 요청 형식으로 전달하는 통합은 후속 작업이다.

### POST /natural-search/text

```json
{"text":"의자에 앉아 오른손을 얼굴 옆에 들고 왼팔은 내린다","top_k":5,"output_scope":"full"}
```

### POST /natural-search/parts

```json
{
  "upper":{"description":"오른팔을 앞으로 굽혀 손을 들고 왼팔은 아래로 내림","observations":["오른팔을 앞으로 굽힘","왼팔은 아래로 내림"],"group":"arms_front_bent"},
  "lower":{"description":"의자에 앉아 양 무릎을 굽힘","group":"chair_sitting"},
  "top_k":5,"output_scope":"full"
}
```

JSON 요청. 한 인물 기준. text는 공백 제외 1~2,000자, 각 description 1~1,000자, observations 최대 8개·각 240자, top_k 정수 1~5. 알 수 없는 필드는 422로 거절한다. 좌우는 화면이 아니라 **캐릭터 자신 기준**이다. group 생략은 `unknown`; 그룹은 강제 필터가 아니다. 모르는 부위는 null/생략. `/parts`는 최소 한 부위 필요, `/text`가 포즈 단서를 못 읽으면 `not_run`.

상체 group: `arms_down`, `arms_front_bent`, `arms_forward`, `arms_sideways`, `arms_raised`, `hands_near_face_head`, `hands_waist_back`, `arms_supporting_body`, `other`, `unknown`.

하체 group: `standing`, `locomotion`, `chair_sitting`, `floor_sitting`, `kneeling_crouching`, `lying`, `off_ground`, `other`, `unknown`.

## 응답에서 UI가 쓸 것

정본은 `/openapi.json` 및 `docs/openapi/natural-search-v1.json`. schema_version은 `natural-part-search-api-v1`.

| 필드 | 의미 |
|---|---|
| request_id, status, reason | 요청 추적·상태 표시 |
| parsed_parts.upper/lower | 해석/입력된 설명. 수정 UI에서 재사용 |
| parser | provider/model/prompt_version/calls. /parts는 calls=0 |
| library | library_version·DB/태그 SHA·전체/검색가능 수 |
| upper/lower.hits | 부위 각각 최대 K개. 미지정은 빈 목록, inferred_default=true |
| candidates | **최종 합친 포즈만** 최대 K개. 중복으로 Top5를 채우지 않음 |
| candidates[].bvh, bvh_sha256 | 뷰어에 로드할 실제 BVH 본문·해시 |
| upper_id/lower_id, inferred_parts, lateral | 출처·기본값 사용·원본/미러 선택 근거 |
| score | 의미 유사도. 확률·정확도·기하 거리 아님 |
| thumbnail_url | **현재 null**. 인체 썸네일 converter는 BFF 후속 연결 |
| export_intent | 출력 범위 요청만. 실제 메시 절단 완료 아님 |
| budget_exhausted | 합성 예산 소진 시 true. 반환 후보가 일부일 수 있음 |

합친 `parts_*`는 내용 해시 기반 preview ID다. 기존 `/pose/{id}/bvh`, `/refine`, export에 보내지 않는다. 합친 BVH는 응답 본문에서 읽는다. 부위 원본은 hits[].bvh_url의 `/natural-search/library/{pose_id}/bvh`로 읽는다. 이는 **원본 전신 BVH**이지 부위별로 자른 메시가 아니다. 미등록·격리 ID는 404, 파일 해시 변경은 503. generated_*를 라이브러리처럼 서비스하지 않는다.

미지정 부위는 기본 자세로 채우되 정답으로 취급하지 않는다. 텍스트 UI 기본 output_scope=full. 상반신만 필요하면 upper_only를 명시한다. remove_lower_body=true, preserve_arms_hands=true, **actual_mesh_cropped=false**로 의도만 반환한다. 허리 아래로 내려온 팔·손은 보존하는 정책이다. FBX 절단·렌더·손가락 프리셋 전달은 실행하지 않는다.

## 상태와 오류

- 200 preview_ready: 1~5개 구조 검사 통과. 의미 일치·피부 관통·손목 꼬임 검증 완료 아님.
- 200 no_result: 검색/합성 후보 없음. 설명 수정·재검색 제안. 자동 생성 실행 안 함.
- 200 not_run: 포즈 단서 없음, 후보 빈 배열.
- 422: 입력 스키마 오류. FastAPI 기본 detail 배열.
- 429 natural_search_busy: 워커 사용 중, Retry-After=2. 중복 클릭 방지·제한된 재시도.
- 502 text_parser_unavailable / invalid_text_parser_response: Gemini/JSON 실패. 가짜 후보 없음.
- 503 natural_search_disabled / natural_search_assets_unavailable / text_parser_unconfigured: 비활성·자산·키 미설정.
- 503 quarantine_changed_restart_required: 격리 목록 변경. 재시작해 색인 재구성.

오류 예: `{"detail":{"code":"natural_search_disabled","retryable":false}}`.

텍스트 해석 15초·SDK 자동 재시도 없음, 검색 합성 soft budget 15초. 초기 색인 로딩·단일 BVH 검사는 예산을 넘길 수 있어 **요청 전체 hard timeout 보장이 아니다**. BFF는 준비 확인 후 약 60초 타임아웃·취소 UI를 고려한다. 클라이언트 취소가 서버 연산을 즉시 중단시키지는 않는다. 워커별 동시 연산 1개, 초과 429.

## 로컬 실행·자산

```bash
git fetch origin
git switch --track origin/feat/natural-part-text-search-api
python -m pip install -r requirements.txt
uvicorn api.natural_search_app:app --host 127.0.0.1 --port 8001
```

자산 없어도 `/docs`와 `/openapi.json`은 열린다. 검색은 정직하게 503을 반환한다. 실검색에 필요한 DB·BVH·태그·ONNX는 Git에 포함하지 않았으며 자동 S3 다운로드/업로드도 없다. **실험 태그는 아직 S3에 없으므로 checkout만으로 4,215개 실검색이 준비되지는 않는다.** 담당자와 같은 버전의 검증 자산 전달/마운트 경로를 맞춘다.

```bash
export NATURAL_SEARCH_ENABLED=1
export NATURAL_SEARCH_BUNDLE_DIR=/mounted/verified-library
export NATURAL_SEARCH_TAGS_PATH=/mounted/part-tags.json
export NATURAL_SEARCH_EMBEDDING_DIR=/mounted/multilingual-e5-small
export NATURAL_SEARCH_CACHE_DIR=/writable/natural-search-cache
export NATURAL_SEARCH_GEMINI_MODEL=gemini-3.8-flash
# GEMINI_API_KEY는 비밀 저장소/로컬 환경으로 주입. Git에 쓰지 않는다.
```

`GET /natural-search/health`는 첫 로딩·자산 검증·라이브러리 버전을 확인한다. 키는 /text만 필요하다. DB·태그 coverage·모든 BVH 해시 및 manifest 버전 불일치는 503. 자산 변경 후 워커 재시작. 기존 api.app:app에도 같은 router를 추가했지만 기본 비활성이고 /analyze 기동 요건은 별도다. UI 연동에는 가벼운 별도 앱을 권장한다.

인증·CORS 정책은 이 router에서 추가하지 않았다. **인터넷 공개 금지**, 기존 BFF 인증 경계·사설망/loopback에서만 실행한다. BFF가 base URL·사용자 인증·요청 제한·썸네일 변환·preview 수명을 담당한다. 정교한 semantic no-match 기준·자체 생성 fallback·스켈레톤 보정·실제 FBX export는 미지원이다. 지원하는 것처럼 UI를 만들지 않는다.

## 이번 브랜치 검증

- API/검색/합성/좌우 및 기존 계약 회귀 158개 통과, 기존 smoke 54/54, compileall 통과.
- 실제 4,215개 라이브러리 `lib-20261015-32af3079`와 검증된 로컬 태그로 API를 실행했다. 같은 set2 15컷 중 검색 가능한 12명의 최종 BVH 60개가 이전 실험과 해시까지 동일했다. 나머지 3명 자세 단서 부족·1명 비인간 박스는 앞단 기권/제외 사례이며 품질 통과로 세지 않는다.
- 텍스트만 넣는 실제 Gemini 3.8 Flash 호출 1회에서 상·하체 문장 분리와 합친 후보 5개 반환을 확인했다. 이는 연결 smoke 검증이지 검색 정확도 검증이 아니다.
- 키·DB·BVH·모델·태그·러프·렌더·실험 결과 파일은 이 브랜치에 포함하지 않았다. 코드 push만 수행하며 PR·배포·S3 업로드는 하지 않는다.
