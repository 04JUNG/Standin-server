# FBX Converter V3.2 제품 통합 핸드오프

> 다음 세션은 이 문서를 정본으로 사용한다. 설계를 다시 시작하지 말고 아래 Phase 0부터
> 순서대로 구현한다.

## 0. 현재 상태와 시작점

- 저장소: `04JUNG/Standin-server`
- 작업 브랜치: `feat/fbx-converter-v3`
- 원격 브랜치: `origin/feat/fbx-converter-v3`
- 문서 작성 시 HEAD: `4279f14aa65fd0b20b2fc1e4aa865995aa6d628b`
- 기반: `origin/develop`의 `a402377f134f584e6b44ae57105d30b2bb5d53df`
- V3.2 상태: 자동 검증과 사용자 원본 FBX 육안 검증을 모두 통과한 동결본
- 제품 상태: solver 동결만 완료. 운영 `converter/`·HTTP API·컨테이너·배포는 미구현

다음 세션은 먼저 아래를 확인한다.

```bash
git fetch origin
git switch feat/fbx-converter-v3
git status --short --branch
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/fbx-converter-v3)"
```

다른 변경이 섞인 작업 트리에서 구현하지 않는다. 필요하면 이 브랜치로 별도 worktree를 만든다.

## 1. 반드시 먼저 읽을 정본

1. `AGENTS.md`
2. `docs/CHAIN_TRANSPORT_V3_2_PELVIS_FREEZE.md`
3. `qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY/README.md`
4. `qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY/DISCOVERY_01_REPORT.md`
5. `docs/API_CONTRACT.md`
6. `docs/EXPORT_CONTRACT.md`
7. `docs/DECISIONS.md`의 BFF·BVH 핸드오프 결정

동결 코드:

```text
qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY/converter/
```

기준 해시:

```text
V3.2 retarget.py
692e975d32f41e3406144763c7c0b7dbf0a586ff07732f09c4365e3233b13693

V3.1 ankle_policy.json
79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f

standin-master-v2.fbx
7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d

검증 Blender
5.2.0 LTS / build hash fbe6228777e7
```

동결 스냅샷 자체는 수정하지 않는다.

현재 테스트 소스 중 일부는 아직 로컬에만 있다.

```text
28건 회귀 원본
/Users/dowon/Downloads/fbx_pipeline 3/tests/

make_fixtures.py
e9b4697013b58a5756f4719772c23f985ae40ee4ff0a5cd486bf89e09cae020b

test_convert.py
1a508c9e417116a9f80ecf3bd2e89e009c04ee232db8a29eefe990472918add2

V3.2 math·fallback·independent verifier 원본
/Users/dowon/dev/Standin-server/out/rest-v32-qa/tools/
```

CI를 만들기 전에 재사용할 테스트 `.py`만 `tests/converter/`로 이관한다. `__pycache__`, 생성
fixture, FBX/BVH/PNG, `outputs/`, `logs/`, `manifests/`는 이관하지 않는다.

## 2. 제품 파이프라인에서 V3.2의 위치

V3.2는 검색이나 refine solver가 아니다. 사용자가 고른 최종 BVH를 캐릭터 FBX에 입혀
최종 FBX를 만드는 출력단 retarget solver다. 제품 응답은 변환 입력이었던 최종 BVH도 함께
보존해 `BVH + FBX` 쌍으로 제공한다.

```text
콘티 이미지
  ↓
추론 API: VLM → 검출 → 포즈 추정 → 검색
  ↓
Top-K BVH 후보
  ↓
사용자 선택
  ↓
선택적 refine
  ├─ refined=true  → RefineResponse.bvh가 최종 BVH
  └─ refined=false → bvh_url에서 받은 base BVH가 최종 BVH
  ↓
BFF: 최종 BVH 바이트 확정·Job 관리
  ↓
내부 Converter API
  ↓
Blender 5.2 background worker
  ├─ 캐릭터 FBX import
  ├─ 최종 BVH import
  ├─ canonical profile·bone mapping
  ├─ V3.2 retarget                     ← solver 위치
  ├─ output mode 적용
  └─ FBX export
  ↓
BFF가 최종 BVH + FBX를 같은 Job에 저장·다운로드 응답
  ↓
CSP/사용자
```

V3.2는 FBX 내보내기를 요청할 때만 실행한다. BVH만 제공하는 흐름은 변경하지 않는다.

## 3. 통합의 핵심 불변식

### 3.1 추론과 변환은 같은 저장소, 다른 런타임

- 기존 `api/`와 `src/`는 추론 서비스다.
- 새 `converter_api/`는 내부 변환 서비스다.
- 두 서비스는 별도 Docker 이미지·ECR repository·ECS service·requirements·CI job을 쓴다.
- 추론 이미지와 추론 Python에는 `bpy`가 없어야 한다.
- converter API 프로세스도 `bpy`를 import하지 않는다.
- `bpy`는 Blender 자식 프로세스 안에서만 로드한다.

### 3.2 작업당 Blender 프로세스 1회

Blender는 전역 scene 상태가 누적된다. 한 프로세스에서 여러 사용자 작업을 순환 처리하지 않는다.

```text
HTTP request 1건
  → 전용 임시 디렉터리
  → Blender child process 1회
  → report·FBX 검증
  → 응답
  → child 종료·임시 파일 삭제
```

### 3.3 동결 수학을 서비스 작업과 섞어 바꾸지 않음

첫 통합 PR에서는 다음을 바꾸지 않는다.

- retarget 수학
- canonical bone map
- V3.1 ankle policy 수치
- V3.2 pelvis boundary
- mirror 수학
- fallback 조건
- 부모→자식 적용 순서
- root translation 처리

서비스 통합 중 품질 문제가 보이면 우선 import 옵션·입력 artifact·output mode·Blender 버전을
검사한다. 즉석에서 solver 임계값을 조정하지 않는다.

## 4. 권장 파일 구조

```text
Standin-server/
├─ api/                              # 기존 추론 API, 유지
├─ src/                              # 기존 추론 로직, 유지
├─ converter/                        # 운영 V3.2 Blender worker 코드
│  ├─ __init__.py
│  ├─ bone_map.py
│  ├─ ankle_policy.json
│  ├─ retarget.py
│  ├─ convert.py
│  └─ worker.py                      # Blender --python 진입점
├─ converter_api/
│  ├─ __init__.py
│  ├─ app.py                         # /convert-bundle, /convert, /characters, /healthz
│  ├─ schemas.py
│  ├─ registry.py                    # character_id → immutable metadata
│  └─ runner.py                      # subprocess·timeout·tempdir·검증
├─ config/
│  └─ characters.example.json        # 실제 URI·비밀은 환경/배포 설정
├─ tests/converter/
│  ├─ test_api_contract.py
│  ├─ test_registry.py
│  ├─ test_runner_failures.py
│  └─ test_refined_bvh_e2e.py
├─ Dockerfile.converter
├─ requirements-converter.txt
└─ .github/workflows/converter-ci.yml
```

`qa/retarget/...`에서 운영 `converter/`로 파일을 복사한 직후 SHA256 동치를 먼저 확인한다.
운영 코드를 동결 디렉터리에서 직접 import하지 않는다.

## 5. 최종 BVH 선택 알고리즘

BFF가 converter를 부르기 전에 최종 BVH를 단 하나로 확정한다.

```python
if refine_response is not None and refine_response.refined:
    assert refine_response.bvh
    final_bvh_bytes = refine_response.bvh.encode("utf-8")
    artifact_kind = "refined"
else:
    final_bvh_bytes = GET(base_bvh_url)
    artifact_kind = "base"
```

주의:

- 현재 `/export-order`는 항상 base `bvh_url`을 만든다.
- refined 결과는 `/export-order`에서 자동 복원되지 않는다.
- `refined=true`인데 `RefineResponse.bvh`를 버리고 base URL을 다시 받으면 refine이 조용히 사라진다.
- BFF는 선택·refine 응답과 최종 BVH의 SHA256을 같은 Job lineage에 기록해야 한다.
- 다인 컷은 인물마다 1개 BVH를 독립 변환한다. `set_id`는 묶음 메타일 뿐 하나의 다인 BVH가 아니다.

## 6. Converter 내부 API v1.1 계약

외부 공개 API가 아니라 BFF가 내부망에서 호출하는 계약이다.

### 6.1 `POST /convert` — FBX 단일 응답 호환 계약

요청은 `multipart/form-data`로 고정한다.

| 필드 | 형식 | MVP 기본값 | 설명 |
|---|---|---:|---|
| `bvh` | file | 필수 | base 또는 refined BVH 바이트 |
| `character_id` | string | `standin-master-v2` | registry에 등록된 값만 허용 |
| `frame` | integer | `0` | 단일 포즈 프레임 |
| `mirror` | boolean | `false` | 반드시 명시적으로 한 번만 적용 |
| `output_mode` | enum | `rigged_rest` | MVP에서는 이 값만 허용 |
| `apply_root_translation` | boolean | `false` | MVP 잠금 |

첫 MVP에서는 `rigged_anim`과 `posed_mesh`를 열지 않는다. 지원 코드는 있어도 V3.2 육안 승인은
`rigged_rest`에만 존재한다.

성공 응답:

```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="<safe-name>.fbx"
X-Standin-Conversion-Id: <uuid>
X-Standin-Solver-Version: chain-transport-v3.2
X-Standin-Source-BVH-SHA256: <sha256>
X-Standin-Artifact-SHA256: <sha256>
X-Standin-Source-Profile: <profile>
X-Standin-Target-Profile: <profile>
X-Standin-Mapped-Bones: <count>
X-Standin-Warning-Count: <count>
```

상세 `ConvertReport`는 `conversion_id`를 키로 구조화 로그에 전부 남긴다. 큰 report를 HTTP
헤더나 base64 JSON에 넣지 않는다. 실패 응답만 JSON 오류 봉투를 사용한다.

권장 오류:

| 상태 | 상황 |
|---:|---|
| 400 | 잘못된 옵션·character_id·파일명 |
| 413 | BVH 크기 제한 초과 |
| 422 | BVH parse·profile·required mapping 실패 |
| 503 | 캐릭터 artifact 또는 Blender 준비 안 됨 |
| 504 | Blender 변환 timeout |
| 500 | report 불일치·FBX 미생성·내부 오류 |

클라이언트가 임의 URL이나 서버 파일 경로를 넘기게 하지 않는다. BVH는 업로드 바이트,
캐릭터는 registry ID만 받는다. SSRF와 path traversal을 동시에 차단한다.

### 6.2 `POST /convert-bundle` — 제품용 BVH + FBX 계약

요청 옵션은 `/convert`와 같고 `artifact_kind=base|refined` 및 BFF가 최종 BVH에서 계산한
`expected_bvh_sha256`을 필수로 추가한다. API는 업로드 직후 해시를 대조해 불일치 시 Blender를
실행하지 않고 409로 실패한다. 성공 응답은 `application/zip`이며 아래 고정 이름의 regular file
세 개만 포함한다.

```text
final.bvh
final.fbx
manifest.json
```

- `final.bvh`는 요청 바이트와 정확히 같아야 한다. 개행·문자열 재직렬화를 하지 않는다.
- `final.fbx`는 같은 요청 바이트를 worker에 전달해 만든 결과다.
- `manifest.json`은 artifact kind, 변환 옵션, 두 파일의 크기·SHA256을 기록한다.
- API는 runner 결과의 conversion id·입력 SHA·출력 SHA를 응답 직전에 독립 재검증한다.
- ZIP 엔트리는 서버가 정한 고정 이름과 일반 파일 권한만 사용해 path traversal·symlink를 막는다.
- ZIP 전체 SHA는 `X-Standin-Artifact-SHA256`과 `X-Standin-Bundle-SHA256`, FBX SHA는
  `X-Standin-FBX-Artifact-SHA256`, BVH SHA는 `X-Standin-Source-BVH-SHA256`으로 반환한다.

BFF는 ZIP과 내부 두 파일 및 manifest를 전부 검증한 후에만 두 산출물을 같은 Job에 publish한다.
불일치나 변환 실패 때는 둘 다 publish하지 않는다.

### 6.3 `GET /characters`

사용 가능한 `character_id`, 표시 이름, rig profile, revision만 반환한다. 실제 로컬 경로·S3 URI는
노출하지 않는다.

### 6.4 `GET /healthz`

다음을 확인한다.

- API 프로세스 정상
- 고정 Blender binary 실행 가능
- 허용 Blender 버전 일치
- 기본 캐릭터 존재와 SHA256 일치
- 임시 디렉터리 쓰기 가능

모델이 없거나 해시가 다르면 `503`이다.

## 7. 캐릭터 registry와 artifact 보관

FBX 본체를 Git이나 Docker image에 넣지 않는다. metadata만 추적한다.

예시:

```json
{
  "schema_version": 1,
  "characters": {
    "standin-master-v2": {
      "display_name": "Standin Master V2",
      "artifact_uri_env": "STANDIN_MASTER_V2_URI",
      "sha256": "7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d",
      "rig_profile": "mixamo",
      "revision": "v2"
    }
  }
}
```

배포 시 절대 경로·`file://` read-only volume 또는 `s3://` object를 지정한다. S3는 ECS task
role로 전용 temp cache에 내려받고, worker 실행 전 registry SHA-256을 검증한다. HTTP(S) URI는
받지 않는다.
동일 `character_id`의 내용이 바뀌면 안 된다. 변경은 새 revision·새 해시로 등록한다.

현재 registry에는 기본 남성 `standin-master-v2`와 beta 운영 승인 여성
`standin-female-v2-lbs` metadata가 있다. beta 배포에는 승인 artifact를 업로드한 뒤
`STANDIN_FEMALE_V2_LBS_URI`를 설정한다. URI가 없는 환경에서는 `/characters`에 노출되지 않는다.

## 8. Blender 실행 계약

초기 운영판은 `pip bpy`가 아니라 전체 Blender 5.2 LTS binary를 쓴다. 기존 검증 경로와 가장
가깝기 때문이다. Linux/amd64 이미지에서도 같은 버전을 고정하고 archive checksum 또는 image
digest를 기록한다.

API 프로세스는 다음과 같은 자식 프로세스를 실행한다.

```text
blender --background --python-exit-code 1 --python converter/worker.py -- --job <job.json>
```

`worker.py`는 repo root를 `sys.path`에 넣고 운영 `converter` package를 import한다. job에는
서버가 만든 절대 임시 경로만 들어가며 사용자 경로를 그대로 전달하지 않는다.

고정 옵션:

```text
frame=0
output_mode=rigged_rest
apply_root_translation=false
embed_textures=false
mirror=false 또는 요청의 명시값
```

runner는 종료코드만 믿지 않는다. 기존 QA에서 Blender가 Python traceback에도 0을 반환한 사례가
있으므로 다음을 모두 확인한다.

1. timeout 없음
2. 로그에 traceback·`[FAIL]` 없음
3. report JSON parse 성공
4. `report.ok == true`
5. solver version 정확히 V3.2
6. output FBX 존재·크기 > 0
7. artifact SHA256 계산 성공
8. output 경로가 해당 job tempdir 내부

timeout이면 process group 전체를 종료하고 임시 파일을 삭제한다.

## 9. 성능과 동시성

로컬 Blender 5.2에서 측정한 현재 참고값:

| 입력 | 전체 wall time |
|---|---:|
| CMU base BVH | 1.33초 |
| refined Mixamo BVH | 0.68초 |
| refined CMU BVH | 0.77초 |
| converter 본체 report elapsed | 약 0.23초 |

이는 macOS 로컬 참고값이며 ECS SLO가 아니다. Linux 배포 이미지에서 다시 측정한다.

초기 목표:

- warm task 20회 기준 p95 3초 이내
- cold start와 실제 변환 시간을 분리 기록
- 최소 task 1개 유지 여부는 비용·UX 결정으로 설정
- task당 동시 Blender 수는 기본 1
- queue 대기시간과 execution 시간을 별도 metric으로 기록

같은 BVH SHA·character SHA·solver version·옵션 조합은 BFF에서 캐시할 수 있다. 캐시 키에
최소 다음이 들어가야 한다.

```text
sha256(final_bvh)
sha256(character_fbx)
solver_version
frame
mirror
output_mode
apply_root_translation
embed_textures
Blender_version
```

## 10. Refine 호환성

refine writer는 원본 BVH의 `HIERARCHY`, `OFFSET`, channel 순서를 그대로 복사하고 MOTION의
회전값만 바꾼다. 따라서 V3.2 입력 계약과 구조적으로 호환된다.

이미 확인한 통합 probe:

| refined 입력 | profile | mapped | chain/pelvis fallback | 결과 |
|---|---|---:|---:|---|
| Rokoko/Mixamo result BVH | mixamo → mixamo | 22/22 | 0 | FBX 생성 성공 |
| CMU refined BVH | cmu_bvh → mixamo | 22/22 | 0 | FBX 생성 성공 |

이 probe는 parse·mapping·solver·export 호환 증거다. 모든 refined 포즈의 메시 육안 안전을
증명한 것은 아니다. 정식 E2E에는 base/refined/mirror 쌍을 포함한다.

## 11. 구현 단계

### Phase 0 — 동결본 승격 준비

- clean branch 확인
- 운영 `converter/` 생성
- 동결 `converter/*`를 byte-identical 복사
- `SHA256SUMS` 검증
- 운영 코드가 `qa/`를 import하지 않는지 검사
- 외부 테스트 원본 해시 확인
- 필요한 테스트 `.py`만 `tests/converter/`로 이관하고 생성물은 제외
- Blender 5.2로 기존 28/28 재실행

Phase 0에서 수학을 수정하지 않는다.

### Phase 1 — 로컬 Blender worker

- `converter/worker.py`
- job JSON validation
- tempdir 격리
- timeout·process group kill
- report·artifact sentinel 검증
- base·refined·mirror CLI E2E

HTTP 없이 worker부터 완성한다.

### Phase 2 — 내부 Converter API

- `converter_api/schemas.py`
- character registry
- `/healthz`, `/characters`, `/convert`, `/convert-bundle`
- 업로드 크기·확장자·내용 최소 검증
- 변환 1건당 child process 1회
- 성공 FBX streaming 및 원자적 BVH+FBX bundle streaming
- 구조화 report logging
- 오류 봉투와 cleanup 테스트

### Phase 3 — 추론/refine 핸드오프 E2E

- base BVH URL 경로
- `RefineResponse.bvh` 인라인 경로
- `refined=false` base fallback 경로
- mirror 경로
- 다인 item별 독립 변환
- 최종 BVH SHA가 BFF lineage와 converter log에서 일치

이 저장소의 실행 가능한 계약과 별도 BFF PR 인계 내용은
`docs/FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`를 따른다.

현재 BFF는 별도 저장소 소유다. 이 저장소에서는 내부 converter API와 계약 문서까지만 구현하고,
BFF 변경은 별도 PR로 전달한다.

### Phase 4 — 컨테이너·CI

- `Dockerfile.converter`
- `requirements-converter.txt`
- Blender 5.2 Linux/amd64 pin과 checksum
- 명시적 `COPY`만 사용하고 `COPY . .` 금지
- converter path 변경 때만 도는 별도 CI
- inference image에 `bpy`가 없음을 검증
- converter image의 Blender 버전·28/28·API smoke 검증

추론 이미지 검사 예시:

```bash
docker run --rm "$INFERENCE_IMAGE" python -c \
  "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('bpy') is None else 1)"
```

converter 이미지 검사는 반대로 고정 Blender 버전과 worker 실행 성공을 요구한다.

### Phase 5 — 배포 준비

- ECR `standin/converter`
- inference와 다른 ECS service/task definition
- 내부망에서만 접근
- health check와 timeout
- CloudWatch structured logs·conversion_id
- path-filtered deploy workflow
- warm/cold latency 측정

구현·운영 변수·내부망 계약은 `docs/FBX_CONVERTER_V3_2_PHASE5_DEPLOYMENT.md`를 따른다.
2026-08-27 `standin-master-v2` 실제 모델의 `g1-move1`·Mixamo control 통합 결과를 사용자가
육안 확인했다. 합성 fixture는 자동 회귀에만 사용하고 실제 품질 판정에는 사용하지 않는다.

자동 배포는 infra 변수와 비용 정책을 확정한 뒤 연다.

## 12. CI·회귀 게이트

최소 게이트:

1. 기존 추론 테스트 전부 통과
2. 기존 inference Docker build 통과
3. inference image에서 `bpy` import 불가
4. V3.2 converter regression 28/28
5. V3.2 math controls 8/8
6. mirror actual path PASS
7. 강제 한쪽 퇴화의 bilateral exact V3.1 fallback PASS
8. character SHA 불일치 시 health/convert 실패
9. malformed BVH·oversize·timeout·Blender traceback negative test
10. base/refined Mixamo/refined CMU E2E 성공
11. job 종료 후 tempdir 누수 없음
12. Git·Docker image에 FBX/BVH 원본·렌더·`out/` 산출물 없음

제품 승격 전 수동 게이트:

- 승인 V3.2 대표 원본 FBX와 통합판 FBX가 같은 자세인지 비교
- g1-move1 골반 자연스러움 유지
- CMU 발 방향 회귀 없음
- UAL2 Slide/SwordHeavy 발목 방향 유지
- refined 대표 자세가 base로 조용히 되돌아가지 않음

## 13. 기존 계약에서 반드시 갱신할 부분

현재 `docs/EXPORT_CONTRACT.md`는 BVH를 동원 Export 단계에 넘기고 CSP 보정을 외부에서 한다는
기존 책임 분리를 기록한다. converter 통합 후 다음을 같은 PR에서 갱신한다.

- BFF가 최종 base/refined BVH를 확정하는 규칙
- 내부 Converter API 호출
- 최종 BVH·FBX 동시 다운로드 경로와 오류 처리
- mirror 소유자 단일화
- converter가 담당하는 rest/chain retarget과 CSP가 담당하는 소재 배치의 경계

mirror를 converter와 CSP 양쪽에서 적용하지 않는다. MVP 기본값은 `mirror=false`이며 UI의 명시적
미러 요청이 어느 계층 소유인지 계약으로 한 번만 결정한다.

## 14. 하지 말 것

- 동결 `qa/retarget/CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY` 직접 수정
- 첫 통합 PR에서 solver 수학·임계값 변경
- `pip bpy`로 검증 없이 교체
- inference `requirements.txt`에 Blender/bpy 추가
- inference Dockerfile에서 Blender 설치
- API 프로세스에서 `import bpy`
- 동일 Blender 프로세스 재사용
- 사용자 입력으로 filesystem path·character URI 허용
- character FBX·BVH 라이브러리·렌더·QA `out/`을 Git이나 image에 포함
- refined=true인데 inline `bvh`를 버리고 base URL 변환
- Blender 종료코드만 보고 성공 판정
- `main` 직접 push

## 15. 권장 커밋 순서

```text
feat(converter): promote frozen v3.2 worker
feat(converter-api): add isolated Blender conversion endpoint
test(converter): lock base refined and mirror E2E
ci(converter): build isolated Blender image
docs(export): document BVH to FBX handoff
```

각 커밋에서 FBX·BVH·PNG·로그가 staged되지 않았는지 확인한다.

## 16. 완료 정의

다음이 모두 충족돼야 “전체 파이프라인 통합 완료”다.

- 사용자 선택 base BVH와 그 BVH로 만든 FBX가 한 쌍으로 다운로드됨
- refine 성공 시 `RefineResponse.bvh` 원문과 이를 실제 반영한 FBX가 한 쌍으로 다운로드됨
- refine 거부 시 base BVH가 정확히 사용됨
- V3.2·V3.1 ankle policy·character·Blender lineage가 report에 남음
- 추론 서비스는 Blender와 독립적으로 기존 배포·테스트를 유지함
- converter는 별도 image/service로 배포 가능함
- 28/28과 실제 base/refined/mirror 회귀 통과
- 사용자 승인 골반·발목 품질이 통합 뒤에도 유지됨
- Git·Docker image에 라이선스 원본·QA 산출물이 없음
- API·EXPORT·배포 문서가 실제 코드와 일치함

## 17. 다음 세션에 바로 줄 실행 요청

```markdown
`docs/FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md`를 전부 읽고 정본으로 따라라.
`feat/fbx-converter-v3`의 clean worktree에서 Phase 0만 수행하라.

목표:
1. frozen V3.2를 운영 `converter/`로 byte-identical 승격
2. SHA256 검증
3. 외부 테스트 원본 해시 확인과 필요한 `.py`만 `tests/converter/`로 이관
4. Blender 5.2에서 기존 28/28 회귀
5. production이 `qa/`를 import하지 않는지 확인

이번 단계에서는 HTTP API·Docker·배포·solver 수학 수정까지 진행하지 마라.
완료 후 변경 파일, 해시, 테스트 결과, 다음 Phase를 보고하고 멈춰라.
```
