# 04J-289 — 1컷 다인 FBX 통합

인물별 최종 rigged-rest FBX를 독립 골격으로 유지하면서 한 컷에 배치한다.
기존 `/convert-bundle` 결과의 `final.fbx`가 입력이다. base/refined 선택, mirror,
후보 카메라는 기존 변환 단계에서 확정하며 합성 단계에서 다시 적용하지 않는다.
동결 V3.2.5 solver와 인물별 export 응답은 변경하지 않는다.

## 내부 API

`POST /compose-scene`, `multipart/form-data`:

| 필드 | 내용 |
|---|---|
| `manifest` | 아래 JSON 문자열 |
| 반복 `fbx` | `persons` 배열과 같은 순서의 binary FBX 파일 |

```json
{
  "schema_version": 1,
  "cut_id": "ep12_cut07",
  "coordinate_system": "threejs-y-up-m",
  "persons": [
    {
      "person_index": 0,
      "fbx_sha256": "<첫 번째 최종 FBX 바이트의 SHA256, 소문자 hex 64자>",
      "position": [-1.0, 0.0, 0.0],
      "rotation_y_degrees": 0.0,
      "scale": 1.0
    },
    {
      "person_index": 2,
      "fbx_sha256": "<두 번째 최종 FBX 바이트의 SHA256, 소문자 hex 64자>",
      "position": [1.2, 0.3, -0.5],
      "rotation_y_degrees": 35.0,
      "scale": 0.75
    }
  ]
}
```

좌표는 Three.js와 같은 오른손 Y-up 미터, `position=[x,y,z]`다. Blender에서는
`[x,-z,y]`로 변환한다. 회전은 Y축 양의 방향, 단위는 도(degree)다. 스케일과 회전은
원본 FBX의 씬 원점을 기준으로 적용하고 그 뒤 위치를 더한다. 모델 높이/중심을 개별
정규화하거나 바닥에 강제 정렬하지 않으므로 앉기·공중 포즈도 유지된다.

`person_index`는 0..9999 범위의 고유 정수다. 순번이 연속일 필요는 없다.
회전 기본값은 0, 스케일 기본값은 1이다. 최대 8인, 파일당 32 MiB, 합계 128 MiB다.
위치는 축당 ±1000m, 스케일은 0.01..10, 회전은 ±360° 범위를 허용한다.
NaN/Infinity, 중복 인물, 음수 스케일, 파일 경로/URL/알 수 없는 필드는 거부한다.

```bash
curl --fail-with-body http://localhost:8001/compose-scene \
  -F 'manifest=<scene.json' \
  -F 'fbx=@person0.fbx;type=application/octet-stream' \
  -F 'fbx=@person2.fbx;type=application/octet-stream' \
  --output cut.zip
```

성공 응답은 `application/zip`이며 `scene.fbx`, `manifest.json` 두 파일을 담는다.
manifest는 정규화된 입력 배치, 입력 FBX SHA, 출력 FBX SHA/크기, 인물별 골격/메시 수,
재임포트 오차, Blender 버전을 기록한다. `fbx_import.threejs_scale_to_meters=0.01`도 명시한다.
`X-Standin-Artifact-SHA256`은 ZIP 전체, `X-Standin-FBX-Artifact-SHA256`은 내부 FBX의 SHA다.
실패 시 오류 JSON만 반환하고 부분 FBX는 제공하지 않는다.

| HTTP | error.code | 의미 |
|---|---|---|
| 400 | `INVALID_SCENE` / `INVALID_FILENAME` | JSON·좌표·파일 개수·이름 오류 |
| 409 | `SOURCE_FBX_SHA256_MISMATCH` | FBX 입력 무결성 오류 |
| 413 | `SCENE_TOO_LARGE` | 입력 크기 제한 초과 |
| 422 | `INVALID_FBX` / `INVALID_SCENE_FBX` | binary FBX 또는 1인 rigged-rest 조건 위반 |
| 500 | `WORKER_INTEGRITY_ERROR` | worker 실패 또는 재임포트 검증 실패 |
| 503 | `BLENDER_UNAVAILABLE` | 고정 Blender 실행 환경 불가 |
| 504 | `CONVERSION_TIMEOUT` | 합성 제한 시간 초과 |

일반 변환과 같은 프로세스 동시 실행 한도, 임시 폴더, Blender 버전/빌드 검사,
`CONVERTER_TIMEOUT_SECONDS`를 쓴다. HTTP 프로세스는 bpy를 import하지 않는다.
합성 코드는 동결 solver manifest 바깥의 별도 worker이며 Docker COPY allowlist에 포함된다.

## 뷰어와 BFF 연결

BFF는 한 컷의 인물별 최종 FBX를 확보한 뒤, 뷰어에서 저장한 3D 배치와 SHA256을 위 API에
전달한다. 인물별 원본 BVH·FBX와 lineage는 유지하고, 컷 전체 다운로드로 ZIP을 추가하면 된다.
이 변경은 converter API와 CLI를 제공한다. 제품 BFF의 저장/다운로드 버튼 연결과 운영 배포는
아직 수행하지 않았다.

FBX는 Y-up / -Z-forward, 센티미터 단위로 저장된다. Blender는 단위 메타를 읽어 미터로
변환한다. Three.js FBXLoader는 단위 메타를 자동으로 적용하지 않으므로 씬 전체에 한 번 적용한다.

```javascript
const cut = new FBXLoader().parse(fbxArrayBuffer, "");
cut.scale.setScalar(manifest.fbx_import.threejs_scale_to_meters); // 0.01
scene.add(cut);
cut.updateMatrixWorld(true);
```

인물별로 bounding box에 맞춰 다시 정규화하면 의도한 상대 크기가 달라질 수 있다.
`person_0000_placement` 같은 Empty 노드가 인물 배치를 소유하며 하위 골격·메시 이름도
인물별 접두사로 구분한다. 본 이름은 각 골격 내부에서 유지한다. 다수 리그를 지원하는
Blender/Three.js에서 검증했으며 CSP/Modeler의 직접 임포트는 이번에 검증하지 않았다.

2D bbox로부터 깊이나 인물 크기를 자동 복원하는 기능은 없다. 카메라/조명·애니메이션·
여러 컷 통합·텍스처 임베딩은 지원 범위 밖이다. 1인 입력은 가능하지만 각 파일에는 골격 1개와
해당 골격에 바인딩된 메시가 있어야 한다. 다시 이 API의 다인 출력을 1인 입력으로 넣을 수 없다.

## 로컬 실행과 검증

API를 기동하지 않고 동일 worker와 검증을 실행할 수 있다. 출력 ZIP이 이미 있으면 덮어쓰지 않는다.

```bash
BLENDER_BINARY=/path/to/blender python scripts/compose_cut_fbx.py \
  --manifest scene.json --fbx person0.fbx --fbx person2.fbx --out cut.zip
python -m pytest -q tests/converter/test_scene_contract.py tests/converter/test_scene_runner.py
```

Blender 통합 회귀는 기존 fixture 생성과 `test_convert.py` 실행 뒤 수행한다.
CI에도 동일 명령을 추가했다.

```bash
export CONVERTER_TEST_ARTIFACT_ROOT=/absolute/path/to/artifacts
/path/to/blender --background --python-exit-code 1 \
  --python tests/converter/make_fixtures.py
/path/to/blender --background --python-exit-code 1 \
  --python tests/converter/test_convert.py
/path/to/blender --background --python-exit-code 1 \
  --python tests/converter/verify_scene_roundtrip.py -- --out /absolute/path/to/verified
```

실제 FBX 두 개로 검사할 때는 `--source a.fbx --source b.fbx`를 추가한다.
1인·2인·동일 원본 반복 3인, 비연속 인물 ID, XYZ 이동·Y축 회전·스케일 차이를 검사한다.
입력 메시/본 위치에 독립 수식을 적용한 값과 최종 FBX를 다시 읽은 값을 비교한다.
worker는 메시 정점, 본 head, 본 부모 관계, 스킨 바인딩 및 오브젝트 구성을 검사하며
월드 좌표 오차가 0.1mm를 넘으면 실패한다.

Three.js를 별도 임시 폴더에 설치한 후 동일 결과를 FBXLoader에서도 검증할 수 있다.

```bash
npm install --prefix /tmp/scene-three three@0.186.1
node tests/converter/verify_scene_three.mjs \
  /tmp/scene-three/node_modules/three a.fbx b.fbx /absolute/path/to/verified
```

2026-10-16 macOS arm64, Blender 5.2.0 LTS `fbe6228777e7`, 실제 65본 캐릭터 검증:

| 샘플 | Blender 최대 오차 | Three.js 0.186.1 최대 오차 |
|---|---|---|
| 1인 | 0.00256mm | 0.00242mm |
| 2인, 다른 포즈·35° 회전·0.75배 | 0.00256mm | 0.00242mm |
| 3인, 원본 반복·비연속 ID·1.2배 | 0.00256mm | 0.00279mm |

실제 multipart API 호출도 HTTP 200, 2인, 출력 SHA 대조와 재임포트 검증을 통과했다.
Three.js는 원본/합성본 모두에서 정점당 가중치 4개 초과 경고를 출력한다. 기존 원본의
특성이며 두 결과의 실제 정점 좌표는 위 오차 이내다. Blender가 기존 terminal bone을 스킨
cluster 배열에 포함해 배열 길이가 달라질 수 있어, Three.js 검증은 실제 본 계층과 스킨 정점
좌표를 비교한다.
