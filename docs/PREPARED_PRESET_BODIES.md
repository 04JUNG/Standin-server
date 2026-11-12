# 기본 체형·프리셋의 중립 체형 사전 준비

## 줄이는 작업과 유지하는 검증

등록된 기본·프리셋의 **포즈 전 중립 FBX와 뼈대 기준 행렬**을 릴리스 준비 단계에서 한 번 만든다.
실제 첫 포즈 생성에서는 중립 체형 생성과 기준 행렬을 읽기 위한 FBX 재임포트를 건너뛴다.
소스 BVH에 포즈를 맞추는 계산, 자세별 피부·접촉 보정, 원본/선택 손, 재질, 출력 범위·카메라 적용,
최종 FBX 저장과 재로드 검증은 기존대로 수행한다. 동결된 retarget/solver·안전 임계값은 바꾸지 않는다.

- 기본·프리셋의 정확한 기본 수치만 대상이다. 명시적으로 기본 수치를 보낸 shape 요청도 같은 원형을 쓴다.
- 사용자가 수치를 변경한 커스텀 체형은 기존 생성 경로를 사용한다. 공용 준비 파일을 추가하거나 디스크에 보관하지 않는다.
- 준비 파일은 작업별 임시 폴더에 사본을 만들어 사용한다. 병렬 요청이 공용 원형을 수정하지 않는다.
- 준비 파일·레지스트리가 없거나 데이터가 손상됐거나 버전이 다르면 기존 중립 체형 생성으로 복구한다.
- 신뢰 경계는 기존의 해시가 고정된 운영자 체형 레지스트리다. HTTP 요청에 준비 파일 경로를 넣을 수 없다.

## 식별·버전·플랫폼

`shape-bake.v2` 레지스트리의 선택적 최상위 `prepared_neutrals`에 캐릭터별 검증 파일 경로와 SHA256을 둔다.
기존 캐릭터 레시피·카탈로그 ID·원본 FBX 해시·체형 수치는 변경하지 않는다.

원형의 key는 원본 asset/Blend/controller SHA, rules·preset·모든 기본 수치, Blender 5.2.0/build,
플랫폼·CPU 아키텍처, 중립 생성/임포트 구현의 해시를 포함한다.
검증 파일과 FBX의 크기·해시, skin-binding 검증, 뼈대 수·유한 행렬을 확인한다.
새 규칙·모델·기본값·런타임에는 새 번들이 필요하다. Mac에서 만든 번들을 Linux에서 그대로 사용하지 않는다.
Linux 배포용 번들은 고정된 Linux/amd64 Converter 이미지 안에서 생성한다. 패키징 호스트는 달라도 된다.

## 생성과 로컬 활성화

명시한 버전 또는 캐릭터만 준비한다. 과거 버전까지 자동으로 전부 생성하지 않는다.
새 출력 폴더가 필요하며 기존 릴리스를 덮어쓰지 않는다. Blender는 표준 고정 버전을 사용한다.

```sh
python scripts/prepare_preset_bodies.py \
  --registry /path/to/shape-bakes.local.json \
  --rules standin-body-0.8.0 --rules standin-female-body-0.4.0 \
  --out /new/prepared-presets --blender /path/to/blender

# 또는 --character male-base-v0-8-0 --character female-base-v0-4-0
# 새 로컬 Converter에서 생성된 레지스트리를 사용한다.
CONVERTER_SHAPE_REGISTRY=/new/prepared-presets/shape-bakes.local.json \
  uvicorn converter_api.app:app --port 8001
```

출력은 `fbx/prepared-<key>.fbx`, `qa/prepared-<key>.json`,
`shape-bakes.local.json`, `results.json`이다. 선택한 모든 항목이 성공했을 때만 레지스트리를 발행한다.
코드에 새 프리셋 이름을 하드코딩하지 않는다. 준비할 데이터의 양은 선택된 프리셋 수에만 비례한다.

기존 자산 릴리스 패키저는 선택적으로 다음을 받는다.

```sh
python scripts/package_shape_staging.py <기존 필수 인수> \
  --prepared-bodies /new/prepared-presets
```

기존 pose QA·소스/asset 검사 이후, 승인 레시피와 일치하는 파일만 복사하고 런타임 경로를 재작성한다.
새 FBX와 검증 JSON은 Converter 전용 role로 기존 경로 allowlist 안에 배포된다. 기존 용량 상한은 그대로다.
준비 명령과 기본 패키징은 업로드·ECS 변경을 수행하지 않는다. 런타임 이미지도 원본 자산을 내장하지 않는다.
번들이 없는 구 레지스트리/구 Converter는 기존 방식대로 동작하므로 배포 순서가 강제되지 않는다.

`CONVERTER_PREPARED_PRESETS_ENABLED=0`이면 준비 파일을 사용하지 않는 복구 경로로 돌아간다.
준비 파일이 등록된 경우 기본값은 1이다. 반환된 `shape_bake.neutral_preparation`은 실제 채택 여부,
소요시간, 생성/재임포트 생략 여부를 기록한다. 실패 이유에 로컬 파일 경로를 노출하지 않는다.

## 실제 실행 결과

스테이징의 고정 해시와 같은 로컬 원본 `standin-body-0.8.0` / `standin-female-body-0.4.0`을 사용했다.
최신 남성 9종·여성 11종, 총 **20종 모두 준비·FBX 재로드 검증 통과**. 준비 FBX 합계 **148.21 MiB**,
오프라인 준비 합계 약 **132.6초**였다. 원본 asset/프리셋 수치/카탈로그 pin은 수정하지 않았다.

완성 FBX·retarget 캐시를 끄고 매번 새 Blender 프로세스에서 처음부터 포즈를 생성했다.
Mac 로컬 5가지 조건을 각각 한 번씩 전후 비교했고, 실행 순서는 번갈아 배치했다.
HTTP/스테이징 지연, PNG, VLM 호출은 포함하지 않는다.

| 체형 | 원본 BVH | 기존 첫 생성 | 사전 준비 사용 | 감소율 |
|---|---|---:|---:|---:|
| male-base-v0-8-0 | ob_001.bvh | 18.03 s | 12.45 s | 30.9% |
| female-base-v0-4-0 | ob_001.bvh | 11.36 s | 8.41 s | 25.9% |
| male-ideal83-v0-8-0 | ob_001.bvh | 20.49 s | 13.65 s | 33.4% |
| male-base-v0-8-0 | ob_015.bvh | 24.56 s | 19.16 s | 22.0% |
| female-base-v0-4-0 | ob_015.bvh | 16.79 s | 13.79 s | 17.9% |

모든 경우 실제 경로가 `prepared`였으며, 최종 FBX를 별도 Blender 프로세스로 다시 읽은 결과
**정점 좌표·뼈대 행렬·메시 행렬·스킨 가중치 차이 0**, 토폴로지도 동일했다. 최종 FBX 검증도 모두 통과했다.
파일 자체의 SHA는 생성 메타데이터 때문에 다를 수 있으며 동일 바이트라고 주장하지 않는다.
`ob_015.bvh`는 앞서 스테이징 Converter 약 50초가 관측된 것과 같은 원본 BVH 해시다.
당시 요청/서버 자원과 다른 로컬 조건이므로, 이 수치로 스테이징 60초가 몇 초가 된다고 환산하지 않는다.

추가 실제 커스텀 smoke: 여성 기본형의 키 +1 요청이 `custom_shape` 경로를 사용했고 요청 수치를 유지했다.
최종 FBX 검증 통과, 공용 준비 항목은 20개 그대로였다.
검증 수치와 모든 대상 목록은 [QA JSON](qa/prepared-presets-20261112/validation.json)에 있다.

재현 명령:

```sh
python scripts/benchmark_prepared_bodies.py \
  --registry /new/prepared-presets/shape-bakes.local.json \
  --characters-root /existing/fbx --character male-base-v0-8-0 \
  --bvh /existing/library/original.bvh --out /new/benchmark --blender /path/to/blender
```

이번 작업은 로컬 구현·준비·검증과 develop 대상 PR까지다. 스테이징/운영의 이미지·자산·설정은 변경하지 않았다.
Linux 실제 20종 준비/동일성 검사와 스테이징 소요시간은 해당 런타임에서 별도 수행해야 한다.

## 운영 적용 (2026-11-12)

운영 로그에서 체형 준비 때마다 `neutral_preparation.status=not_prepared`였다. 운영 Converter 체형
릴리스에 `prepared_neutrals`가 없어서 남성 기본형은 매번 12~26초, 여성 기본형은 6~10초를 중립 체형 생성에 썼다.

- **생성 환경**: 운영 태스크가 실행 중인 이미지(`standin/converter:8170ae1…`, index `sha256:85ca9e3d…`,
  linux/amd64 `sha256:84cb7c76…`). Blender 5.2.0 / `fbe6228777e7`.
  식별값에 들어가는 코드 6개 파일은 `origin/main`과 같다.
- **대상**: `male-base-v0-8-0`, `female-base-v0-4-0`(운영 기본형).
  - 나머지 18종은 그대로 `not_prepared`이고 기존 경로를 쓴다.
  - 20종 전체를 넣으면 체형 설치 상한(768 MiB)을 넘는다.
    기존 릴리스가 754.0 MB이고, 사전 준비 FBX는 남성 약 9.1 MB, 여성 약 6.7 MB다.
- **릴리스**: 기존 Converter 릴리스 아래 하위 릴리스
  `body-models/releases/anatomy-composed-skins-20261112-8149c1568539-converter/presets-ecb22102fd47/manifest.json`
  (SHA256 `ecb22102fd47658f6fed2f6421166dccf83ffa8b866c2e7a686f9bb8abd985cd`).
  - 기존 192개 파일은 경로·크기·SHA256·역할이 그대로다. `shape-bakes.json`에는 `prepared_neutrals`만 추가된다.
  - 추가 파일은 `fbx/prepared-*.fbx` 2개, `qa/prepared-*.json` 2개다. 총 769.9 MB이고 상한까지 35.5 MB가 남는다.
  - 하위 경로를 쓰므로 Converter 태스크 역할의 기존 읽기 권한(`…-converter/*`) 안에 들어간다.
- **생성 명령**: 이미지 안에서 실행한다.
  1. `scripts/prepare_preset_bodies.py --character male-base-v0-8-0 --character female-base-v0-4-0`
  2. `scripts/add_prepared_bodies_to_release.py --release <운영 manifest/shape-bakes 사본> --prepared <1의 출력>`
     — 업로드는 하지 않고, 새 manifest와 `upload-plan.json`을 만든다.
- **검증**:
  - 같은 이미지에서 기본형 2종 × BVH 3개(타격, 누운 자세, 점프 분할)를 사전 준비 사용/미사용으로 변환했다.
    최종 FBX를 독립 재로드해 비교한 결과 6쌍 모두 차이가 0이었다(정점·뼈대 행렬·메시 행렬·가중치).
    skin 검증과 shape 제어값도 같았다.
  - 중립 체형 준비 시간: 로컬 Docker(Windows 호스트)에서 남성 29.6~64.6 s → 1.3~4.4 s, 여성 17.1~29.1 s → 1.0~3.0 s.
  - 운영과 같은 배치(`/app/body-staging`, 60종 전체 레지스트리, 패키지의 실제 경로)에서 다시 확인했다.
    기본형 2종은 `prepared`, 다른 프리셋은 `not_prepared`였고 변환 결과 차이는 0이었다.
  - 운영 절감 예상치: 첫 체형 준비 1회당 남성 약 11~25 s, 여성 약 5~9 s.
    로컬 수치를 운영 수치로 환산하지 않는다. 적용 후 운영 로그로 확인한다.
- **적용**:
  - infra `cdk deploy`를 쓰지 않는다. `cdk.json`의 Converter 이미지가 운영 실행 이미지보다 오래돼, 배포하면 코드가 되돌아간다.
  - 운영 태스크 정의를 복사해 `body-assets-init`의 `BODY_ASSETS_MANIFEST_KEY/SHA256` 두 값만 바꾼 새 리비전으로 서비스를 갱신한다.
    Converter 배포 워크플로가 이미지만 바꾸는 방식과 같다.
  - Converter는 최소 정상 비율이 0%라 교체 중 잠시 중단되고, interactive 메모리 캐시가 비워진다.
  - 롤백: 이전 태스크 정의 리비전으로 `update-service`.
  - 고정값은 네 곳이 같아야 한다.
    1. 운영 태스크 정의
    2. 이 저장소 `config/production-runtime-parity.json` (다음 Converter 배포의 검사 기준)
    3. infra `config/production-body-assets.json`
    4. infra `config/production-runtime-parity.json`
  - 태스크 정의를 바꾼 뒤 이 저장소의 변경이 main에 들어가기 전까지는 Converter 배포가 parity 검사에서 멈춘다(배포 전 실패).
- **재생성 조건**: `converter/{prepared_body,shape_bake,shape_spec,retarget,bone_map,final_artifact}.py`,
  Blender 빌드, 원본 asset/blend/controller, 프리셋 기본값 중 하나라도 바뀌면 식별값이 달라진다.
  이때 런타임은 `invalid_prepared`를 기록하고 안전하게 기존 생성 경로로 돌아간다(속도만 원래대로).
  위 명령으로 새 하위 릴리스를 만들어 다시 고정한다.
- **적용 기록 (2026-11-12 22:42~22:46 KST)**:
  - S3: 운영 버킷 하위 릴리스에 198개 객체를 썼다. 기존 192개는 서버 측 복사, 새 파일 5개와 manifest는 조건부 쓰기로 올렸다.
  - 사전 설치 검증: 운영 이미지에서 태스크 정의의 인라인 설치 스크립트로 197개 파일 해시를 확인했다.
    기본형 2종 `prepared`, `male-muscle-max-v0-8-0`와 `male-base-v0-7-2`는 `not_prepared`.
  - ECS: `StandinAppConverterTask0B8C1552:16` → `:17`. 바뀐 값은 `body-assets-init`의 manifest key/SHA256 두 값뿐이다.
    이미지는 그대로 `8170ae1`.
  - 새 태스크: `body_assets_ready`(version `anatomy-all-skins-outline-20261112-v2-prepared-defaults-v1`, files 197),
    init 종료 코드 0, 컨테이너 HEALTHY.
    마지막 이전 상태 확인(22:42:41)부터 기동 완료(22:44:56)까지 약 2분 15초 중단.
  - 롤백: `aws ecs update-service --task-definition StandinAppConverterTask0B8C1552:16`.
