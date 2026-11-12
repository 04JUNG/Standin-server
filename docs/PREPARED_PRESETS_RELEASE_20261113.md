# 사용자 노출 기본·프리셋 20종의 prepared neutral 릴리스

기존 `prepare_preset_bodies.py` / `converter.prepared_body`를 사용해 포즈 전 몸·리그·기준 행렬만 준비한다. 포즈별 완성 파일 캐시를 추가하지 않는다. solver, 포즈 보정, 최종 검증 임계값, 원본 자산, 기본 모델, 카탈로그 pin, 스킨·외곽선은 변경하지 않는다.

## 대상과 원본

승인된 `body-models.json`의 `role=default|preset` 항목을 registry와 대조한다. 과거 버전은 카탈로그에 있어도 노출 역할이 없으므로 포함하지 않는다. 이름/버전 문자열을 추정해 선택하지 않는다.

- 카탈로그 SHA256: `76ff5d66d13097839293fe8ae7464fae9a22463081582b27246a4b2d4344b635`.
- 남성 9 / 여성 11, 합계 20종과 character ID 중복 없음, 원본 FBX·Blend·controller 해시 및 모든 control 기본값의 일치를 요구한다.
- 원본 Converter 릴리스 manifest SHA256: `8149c15685398ba279550a9887fa8d05b5208e26b60264e6fa55a6cc20a09771`, 193 files / 754,044,218 bytes. 로컬 193개 모두 실제 크기·SHA를 확인했다.
- 기존 두 기본형도 같은 승인 레시피로 포함한다. 입력 릴리스에 이전 prepared 두 쌍이 있으면 파일 경로·역할·크기·해시를 유지하고 현재 런타임 index만 확장한다. 이미 같은 key가 있으면 다시 추가하지 않는다. 이전 런타임의 두 쌍을 보존할 수 있도록 최대 22쌍을 허용한다.
- 참고용 Mac `prepared-defaults-current` 원본 6개 파일은 변경하지 않았다. Mac 준비 파일을 Linux 번들에 재사용하지 않았다.

## 768 MiB 때문에 누락됐던 18종

기존 전체 릴리스 설치기는 768 MiB가 상한이다. 20종 FBX만 약 148.21 MiB이므로 원본 릴리스와 합치면 이를 넘는다. 생성 성공만으로 배포 가능한 상태가 되지는 않는다.

명시적 manifest `budgetProfile=prepared-neutral-20.v1`에만 다음 유한 예산을 적용한다.

| 항목 | 상한 |
|---|---:|
| 기존 일반 자산 | 768 MiB (유지) |
| `fbx/prepared-<64hex>.fbx` + `qa/prepared-<64hex>.json` | 합계 192 MiB |
| 전체 | 960 MiB |
| 준비 FBX / receipt 1개 | 64 MiB / 256 KiB |
| 고유 FBX·receipt 쌍 | 22 이하 (현재 20 + 보존 가능한 이전 기본형 2) |

프로필이 없으면 기존 768 MiB 그대로다. 이 프로필은 Converter role만 허용한다. hash 이름·동일 key의 FBX/receipt 쌍·정확한 Converter role·기존 경로 allowlist·파일별 SHA·크기·manifest 64 KiB 제한을 모두 검사한다. 임의 상한 환경변수나 해시 검증 우회는 없다.

기존 `add_prepared_bodies_to_release.py`는 main에서 가져온 의존성을 별도 커밋으로 분리했다. 새 패키저는 기존 `stage_prepared`를 재사용한다. 승인 manifest pin을 확인하고 registry의 `prepared_neutrals` 외 레시피 변경을 거부한다. Linux/amd64 외 준비 파일, 누락된 프리셋, 손상된 receipt/FBX, 중복 경로를 거부한다.

실제 배포 설치기는 인프라 `scripts/body-assets-init.py`가 ECS task command에 인라인으로 들어간다. 서버 스크립트만 바꾸면 이 큰 번들은 여전히 거부된다. 따라서 인프라의 대응 PR이 배포 의존성이다. 인프라 PR은 설치기·검증·문서만 변경하며 #90의 BFF 자원 변경과 섞지 않는다. 릴리스 pin, task definition, ECS, S3는 이 작업에서 변경하지 않았다.

설치기는 파일을 한 번에 하나씩 읽는다. 기존 300 MiB 일반 파일 한도와 init memory reservation 128 MiB는 그대로이며, body assets가 있는 Converter task의 기존 예산은 2 vCPU / 4 GiB다. 새 볼륨 설치 상한은 960 MiB, 기존 설치와 새 staging이 함께 있는 재설치의 body 자산 최악치는 1,920 MiB (1.875 GiB)다. Converter task는 별도 ephemeral storage 확장을 지정하지 않는다. Fargate Linux 1.4+ 기본 20 GiB에서 이미지 압축·해제 크기와 다른 볼륨도 차감해야 한다([AWS 저장소 문서](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-task-storage.html)). 이 변경은 body 자산 상한만 제한하며 전체 task의 자유 공간을 실측했다고 주장하지 않는다.

## 고정 런타임과 브랜치 분리

실제 검증은 Mac arm64 호스트의 **Linux/amd64 컨테이너 안**에서 Blender `5.2.0` / build `fbe6228777e7`로 수행했다. 로컬 기본 이미지 ID는 `sha256:497d46fa92ade673b0b1f943cf2aa9e6290d90973ec074427a0aa785de9ebef4`이며 해당 작업 트리를 read-only로 mount했다. 컨테이너 network는 `none`, Blender 실행은 순차, CPU 한도는 4다. 네이티브 Linux 서버 성능이나 운영 지연으로 환산하지 않는다.

`develop`과 `main`은 현재 중립 준비 식별값에 포함되는 `shape_bake.py`가 다르다. 관련 없는 main 코드를 develop PR에 병합하지 않고 각각 별도 번들을 생성한다.

| 기준 코드 | neutral 구현 SHA256 |
|---|---|
| develop `e9de78d6c` + 기능 PR | `2b2595e06ba460b840059be52e2b0f6f63b21c855f25b33b29061b30a9c4996d` |
| main `fa5919798` + 기능 PR | `143fac771bd45decc8bfbb973c13ba9830a53d9d32e207212815c92beee8937d` |

이 해시는 기존 `identity()`가 계산하는 6개 converter 파일 해시다. 실제 배포 이미지에서도 Blender/build, OS/architecture, 이 구현 해시, 자산·기본값이 모두 같아야 채택된다. 다른 이미지나 두 브랜치 사이에서 그대로 사용한다고 주장하지 않는다. 달라지면 `invalid_prepared`로 기존 안전 생성에 복귀하며, 승인할 최종 이미지 안에서 다시 준비해야 한다.

## 생성·설치·벤치 재현

다음은 컨테이너 내부 명령이다. 대상 브랜치 작업 트리는 `/work:ro`, 승인 원본 릴리스는 `/app/body-staging:ro`, 승인 카탈로그는 `/catalog.json:ro`, 새 출력은 `/output:rw`로 mount하고 `/work`에서 실행한다. `--platform linux/amd64 --network none --cpus 4 --entrypoint python`을 사용한다. `Dockerfile.converter`가 고정한 Blender archive SHA256은 `96f6c181a30f4950607839dc84d42a354b250d8a0231b098b59b7bc69c351c48`다.

```sh
python scripts/prepare_preset_bodies.py \
  --registry /app/body-staging/shape-bakes.json \
  --catalog /catalog.json \
  --catalog-sha256 76ff5d66d13097839293fe8ae7464fae9a22463081582b27246a4b2d4344b635 \
  --out /output/prepared --blender /opt/blender/blender

python scripts/add_prepared_bodies_to_release.py \
  --release /app/body-staging \
  --release-sha256 8149c15685398ba279550a9887fa8d05b5208e26b60264e6fa55a6cc20a09771 \
  --prepared /output/prepared --version prepared-all20-20261113-v1 \
  --catalog /catalog.json \
  --catalog-sha256 76ff5d66d13097839293fe8ae7464fae9a22463081582b27246a4b2d4344b635 \
  --budget-profile prepared-neutral-20.v1 --out /output/release
```

설치는 별도 새 컨테이너에서 수행한다. 승인 원본은 `/input-base:ro`, **비어 있는 새 로컬 설치 폴더**를 `/app/body-staging:rw`로 mount한다. 다음 명령은 네트워크 대신 로컬 base/new 파일을 실제 설치기에 공급하고, 실제 배포 경로에서 20종 모두 `restore(...).status=prepared`인지 검사한다.

```sh
python scripts/verify_prepared_release.py \
  --base /input-base --release /output/release \
  --catalog /catalog.json \
  --catalog-sha256 76ff5d66d13097839293fe8ae7464fae9a22463081582b27246a4b2d4344b635 \
  --result /output/install-verification.json
```

설치가 끝난 폴더를 `/app/body-staging:ro`로 바꿔 다음을 실행한다. BVH는 원본 파일을 read-only로 mount한다. 완성 FBX·retarget 캐시를 끄고 요청마다 새 Blender를 사용한다. 순서를 번갈아 실행하며 최종 FBX 두 개를 독립 Blender 프로세스에서 재로드해 topology·정점·bones·행렬·가중치를 비교한다. 기존 비교 임계값은 변경하지 않는다.

```sh
python scripts/benchmark_prepared_bodies.py \
  --registry /app/body-staging/shape-bakes.json \
  --characters-root /app/body-staging/fbx \
  --catalog /catalog.json \
  --catalog-sha256 76ff5d66d13097839293fe8ae7464fae9a22463081582b27246a4b2d4344b635 \
  --bvh /ob_015.bvh --out /output/benchmark-ob015 --blender /opt/blender/blender

# 대표 프리셋은 실제 승인 목록에서 골라 --character를 반복한다.
python scripts/benchmark_prepared_http.py \
  --assets /app/body-staging --bvh /ob_015.bvh \
  --character male-base-v0-8-0 --character male-fat-max-v0-8-0 \
  --character male-muscle-max-v0-8-0 --character female-base-v0-4-0 \
  --character female-fat-max-v0-4-0 --character female-muscle-max-v0-4-0 \
  --out /output/http-ob015 --port 19971
```

HTTP 검사는 각 조건마다 신규 API 프로세스·private/cache/character 폴더를 사용한다. `/convert-framed`의 `preview_format=none`, 손 없음, full scope다. 응답의 실제 `neutral_preparation`과 HTTP POST부터 FBX decode·SHA 검증·디스크 저장까지의 file-ready를 분리한다. 같은 BVH·body의 baseline/prepared를 다시 독립 FBX 재로드 비교한다. BFF·네이티브 UI·PNG·스킨 선택 후처리·네트워크 운영 성능은 이 측정에 포함하지 않는다. OS 파일 캐시는 통제하지 않았고 조건별 1회 표본이다.

## 이후 승인된 배포에서 사용할 준비물

1. 서버와 대응 인프라 PR을 각각 최소 1명 리뷰 후 통합한다. 해당 브랜치에 맞는 최종 Converter 이미지에서 위 생성·검증을 다시 수행해 runtime 식별값을 고정한다.
2. 생성된 `upload-plan.json`의 `copyUnchanged`는 기존 객체의 경로·크기·SHA·role 그대로 새 immutable prefix로 복사한다. `putNew`만 조건부 새 객체로 쓰고 모든 SHA를 재확인한 뒤 `manifest.json`을 마지막에 발행한다. 새 prefix는 기존 Converter 읽기 권한 안의 승인된 경로를 사용한다.
3. 예산 프로필을 이해하는 **인프라 인라인 initializer**와 manifest key/SHA를 함께 검토한다. 이미지·카탈로그·스킨 pin을 임의로 바꾸지 않는다. 다음 배포가 검사하는 서버/인프라 parity pin도 같은 승인 계획에 포함한다.
4. 새 task의 initializer 성공, 20종 `prepared`, 커스텀 `custom_shape`, 실제 FBX 검증을 확인한다. 기존 task definition과 manifest pin을 보존한다.
5. rollback은 이전 task definition/manifest pin으로 되돌린다. `CONVERTER_PREPARED_PRESETS_ENABLED=0`도 기존 안전 생성으로 복귀하는 경로다. 새 FBX를 이전 registry가 참조하지 않으므로 파일을 삭제할 필요가 없다.

이 작업은 로컬 번들과 새 PR까지다. S3 쓰기, ECS/SSM 변경, main/develop 병합, 운영·스테이징 배포, 릴리스를 실행하지 않는다. 큰 FBX/Blend, 사용자 파일, 비밀 env/token은 Git에 넣지 않는다.

## 승인된 대상 목록

| 캐릭터 | 분류 | 역할 |
|---|---|---|
| `male-base-v0-8-0` | male | default |
| `male-ideal83-v0-8-0` | male | preset |
| `male-ideal70-v0-8-0` | male | preset |
| `male-toon64-v0-8-0` | male | preset |
| `male-muscular-v0-8-0` | male | preset |
| `male-muscle-max-v0-8-0` | male | preset |
| `male-bulk-v0-8-0` | male | preset |
| `male-soft-v0-8-0` | male | preset |
| `male-fat-max-v0-8-0` | male | preset |
| `female-base-v0-4-0` | female | default |
| `female-slim-v0-4-0` | female | preset |
| `female-ideal83-v0-4-0` | female | preset |
| `female-ideal70-v0-4-0` | female | preset |
| `female-toon64-v0-4-0` | female | preset |
| `female-muscular-v0-4-0` | female | preset |
| `female-muscle-max-v0-4-0` | female | preset |
| `female-soft-v0-4-0` | female | preset |
| `female-fat-max-v0-4-0` | female | preset |
| `female-new-fat-v0-4-0` | female | preset |
| `female-new-muscular-v0-4-0` | female | preset |

각각의 원본 해시·정확한 기본값은 [scope JSON](qa/prepared-all20-20261113/catalog-scope.json), Linux key/runtime/크기/중립 검증은 [develop 설치 결과](qa/prepared-all20-20261113/develop-install.json)에 기록했다. develop 20종의 ob_015 독립 비교도 모두 통과했고 모든 정점·뼈대·메시 행렬·가중치 최대 차이는 0이다([결과](qa/prepared-all20-20261113/develop-ob015.json)). 오프라인 준비 합계는 199.512초이며 사용자 요청 시간과 분리했다. main 전체 검증과 대표 HTTP 검사는 진행 중이다.
