# 새 기본 모델의 스테이징 승격 조건

기존 모델·카드·작업의 ID와 파일을 보존하면서 남성 9종과 여성 11종을 새 ID로 추가한다. 기본형·프리셋·슬라이더 미리보기·최종 FBX가 같은 새 source와 asset SHA를 사용해야 한다. 원본 전달용 S3 업로드는 런타임 승격 증거가 아니다.

## 공개 슬라이더 범위의 미리보기 검증

`check_sampled_runtime.py --catalog <body-models.json> --catalog-sha256 <SHA>`는 카탈로그의 정확한 source pin과 runtime preset을 대조한다. 모든 runtime preset이 중복 없이 있어야 하며, 각 프리셋에 실제 공개된 슬라이더 범위와 step으로 독립 검사를 수행한다. 카탈로그 pin·source·프리셋·기본값·키·범위·step이 다르면 거절한다. 최종 입력 검사는 카탈로그의 runtime SHA·크기·형식·규칙도 실제 입력과 대조한다. preview 정점·면 개수와 인덱스는 최종 source에서 직접 재검사한다. 기존 preset 0.5 / few 5 / basic 10 / all 20 mm와 32 MiB 기준은 그대로이며, 통과 여부는 반올림 전 수치로 판단한다.

생성기의 학습·가지치기용 결과와 독립 검사는 별개다. 최종 SSR2의 실제 해시·source·카탈로그 해시와 독립 결과를 함께 남긴다. 기존 SSR2의 승인 값을 새 source에 복사하지 않는다.

## 기존 파일을 보존하는 설치 예산

명시적 `budgetProfile: lod-defaults-20.v1`은 Converter에만 일반 자산 1,280 MiB와 prepared 자산 192 MiB, 합계 1,472 MiB를 허용한다. 이전 모델·authoring source·UV 파일을 삭제하지 않고 새 20종을 추가하기 위한 유한 상한이다. 일반 릴리스 768 MiB와 기존 `prepared-neutral-20.v1`의 768+192 MiB 정책은 유지한다.

파일당 300 MiB, prepared FBX 64 MiB / receipt 256 KiB / 최대 22쌍, 경로·역할·SHA·manifest 64 KiB 검사는 그대로다. BFF와 Inference는 확장 예산을 사용할 수 없다. 실제 최종 파일 목록의 크기를 검증하고 같은 Linux 설치기로 설치한 뒤 prepared identity를 재검사해야 한다. manifest와 배포 pin 변경은 별도 검토 대상이다.

설치기는 파일 게시 전에 실제 `converter.prepared_body.read_prepared`로 source·character SHA·preset controls·코드 지문·Blender build·Linux platform·artifact receipt를 검증한다. registry와 파일 목록에 없는 prepared 파일이나 source 경로는 거절한다. 설치용 Converter 이미지에는 이 검증 모듈이 있어야 한다.

2026-11-21 로컬 후보 검증에서는 새 prepared 20개와 과거 남녀 기본형 2개를 보존한 280개 파일을 Linux/amd64에서 설치했다. manifest는 48,482 bytes, 전체 파일은 1,294,660,307 bytes이며 `preparedIdentitiesVerified=22`를 확인했다. 고정 이미지 자체의 코드로 ECS와 같은 `python -c` 실행을 재현했다. 이 후보는 S3에 게시하거나 활성화하지 않았다.

## 모델 QA와 배포의 경계

최초 남성 v5는 공개 커스텀 조합에서 승인 원본에 없는 뒤집힌 면이 확인됐다. 원본 전달 파일은 유지하며, 면 수정 후보는 별도 ID·source SHA·규칙 버전으로 다룬다. 면 수정에도 정점·셰이프키·웨이트·리그·손의 불변식과 기존 표면 오차 기준을 적용한다. 후보를 생성하거나 로컬 수치 검사를 통과한 사실만으로 승인을 주장하지 않는다.

최종 후보에는 실제 Linux Converter, 최종 FBX 독립 재로드, 스킨·외곽선, CSP 저장·재열기, 전체 프리셋 SSR2 독립 검사, 같은 조건의 시간 비교와 필수 코드 리뷰가 필요하다. 로컬 직접 파일 검증과 API 검증, 실제 네이티브 앱 관찰은 구분한다. 모든 필수 게이트가 통과한 뒤 최신 main에서 필요한 변경만 별도 PR로 옮긴다. 현재 코드 변경만으로 자산·기본 ID·S3 pin·ECS task를 활성화하지 않으며 main merge·운영 배포·운영 앱 게시를 수행하지 않는다.
