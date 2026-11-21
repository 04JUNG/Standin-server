# API 동일 조건 비교 · 36회 / 18쌍

전체 API 경로의 조건별 1회 관찰값입니다. 반복 분산이나 운영 지연 시간으로 확대하지 않습니다. 실제 앱 대표 조건의 3회 반복은 별도 검증이며 현재 16/36회입니다.

같은 실제 BVH·공개 체형값·손·스킨·외곽선·실제 카메라 행렬을 대조했습니다. 기본형·8.3등신 프리셋은 기본 스킨/외곽선0, 큰 커스텀은 살색가이드/외곽선2입니다. source cold는 prepared를 끄며 prepared cold와 같은 결과를 재사용하는 warm을 분리합니다. 큰 커스텀은 prepared 설정을 켜도 실제 neutral 생성 경로가 필요합니다.

로컬 Converter는 Linux/amd64, 2 CPU/4GiB이며 BFF는 ARM64, 0.5 CPU/1GiB입니다. 운영 Converter도 2 CPU/4GiB지만 운영 BFF는1 CPU/2GiB이고 Mac/x86 실행 환경도 다릅니다. 과거 운영 40초대와 직접 대조할 수 없습니다. OS 파일 캐시를 통제하지 않았으며 운영 캐시는 지우지 않았습니다.

단위: 초. file-ready / fsync를 마친 로컬 저장 완료.

| 성별 / 체형 | neutral | cache | 기존 ready / save | 신규 ready / save |
| --- | --- | --- | ---: | ---: |
| female / base | prepared | cold | 50.006 / 51.021 | 52.586 / 53.550 |
| female / base | prepared | warm | 6.218 / 7.035 | 9.922 / 10.943 |
| female / base | source | cold | 58.225 / 59.081 | 61.102 / 62.130 |
| female / custom | prepared | cold | 123.867 / 127.862 | 121.311 / 124.474 |
| female / custom | prepared | warm | 16.733 / 21.057 | 20.220 / 23.884 |
| female / custom | source | cold | 124.386 / 128.191 | 123.625 / 127.430 |
| female / preset | prepared | cold | 52.906 / 53.832 | 53.794 / 55.338 |
| female / preset | prepared | warm | 6.909 / 7.454 | 10.208 / 11.247 |
| female / preset | source | cold | 62.635 / 63.433 | 66.100 / 67.580 |
| male / base | prepared | cold | 69.828 / 70.782 | 56.313 / 57.343 |
| male / base | prepared | warm | 7.517 / 8.343 | 10.498 / 11.527 |
| male / base | source | cold | 81.880 / 82.929 | 68.114 / 69.089 |
| male / custom | prepared | cold | 174.510 / 178.563 | 142.536 / 146.279 |
| male / custom | prepared | warm | 18.716 / 23.526 | 20.116 / 24.136 |
| male / custom | source | cold | 171.425 / 176.416 | 141.603 / 145.510 |
| male / preset | prepared | cold | 70.521 / 71.550 | 58.690 / 59.546 |
| male / preset | prepared | warm | 7.022 / 7.905 | 10.493 / 11.747 |
| male / preset | source | cold | 88.899 / 89.763 | 71.911 / 72.886 |

원시 [wall-clock CSV](performance/api/wall-clock.csv), [중첩 단계 CSV](performance/api/stages.csv), [JSON](performance/api/performance-summary.json). 중첩 worker/BFF/S3 구간은 더하지 않습니다.

[QA 스킨 연결 수정 및 기존 파일 동등성](provenance/performance-approved-skin-setup-amendment.json), [반복 범위 결정](provenance/performance-coverage-decision.json), [운영 자원 읽기 전용 비교](provenance/performance-production-resource-comparison.json). 기본 스킨만 사용한 기존 완료28개 영수증은 스킨/외곽선 후처리 proof가 없고 계측 함수가 동일함을 재확인했습니다. 실패·잠금 중단 시도는 보존하고 완료 표에서 제외했습니다.

API 전체 통과는 실제 앱의 남은 20회 저장이나 main PR 완료를 대신하지 않습니다. 운영 배포·기본값 전환·main 병합·운영 앱 릴리스는 하지 않았습니다.
