# 여성 v4의 명시적 SSR2 reference codec

여성 v4의 일부 극단 조합은 PCA 표 보간으로 원본의 국소 면 보정을 안정적으로 재현하지 못했다. 새 `female-reference-v4-zlib` encoding은 승인된 원본의 형상 데이터와 정적인 보정 필드를 저장하고, 고정된 컨트롤러 계산을 클라이언트에서 실행한다. 남성 SSR2와 기존 SSR1/SSR2 보간 reader는 유지한다.

내보내는 여성 모델은 계속 94,901정점이다. 원본 컨트롤러 내부의 reference graph 105,430정점·210,856면에서 계산한 뒤, 원래의 투영을 합성해 50,000정점 미리보기로 옮긴다. source blend·controller·셰이프·리그·가중치·export 형상은 수정하지 않는다. 원본의 정규화, 연산 순서, adjacency 순서와 fold guard를 보존한다.

## 형식과 검증

`SSR2 | u32 headerBytes | JSON | zlib graph` 프레이밍을 사용한다. header의 `encoding=female-reference-v4-zlib`, `dataVersion=1`은 이 고정된 female-v4 규칙과 source/reference-graph SHA만 허용한다. 다운로드 파일은 32MiB, header는 256KiB, 압축 해제 결과는 192MiB 이하이다. 1KiB 입력 단위로 해제하면서 출력 상한을 확인한다. 정점·면·배열 길이·offset·dtype·중복·비유한 값·인덱스·adjacency degree·투영 가중치와 프리셋/컨트롤 전체 coverage를 확인한다. 실행 코드는 자산에 담지 않는다.

현재 고정 후보는 17,392,897 bytes, SHA256 `c417ac1e59b9350d1745a4061e75ecfa35e6389bb44797cfcd3c4053929d7592`이다. decoded 데이터는 174,567,024 bytes이며 다운로드 크기와 구분한다. `build_reference_female_runtime.py`는 source와 입력 카탈로그 SHA를 확인해 생성하고, 생성 자체에 독립 승인 의미를 부여하지 않는다.

`check_reference_runtime.py`는 정확한 최종 카탈로그의 공개 slider grid에서 원래 checker와 같은 preset/few/basic/all 표본을 만들고, 새 Blender 실행의 실제 source 결과를 실제 client parser/evaluator 출력과 비교한다. 판정은 원시 full precision으로 preset≤0.5/few≤5/basic≤10/all≤20mm, 32MiB를 유지한다. 이전 실패 시드 6개는 개발 회귀 검사이며, 별도 새 시드 271828183·577215665로 최종 카탈로그를 검사한다. 결과 JSON은 runtime/catalog/source와 실제 client evaluator SHA를 기록한다.

클라이언트 CLI는 `scripts/check-reference-runtime.ts`를 esbuild의 `--bundle --platform=node --format=esm`으로 컴파일해 실행한다. 이 경로는 계산을 대체하는 Python 근사 구현을 사용하지 않는다.

## 앱 실행과 활성화 순서

큰 데이터의 해제·검증과 실제 변형은 worker에서 실행한다. UI 캐시는 압축 데이터와 작은 preview index만 유지한다. 한 개의 계산 worker가 요청을 직렬화해 썸네일 수만큼 큰 graph가 복제되지 않게 하고, live preview를 썸네일보다 먼저 처리한다. 최신 slider 값보다 오래된 결과와 닫힌 화면의 결과는 그리지 않으며, 유휴 worker는 30초 후 해제한다. worker 실패는 미리보기 실패로 처리한다.

구형 reader는 이 encoding을 거부한다. 활성화 전 호환 클라이언트를 먼저 배포하고, 그 다음 검증된 immutable catalog/manifest pin을 적용해야 한다. 이 작업에서는 운영 앱 릴리스·업데이트 feed·운영 catalog/default·AWS 서비스 활성화를 하지 않는다. 최종 앱 다운로드/CSP·동일 조건 성능·세 역할 installer·필수 리뷰/CI를 별도로 통과해야 하며, 코드/자산 생성 성공을 운영 준비 완료로 간주하지 않는다.
