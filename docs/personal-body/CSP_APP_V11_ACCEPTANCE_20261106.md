# v11 앱 실제 저장본 4개 — CSP import 수용 (2026-11-06)

새 QA 앱의 실제 picker 등록 → 기록 BVH 적용 → 제품 저장으로 만든 남녀 identity/yaw35 FBX 네 개 모두 CSP 5.1.1 / macOS 26.6.2 arm64에서 **가져오기·표시 통과**했다. CSP 조작자는 전달된 원본과 바이트가 같은 사본을 새 QA 캔버스에 각각 가져왔으며 입력·사본·사진 SHA를 전후 확인했다. 새 crash log는 없었다. 이전 25회 원인 분리 실험과 별도 stage다.

앱 source `35e0136cf9cd230b18f0772116bd955f4228bf68`, 연결 통합 `bed1d85`, binary `265d37b474084916da59ecf6638228aed3b0f7f924b0473f29ec0351fe85b5f7`, 앱 tree `3fe201675c9f551066ef4a9f7cbfcc45506de3dae5e2645cdb502e38b049ef3d`를 고정했다. CSP 담당은 실행 전과 후 binary·전체 앱 6,610개 파일·내장 runtime 6,606개 파일을 독립 재검증했다. 앱 생산 코드와 publisher head의 일치는 연결 담당의 diff 증거이며 CSP 담당이 앱을 재빌드한 것은 아니다.

`personal-converter/11`, contract revision 6, runtime manifest `2540a161e19fa34ad1d9c38bcc79751d471b938f6de5b0757923e303def5789b`, fingerprint `5fdc61c7775279392171e8002aece0e87744fe6187855746fe7411b4004458d1`이다. 양 Mock flag는 false지만 BVH는 허용된 기록 자료를 제공하는 통제된 loopback fixture다. 운영 live BFF나 실제 추론 카메라 수용으로 확대하지 않는다.

| 실제 앱 저장본 | FBX SHA256 | CSP |
|---|---|---|
| 남성 identity | `04961228c2acd4532adc4066b0962bd402f2d51f1380dfd226f447d7a67500ea` | 표시 |
| 남성 yaw35 | `6ffe3b3b79e61725f0145919ff6328c15266af98b40ef8f5fa69d4d76e99864c` | 표시 |
| 여성 yaw35 | `e9a7500df3fed7f1e27b1e92918599cefc67424dc08d89354b6a29a1c9b9021d` | 표시 |
| 여성 identity | `016211a2800c6e1da70213e991ff30680de3fcade5cffe9408bf67b8d607d9f6` | 표시 |

남성은 최대 9개, 여성은 최대 8개의 nonzero influences를 실제 파일에서 다시 확인했다. 남성의 `root_group_` Model 이름 별칭만 역치환하면 각 출력의 기록된 preimage SHA와 일치했다. `weightsTruncated:false`이며 여성 파일에는 별칭 변환이 없다. 입력 모델·BVH·카메라·출력·사진의 전체 대응은 [증거 JSON](csp-app-v11-acceptance-20261106.json)에 있다. [원인 역전 실험과 최소 수정](CSP_CRASH_20261106.md)의 범위를 유지한다.

## 별도 결과와 한계

- **조형 품질은 여전히 not-qualified**다. 원래 골반/관절 결함이 보이며 importer 성공이 자동 보정의 품질 합격이 아니다.
- 네 import 후 QA 여성 identity 캔버스에서 오브젝트 drag를 실제 시도했으나 CUA가 `noWindowsAvailable`을 반환했다. 오브젝트 이동·회전은 **미검증**이며 CSP 기능 실패나 성공으로 단정하지 않는다.
- 같은 캔버스에서 실제 Save As를 실행하자 라이선스 취득/등록 요구 안내가 나타났다. **아니오**로 닫았다. `.clip` 저장은 **라이선스 차단**, 새 저장본 재열기는 **미검증**이다. 계정·라이선스·결제 변경은 없다. 내부 관절 재편집은 현 FBX 계약 밖이다.
- 기존 문서를 닫거나 편집하지 않았다. 독점 UI lock과 로컬 복구 사본을 사용했다. raw FBX/모델/decoder/runtime, 원본 crash log와 사진은 Git에 추가하지 않았다. 사진은 승인된 공용 로컬 acceptance 폴더에 SHA와 함께 보관한다.

Draft / default-off / `releaseReady:false`다. 운영·스테이징 배포, 활성화, 병합, 원격 PR 직접 push는 하지 않았다. 서명/배포권·Windows/Intel 실기·생산 BFF는 이 검사의 합격 항목이 아니다.
