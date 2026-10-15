# 체형 수치 굽기 · 스테이징 인계 (2026-10-15)

## 확정 사항

- 남성 0.6.7, 여성 0.2.2의 원본 blend·컨트롤러는 변경하지 않는다.
- 사용자 결정: 기본 사진, 체형 비교, 실시간 체형 모델은 **T 포즈**다.
- 실제 포즈 미리보기와 최종 FBX는 같은 체형 수치를 굽고 기존 자세·접촉 보정을 적용한 결과에서 생성한다.
- 남성 8개, 여성 11개 프리셋. 여성 기본 항목 7개, 남성 5개. 여성 지방·지방 볼륨은 서로 다른 조절 항목이다.

## 구현 위치

| 역할 | 워크트리 | 브랜치 |
|---|---|---|
| converter | `/Users/dowon/.codex/worktrees/shape-converter-staging/Standin-server` | `codex/shape-converter-staging` |
| BFF | `/Users/dowon/dev/Standin-app-server-shape-bake-staging` | `codex/shape-bake-staging` |
| 클라이언트 | `/Users/dowon/dev/Standin-client-shape-bake-staging` | `codex/shape-t-pose-staging` |
| 인프라 | `/Users/dowon/dev/Standin-infra-shape-bake-staging` | `codex/shape-runtime-staging` |

클로드 원본 작업 위치는 `Standin-client-shapekey-live`, `Standin-app-server-shapekey-live`다. 해당 원본을 수정하지 않고 최신 여성 런타임·UX 변경을 격리 워크트리에 복사했다. 클라이언트에서 이 작업이 추가한 기능 변경은 T 포즈 카탈로그 허용이며, 나머지 UX·여성 런타임 구현은 클로드 작업 스냅샷이다.

## 게시 릴리스

- 버킷: `standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk`
- manifest: `body-models/releases/shape-20261015-bad69d7b58f2/manifest.json`
- SHA-256: `bad69d7b58f27eaf0d0642c0ec086baed6357d87bfefe0152c7be75085c85f9d`
- 70개 파일 + manifest. 19개 FBX·T 포즈 PNG, 확정 원본 4개, 남녀 SSR 런타임, 설정·검증 자료.
- 이전 차렷 릴리스 `shape-20261015-efefa6523a0f`도 보존한다.
- ECS 서비스나 운영 포인터는 변경하지 않았다.

## 검증

실제 converter API 103개: 19프리셋 × 실제 BVH 5종 + 중간값·최댓값·GLB·후면 8개. 무릎 약90도 앉기와 몸통 약78도 깊은 굽힘도 모든 프리셋에서 실행했다. PNG/GLB는 같은 최종 FBX에서 생성한다.

앱 런타임과 중립 FBX의 19개 전체 메시 비교 최대 오차 0.003914mm. 실제 BFF HTTP 연동 PNG/FBX 4쌍, GLB/FBX 2쌍, 중복 요청 병합을 확인했다.

converter 계약 177개, BFF 363개(2개 skip), 클라이언트 503개 통과. Node26 클라이언트 테스트는 `NODE_OPTIONS=--no-experimental-webstorage npm test`로 실행했다. TypeScript 타입 검사와 새 T 포즈 릴리스의 CDK 합성, 역할별 파일 설치 검증을 통과했다.

실측은 macOS arm64 / Blender5.2.0 환경이다. Linux 컨테이너·CSP 실앱 검수, 전체 포즈 라이브러리 검수와 무관통 인증은 포함하지 않는다. 원본 조형과 웨이트의 기존 한계는 변경하지 않았다.

## 실행 및 배포 연결

`CONVERTER_SHAPE_REGISTRY=/app/body-staging/shape-bakes.json`으로 승인한 원본과 수치를 선택한다. 클라이언트 요청은 원본 파일 경로를 선택할 수 없다. BFF의 `BODY_SHAPE_BAKE_ENABLED`는 기본 false이며 새 코드·에셋이 함께 있을 때만 활성화한다.

일치하는 converter·BFF·클라이언트 코드를 먼저 배포하고, 인프라의 `stagingShapeAssetsEnabled=true`로 새 manifest를 선택한다. 자동 검색용 body catalog는 draft로 유지한다. 기존 기본 릴리스는 opt-in 이전까지 그대로다.

로컬 산출물과 최종 해시 검증: 이 워크트리의 `artifacts/shape-staging/REPORT.html`, `release.json`, `upload-verification.json`, `source-preservation.json`, `SUMMARY.json`.
